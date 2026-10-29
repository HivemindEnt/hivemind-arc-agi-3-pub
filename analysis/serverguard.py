"""FI-939 DUCK 38ed: model-server guard for the M2 notebook. Robustness only, no policy change.

Why: the standing default scored 0.00 on one hidden draw out of 11 (10-22) with no code change. The notebook's
startup loop notices a dead server (proc.poll() is not None), prints the log tail, BREAKS and runs the benchmark
with no model. Nothing relaunches it later either. This guard:
  1. at install time: if the server process has already exited, kill its process group, relaunch it with the
     same argv/env, and wait for /health (bounded);
  2. afterwards: a daemon thread polls the process every POLL_S seconds and relaunches it the same way if it
     exits mid-run. Requests made while it reloads fail and the harness treats them as it treats any failed call.
It acts ONLY on process exit. A live-but-slow server is never touched: under load /health can lag, and killing
a busy server would turn a slow turn into a lost one. At most MAX_RELAUNCH relaunches per run.

Everything that touches the OS is injected, so test_serverguard.py drives it with fakes.
"""
import os
import signal
import subprocess
import threading
import time
import urllib.request

POLL_S = 20
MAX_RELAUNCH = 2
READY_TIMEOUT_S = 900


def _health_ok(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _kill_group(proc):
    # The notebook launches with start_new_session=True, so the server's pgid is its pid. SGLang leaves scheduler
    # and detokenizer children; if they survive the parent they keep the GPU memory and the port.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass


class ServerGuard:
    def __init__(self, proc, argv, env, log_path, health_url, *, popen=None, health=None, kill=None,
                 sleep=time.sleep, clock=time.monotonic, say=print, max_relaunch=MAX_RELAUNCH,
                 ready_timeout=READY_TIMEOUT_S, poll_s=POLL_S):
        self.proc, self.argv, self.env, self.log_path, self.health_url = proc, list(argv), env, log_path, health_url
        self._popen = popen or self._default_popen
        self._health = health or _health_ok
        self._kill = kill or _kill_group
        self._sleep, self._clock, self._say = sleep, clock, say
        self.max_relaunch, self.ready_timeout, self.poll_s = max_relaunch, ready_timeout, poll_s
        self.relaunches = 0
        self.events = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _default_popen(self):
        logf = open(self.log_path, "ab", buffering=0)
        return subprocess.Popen(self.argv, env=self.env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)

    def dead(self):
        return self.proc is None or self.proc.poll() is not None

    def wait_ready(self):
        t0 = self._clock()
        while self._clock() - t0 < self.ready_timeout:
            if self.dead():
                return "died"
            if self._health(self.health_url):
                return "ready"
            self._sleep(3)
        return "timeout"

    def relaunch(self, why):
        with self._lock:
            if not self.dead():
                return "alive"
            if self.relaunches >= self.max_relaunch:
                self.events.append(("exhausted", why))
                return "exhausted"
            code = None if self.proc is None else self.proc.poll()
            if self.proc is not None:
                self._kill(self.proc)
            self.relaunches += 1
            self._say("[SERVERGUARD] model server exited (code %s, %s); relaunch %d/%d"
                      % (code, why, self.relaunches, self.max_relaunch), flush=True)
            self.proc = self._popen()
        r = self.wait_ready()
        self.events.append(("relaunch", why, r))
        self._say("[SERVERGUARD] relaunch %d -> %s" % (self.relaunches, r), flush=True)
        return r

    def check_once(self, why="watchdog"):
        """One watchdog tick. Relaunches until the server is up or the budget is spent."""
        r = "alive"
        while self.dead():
            r = self.relaunch(why)
            if r in ("ready", "exhausted", "timeout", "alive"):
                break
        return r

    def start(self):
        def loop():
            while not self._stop.is_set():
                try:
                    self.check_once()
                except Exception as e:  # never let the guard take the run down
                    self._say("[SERVERGUARD] watchdog error: %r" % (e,), flush=True)
                self._stop.wait(self.poll_s)
        t = threading.Thread(target=loop, name="serverguard", daemon=True)
        t.start()
        return t

    def stop(self):
        self._stop.set()


def install(g, **kw):
    """Install from the notebook's globals: needs proc, args, env, LOG, SERVED_MODEL_PORT (all set by cell 5)."""
    for k in ("proc", "args", "env", "LOG", "SERVED_MODEL_PORT"):
        if k not in g:
            raise RuntimeError("[SERVERGUARD] notebook global %r missing: server cell changed" % k)
    url = "http://127.0.0.1:%s/health" % g["SERVED_MODEL_PORT"]
    guard = ServerGuard(g["proc"], g["args"], g["env"], g["LOG"], url, **kw)
    first = guard.check_once("dead before benchmark") if guard.dead() else "alive"
    g["proc"] = guard.proc
    guard.start()
    g["_SERVERGUARD"] = guard
    return "[SERVERGUARD] active (DUCK 38ed): server at install = %s; watchdog every %ss, max %d relaunches" % (
        first, guard.poll_s, guard.max_relaunch)
