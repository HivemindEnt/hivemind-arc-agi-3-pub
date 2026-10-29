"""FI-939 DUCK 38ep: loadguard - keep a scored rerun alive through a slow model load. Robustness only, no policy change.

Why: 3 of our last 4 hidden draws (10-22, 10-24, 10-25) scored 0.00 with byte-identical 3,411-byte submissions,
across three different notebook versions. The competition forum (discussion 751598, 2026-10-21..25) reports the
same signature from many M2-derived teams. Kaggle staff confirmed on 10-23 that "the dataset cache was unstable in
some regions", causing slow model reads. Teams measured weight loads of 35-63 minutes, where they used to take
minutes. M2 v3's startup path then loses the run:
  1. the launcher waits only SERVER_STARTUP_TIMEOUT = 12 min from notebook start, then releases the benchmark;
  2. one game sends a warmup RESET (ARC3_WARMUP_ACTION_GAMES=1) so the run has an action inside 15 minutes;
  3. every game then calls the analyzer. The first request retries for up to ARC3_HTTP_RETRY_INITIAL_SECONDS
     (900 s), and after that each game gives up after ARC3_MAX_ANALYZER_FAILURES (10) failures in a row, and
  4. the gateway closes a scorecard that sees no action for 15 minutes. Later actions are refused, every game
     ends with nothing, and the run is COMPLETE at ~30 min with 0.00.
The forum reports match this: COMPLETE about 30 minutes after submission instead of ~9 h.

What this does, on the scored rerun only:
  - Before a game's first analyzer call, HOLD it until the model server's /health returns 200. Waiting is
    cheap: no analyzer failures are burned, and the hold counts against the game's own runtime budget
    (started_at is set when the session is created), so the 9 h notebook limit is not extended.
  - While games are held, ONE of them (the first to arrive, the "leader") sends a RESET on its own game at
    once and then every KEEPALIVE_S (7 min), so the scorecard never idles for 15 minutes. The RESET goes
    through the same path as M2's own warmup RESET.
  - The hold is capped at HOLD_CAP_S (150 min) from notebook start. A server that never comes up still ends
    the run as before instead of idling for 9 hours.
A healthy run (server ready before the benchmark starts) is unchanged: hold() returns at once.

Every OS or clock dependency is injected, so test_loadguard.py drives it with fakes. install() patches
inference.framework.solver._HarnessGameSession._play_inner, and a test pins that against the real harness.
"""
import threading
import time
import urllib.request

HOLD_CAP_S = 150 * 60
KEEPALIVE_S = 7 * 60
POLL_S = 5.0


def _health_ok(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class LoadGuard:
    def __init__(self, health_url, notebook_start, *, health=None, clock=time.time, wait=None, say=print,
                 hold_cap_s=HOLD_CAP_S, keepalive_s=KEEPALIVE_S, poll_s=POLL_S):
        self.health_url, self.notebook_start = health_url, float(notebook_start)
        self._health = health or _health_ok
        self._clock, self._say = clock, say
        self.hold_cap_s, self.keepalive_s, self.poll_s = hold_cap_s, keepalive_s, poll_s
        self.ready = threading.Event()
        self._wait = wait or self.ready.wait  # wait(seconds); returns early when ready is set
        self._lock = threading.Lock()
        self._leader = None
        self.keepalives = 0
        self.held = 0
        self.ready_at = None
        self.capped = False

    def cap_at(self):
        return self.notebook_start + self.hold_cap_s

    def poll_once(self):
        if self.ready.is_set():
            return True
        try:
            ok = self._health(self.health_url)
        except Exception:  # a probe error reads as not ready, never as a crash
            ok = False
        if ok:
            with self._lock:
                if not self.ready.is_set():
                    self.ready_at = self._clock() - self.notebook_start
                    self.ready.set()
                    self._say("[LOADGUARD] model server ready at %ds from notebook start; %d game(s) were held, "
                              "%d keepalive RESET(s) sent" % (self.ready_at, self.held, self.keepalives), flush=True)
            return True
        return False

    def hold(self, session, reset):
        """Run on a game thread before the game's first analyzer call. Returns 'immediate', 'ready' or 'cap'."""
        if self.poll_once():
            return "immediate"
        with self._lock:
            leader = self._leader is None
            if leader:
                self._leader = id(session)
            self.held += 1
        last = None
        if leader:
            self._say("[LOADGUARD] server still loading at %ds; holding games, keepalive RESET every %ds"
                      % (self._clock() - self.notebook_start, self.keepalive_s), flush=True)
        while True:
            if self.poll_once():
                return "ready"
            now = self._clock()
            if now >= self.cap_at():
                with self._lock:
                    if not self.capped:
                        self.capped = True
                        self._say("[LOADGUARD] hold cap %ds reached with the server not ready; releasing games"
                                  % self.hold_cap_s, flush=True)
                return "cap"
            if leader and (last is None or now - last >= self.keepalive_s):
                try:
                    reset()
                    self.keepalives += 1
                except Exception as e:  # a keepalive must never take the game down
                    self._say("[LOADGUARD] keepalive RESET failed: %r" % (e,), flush=True)
                last = now
            self._wait(self.poll_s)


def install(g, session_cls=None, **kw):
    """Patch the harness from the notebook's globals (needs SERVED_MODEL_PORT and NOTEBOOK_START_TIME)."""
    for k in ("SERVED_MODEL_PORT", "NOTEBOOK_START_TIME"):
        if k not in g:
            raise RuntimeError("[LOADGUARD] notebook global %r missing: launcher cell changed" % k)
    if session_cls is None:
        import inference.framework.solver as S
        session_cls = S._HarnessGameSession
    orig = session_cls._play_inner
    if getattr(orig, "__loadguard__", False):
        return "[LOADGUARD] already installed"
    for name in ("_execute_auto_reset",):
        if not callable(getattr(session_cls, name, None)):
            raise RuntimeError("[LOADGUARD] %s.%s missing: harness changed" % (session_cls.__name__, name))
    guard = LoadGuard("http://127.0.0.1:%s/health" % g["SERVED_MODEL_PORT"], g["NOTEBOOK_START_TIME"], **kw)

    def _play_inner(self):
        try:
            guard.hold(self, self._execute_auto_reset)
        except Exception as e:  # the guard must never be what ends a game
            guard._say("[LOADGUARD] hold error, continuing unguarded: %r" % (e,), flush=True)
        return orig(self)

    _play_inner.__loadguard__ = True
    _play_inner.__wrapped__ = orig
    session_cls._play_inner = _play_inner
    g["_LOADGUARD"] = guard
    state = "ready" if guard.poll_once() else "still loading"
    return ("[LOADGUARD] installed (DUCK 38ep): server %s at %ds; hold cap %ds from notebook start, keepalive "
            "every %ds" % (state, guard._clock() - guard.notebook_start, guard.hold_cap_s, guard.keepalive_s))
