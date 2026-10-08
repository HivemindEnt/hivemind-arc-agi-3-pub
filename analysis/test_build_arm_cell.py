"""Tests for build_arm_cell.py (FI-939 DUCK 38ar). Offline: fake notebooks and fake harness modules."""
import json, os, sys, tempfile, types, unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_arm_cell as bac  # noqa: E402


def nb(cells):
    return {"cells": [{"cell_type": t, "source": s, "metadata": {}, "outputs": [], "execution_count": None}
                      for t, s in cells], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


BASE = [("markdown", "# title"), ("code", "import os\nENV = {}"), ("code", bac.BENCH_MARKER + "\nrun()"),
        ("code", "show()")]


def fake_harness(prompt="OLD"):
    pkg = types.ModuleType("inference"); agent = types.ModuleType("inference.agent")
    ta = types.ModuleType("inference.agent.tool_agent"); pr = types.ModuleType("inference.agent.prompts")
    ta.LEVEL_START_USER_PROMPT = prompt; pr.LEVEL_START_USER_PROMPT = prompt
    pkg.agent = agent; agent.tool_agent = ta; agent.prompts = pr
    mods = {"inference": pkg, "inference.agent": agent, "inference.agent.tool_agent": ta, "inference.agent.prompts": pr}
    return mods, ta, pr


class Insert(unittest.TestCase):
    def test_inserts_before_benchmark(self):
        n = nb(BASE)
        at = bac.insert_cell(n, bac.CELLS["unanchor"])
        self.assertEqual(at, 2)
        self.assertTrue(bac.src_of(n["cells"][2]).startswith("# --- FI-939 arm unanchor"))
        self.assertTrue(bac.src_of(n["cells"][3]).startswith(bac.BENCH_MARKER))
        self.assertEqual(len(n["cells"]), 5)

    def test_refuses_duplicate(self):
        n = nb(BASE)
        bac.insert_cell(n, bac.CELLS["unanchor"])
        with self.assertRaises(ValueError):
            bac.insert_cell(n, bac.CELLS["unanchor"])

    def test_refuses_missing_or_repeated_marker(self):
        with self.assertRaises(ValueError):
            bac.insert_cell(nb([("code", "x")]), bac.CELLS["unanchor"])
        with self.assertRaises(ValueError):
            bac.insert_cell(nb([("code", bac.BENCH_MARKER), ("code", bac.BENCH_MARKER + " ")]), bac.CELLS["unanchor"])


class CellBehaviour(unittest.TestCase):
    def run_cell(self, env_on=True, prompt="OLD"):
        mods, ta, pr = fake_harness(prompt)
        saved = {k: sys.modules.get(k) for k in mods}
        old_env = os.environ.get("ARC3_LEVEL_TRANSFER_GUIDANCE")
        try:
            sys.modules.update(mods)
            if env_on:
                os.environ["ARC3_LEVEL_TRANSFER_GUIDANCE"] = "1"
            else:
                os.environ.pop("ARC3_LEVEL_TRANSFER_GUIDANCE", None)
            exec(compile(bac.CELLS["unanchor"], "cell", "exec"), {})
            return ta, pr
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            if old_env is None:
                os.environ.pop("ARC3_LEVEL_TRANSFER_GUIDANCE", None)
            else:
                os.environ["ARC3_LEVEL_TRANSFER_GUIDANCE"] = old_env

    def test_patches_both_modules(self):
        ta, pr = self.run_cell()
        self.assertEqual(ta.LEVEL_START_USER_PROMPT, bac.UNANCHOR_PROMPT)
        self.assertEqual(pr.LEVEL_START_USER_PROMPT, bac.UNANCHOR_PROMPT)

    def test_refuses_when_guidance_off(self):
        with self.assertRaises(RuntimeError):
            self.run_cell(env_on=False)

    def test_refuses_when_harness_changed(self):
        with self.assertRaises(RuntimeError):
            self.run_cell(prompt=None)

    def test_prompt_content(self):
        p = bac.UNANCHOR_PROMPT
        self.assertNotIn("do not rediscover", p)
        for must in ("Win attribution", "two different explanations", "Re-test", "one alternative", "current_frame"):
            self.assertIn(must, p)
        self.assertNotIn("—", p)


class Build(unittest.TestCase):
    def test_build_writes_arm(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src"); os.makedirs(src)
            json.dump(nb(BASE), open(os.path.join(src, "base.ipynb"), "w"))
            json.dump({"id": "hivemindadmin/duck-franzen-m2-loadsim", "title": "x", "code_file": "base.ipynb",
                       "docker_image": "img"}, open(os.path.join(src, "kernel-metadata.json"), "w"))
            out, at = bac.build("unanchor", src, d, "unanchor-loadsim")
            meta = json.load(open(os.path.join(out, "kernel-metadata.json")))
            self.assertEqual(meta["id"], "hivemindadmin/duck-franzen-m2-unanchor-loadsim")
            self.assertEqual(meta["docker_image"], "img")
            built = json.load(open(os.path.join(out, meta["code_file"])))
            self.assertEqual(len(built["cells"]), 5)
            with self.assertRaises(ValueError):
                bac.build("nope", src, d, "x")


class CtrlprobeCell(unittest.TestCase):
    BOOT = "import json\ndef main():\n    runtime_globals = {}\n    def action(a):\n        return {}\n    runtime_globals[\"action\"] = action\n    return runtime_globals\n"

    def run_cell(self, boot):
        sb = types.ModuleType("inference.agent.python_tool_sandbox"); sb._SANDBOX_BOOTSTRAP = boot
        ta = types.ModuleType("inference.agent.tool_agent"); ta._build_system_prompt = lambda **k: "BASE"
        pkg = types.ModuleType("inference"); agent = types.ModuleType("inference.agent")
        pkg.agent = agent; agent.python_tool_sandbox = sb; agent.tool_agent = ta
        mods = {"inference": pkg, "inference.agent": agent, "inference.agent.python_tool_sandbox": sb,
                "inference.agent.tool_agent": ta}
        saved = {k: sys.modules.get(k) for k in mods}
        try:
            sys.modules.update(mods)
            exec(compile(bac.cell_for("ctrlprobe"), "cell", "exec"), {})
            return sb, ta
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_injects_probe_into_bootstrap_main(self):
        sb, ta = self.run_cell(self.BOOT)
        ns = {}
        exec(compile(sb._SANDBOX_BOOTSTRAP, "boot", "exec"), ns)
        rg = ns["main"]()
        self.assertTrue(callable(rg["probe_controls"]))
        self.assertTrue(ta._build_system_prompt(tool_output_tokens=1).startswith("BASE"))
        self.assertIn("probe_controls(", ta._build_system_prompt(tool_output_tokens=1))

    def test_v2_passes_stale_state_when_the_sandbox_has_it(self):
        boot = ("import json\ndef main():\n    runtime_globals = {}\n    _stale_state: list[str] = ['UP']\n"
                "    def action(a):\n        return {}\n    runtime_globals[\"action\"] = action\n"
                "    return runtime_globals, _stale_state\n")
        sb, _ = self.run_cell(boot)
        self.assertIn("make_probe(runtime_globals, action, _stale_state)", sb._SANDBOX_BOOTSTRAP)
        ns = {}
        exec(compile(sb._SANDBOX_BOOTSTRAP, "boot", "exec"), ns)
        rg, stale = ns["main"]()
        rg["valid_actions"] = ["UP"]
        rg["probe_controls"](verbose=False, clicks=False)
        self.assertEqual(stale, [])  # cleared by the probe before its action
        sb1, _ = self.run_cell(self.BOOT)
        self.assertIn("make_probe(runtime_globals, action)\n", sb1._SANDBOX_BOOTSTRAP)

    def test_v2_against_the_real_m2_bootstrap(self):
        import ast, textwrap
        path = os.path.expanduser("~/m2/arc-agi-3-solution/ARC3-Inference/inference/agent/python_tool_sandbox.py")
        if not os.path.exists(path):
            self.skipTest("M2 source not present on this host")
        tree = ast.parse(open(path, encoding="utf-8").read())
        boot = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_SANDBOX_BOOTSTRAP" for t in node.targets):
                boot = [textwrap.dedent(c.args[0].value) for c in ast.walk(node.value) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "dedent"][0]  # chained .replace() fill-ins are not needed here
        self.assertIsNotNone(boot)
        sb, _ = self.run_cell(boot)
        out = sb._SANDBOX_BOOTSTRAP
        line = [l for l in out.split("\n") if "make_probe(runtime_globals, action" in l and "probe_controls" in l]
        self.assertEqual(len(line), 1)
        self.assertIn("_stale_state", line[0])
        ind = lambda l: len(l) - len(l.lstrip())
        stale_decl = [l for l in out.split("\n") if l.strip().startswith("_stale_state: list")][0]
        self.assertEqual(ind(line[0]), ind(stale_decl))  # same scope as the closure variable

    def test_ctrlprobe2_on_demand_guidance(self):
        c2 = bac.cell_for("ctrlprobe2")
        self.assertIn("def make_probe(", c2)
        self.assertIn("ctrlprobe2", c2)
        self.assertIn("Do not call it routinely", bac.CTRLPROBE2_GUIDANCE)
        self.assertNotIn("start of level 1", bac.CTRLPROBE2_GUIDANCE)
        self.assertIn("start of level 1", bac.CTRLPROBE_GUIDANCE)  # v1 unchanged
        self.assertNotIn("\u2014", bac.CTRLPROBE2_GUIDANCE)
        boot = ("import json\ndef main():\n    runtime_globals = {}\n    _stale_state: list[str] = []\n"
                "    def action(a):\n        return {}\n    runtime_globals[\"action\"] = action\n    return runtime_globals\n")
        sb = types.ModuleType("inference.agent.python_tool_sandbox"); sb._SANDBOX_BOOTSTRAP = boot
        ta = types.ModuleType("inference.agent.tool_agent"); ta._build_system_prompt = lambda **k: "BASE"
        pkg = types.ModuleType("inference"); agent = types.ModuleType("inference.agent")
        pkg.agent = agent; agent.python_tool_sandbox = sb; agent.tool_agent = ta
        mods = {"inference": pkg, "inference.agent": agent, "inference.agent.python_tool_sandbox": sb,
                "inference.agent.tool_agent": ta}
        saved = {k: sys.modules.get(k) for k in mods}
        try:
            sys.modules.update(mods)
            exec(compile(c2, "cell", "exec"), {})
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        self.assertIn("only when you are stuck", ta._build_system_prompt())
        self.assertIn("_stale_state)", sb._SANDBOX_BOOTSTRAP)

    def test_refuses_when_anchor_missing_or_repeated(self):
        with self.assertRaises(RuntimeError):
            self.run_cell("def main():\n    pass\n")
        twice = self.BOOT + "def other():\n    runtime_globals[\"action\"] = action\n"
        with self.assertRaises(RuntimeError):
            self.run_cell(twice)

    def test_inventory_cell_sets_knob(self):
        import sys, types, os as _os
        pkg = types.ModuleType("inference"); ag = types.ModuleType("inference.agent"); ta = types.ModuleType("inference.agent.tool_agent")
        ta._level_inventory_enabled = lambda: _os.environ.get("ARC3_LEVEL_INVENTORY", "").lower() in ("1", "true")
        saved = {k: sys.modules.get(k) for k in ("inference", "inference.agent", "inference.agent.tool_agent")}
        sys.modules.update({"inference": pkg, "inference.agent": ag, "inference.agent.tool_agent": ta})
        pkg.agent = ag; ag.tool_agent = ta
        old_env = _os.environ.pop("ARC3_LEVEL_INVENTORY", None)
        try:
            exec(compile(bac.cell_for("inventory"), "cell", "exec"), {})
            self.assertEqual(_os.environ.get("ARC3_LEVEL_INVENTORY"), "1")
            del ta._level_inventory_enabled
            with self.assertRaises(RuntimeError):
                exec(compile(bac.cell_for("inventory"), "cell", "exec"), {})
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            if old_env is None:
                _os.environ.pop("ARC3_LEVEL_INVENTORY", None)
            else:
                _os.environ["ARC3_LEVEL_INVENTORY"] = old_env

    def test_inventory_cell_marker_and_no_em_dash(self):
        c = bac.cell_for("inventory")
        self.assertIn("[INVENTORY]", c)
        self.assertNotIn("\u2014", c)

    def test_cell_embeds_current_source(self):
        cell = bac.cell_for("ctrlprobe")
        self.assertIn("def make_probe(", cell)
        self.assertIn("[CTRLPROBE]", cell)
        self.assertNotIn("\u2014", bac.CTRLPROBE_GUIDANCE)



class TestKnobs6AndBypass(unittest.TestCase):
    """DUCK 38cg"""

    def _fake(self):
        import types
        ta = types.ModuleType("inference.agent.tool_agent")
        env = os.environ
        ta._AUTO_FRAME_DIFF = False
        ta._AUTO_FRAME_DIFF_BUDGET = 0
        ta._get_env_bool = lambda k, d=False: env.get(k, "").strip().lower() in ("1", "true", "yes", "on")
        ta._repeat_state_guard_enabled = lambda: ta._get_env_bool("ARC3_REPEAT_STATE_GUARD", False)
        ta._level_inventory_enabled = lambda: ta._get_env_bool("ARC3_LEVEL_INVENTORY", False)
        sv = types.ModuleType("inference.framework.solver")
        sv.GUARDS_FROM_LEVEL = 2
        ta._guards_active = lambda level: int(level) >= max(1, int(env.get("ARC3_GUARDS_FROM_LEVEL", "1") or 1))
        pkg = types.ModuleType("inference"); ag = types.ModuleType("inference.agent"); fw = types.ModuleType("inference.framework")
        pkg.agent, pkg.framework, ag.tool_agent, fw.solver = ag, fw, ta, sv
        return {"inference": pkg, "inference.agent": ag, "inference.framework": fw,
                "inference.agent.tool_agent": ta, "inference.framework.solver": sv}

    def test_knobs6_cell_turns_all_switches_on(self):
        mods = self._fake()
        saved_env = dict(os.environ)
        with unittest.mock.patch.dict(sys.modules, mods):
            try:
                exec(compile(bac.cell_for("knobs6"), "knobs6", "exec"), {})
                ta, sv = mods["inference.agent.tool_agent"], mods["inference.framework.solver"]
                self.assertTrue(ta._AUTO_FRAME_DIFF)
                self.assertEqual(ta._AUTO_FRAME_DIFF_BUDGET, 300)
                self.assertEqual(sv.GUARDS_FROM_LEVEL, 1)
                self.assertTrue(ta._repeat_state_guard_enabled() and ta._level_inventory_enabled())
            finally:
                os.environ.clear(); os.environ.update(saved_env)

    def test_knobs6_refuses_on_changed_harness(self):
        mods = self._fake()
        del mods["inference.framework.solver"].GUARDS_FROM_LEVEL
        saved_env = dict(os.environ)
        with unittest.mock.patch.dict(sys.modules, mods):
            try:
                with self.assertRaises(RuntimeError):
                    exec(compile(bac.cell_for("knobs6"), "knobs6", "exec"), {})
            finally:
                os.environ.clear(); os.environ.update(saved_env)

    def test_build_with_bypass_on_real_notebook(self):
        src = os.path.expanduser("~/m2/kaggle-franzen-m2-loadsim")
        if not os.path.isdir(src):
            self.skipTest("no M2 loadsim source here")
        root = tempfile.mkdtemp()
        out, at = bac.build("knobs6", src, root, "knobs6-test", bypass=True)
        meta = json.load(open(os.path.join(out, "kernel-metadata.json")))
        nb = json.load(open(os.path.join(out, meta["code_file"]), encoding="utf-8"))
        text = json.dumps(nb)
        self.assertIn("[KNOBS6]", text)
        self.assertIn("FI-939 commit bypass (DUCK 38cg)", text)

if __name__ == "__main__":
    unittest.main()
