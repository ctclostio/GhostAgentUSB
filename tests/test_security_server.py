"""The model server is locked down: a new API key for every start (handed over in the environment,
never on the command line or in a log), browsers may call it only from localhost pages, no web chat
page or /slots unless asked for, and none of the PC's LLAMA_* / AIP_* variables reach it. Playwright's
driver gets a cleaned environment too.

The real llama-server is only started from the CPU build without a model (no GPU, nothing loaded)."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

import helpers
from agent_support import AGENT, StubBrowser, call, quiet, read_jsonl, run_cli, scratch_paths, settings


REAL_POPEN = subprocess.Popen  # the tests below replace subprocess.Popen while the agent starts its server


class Proc:
    """Popen's stand-in when nothing should really start."""
    _handle = 0

    def poll(self):
        return None


class ServerStartTests(unittest.TestCase):
    def start_server(self, test_settings, popen=None, environ=None):
        """LlamaServer.start (CPU, no GPU probe, no model files): returns (args, Popen keywords, server, output)."""
        folder = scratch_paths(self)
        exe = folder / "bin" / "cpu" / "llama-server.exe"
        exe.parent.mkdir(parents=True, exist_ok=True)
        launched = []

        def record(args, **kw):
            launched.append((args, kw))
            return popen(args, **kw) if popen else Proc()

        server = AGENT.LlamaServer(test_settings)
        with mock.patch.object(AGENT, "pick_backend", return_value=("cpu", exe, None)), \
                mock.patch.object(AGENT, "model_files", return_value=[Path("m.gguf"), Path("p.gguf")]), \
                mock.patch.object(AGENT.subprocess, "Popen", record), \
                mock.patch.object(AGENT, "port_is_free", return_value=True), \
                mock.patch.dict(os.environ, environ or {}), quiet() as out:
            server.start()
        self.assertEqual(len(launched), 1)
        args, kw = launched[0]
        return args, kw, server, out.getvalue()

    def test_locked_down_by_default(self):
        environ = {"LLAMA_ARG_TOOLS": "all", "llama_arg_webui": "1", "LLAMA_SERVER_SLOTS_DEBUG": "1",
                   "AIP_MODE": "1", "AIP_HTTP_PORT": "9", "LLAMA_API_KEY": "the-pc-s-own-key", "GGML_SOMETHING": "kept"}
        args, kw, server, out = self.start_server(settings(), environ=environ)
        self.assertEqual(args[args.index("--cors-origins") + 1], "localhost")
        self.assertIn("--no-slots", args)
        self.assertIn("--no-webui", args)
        self.assertNotIn("--api-key", args)
        self.assertFalse(any(server.api_key in a for a in args))  # the key is not on the command line
        self.assertGreaterEqual(len(server.api_key), 32)
        env = kw["env"]
        self.assertEqual(env["LLAMA_API_KEY"], server.api_key)
        for name in ("LLAMA_ARG_TOOLS", "llama_arg_webui", "LLAMA_SERVER_SLOTS_DEBUG", "AIP_MODE", "AIP_HTTP_PORT"):
            self.assertNotIn(name, env)
        self.assertEqual(env["GGML_SOMETHING"], "kept")
        self.assertEqual(env.get("PATH"), os.environ.get("PATH"))
        log = server.log_path.read_text(encoding="utf-8")
        self.assertNotIn(server.api_key, log)
        self.assertNotIn(server.api_key, out)
        first_key = server.api_key
        _, _, server, _ = self.start_server(settings())
        self.assertNotEqual(server.api_key, first_key)  # a new one on every start

    def test_web_ui_on_request(self):
        args, kw, server, out = self.start_server(settings(web_ui=True))
        self.assertNotIn("--no-webui", args)
        self.assertIn("--cors-origins", args)
        self.assertIn(server.api_key, out)  # shown in the console, for the chat page
        self.assertNotIn(server.api_key, server.log_path.read_text(encoding="utf-8"))

    def test_own_key_and_loosening_flags_in_extra_server_args(self):
        args, kw, server, out = self.start_server(settings(extra_server_args=["--api-key", "my-key,second", "--tools", "all"]))
        self.assertEqual(server.api_key, "my-key")  # the server takes the command line's key, so the agent does too
        self.assertIn("extra_server_args contains --api-key", out)
        self.assertIn("extra_server_args contains --tools", out)
        log = server.log_path.read_text(encoding="utf-8")
        self.assertNotIn("my-key", log)
        self.assertIn("--api-key <hidden>", log)

        key_file = helpers.temp_dir(self) / "key.txt"
        key_file.write_text("# a comment\n\nfile-key\n", encoding="utf-8")
        _, _, server, out = self.start_server(settings(extra_server_args=[f"--api-key-file={key_file}"]))
        self.assertEqual(server.api_key, "file-key")
        self.assertIn("--api-key-file", out)

    def test_the_log_line_hides_keys_and_the_home_folder(self):
        home = str(Path.home())
        line = AGENT.loggable_command(["x.exe", "--api-key", "secret1", "--api-key=secret2", "-m", home + "\\m.gguf"])
        self.assertNotIn("secret1", line)
        self.assertNotIn("secret2", line)
        self.assertNotIn(home, line)
        self.assertIn("%USERPROFILE%\\m.gguf", line)

    def test_server_environment(self):
        with mock.patch.dict(os.environ, {"LLAMA_ARG_MODEL": "x", "AIP_MODE": "1", "KEEP_ME": "1"}):
            env = AGENT.server_environment("k")
            self.assertEqual((env.get("LLAMA_API_KEY"), env.get("KEEP_ME")), ("k", "1"))
            self.assertFalse(any(k.upper().startswith(("LLAMA_ARG", "AIP_")) for k in env))
            self.assertNotIn("LLAMA_API_KEY", AGENT.server_environment(None))

    def test_a_stopped_servers_log_names_no_account(self):
        with helpers.isolated_env():
            home = str(Path.home())
            server = AGENT.LlamaServer(settings())
            server.log_path = helpers.temp_dir(self) / "server.log"
            server.log_path.write_text(f"llama_model_loader: loaded from {home}\\AppData\\Local\\GhostAgent\\models\\m.gguf\n",
                                       encoding="utf-8")
            server.stop()
            text = server.log_path.read_text(encoding="utf-8")
        self.assertNotIn(home, text)
        self.assertIn("from %USERPROFILE%\\AppData\\Local\\GhostAgent", text)

    def test_the_server_tail_skips_expected_noise(self):
        server = AGENT.LlamaServer(settings())
        server.log_path = helpers.temp_dir(self) / "server.log"
        noise = "slot update: find_slot: non-consecutive token position 12 after 7"
        server.log_path.write_text("\n".join(["start", noise, "real problem", noise]) + "\n", encoding="utf-8")
        self.assertEqual(server.tail(5), ["start", "real problem"])
        self.assertEqual(server.log_path.read_text(encoding="utf-8").count("non-consecutive"), 2)  # the file keeps them

    def test_the_agent_talks_to_its_server_with_the_key(self):
        """The real command line, started as fake_llama (which reads LLAMA_API_KEY like llama-server)."""
        def popen(args, **kw):
            cmd = [str(helpers.PYTHON), "-I", str(helpers.BENCH / "fake_llama.py")] + [str(a) for a in args[1:]]
            proc = REAL_POPEN(cmd, stdout=kw["stdout"], stderr=kw["stderr"], env=kw["env"],
                              creationflags=kw["creationflags"])
            self.addCleanup(proc.wait, 10)
            self.addCleanup(proc.kill)
            return proc

        args, kw, server, _ = self.start_server(settings(port=helpers.free_port()), popen=popen)
        self.assertTrue(helpers.wait_until(lambda: AGENT.server_healthy(server.base), timeout=20))  # /health is public
        chat = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
        self.assertEqual(helpers.sse_request(server.base, chat)["status"], 401)
        self.assertEqual(helpers.sse_request(server.base, chat, api_key="wrong")["status"], 401)
        self.assertEqual(helpers.sse_request(server.base, chat, api_key=server.api_key)["status"], 200)

        browser = StubBrowser()
        agent = AGENT.Agent(settings(), server, browser)
        with quiet():
            self.assertEqual(agent.run("Say hello."), "Done (fake model).")  # fake_llama's reply when it has no script
        server.stop()


class ExternalServerTests(unittest.TestCase):
    def setUp(self):
        scratch_paths(self)

    def test_only_the_configured_key_is_sent(self):
        self.assertIsNone(AGENT.ExternalServer("http://127.0.0.1:1").api_key)
        self.assertIsNone(AGENT.ExternalServer("http://127.0.0.1:1", "").api_key)
        self.assertEqual(AGENT.ExternalServer("http://127.0.0.1:1", "abc").api_key, "abc")

        fake = helpers.start_fake_llama(self, script=[{"tool_calls": [call("answer", content="hello")]}], api_key="abc")
        agent = AGENT.Agent(settings(), AGENT.ExternalServer(fake.base, "abc"), StubBrowser())
        with quiet():
            self.assertEqual(agent.run("Say hello."), "hello")
        chats = [r for r in fake.requests if r["path"].endswith("/chat/completions")]
        self.assertEqual(chats[-1]["headers"].get("Authorization"), "Bearer abc")

        fake = helpers.start_fake_llama(self, script=[{"tool_calls": [call("answer", content="hello")]}])
        agent = AGENT.Agent(settings(), AGENT.ExternalServer(fake.base), StubBrowser())
        with quiet():
            agent.run("Say hello.")
        chats = [r for r in fake.requests if r["path"].endswith("/chat/completions")]
        self.assertNotIn("Authorization", chats[-1]["headers"])

    def test_the_setting_reaches_the_server_from_the_command_line(self):
        fake = helpers.start_fake_llama(self, script=[{"tool_calls": [call("answer", content="keyed answer")]}],
                                        api_key="cli-key")
        d = helpers.temp_dir(self)
        override = {"external_server": fake.base, "external_server_key": "cli-key", "start_url": "about:blank",
                    "settle_seconds": 0.1}
        r = run_cli(["--headless-browser", "--data-dir", d, "--task", "Say something.", "--results", d / "r.jsonl",
                     "--settings-override", json.dumps(override)], timeout=120,
                    env=dict(os.environ, TEMP=str(d), TMP=str(d)))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(read_jsonl(d / "r.jsonl")[0]["answer"], "keyed answer")
        self.assertNotIn("cli-key", r.stdout + r.stderr)


class PlaywrightEnvironmentTests(unittest.TestCase):
    def test_the_driver_gets_a_clean_environment(self):
        with mock.patch.dict(os.environ, {"NODE_OPTIONS": "--require evil.js", "PWDEBUG": "1",
                                          "PLAYWRIGHT_BROWSERS_PATH": "C:\\elsewhere", "playwright_driver_x": "1",
                                          "PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS": "1", "KEEP_ME": "1"}):
            AGENT._scrub_playwright_environment()
            for name in ("NODE_OPTIONS", "PWDEBUG", "playwright_driver_x", "PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS"):
                self.assertNotIn(name, os.environ)
            self.assertEqual(os.environ["PLAYWRIGHT_BROWSERS_PATH"], str(AGENT.ROOT / "browsers"))
            self.assertEqual(os.environ["KEEP_ME"], "1")

    def test_the_browsers_folder_is_always_the_one_on_the_drive(self):
        code = ("import os, sys; sys.path.insert(0, sys.argv[1]); import ghost_agent; "
                "print(os.environ['PLAYWRIGHT_BROWSERS_PATH'])")
        r = helpers.run_python(["-B", "-c", code, helpers.AGENT_DIR],  # -B: no __pycache__ next to the agent
                               env=dict(os.environ, PLAYWRIGHT_BROWSERS_PATH="C:\\elsewhere"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(Path(r.stdout.strip()), helpers.REPO / "GhostAgent" / "browsers")


CPU_SERVER = helpers.REPO / "GhostAgent" / "bin" / "cpu" / "llama-server.exe"


def cpu_only_build():
    """True when bin\\cpu holds a llama-server without any GPU backend (so starting it cannot touch a GPU)."""
    if not CPU_SERVER.is_file():
        return False
    return not any(p.name.lower().startswith(("ggml-cuda", "ggml-vulkan", "ggml-hip", "ggml-sycl", "ggml-opencl"))
                   for p in CPU_SERVER.parent.iterdir())


@unittest.skipUnless(cpu_only_build(), "needs the CPU build of llama-server in GhostAgent\\bin\\cpu")
class RealServerTests(unittest.TestCase):
    """llama-server itself honours the lock-down. Started in router mode with an empty models folder and
    no automatic loading: it serves its HTTP endpoints but loads no model."""

    def test_the_cpu_build_honours_the_lock_down(self):
        d = Path(tempfile.mkdtemp(prefix="llama-", dir=helpers.TMP))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        (d / "models").mkdir()
        port = helpers.free_port()
        key = "test-key-" + os.urandom(8).hex()
        with mock.patch.dict(os.environ, {"LLAMA_ARG_TOOLS": "all", "LLAMA_ARG_WEBUI": "1"}):
            env = AGENT.server_environment(key)
        env["LLAMA_CACHE"] = str(d / "models")  # keeps the router away from any real model cache on this PC
        log_path = d / "server.log"
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen([str(CPU_SERVER), "--host", "127.0.0.1", "--port", str(port),
                                     "--models-dir", str(d / "models"), "--no-models-autoload",
                                     "--cors-origins", "localhost", "--no-slots", "--no-webui"],
                                    stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(CPU_SERVER.parent),
                                    creationflags=helpers.CREATE_NO_WINDOW)
        try:
            base = f"http://127.0.0.1:{port}"
            self.assertTrue(helpers.wait_until(lambda: AGENT.server_healthy(base), timeout=30), log_path.read_text())
            chat = {"model": "none", "messages": [{"role": "user", "content": "hi"}]}
            status, _, _ = helpers.http_request("POST", base + "/v1/chat/completions", chat)
            self.assertEqual(status, 401)
            status, _, _ = helpers.http_request("POST", base + "/v1/chat/completions", chat,
                                                headers={"Authorization": "Bearer " + key})
            self.assertNotIn(status, (401, 403))  # past the key check (no model is loaded, so not a 200)
            auth = {"Authorization": "Bearer " + key}
            status, headers, _ = helpers.http_request("GET", base + "/v1/models",
                                                      headers=dict(auth, Origin="https://evil.example"))
            self.assertNotIn("access-control-allow-origin", {k.lower() for k in headers})
            status, headers, _ = helpers.http_request("OPTIONS", base + "/v1/chat/completions", headers={
                "Origin": "https://evil.example", "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type"})
            self.assertNotIn("access-control-allow-origin", {k.lower() for k in headers})
            status, _, _ = helpers.http_request("GET", base + "/", headers=auth)
            self.assertEqual(status, 404)  # no web chat page
            status, _, _ = helpers.http_request("GET", base + "/tools", headers=auth)
            self.assertNotEqual(status, 200)  # LLAMA_ARG_TOOLS did not reach it
        finally:
            proc.terminate()
            proc.wait(15)
        self.assertNotIn(key, log_path.read_text(encoding="utf-8", errors="replace"))


if __name__ == "__main__":
    unittest.main()
