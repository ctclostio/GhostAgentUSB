"""Unattended runs tell a model server that could not be started apart from other failures by exit
code 2, also when the start fails outright (a blocked program, no usable port) instead of the
server exiting while it loads. No browser, no model: stand-ins for both."""

import unittest

from agent_support import AGENT, quiet, scratch_paths, settings


class _Browser:
    def __init__(self):
        self.started = False
        self.closed = False

    def start(self, start_url=None):
        self.started = True

    def close(self):
        self.closed = True


class _FailingServer:
    """A model server whose start raises SystemExit, the way LlamaServer does for a blocked program."""

    def __init__(self, where, message):
        self.where = where
        self.message = message
        self.stopped = False
        self.explicit_layout = False

    def begin(self, interactive=True):
        if self.where == "begin":
            raise SystemExit(self.message)
        return "cpu"

    def start(self, auto_fit=False, quiet=False):
        return "cpu"

    def wait_ready(self):
        if self.where == "wait":
            raise SystemExit(self.message)
        return 1.0

    def describe(self):
        return "stand-in"

    def stop(self):
        self.stopped = True


class ServerStartFailureTests(unittest.TestCase):
    def run_tasks(self, server):
        folder = scratch_paths(self)
        opts = AGENT.parse_args(["--task", "one", "--results", str(folder / "r.jsonl")])
        browser = _Browser()
        with quiet() as out:
            code = AGENT.run_tasks(opts, settings(), server, browser, AGENT.load_tasks(opts))
        return code, out.getvalue(), browser, folder

    def test_a_start_that_fails_outright_means_exit_code_2(self):
        msg = "Windows App Control (Smart App Control or your organization's policy) blocked llama-server.exe."
        server = _FailingServer("begin", msg)
        code, out, browser, folder = self.run_tasks(server)
        self.assertEqual(code, 2)
        self.assertIn("blocked llama-server.exe", out)
        self.assertIn("no task was run", out)
        self.assertFalse(browser.started)  # the browser is not started for nothing
        self.assertTrue(server.stopped and browser.closed)
        self.assertFalse((folder / "r.jsonl").exists())

    def test_a_failure_while_waiting_for_the_server_means_exit_code_2(self):
        server = _FailingServer("wait", "The model server could not be started.")
        code, out, browser, _ = self.run_tasks(server)
        self.assertEqual(code, 2)
        self.assertIn("could not be started", out)
        self.assertTrue(server.stopped and browser.closed)


if __name__ == "__main__":
    unittest.main()
