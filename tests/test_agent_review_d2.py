"""Regression tests for problems found in review of the agent loop changes (CS-07, CS-15, CS-17,
CS-18, CS-27, CS-29): an answer whose content is null, a trim that only removed old reasoning, the
summary of a pasted multi-line task, the "page did not move" hint under tool_schema "h", and a
conversation whose browser changed between tasks. Fake model, stub browser, no window, no GPU."""

import unittest

import test_agent_loop as loop  # (the module only: its tests are not collected twice)
from agent_support import (AGENT, StubBrowser, assert_requests_valid, call, make_agent, quiet, read_jsonl,
                           reset_stop, scratch_paths, settings, steps_file)


def run(agent, task):
    with quiet():
        return agent.run(task)


class ReviewFixTests(unittest.TestCase):
    def setUp(self):
        reset_stop(self)

    def test_a_null_answer_falls_back_to_the_message(self):
        script = [{"content": "The ticket costs 12 dollars.", "tool_calls": [call("answer", content=None)]}]
        agent, fake, folder = make_agent(self, script)
        self.assertEqual(run(agent, "price?"), "The ticket costs 12 dollars.")
        script = [{"content": "", "tool_calls": [call("answer", content=None)]}]
        agent, fake, folder = make_agent(self, script)
        self.assertEqual(run(agent, "price?"), "")  # never the word "None"

    def test_removing_old_reasoning_alone_brings_no_summary(self):
        script = [{"reasoning": f"thinking {n}", "tool_calls": [call("scroll_web", direction="down")]} for n in range(4)]
        script += [{"tool_calls": [call("answer", content="first done")]},
                   {"tool_calls": [call("answer", content="second done")]},
                   {"tool_calls": [call("answer", content="third done")]}]
        agent, fake, folder = make_agent(self, script, preserve_reasoning=True)
        run(agent, "first task")
        self.assertEqual(run(agent, "second task"), "second done")
        agent.used = 10 ** 9  # as if the context were nearly full: the next step trims
        self.assertEqual(run(agent, "third task"), "third done")
        steps = [line for line in read_jsonl(steps_file(folder)) if line["kind"] == "step"]
        self.assertTrue(steps[-1]["trimmed"])
        self.assertEqual(agent.dropped, 0)  # only reasoning went; every step is still there
        self.assertFalse(agent.summary_due)
        self.assertIn("thinking 3", [m.get("reasoning_content") for m in agent.messages])
        self.assertNotIn("thinking 0", [m.get("reasoning_content") for m in agent.messages])
        assert_requests_valid(self, fake)

    def test_a_multi_line_task_is_one_line_in_the_summary(self):
        text = AGENT.task_message(settings(), "next one", True, [("line one\nline two", "an\nanswer")])
        self.assertIn("- Task: line one line two\n  Your answer: an answer\n", text)
        msg = {"role": "user", "content": text}
        self.assertTrue(AGENT.is_follow_up(msg))
        self.assertEqual(AGENT.follow_up_text(msg), "next one")
        pasted = AGENT.task_message(settings(), "new", False, [("a\n" + AGENT.FOLLOW_UP_PREFIX + "x", "ok")])
        self.assertFalse(AGENT.is_follow_up({"role": "user", "content": pasted}))

    def test_the_no_move_hint_names_the_schema_argument(self):
        browser = AGENT.Browser.__new__(AGENT.Browser)
        browser.s = settings()
        self.assertIn("with x,y", browser._no_move)
        browser.s = settings(tool_schema="h")
        self.assertIn("with scroll_element", browser._no_move)
        self.assertNotIn("x,y", browser._no_move)

    def test_a_changed_browser_starts_a_new_conversation(self):
        browser = StubBrowser()
        browser.prompt_note = "\n- You work in your own window of the user's everyday Chrome."
        script = [{"tool_calls": [call("answer", content="one")]}, {"tool_calls": [call("answer", content="two")]},
                  {"tool_calls": [call("answer", content="three")]}]
        agent, fake, folder = make_agent(self, script, browser=browser)
        run(agent, "first")
        browser.prompt_note = ""  # Chrome was gone; the browser on the drive took over
        run(agent, "second")
        sent = fake.chat_requests()[1]["body"]["messages"]
        self.assertEqual([m["role"] for m in sent], ["system", "user", "user"])
        self.assertNotIn("everyday Chrome", sent[0]["content"])
        self.assertIn(AGENT.SUMMARY_LABEL, sent[1]["content"])
        self.assertTrue(sent[1]["content"].endswith("New task: second"))
        run(agent, "third")  # same browser again: a follow-up
        sent = fake.chat_requests()[2]["body"]["messages"]
        self.assertIn(AGENT.FOLLOW_UP_PREFIX + "third", [m["content"] for m in sent if isinstance(m["content"], str)])
        assert_requests_valid(self, fake)


class InteractiveErrorTests(unittest.TestCase):
    def setUp(self):
        reset_stop(self)
        scratch_paths(self)

    def test_a_failed_task_says_the_next_one_starts_afresh(self):
        script = [{"content": "hmm"}] * AGENT.MAX_FAILED_REPLIES + [{"tool_calls": [call("answer", content="ok")]}]
        code, out, fake = loop.InteractiveTests.run_prompt(self, ["a task that breaks", "yes, go on", "/quit"], script)
        self.assertIn("Task failed:", out)
        self.assertIn("The next task starts a new conversation, with a short summary of the earlier tasks.", out)
        sent = fake.chat_requests()[-1]["body"]["messages"]
        self.assertTrue(sent[1]["content"].startswith(AGENT.SUMMARY_LABEL))
        assert_requests_valid(self, fake)


if __name__ == "__main__":
    unittest.main()
