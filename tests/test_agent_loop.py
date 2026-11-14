"""The agent loop's conversation, against the fake model server (strict about tool-call pairing) and
a stub browser: every tool call of a step runs in order, and the ones after a failure, a page change
or the answer are skipped (CS-07); a later task continues the conversation as a follow-up, /new
starts a new one with a summary, a pasted multi-line task is one task (CS-18); the model's earlier
reasoning can go back into its context (CS-15); the "h" prompt and tool schema (CS-17, CS-29).

Every message list the fake receives is checked for a valid pairing (agent_support's
assert_requests_valid). Nothing here opens a window or needs a GPU."""

import builtins
import copy
import json
import unittest
import unittest.mock as mock
from datetime import datetime, timedelta

import helpers
import test_server_runtime as server_runtime  # (the module only: its tests are not collected twice)
from agent_support import (AGENT, StubBrowser, StubServer, assert_requests_valid, assert_valid_pairing, call,
                           make_agent, quiet, read_jsonl, reset_stop, scratch_paths, settings, steps_file)


def run(agent, task):
    with quiet():
        return agent.run(task)


def last_step_tools(messages):
    """(the last assistant message with tool calls, the tool messages that answer it)."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant" and messages[i].get("tool_calls"):
            j = i + 1
            while j < len(messages) and messages[j]["role"] == "tool":
                j += 1
            return messages[i], messages[i + 1:j]
    raise AssertionError("no assistant message with tool calls")


def tool_contents(messages):
    return [m["content"] for m in messages if m["role"] == "tool"]


def user_texts(messages):
    return [m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], str)]


def step_lines(folder):
    return [e for e in read_jsonl(steps_file(folder)) if e.get("kind") == "step"]


class MultiCallTests(unittest.TestCase):
    """CS-07: all calls of one reply run, in order, each answered by its own tool message."""

    def setUp(self):
        reset_stop(self)

    def test_every_call_of_a_step_runs_in_order(self):
        script = [{"content": "Filling in the form.",
                   "tool_calls": [call("write", element="name field", x=100, y=200, text="Ada"),
                                  call("click_web", element="Next button", x=300, y=400),
                                  call("answer", content="Done: the form is sent.")]},
                  {"tool_calls": [call("answer", content="Nothing else to do.")]}]
        browser = StubBrowser()
        agent, fake, folder = make_agent(self, script, browser=browser)
        self.assertEqual(run(agent, "fill in the form"), "Done: the form is sent.")
        self.assertEqual([name for name, _ in browser.calls], ["write", "click_web"])
        # The executor gets write's switches under one set of names, filled in.
        self.assertEqual(browser.calls[0][1], {"element": "name field", "x": 100, "y": 200, "text": "Ada",
                                               "overwrite": True, "enter": False})
        assistant, results = last_step_tools(agent.messages)
        self.assertEqual([r["tool_call_id"] for r in results], [tc["id"] for tc in assistant["tool_calls"]])
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0]["content"].startswith("Executed."))
        self.assertTrue(results[1]["content"].startswith("Executed."))
        self.assertEqual(results[2]["content"], AGENT.ANSWER_DELIVERED)
        assert_valid_pairing(self, agent.messages, "the conversation")
        line = step_lines(folder)[-1]
        self.assertEqual(line["tool"], "write,click_web,answer")
        self.assertEqual(line["args_summary"].count(" ; "), 2)
        self.assertIsNotNone(line["t_execute"])
        # A follow-up sends the whole step back: the strict fake takes the three answered calls.
        self.assertEqual(run(agent, "anything else?"), "Nothing else to do.")
        assert_requests_valid(self, fake)

    def test_a_call_that_changes_the_page_skips_the_rest(self):
        browser = StubBrowser(navigate_on={"click_web": "https://example.test/results"})
        script = [{"tool_calls": [call("click_web", element="Search", x=10, y=20),
                                  call("write", element="filter box", x=30, y=40, text="red")]},
                  {"tool_calls": [call("answer", content="ok")]}]
        agent, fake, folder = make_agent(self, script, browser=browser)
        self.assertEqual(run(agent, "search"), "ok")
        self.assertEqual([name for name, _ in browser.calls], ["click_web"])
        self.assertEqual(tool_contents(agent.messages)[:2],
                         ["Executed. Current URL: https://example.test/results", AGENT.SKIPPED_AFTER_CHANGE])
        assert_requests_valid(self, fake)

    def test_a_failed_or_refused_call_skips_the_rest(self):
        browser = StubBrowser(fail_on={"click_web"}, refuse_on={"press_keys_web"})
        script = [{"tool_calls": [call("click_web", element="Buy", x=10, y=20),
                                  call("write", element="note", x=30, y=40, text="gift")]},
                  {"tool_calls": [call("press_keys_web", keys="Enter"), call("go_to_web", url="https://example.test/")]},
                  {"tool_calls": [call("answer", content="I could not do it.")]}]
        agent, fake, folder = make_agent(self, script, browser=browser)
        self.assertEqual(run(agent, "buy it"), "I could not do it.")
        self.assertEqual([name for name, _ in browser.calls], ["click_web", "press_keys_web"])
        results = tool_contents(agent.messages)
        self.assertTrue(results[0].startswith("Error: click_web failed on purpose"))
        self.assertNotIn("second line", results[0])
        self.assertEqual(results[1:4], [AGENT.SKIPPED_AFTER_FAILURE, "The user did not allow this action.",
                                        AGENT.SKIPPED_AFTER_FAILURE])
        assert_requests_valid(self, fake)

    def test_calls_after_the_answer_are_skipped(self):
        browser = StubBrowser()
        script = [{"tool_calls": [call("answer", content="Here it is."), call("click_web", element="x", x=1, y=2)]}]
        agent, fake, folder = make_agent(self, script, browser=browser)
        self.assertEqual(run(agent, "find it"), "Here it is.")
        self.assertEqual(browser.calls, [])
        self.assertEqual(tool_contents(agent.messages), [AGENT.ANSWER_DELIVERED, AGENT.SKIPPED_AFTER_ANSWER])
        assert_valid_pairing(self, agent.messages)
        self.assertEqual(agent.last["outcome"], "answer")

    def test_repeated_call_ids_are_made_unique(self):
        script = [{"tool_calls": [dict(call("write", element="a", x=1, y=2, text="x"), id="same"),
                                  dict(call("click_web", element="b", x=3, y=4), id="same")]},
                  {"tool_calls": [call("answer", content="ok")]}]
        agent, fake, folder = make_agent(self, script)
        self.assertEqual(run(agent, "go"), "ok")
        assert_requests_valid(self, fake)
        ids = [tc["id"] for tc in agent.messages[3]["tool_calls"]]
        self.assertEqual(len(set(ids)), 2)
        calls = AGENT.unique_call_ids([{"id": "a"}, {"id": "a"}, {}, {"id": ""}], 7)
        self.assertEqual([c["id"] for c in calls], ["a", "call_7_1", "call_7_2", "call_7_3"])

    def test_answer_outcome_and_an_empty_answer(self):
        script = [{"content": "The ticket costs 12 dollars.", "tool_calls": [call("answer", content="", outcome="Partial ")]},
                  {"tool_calls": [call("answer", content="Done.", outcome="maybe")]}]
        agent, fake, folder = make_agent(self, script, tool_schema="h")
        self.assertEqual(run(agent, "price?"), "The ticket costs 12 dollars.")  # the message stands in
        self.assertEqual(agent.last["answer_outcome"], "partial")
        self.assertEqual(run(agent, "thanks"), "Done.")
        self.assertIsNone(agent.last["answer_outcome"])  # not one of the four: left out
        body = fake.chat_requests()[0]["body"]
        self.assertEqual([t["function"]["name"] for t in body["tools"]],
                         [t["function"]["name"] for t in AGENT.TOOLS_H])
        assert_requests_valid(self, fake)

    def test_a_task_stopped_mid_step_and_then_continued(self):
        class StoppingBrowser(StubBrowser):
            def execute(self, name, args):
                result = super().execute(name, args)
                if name == "click_web":
                    AGENT.STOP.set()  # Ctrl+C while the first call ran
                return result

        browser = StoppingBrowser()
        script = [{"tool_calls": [call("click_web", element="Next", x=1, y=2),
                                  call("write", element="box", x=3, y=4, text="hi")]},
                  {"tool_calls": [call("answer", content="Carried on.")]}]
        agent, fake, folder = make_agent(self, script, browser=browser)
        with self.assertRaises(AGENT.StepCancelled):
            run(agent, "first task")
        self.assertEqual(agent.last["outcome"], "cancelled")
        self.assertEqual([name for name, _ in browser.calls], ["click_web"])
        self.assertEqual(tool_contents(agent.messages)[-1], AGENT.NOT_RUN_STOPPED)
        assert_valid_pairing(self, agent.messages, "after the stop")
        AGENT.STOP.clear()
        self.assertEqual(run(agent, "go on"), "Carried on.")
        assert_requests_valid(self, fake)
        sent = fake.chat_requests()[-1]["body"]["messages"]
        self.assertIn(AGENT.FOLLOW_UP_PREFIX + "go on", user_texts(sent))
        self.assertEqual(agent.history[0], ("first task", "(the user stopped this task before an answer)"))


class ConversationTests(unittest.TestCase):
    """CS-18: follow-ups, /new, a fresh start after an error, and reset() for unattended runs."""

    def setUp(self):
        reset_stop(self)

    def test_a_follow_up_continues_the_conversation(self):
        script = [{"tool_calls": [call("answer", content="Found 3 flights. Shall I book the cheapest?")]},
                  {"tool_calls": [call("answer", content="Booked.")]}]
        agent, fake, folder = make_agent(self, script)
        run(agent, "find flights to Boston")
        self.assertEqual(run(agent, "yes, book it"), "Booked.")
        first, second = [r["body"]["messages"] for r in fake.chat_requests()]
        self.assertEqual(second[:len(first)], first)  # the same prefix: the server reuses its cache
        self.assertEqual([m["role"] for m in second].count("system"), 1)
        self.assertIn(AGENT.ANSWER_DELIVERED, tool_contents(second))
        self.assertEqual(user_texts(second)[-1], AGENT.FOLLOW_UP_PREFIX + "yes, book it")
        self.assertEqual(agent.tasks, 2)
        self.assertNotIn(AGENT.SUMMARY_LABEL, json.dumps(second))
        assert_requests_valid(self, fake)

    def test_new_starts_a_new_conversation_with_a_summary(self):
        script = [{"tool_calls": [call("answer", content="It is 21 degrees.")]},
                  {"tool_calls": [call("answer", content="Second.")]},
                  {"tool_calls": [call("answer", content="Third.")]}]
        agent, fake, folder = make_agent(self, script)
        run(agent, "weather in Denver")
        agent.new_conversation()
        run(agent, "and in Boston?")
        sent = fake.chat_requests()[1]["body"]["messages"]
        self.assertEqual([m["role"] for m in sent], ["system", "user", "user"])
        self.assertEqual(sent[1]["content"], f"{AGENT.SUMMARY_LABEL}\n- Task: weather in Denver\n"
                                             f"  Your answer: It is 21 degrees.\n\nNew task: and in Boston?")
        agent.reset()
        run(agent, "a new start")
        sent = fake.chat_requests()[2]["body"]["messages"]
        self.assertEqual(sent[1]["content"], "a new start")
        assert_requests_valid(self, fake)

    def test_new_without_earlier_tasks_has_no_summary(self):
        agent, fake, folder = make_agent(self, [{"tool_calls": [call("answer", content="ok")]}])
        agent.new_conversation()
        run(agent, "only task")
        self.assertEqual(fake.chat_requests()[0]["body"]["messages"][1]["content"], "only task")

    def test_an_error_starts_a_new_conversation(self):
        script = [{"content": "hmm"}] * AGENT.MAX_FAILED_REPLIES + [{"tool_calls": [call("answer", content="Fine now.")]}]
        agent, fake, folder = make_agent(self, script)
        with self.assertRaises(RuntimeError):
            run(agent, "a task that breaks")
        self.assertEqual(agent.messages, [])
        self.assertEqual(run(agent, "try again"), "Fine now.")
        sent = fake.chat_requests()[-1]["body"]["messages"]
        self.assertEqual(len(sent), 3)
        self.assertIn("- Task: a task that breaks\n  Your answer: (no answer: the task ended with an error)",
                      sent[1]["content"])
        self.assertTrue(sent[1]["content"].endswith("New task: try again"))
        assert_requests_valid(self, fake)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def nap(self, seconds):
        self.now += seconds


class PasteKeyboard:
    """Lines that reach the console at given times (seconds on a FakeClock): kbhit says whether one
    is waiting, and input() (patched) takes it."""

    def __init__(self, clock, arrivals):
        self.clock = clock
        self.arrivals = list(arrivals)

    def kbhit(self):
        return bool(self.arrivals) and self.arrivals[0][0] <= self.clock.now

    def take(self):
        return self.arrivals.pop(0)[1]


class ReadTaskTests(unittest.TestCase):
    """CS-18: lines pasted together are one task; a line typed later is not part of it."""

    def read(self, first, arrivals, keyboard=True):
        clock = FakeClock()
        kb = PasteKeyboard(clock, arrivals)
        reads = []

        def fake_input(prompt=""):
            reads.append(prompt)
            if len(reads) == 1:
                return first
            if not kb.arrivals:
                raise EOFError
            return kb.take()

        with mock.patch.object(builtins, "input", fake_input):
            text = AGENT.read_task("Task> ", kb if keyboard else None, clock=clock, nap=clock.nap)
        return text, reads, clock, kb

    def test_one_typed_line(self):
        text, reads, clock, kb = self.read("find the weather", [])
        self.assertEqual((text, len(reads)), ("find the weather", 1))
        self.assertLess(clock.now, AGENT.PASTE_GAP_SECONDS + 0.05)  # waited only the short gap

    def test_a_pasted_block_is_one_task(self):
        text, reads, clock, kb = self.read("compare these:  ", [(0.0, "first item"), (0.01, ""), (0.02, "third item")])
        self.assertEqual(text, "compare these:\nfirst item\n\nthird item")

    def test_a_line_typed_later_is_the_next_task(self):
        text, reads, clock, kb = self.read("first task", [(1.0, "second task")])
        self.assertEqual(text, "first task")
        self.assertEqual(len(kb.arrivals), 1)

    def test_no_console_means_no_waiting(self):
        with mock.patch.object(AGENT, "_keyboard", lambda: None):
            text, reads, clock, kb = self.read("piped task", [(0.0, "more")], keyboard=False)
        self.assertEqual((text, clock.now), ("piped task", 0.0))


class InteractiveTests(unittest.TestCase):
    """run_interactive with a paste, /new and an answer outcome."""

    def setUp(self):
        reset_stop(self)
        scratch_paths(self)

    def run_prompt(self, entries, script, **over):
        """entries: what is typed at the prompt; a tuple is a paste of several lines."""
        fake = helpers.start_fake_llama(self, script=script, validate="strict")
        browser = StubBrowser()
        browser.mode = "bundled"
        browser.start = lambda start_url=None: None
        browser.close = lambda: None
        queue = []
        for entry in entries:
            lines = list(entry) if isinstance(entry, tuple) else [entry]
            queue += [(line, i > 0) for i, line in enumerate(lines)]

        class Keyboard:
            @staticmethod
            def kbhit():
                return bool(queue) and queue[0][1]  # the rest of a paste is waiting

        def fake_input(prompt=""):
            if not queue:
                raise EOFError
            return queue.pop(0)[0]

        with quiet() as out, mock.patch.object(builtins, "input", fake_input), \
                mock.patch.object(AGENT, "_keyboard", lambda: Keyboard):
            code = AGENT.run_interactive(settings(**over), AGENT.ExternalServer(fake.base), browser)
        return code, out.getvalue(), fake

    def test_paste_new_and_outcome(self):
        script = [{"tool_calls": [call("answer", content="Please sign in first.", outcome="blocked")]},
                  {"tool_calls": [call("answer", content="Hotels found.", outcome="success")]}]
        code, out, fake = self.run_prompt([("find flights", "from Denver", "to Boston"), "/new", "and hotels", "/quit"],
                                          script, tool_schema="h")
        self.assertEqual(code, 0)
        self.assertIn("Task received (3 lines)", out)
        self.assertIn("Answer (blocked):", out)
        self.assertIn("Answer (success):", out)
        self.assertIn("The next task starts a new conversation (with a short summary of the earlier tasks).", out)
        first, second = [r["body"]["messages"] for r in fake.chat_requests()]
        self.assertEqual(first[1]["content"], "find flights\nfrom Denver\nto Boston")
        self.assertTrue(second[1]["content"].startswith(AGENT.SUMMARY_LABEL))
        # (a pasted task is one line in the summary)
        self.assertIn("- Task: find flights from Denver to Boston\n  Your answer: Please sign in first.",
                      second[1]["content"])
        assert_requests_valid(self, fake)

    def test_a_follow_up_at_the_prompt(self):
        script = [{"tool_calls": [call("answer", content="Shall I send it?")]},
                  {"tool_calls": [call("answer", content="Sent.")]}]
        code, out, fake = self.run_prompt(["write the email", "yes, send it", "/quit"], script)
        self.assertIn("Answer:", out)
        self.assertNotIn("Answer (", out)
        sent = fake.chat_requests()[1]["body"]["messages"]
        self.assertEqual(user_texts(sent)[-1], AGENT.FOLLOW_UP_PREFIX + "yes, send it")
        assert_requests_valid(self, fake)


class PreserveReasoningTests(unittest.TestCase):
    """CS-15: preserve_reasoning sends the model's reasoning back with later requests."""

    def setUp(self):
        reset_stop(self)

    SCRIPT = [{"reasoning": "The search box is at the top; the price list is below it.",
               "content": "Opening the list.", "tool_calls": [call("click_web", element="Prices", x=5, y=6)]},
              {"tool_calls": [call("answer", content="12 dollars.")]}]

    def test_kept_reasoning_goes_back(self):
        agent, fake, folder = make_agent(self, copy.deepcopy(self.SCRIPT), preserve_reasoning=True)
        run(agent, "find the price")
        first, second = [r["body"] for r in fake.chat_requests()]
        self.assertTrue(second["chat_template_kwargs"]["preserve_thinking"])
        assistant = [m for m in second["messages"] if m["role"] == "assistant"][0]
        self.assertEqual(assistant["reasoning_content"], "The search box is at the top; the price list is below it.")
        self.assertIn(AGENT.KEPT_REASONING_LINE, first["messages"][0]["content"])
        self.assertNotIn(AGENT.CLASSIC_REASONING_LINE, first["messages"][0]["content"])
        assert_requests_valid(self, fake)

    def test_off_by_default_on_the_fast_profile(self):
        agent, fake, folder = make_agent(self, copy.deepcopy(self.SCRIPT))
        run(agent, "find the price")
        first, second = [r["body"] for r in fake.chat_requests()]
        self.assertNotIn("preserve_thinking", second["chat_template_kwargs"])
        self.assertNotIn("reasoning_content", json.dumps(second["messages"]))
        today = datetime.now().strftime("%A %d %B %Y")
        self.assertEqual(first["messages"][0]["content"], AGENT.SYSTEM_PROMPT.format(today=today))

    def test_auto_follows_the_profile(self):
        p = AGENT.preserves_reasoning
        self.assertEqual(settings()["preserve_reasoning"], "auto")
        self.assertFalse(p(settings()))
        self.assertFalse(p(settings(model_profile="fast-q8")))
        self.assertTrue(p(settings(model_profile="accurate")))
        self.assertFalse(p(settings(model_profile="accurate", preserve_reasoning=False)))
        self.assertTrue(p(settings(preserve_reasoning=True)))
        with mock.patch.object(AGENT, "active_profile", return_value=("x", {"arch": "moe", "preserve_reasoning": True})):
            self.assertTrue(p(settings()))

    def test_setting_values(self):
        c = AGENT.coerce_setting
        self.assertEqual([c("preserve_reasoning", v, "auto") for v in ("auto", "", None, " AUTO ", "true", False, "off")],
                         ["auto", "auto", "auto", "auto", True, False, False])
        with self.assertRaises(AGENT.BadSetting):
            c("preserve_reasoning", "sometimes", "auto")
        for key in ("prompt_version", "tool_schema"):
            self.assertEqual(AGENT.check_setting(key, " H "), "h")
            with self.assertRaises(AGENT.BadSetting):
                AGENT.check_setting(key, "v2")

    def test_server_flag(self):
        # test_server_runtime's harness: LlamaServer.start with Popen and the model files replaced.
        start = server_runtime.ServerArgsTests.start
        args, *_ = start(self, settings())
        self.assertIn("--no-reasoning-preserve", args)
        args, *_ = start(self, settings(preserve_reasoning=True))
        self.assertNotIn("--no-reasoning-preserve", args)
        args, *_ = start(self, settings(model_profile="accurate", preserve_reasoning=False))
        self.assertNotIn("--no-reasoning-preserve", args)  # the dense model's server is never given it


class PromptTests(unittest.TestCase):
    """CS-17: the "h" prompt and the task message's time of day."""

    def test_classic_prompt_is_unchanged(self):
        today = datetime.now().strftime("%A %d %B %Y")
        self.assertEqual(AGENT.system_prompt(settings()), AGENT.SYSTEM_PROMPT.format(today=today))
        self.assertEqual(AGENT.task_message(settings(), "do it"), "do it")

    def test_h_prompt(self):
        note = "\n- You are working in the user's own Chrome window."
        text = AGENT.system_prompt(settings(prompt_version="h"), note)
        self.assertTrue(text.startswith("You are GhostAgent"))
        self.assertTrue(text.endswith(note))
        self.assertNotIn("{", text)
        for needed in ("<untrusted_page_text>", "</untrusted_page_text>", "<system_alert>", "confirm",
                       "press_enter=true", "tab_index", "CAPTCHA", AGENT.CLASSIC_REASONING_LINE):
            self.assertIn(needed, text)
        for absent in ("scroll_element", "drag_and_drop_web", "tab_id", "outcome"):
            self.assertNotIn(absent, text)  # the classic tools have no such things
        text = AGENT.system_prompt(settings(prompt_version="h", tool_schema="h", observation_format="header",
                                            write_enter_default=True, preserve_reasoning=True))
        for needed in ("scroll_element", "drag_and_drop_web", "tab_id", "outcome: success, partial", "set enter=false",
                       "header", AGENT.KEPT_REASONING_LINE):
            self.assertIn(needed, text)
        self.assertLess(len(text), 9000)

    def test_h_task_message_has_the_time(self):
        s = settings(prompt_version="h")
        text = AGENT.task_message(s, "book it", follow_up=True, started=datetime.now())
        self.assertTrue(text.startswith(AGENT.FOLLOW_UP_PREFIX + "book it\n\n(Local time: "))
        self.assertEqual(AGENT.follow_up_text({"role": "user", "content": text}), "book it")
        yesterday = AGENT.task_message(s, "book it", started=datetime.now() - timedelta(days=1))
        self.assertIn(datetime.now().strftime("%A %d %B %Y"), yesterday)

    def test_h_prompt_and_tools_through_the_loop(self):
        reset_stop(self)
        script = [{"tool_calls": [call("scroll_web", direction="down", scroll_element={"element": "list", "x": 50, "y": 60},
                                       num_pixels=300),
                                  call("write", element="box", x=1, y=2, text="hi", enter=True, overwrite=False)]},
                  {"tool_calls": [call("answer", content="ok", outcome="success")]}]
        browser = StubBrowser()
        agent, fake, folder = make_agent(self, script, browser=browser, prompt_version="h", tool_schema="h")
        run(agent, "scroll the list")
        self.assertEqual(browser.calls[0], ("scroll_web", {"direction": "down", "x": 50, "y": 60, "element": "list",
                                                           "num_pixels": 300}))
        self.assertEqual(browser.calls[1][1]["enter"], True)
        self.assertEqual(browser.calls[1][1]["overwrite"], False)
        body = fake.chat_requests()[0]["body"]
        self.assertTrue(body["messages"][0]["content"].startswith("You are GhostAgent"))
        self.assertIn("(Local time: ", body["messages"][1]["content"])
        assert_requests_valid(self, fake)


class ToolSchemaTests(unittest.TestCase):
    """CS-29: the "h" tools are well formed, and the old argument names still work."""

    def check_tools(self, tools):
        names = [t["function"]["name"] for t in tools]
        self.assertEqual(len(names), len(set(names)))
        json.dumps(tools)
        for t in tools:
            params = t["function"]["parameters"]
            self.assertEqual(params["type"], "object")
            self.assertLessEqual(set(params["required"]), set(params["properties"]), t["function"]["name"])
            for name, prop in params["properties"].items():
                self.assertIn(prop["type"], ("string", "integer", "number", "boolean", "object"), name)
                if prop["type"] == "object":
                    self.assertLessEqual(set(prop["required"]), set(prop["properties"]))
        return names

    def test_tool_lists(self):
        classic = self.check_tools(AGENT.TOOLS)
        h = self.check_tools(AGENT.TOOLS_H)
        self.assertEqual(set(h) - set(classic), {"drag_and_drop_web"})
        answer = [t for t in AGENT.TOOLS_H if t["function"]["name"] == "answer"][0]["function"]["parameters"]
        self.assertEqual(answer["properties"]["outcome"]["enum"], ["success", "partial", "infeasible", "blocked"])
        self.assertEqual(answer["required"], ["content"])
        self.assertIs(AGENT.tool_list(settings()), AGENT.TOOLS)  # the default: today's tools, untouched
        before = json.dumps(AGENT.TOOLS)
        tools = AGENT.tool_list(settings(write_enter_default=True))
        write = [t for t in tools if t["function"]["name"] == "write"][0]["function"]["parameters"]["properties"]
        self.assertIn("default true", write["press_enter"]["description"])
        self.assertEqual(json.dumps(AGENT.TOOLS), before)
        tools = AGENT.tool_list(settings(tool_schema="h"))
        write = [t for t in tools if t["function"]["name"] == "write"][0]["function"]["parameters"]["properties"]
        self.assertEqual(set(write), {"element", "x", "y", "text", "overwrite", "enter"})
        self.assertIn("default false", write["enter"]["description"])

    def test_old_and_new_argument_names(self):
        ca = AGENT.canonical_args
        self.assertEqual(ca("write", {"text": "a", "clear": False, "press_enter": True}),
                         {"text": "a", "overwrite": False, "enter": True})
        self.assertEqual(ca("write", {"text": "a"}), {"text": "a", "overwrite": True, "enter": False})
        self.assertEqual(ca("write", {"text": "a"}, enter_default=True)["enter"], True)
        self.assertEqual(ca("write", {"text": "a", "enter": "false", "press_enter": True})["enter"], False)
        self.assertEqual(ca("write", {"text": "a", "press_enter": "yes"})["enter"], True)
        self.assertEqual(ca("switch_tab_web", {"tab_index": 2}), {"tab_id": 2})
        self.assertEqual(ca("switch_tab_web", {"tab_id": 1, "tab_index": 3}), {"tab_id": 1})
        self.assertEqual(ca("switch_tab_web", {"new_tab_url": "https://example.test/"}), {"url": "https://example.test/"})
        self.assertEqual(ca("switch_tab_web", {"new_tab_url": ""}), {})
        self.assertEqual(ca("close_tab_web", {"tab_index": 1}), {"tab_id": 1})
        self.assertEqual(ca("close_tab_web", {}), {})
        target = {"element": "list", "x": 10, "y": 20}
        self.assertEqual(ca("scroll_web", {"direction": "down", "scroll_element": target, "num_pixels": 300}),
                         {"direction": "down", "x": 10, "y": 20, "element": "list", "num_pixels": 300})
        self.assertEqual(ca("scroll_web", {"direction": "up", "scroll_element": json.dumps(target)})["x"], 10)
        self.assertEqual(ca("scroll_web", {"direction": "up", "scroll_element": {"element": "list"}}), {"direction": "up"})
        self.assertEqual(ca("scroll_web", {"direction": "up", "x": 5, "y": 6}), {"direction": "up", "x": 5, "y": 6})
        self.assertEqual(ca("drag_and_drop_web", {"element": "knob", "x1": 1})["description"], "knob")
        self.assertEqual(ca("answer", {"content": "c", "outcome": " Blocked"}), {"content": "c", "outcome": "blocked"})
        self.assertEqual(ca("answer", {"content": "c", "outcome": "maybe"}), {"content": "c"})
        args = {"text": "a", "clear": False}
        ca("write", args)
        self.assertEqual(args, {"text": "a", "clear": False})  # the model's own arguments are left as they were

    def test_describe_call(self):
        d = AGENT.describe_call
        self.assertTrue(d("write", {"element": "box", "x": 1, "y": 2, "text": "a", "enter": True}).endswith("+ Enter"))
        self.assertTrue(d("write", {"element": "box", "x": 1, "y": 2, "text": "a", "press_enter": True}).endswith("+ Enter"))
        self.assertFalse(d("write", {"element": "box", "x": 1, "y": 2, "text": "a", "enter": "false"}).endswith("+ Enter"))
        self.assertEqual(d("drag_and_drop_web", {"description": "slider knob", "x1": 1, "y1": 2, "x2": 3, "y2": 4}),
                         "drag_and_drop_web 'slider knob' from (1, 2) to (3, 4)")

if __name__ == "__main__":
    unittest.main()
