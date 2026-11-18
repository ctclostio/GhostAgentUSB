"""CS-27 (b)-(d), CS-07 and the CS-15 part of trimming: what the conversation's history keeps as it
grows. A history that outgrows the context keeps a 'Notes from earlier steps:' block, long tool
results from older steps are shortened, screenshots leave one at a time or in blocks, and with
preserve_reasoning the oldest reasoning goes first. Mostly pure functions on synthetic histories;
the last tests run the real agent loop against the fake model (and, for reader_mode, the agent's own
headless browser on the benchmark site). Every history is checked for valid tool-call pairing."""

import json
import re
import unittest

import helpers
from agent_support import (AGENT, StubServer, assert_requests_valid, assert_valid_pairing, call, make_agent,
                           pairing_problems, quiet, read_jsonl, reset_stop, settings, steps_file)
from security_support import SiteBrowserCase

PNG = helpers.make_png(32, 20)
LONG_PAGE = ("Page: <untrusted_page_text>A title</untrusted_page_text> (https://example.test/)\n\n"
             "<untrusted_page_text>\n" + "word " * 1200 + "\n</untrusted_page_text>\n[Showing chars 0-6000 of 6000 total]")


def observation(n):
    return {"role": "user", "content": [
        {"type": "text", "text": "<observation>\n"},
        {"type": "image_url", "image_url": {"url": helpers.png_data_url(PNG)}},
        {"type": "text", "text": f"\nURL: https://example.test/{n}\n</observation>"}]}


def tool_call(cid, name, **args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def synthetic(steps, follow_ups=(), reasoning=False, long_results=False, content=None, budget=3):
    """A history as the agent builds it: system prompt, first task, then per step an observation,
    the model's reply and one tool message per call. Every 7th step chains two calls, every 11th
    reply had no usable call (the agent's note follows), and follow-up tasks come before the steps
    named in follow_ups. Screenshots are trimmed to `budget` as the loop would."""
    msgs = [{"role": "system", "content": "You are a web agent."}, {"role": "user", "content": "First task."}]
    for n in range(1, steps + 1):
        if n in follow_ups:
            msgs.append({"role": "user", "content": AGENT.FOLLOW_UP_PREFIX + f"task given before step {n}"})
        msgs.append(observation(n))
        AGENT.trim_images(msgs, budget)
        text = content(n) if content else f"note {n}"
        if n % 11 == 0:
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user", "content": AGENT.NO_TOOL_NOTE})
            continue
        calls = [tool_call(f"s{n}a", "click_web", element=f"button {n}", x=n, y=n)]
        if n % 7 == 0:
            calls.append(tool_call(f"s{n}b", "write", element="box", x=1, y=2, text=f"text {n}"))
        reply = {"role": "assistant", "content": text, "tool_calls": calls}
        if reasoning:
            reply["reasoning_content"] = f"reasoning of step {n}"
        msgs.append(reply)
        for k, c in enumerate(calls):
            result = LONG_PAGE if long_results else f"Executed. Current URL: https://example.test/{n}"
            if k == 1:
                result = AGENT.SKIPPED_AFTER_CHANGE
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    return msgs


def notes_of(msgs):
    blocks = [m for m in msgs if AGENT.is_notes(m)]
    return blocks[0]["content"] if blocks else None


class DropOldestStepsTests(unittest.TestCase):
    def test_a_60_step_history_is_trimmed_validly(self):
        msgs = synthetic(60, follow_ups=(20, 45))
        assert_valid_pairing(self, msgs)
        head = [dict(m) for m in msgs[:2]]
        dropped, changed = AGENT.drop_oldest_steps(msgs, 0)
        self.assertTrue(changed)
        self.assertEqual(dropped, 30)
        assert_valid_pairing(self, msgs)
        self.assertEqual(msgs[:2], head)  # the system prompt and the first task stay
        self.assertTrue(AGENT.is_notes(msgs[2]))
        notes = notes_of(msgs)
        self.assertLessEqual(len(notes), AGENT.NOTES_MAX_CHARS)
        self.assertTrue(notes.startswith("Notes from earlier steps:\n1. click_web 'button 1' at (1, 1) - note 1\n"))
        self.assertIn("\n7. click_web 'button 7' at (7, 7); write 'box' at (1, 2) <- \"text 7\" (not run) - note 7\n", notes)
        self.assertIn("\n11. (no action) - note 11\n", notes)
        self.assertIn("\nThe user then asked: task given before step 20\n", notes)
        self.assertIn("\n30. click_web 'button 30' at (30, 30) - note 30\n", notes)
        self.assertNotIn("\n31. ", notes)
        self.assertTrue(notes.endswith("(The screenshots and tool results of these 30 steps were removed to save memory. "
                                       "Continue from the current screen.)"))
        self.assertLessEqual(AGENT.count_images(msgs), 3)
        self.assertEqual(len(AGENT.step_starts(msgs)), 30)
        self.assertEqual(sum(1 for m in msgs if AGENT.is_follow_up(m)), 1)  # step 45's task is still in place

        # A second trim carries the lines over and numbers on.
        dropped, changed = AGENT.drop_oldest_steps(msgs, dropped)
        self.assertEqual(dropped, 45)
        assert_valid_pairing(self, msgs)
        notes = notes_of(msgs)
        self.assertEqual(sum(1 for m in msgs if AGENT.is_notes(m)), 1)
        self.assertIn("\n1. click_web 'button 1'", notes)
        self.assertIn("\n45. click_web 'button 45' at (45, 45) - note 45\n", notes)
        self.assertIn("these 45 steps", notes)
        self.assertLessEqual(len(notes), AGENT.NOTES_MAX_CHARS)
        # The current task (given before step 45, which is now a note) is kept word for word after it.
        self.assertTrue(AGENT.is_follow_up(msgs[3]))
        self.assertIn("task given before step 45", msgs[3]["content"])
        self.assertNotIn("asked: task given before step 45", notes)

    def test_short_histories_are_left_alone(self):
        msgs = synthetic(3)
        before = json.dumps(msgs)
        self.assertEqual(AGENT.drop_oldest_steps(msgs, 0), (0, False))
        self.assertEqual(json.dumps(msgs), before)

    def test_the_newest_lines_are_kept_within_the_cap(self):
        msgs = synthetic(60, content=lambda n: f"step {n} " + "detail " * 120)  # about 850 chars a line
        dropped, _ = AGENT.drop_oldest_steps(msgs, 0)
        notes = notes_of(msgs)
        self.assertLessEqual(len(notes), AGENT.NOTES_MAX_CHARS)
        self.assertIn("\n30. click_web 'button 30'", notes)
        self.assertNotIn("\n1. click_web", notes)
        lines = notes.split("\n")[1:-1]
        numbers = [int(re.match(r"(\d+)\.", line).group(1)) for line in lines]
        self.assertEqual(numbers, list(range(numbers[0], 31)))  # a contiguous run up to the newest
        assert_valid_pairing(self, msgs)

    def test_one_huge_message_is_cut_to_its_line_limit(self):
        msgs = synthetic(10, content=lambda n: "x" * 20000)
        AGENT.drop_oldest_steps(msgs, 0)
        for line in notes_of(msgs).split("\n"):
            self.assertLessEqual(len(line), AGENT.NOTE_LINE_MAX)

    def test_failed_calls_are_marked(self):
        msgs = synthetic(8)
        msgs[4]["content"] = "Error: the element was gone"  # step 1's click
        AGENT.drop_oldest_steps(msgs, 0)
        self.assertIn("1. click_web 'button 1' at (1, 1) (failed) - note 1", notes_of(msgs))

    def test_reasoning_goes_before_any_step(self):
        msgs = synthetic(20, reasoning=True)
        count = len(msgs)
        dropped, changed = AGENT.drop_oldest_steps(msgs, 0, reasoning_first=True)
        self.assertEqual((dropped, changed, len(msgs)), (0, True, count))
        cut = AGENT.step_starts(msgs)[len(AGENT.step_starts(msgs)) // 2]
        replies = [(i, m) for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("tool_calls")]
        self.assertTrue(all("reasoning_content" not in m for i, m in replies if i < cut))
        self.assertTrue(all(m.get("reasoning_content") for i, m in replies if i > cut))
        # Next time the steps go (their reasoning is gone already).
        dropped, changed = AGENT.drop_oldest_steps(msgs, 0, reasoning_first=True)
        self.assertEqual((dropped, changed), (10, True))
        self.assertNotIn("reasoning of step", notes_of(msgs))
        assert_valid_pairing(self, msgs)

    def test_a_stopped_step_is_closed_before_the_conversation_goes_on(self):
        msgs = synthetic(4)
        msgs.append(observation(5))
        msgs.append({"role": "assistant", "content": "", "tool_calls": [tool_call("x1", "click_web", x=1, y=1),
                                                                         tool_call("x2", "answer", content="hi")]})
        msgs.append({"role": "tool", "tool_call_id": "x1", "content": "Executed."})
        self.assertTrue(pairing_problems(msgs))
        self.assertEqual(AGENT.answer_open_calls(msgs, AGENT.NOT_RUN_STOPPED), 1)
        assert_valid_pairing(self, msgs)
        self.assertEqual(msgs[-1], {"role": "tool", "tool_call_id": "x2", "content": AGENT.NOT_RUN_STOPPED})
        self.assertEqual(AGENT.answer_open_calls(msgs, AGENT.NOT_RUN_STOPPED), 0)


class ShortenOldResultsTests(unittest.TestCase):
    def results(self, msgs):
        return [m["content"] for m in msgs if m["role"] == "tool" and m["content"] != AGENT.SKIPPED_AFTER_CHANGE]

    def test_results_from_three_steps_ago_are_cut(self):
        msgs = synthetic(10, long_results=True)
        self.assertEqual(AGENT.shorten_old_results(msgs), 7)  # steps 1-7 (step 7's skipped call is short)
        results = self.results(msgs)
        for text in results[:-3]:
            self.assertTrue(text.endswith(" [truncated]"), text[-40:])
            self.assertLessEqual(len(text), AGENT.OLD_RESULT_CHARS + len(AGENT.UNTRUSTED_CLOSE) + len(AGENT.TRUNCATED))
            # An untrusted block the cut left open is closed again; no tag is cut in half.
            self.assertEqual(text.count(AGENT.UNTRUSTED_OPEN), text.count(AGENT.UNTRUSTED_CLOSE))
            self.assertNotRegex(text, r"<[^>]*$")
        for text in results[-3:]:
            self.assertEqual(text, LONG_PAGE)  # the last 3 steps keep theirs whole
        self.assertEqual(AGENT.shorten_old_results(msgs), 0)  # done once
        assert_valid_pairing(self, msgs)

    def test_block_mode_cuts_only_behind_the_screenshots_kept(self):
        msgs = synthetic(9, long_results=True, budget=99)
        self.assertTrue(AGENT.trim_images(msgs, 3, "block", 2))  # 9 screenshots: the newest 2 stay
        oldest_kept = AGENT.image_chunks(msgs)[0][0]
        AGENT.shorten_old_results(msgs)
        for i, m in enumerate(msgs):
            if m["role"] == "tool" and m["content"] != AGENT.SKIPPED_AFTER_CHANGE:
                self.assertEqual(m["content"].endswith(" [truncated]"), i < AGENT.step_starts(msgs)[-3], i)
        self.assertLess(AGENT.step_starts(msgs)[-3], oldest_kept)

        # A screenshot kept further back holds the cut back too: nothing before it changes.
        msgs = synthetic(9, long_results=True, budget=99)
        AGENT.trim_images(msgs, 6, "rolling")
        AGENT.shorten_old_results(msgs)
        oldest_kept = AGENT.image_chunks(msgs)[0][0]
        for i, m in enumerate(msgs):
            if m["role"] == "tool" and m["content"] != AGENT.SKIPPED_AFTER_CHANGE:
                self.assertEqual(m["content"].endswith(" [truncated]"), i < oldest_kept, i)
        # ... unless the history was trimmed, which changes everything after the first task anyway.
        AGENT.shorten_old_results(msgs, everywhere=True)
        cut_off = [m for m in msgs[:AGENT.step_starts(msgs)[-3]] if m["role"] == "tool"
                   and m["content"] != AGENT.SKIPPED_AFTER_CHANGE]
        self.assertTrue(cut_off and all(m["content"].endswith(" [truncated]") for m in cut_off))


class ImageEvictionTests(unittest.TestCase):
    def visible(self, mode, keep, steps=10, budget=3):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
        counts, evicted = [], []
        for n in range(1, steps + 1):
            msgs.append(observation(n))
            evicted.append(AGENT.trim_images(msgs, budget, mode, keep))
            counts.append(AGENT.count_images(msgs))
        return counts, evicted, msgs

    def test_block_mode_keep_2_and_keep_1(self):
        counts, evicted, _ = self.visible("block", 2)
        self.assertEqual(counts, [1, 2, 3, 2, 3, 2, 3, 2, 3, 2])
        self.assertEqual(evicted, [False, False, False, True, False, True, False, True, False, True])
        counts, evicted, _ = self.visible("block", 1)
        self.assertEqual(counts, [1, 2, 3, 1, 2, 3, 1, 2, 3, 1])

    def test_rolling_mode_is_unchanged(self):
        counts, evicted, msgs = self.visible("rolling", 2)
        self.assertEqual(counts, [1, 2, 3, 3, 3, 3, 3, 3, 3, 3])
        self.assertEqual(evicted, [False] * 3 + [True] * 7)
        # An evicted screenshot becomes a short note; its observation text stays around it.
        first = msgs[2]["content"]
        self.assertEqual(first[1], {"type": "text", "text": "[screenshot evicted]"})
        self.assertEqual(first[0]["text"], "<observation>\n")
        self.assertTrue(AGENT.is_observation(msgs[2]))
        # The newest three are the ones kept.
        self.assertEqual([i for i, _ in AGENT.image_chunks(msgs)], [9, 10, 11])

    def test_keep_is_held_below_the_budget(self):
        counts, _, _ = self.visible("block", 5)  # more than budget - 1: works as keep 2
        self.assertEqual(counts[:6], [1, 2, 3, 2, 3, 2])
        counts, _, _ = self.visible("block", 2, budget=1)
        self.assertEqual(counts, [1] * 10)

    def test_settings_check_keep_against_the_budget(self):
        with quiet() as out:
            s = AGENT.check_related_settings(settings(image_eviction="block", image_budget=2, image_keep_after_evict=2))
        self.assertEqual(s["image_keep_after_evict"], 1)
        self.assertIn('"image_keep_after_evict" is 2', out.getvalue())
        with quiet() as out:
            s = AGENT.check_related_settings(settings(image_eviction="rolling", image_budget=2, image_keep_after_evict=2))
        self.assertEqual((s["image_keep_after_evict"], out.getvalue()), (2, ""))  # rolling: not used, no warning
        # The default, block keeping 1, fits any budget of 2 or more ...
        with quiet() as out:
            s = AGENT.check_related_settings(settings(image_budget=2))
        self.assertEqual((s["image_eviction"], s["image_keep_after_evict"], out.getvalue()), ("block", 1, ""))
        # ... and a keep raised to 2 is checked against the budget under the default eviction.
        with quiet() as out:
            s = AGENT.check_related_settings(settings(image_budget=2, image_keep_after_evict=2))
        self.assertEqual(s["image_keep_after_evict"], 1)
        self.assertIn('"image_keep_after_evict" is 2', out.getvalue())


class LoopTests(unittest.TestCase):
    """The agent loop with the fake model (which simulates llama.cpp's prompt cache: cache_n is the
    part of a request it shares with the previous one)."""

    def setUp(self):
        reset_stop(self)

    def run_steps(self, steps, **over):
        script = [{"reasoning": "go", "tool_calls": [call("scroll_web", direction="down")]}] * (steps - 1)
        script.append({"tool_calls": [call("answer", content="done")]})
        agent, fake, folder = make_agent(self, script=script, **over)
        with quiet():
            self.assertEqual(agent.run("Scroll a while."), "done")
        assert_requests_valid(self, fake)
        return [l for l in read_jsonl(steps_file(folder)) if l["kind"] == "step"], fake

    def test_block_eviction_keeps_the_prompt_append_only_between_batches(self):
        rolling, _ = self.run_steps(9, image_eviction="rolling")
        block, fake = self.run_steps(9, image_eviction="block", image_keep_after_evict=2)
        self.assertEqual([l["images_in_prompt"] for l in block], [1, 2, 3, 2, 3, 2, 3, 2, 3])
        self.assertEqual([l["evicted"] for l in block], [False, False, False, True, False, True, False, True, False])
        # Between evictions only the newest step is new to the server (about one screenshot);
        # rolling re-reads all three screenshots on every step past the third.
        for l in block:
            if l["step"] > 1 and not l["evicted"]:
                self.assertLess(l["prompt_n"], 1500, l)
        for l in rolling[3:]:
            self.assertGreater(l["prompt_n"], 3000, l)
        self.assertEqual([AGENT.count_images(r["body"]["messages"]) for r in fake.chat_requests()],
                         [1, 2, 3, 2, 3, 2, 3, 2, 3])

    def test_a_long_conversation_is_trimmed_and_goes_on(self):
        # A small context: the trim happens after a few steps, and the requests after it stay valid.
        script = [{"content": f"step {n} note", "tool_calls": [call("click_web", element=f"b{n}", x=n, y=n)]}
                  for n in range(1, 12)] + [{"tool_calls": [call("answer", content="done")]}]
        # The classic prompt: the first task message is the task alone (h adds the time of day).
        agent, fake, folder = make_agent(self, script=script, ctx_size=4096, prompt_version="classic")
        with quiet():
            self.assertEqual(agent.run("Click around."), "done")
        assert_requests_valid(self, fake)
        lines = [l for l in read_jsonl(steps_file(folder)) if l["kind"] == "step"]
        self.assertTrue(any(l["trimmed"] for l in lines))
        last = fake.chat_requests()[-1]["body"]["messages"]
        notes = [m["content"] for m in last if isinstance(m["content"], str) and m["content"].startswith("Notes from")]
        self.assertEqual(len(notes), 1)
        self.assertIn("1. click_web 'b1' at (1, 1) - step 1 note", notes[0])
        self.assertEqual(last[1]["content"], "Click around.")


class ReaderPagingTests(SiteBrowserCase):
    """reader_mode through the agent loop, in the agent's headless browser: a 50k-character article
    in 7 calls, each footer pointing at the next part, and the older parts shortened in the history."""

    def test_a_long_article_through_the_loop(self):
        script = [{"tool_calls": [call("reader_mode", start=8000 * k)]} for k in range(7)]
        script.append({"tool_calls": [call("answer", content="read it all")]})
        b = self.start("/fixtures/reader.html?paras=49")
        fake = helpers.start_fake_llama(self, script=script, validate="strict")
        # Rolling eviction: a screenshot leaves at every step past the third, so each step's shortening
        # reaches the results 3 steps back (block cuts only behind its batches; see ShortenOldResultsTests).
        agent = AGENT.Agent(settings(settle_seconds=0.1, image_eviction="rolling"), StubServer(fake.base), b)
        with quiet():
            self.assertEqual(agent.run("Read the article."), "read it all")
        assert_requests_valid(self, fake)
        chats = fake.chat_requests()
        self.assertEqual(len(chats), 8)
        total = None
        for k in range(7):
            # Part k as the model got it on the next step, the newest result then.
            text = [m["content"] for m in chats[k + 1]["body"]["messages"] if m["role"] == "tool"][-1]
            m = re.search(r"\[Showing chars (\d+)-(\d+) of (\d+) total\]", text)
            self.assertIsNotNone(m, text[-200:])
            a, end, n = map(int, m.groups())
            self.assertEqual(a, 8000 * k)
            total = total or n
            self.assertEqual(n, total)
            if k < 6:
                self.assertEqual(end, 8000 * (k + 1))
                self.assertIn(f"call reader_mode with start={end}", text)
            else:
                self.assertEqual(end, total)
                self.assertNotIn("For the next part", text)
        self.assertTrue(48000 < total <= 56000, total)
        # In the last request (step 8), the parts read 3 or more steps earlier (steps 1-5) are cut
        # short; those of steps 6 and 7 are whole.
        results = [m["content"] for m in chats[-1]["body"]["messages"] if m["role"] == "tool"]
        self.assertEqual(len(results), 7)
        for text in results[:5]:
            self.assertTrue(text.endswith(AGENT.TRUNCATED), text[-60:])
            self.assertLessEqual(len(text), AGENT.OLD_RESULT_CHARS + len(AGENT.UNTRUSTED_CLOSE) + len(AGENT.TRUNCATED))
            self.assertEqual(text.count(AGENT.UNTRUSTED_OPEN), text.count(AGENT.UNTRUSTED_CLOSE))
        for text in results[5:]:
            self.assertGreater(len(text), 1000)


if __name__ == "__main__":
    unittest.main()
