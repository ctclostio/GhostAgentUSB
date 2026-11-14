"""Batch D1 in the agent's own headless browser on the benchmark site's fixture pages: honest action
results, stall alerts, the edge of the page, tabs and pop-ups, file pickers, busy pages, the
"header" observation, reader_mode as markdown, whole-tile screenshots and the adaptive settle."""

import re
import time
import unittest

import helpers
from agent_support import AGENT, call, quiet, StubServer, settings
from security_support import SiteBrowserCase


def texts(obs):
    return [p["text"] for p in obs["content"] if p.get("type") == "text"]


def has_image(obs):
    return any(p.get("type") == "image_url" for p in obs["content"])


class ActionResultTests(SiteBrowserCase):
    def test_a_click_at_the_far_corner_lands_inside_the_page(self):
        b = self.start("/fixtures/clicks.html")
        result = self.act(b, "click_web", element="corner", x=1000, y=1000)
        self.assertTrue(result.startswith("Executed."), result)
        last = b.page.evaluate("window.clicks")[-1]
        self.assertEqual((last["x"], last["y"]), (b.w - 1, b.h - 1))
        self.assertGreaterEqual(last["held_ms"], 40)  # pressed for a moment, like a real click

    def test_a_dead_button_is_reported_and_then_alerted(self):
        b = self.start("/fixtures/dead-button.html")
        self.click(b, "#more", "Load more")  # the first click moves the focus to the button: a change
        for i in range(3):
            result = self.click(b, "#more", "Load more")
            self.assertIn("The page did not change.", result)
            self.assertEqual(b._alert is not None, i == 2)
        with quiet():
            obs = b.observation()
        first = texts(obs)[0]
        self.assertTrue(first.startswith("<system_alert>Your last 3 actions did not change the page."), first)
        self.assertTrue(AGENT.is_observation(obs))
        with quiet():
            again = b.observation()
        self.assertFalse(texts(again)[0].startswith("<system_alert>"))  # said once

    def test_writing_where_there_is_no_field(self):
        b = self.start("/fixtures/keys.html")
        result = self.act(b, "write", element="plain text", text="hello", **self.at(b, "#plain"))
        self.assertIn("Warning: no text field had focus at (", result)
        self.assertIn("nothing was typed", result)
        self.assertFalse(any(k.get("key") == "h" for k in b.page.evaluate("window.keys")))

    def test_writing_reads_the_field_back(self):
        b = self.start("/fixtures/keys.html")
        result = self.act(b, "write", element="Target field", text="hello there", **self.at(b, "#target"))
        self.assertIn('Typed into the text field "Paste or type here"; it now reads "hello there".', result)
        self.assertEqual(b.page.input_value("#target"), "hello there")
        result = self.act(b, "write", element="editor", text="Rich", **self.at(b, "#editor"))
        self.assertIn('it now reads "Rich"', result)

    def test_a_scroll_that_moves_nothing(self):
        b = self.start("/fixtures/noscroll.html")
        result = self.act(b, "scroll_web", direction="down")
        self.assertIn("The page did not move (already at the end", result)
        self.assertNotIn("The page did not change.", result)

    def test_a_scroll_that_moves_the_page(self):
        b = self.start("/fixtures/long.html")
        result = self.act(b, "scroll_web", direction="down")
        self.assertNotIn("did not move", result)
        self.assertNotIn("did not change", result)

    def test_a_scroll_inside_a_panel_counts_as_a_change(self):
        b = self.start("/panel.html")
        result = self.act(b, "scroll_web", direction="down", **self.at(b, "#notes"))
        self.assertNotIn("did not move", result)
        self.assertNotIn("did not change", result)
        self.assertGreater(b.page.evaluate("document.getElementById('notes').scrollTop"), 0)

    def test_a_click_during_a_slow_navigation(self):
        b = self.start("/fixtures/navigate.html")
        button = self.at(b, "#b")
        self.click(b, "#slow", "Open the slow page")
        result = self.act(b, "click_web", element="A button", **button)  # the slow page is still on its way
        self.assertTrue(result.startswith("Executed."), result)

    def test_find_looks_inside_frames_of_the_same_site(self):
        b = self.start("/fixtures/iframe.html")
        self.assertIn("Match found", self.act(b, "ctrl_f_web", query="PELICAN"))
        self.assertIn("No match", self.act(b, "ctrl_f_web", query="FLAMINGO"))

    def test_unknown_keys_press_nothing(self):
        b = self.start("/fixtures/keys.html")
        b.page.focus("#target")
        result = self.act(b, "press_keys_web", keys="ctrl + Bogus")
        self.assertIn('Unknown key "Bogus"; nothing was pressed.', result)
        self.assertEqual(b.page.evaluate("window.keys"), [])
        self.act(b, "press_keys_web", keys="shift + h")
        self.assertEqual(b.page.input_value("#target"), "H")


class TabTests(SiteBrowserCase):
    def test_a_tab_opened_by_a_click_becomes_current(self):
        b = self.start("/fixtures/popups.html?timer=off")
        result = self.click(b, "#open", "Open help")
        self.assertIn("A new tab opened and is now the current tab.", result)
        self.assertIn("from=click", b.page.url)

    def test_closing_a_background_tab_keeps_the_current_one(self):
        b = self.start("/fixtures/popups.html?timer=off")
        self.click(b, "#open", "Open help")
        popup = b.page
        self.act(b, "close_tab_web", tab_index=0)
        self.assertIs(b.page, popup)
        self.assertEqual(len(b._tabs()), 1)

    def test_closing_another_tab_from_the_first_one(self):
        b = self.start("/fixtures/popups.html?timer=off")
        self.click(b, "#open", "Open help")
        self.act(b, "switch_tab_web", tab_index=0)
        first = b.page
        self.act(b, "close_tab_web", tab_index=1)
        self.assertIs(b.page, first)

    def test_a_sign_in_pop_up_that_closes_returns_to_its_opener(self):
        b = self.start("/fixtures/oauth.html")
        opener = b.page
        self.click(b, "#signin", "Sign in")
        self.assertIn("oauth-popup", b.page.url)
        result = self.act(b, "wait_web", seconds=3)
        self.assertIs(b.page, opener)
        self.assertIn("the tab that opened it is the current tab again", result)
        self.assertEqual(b.page.title(), "signed-in")

    def test_a_tab_the_page_opens_later_is_listed_not_switched_to(self):
        b = self.start("/fixtures/popups.html?timer=off&later=2500")
        page = b.page
        result = self.click(b, "#later", "Open a tab later")
        self.assertNotIn("new tab", result)
        result = self.act(b, "wait_web", seconds=3.5)
        self.assertIn("The page opened a new tab by itself; it was not switched to", result)
        self.assertIs(b.page, page)
        with quiet():
            obs = b.observation()
        self.assertIn("(opened by the page)", texts(obs)[-1])
        self.act(b, "switch_tab_web", tab_index=1)
        with quiet():
            obs = b.observation()
        self.assertNotIn("(opened by the page)", texts(obs)[-1])

    def test_a_file_picker_is_reported_during_a_task_only(self):
        b = self.start("/fixtures/upload.html")
        result = self.click(b, "#pick", "Browse")
        self.assertIn("A file picker opened; uploading local files is not supported. Ask the user to choose the file.",
                      result)
        b.end_task()
        self.assertEqual(b._choosers, set())  # between tasks the user's own uploads work


class BusyPageTests(SiteBrowserCase):
    def test_a_busy_page_answers_within_a_few_seconds(self):
        b = self.start("/fixtures/busy.html?ms=9000")
        b.page.evaluate("() => setTimeout(() => window.freeze(), 0)")
        b.page.wait_for_timeout(300)
        t0 = time.monotonic()
        result = self.act(b, "click_web", element="Freeze", **{"x": 500, "y": 500})
        took = time.monotonic() - t0
        self.assertTrue(result.startswith(AGENT.BUSY_RESULT), result)
        self.assertLess(took, 4.5)
        t0 = time.monotonic()
        result = self.act(b, "wait_web", seconds=1)  # waiting is what a busy page needs: not refused
        self.assertTrue(result.startswith("Executed."), result)
        self.assertLess(time.monotonic() - t0, 3.0)


class HeaderObservationTests(SiteBrowserCase):
    def test_header_on_a_long_page(self):
        b = self.start("/fixtures/long.html", observation_format="header")
        with quiet():
            obs = b.observation()
        head, foot = texts(obs)[0], texts(obs)[-1]
        self.assertEqual([p["type"] for p in obs["content"]], ["text", "image_url", "text"])
        self.assertTrue(head.startswith("<observation>\n====================\nURL: "), head)
        self.assertIn("Title: <untrusted_page_text>Long page</untrusted_page_text>", head)
        self.assertIn("Viewport: 1280×800px", head)
        self.assertRegex(head, r"Position: 0\.0% \(0px / \d+px\)")
        self.assertRegex(head, r"Visible: \d+% of page")
        self.assertRegex(foot, r"^\n↓ \d+px below \(scroll down to continue\)\n</observation>$")
        b.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        with quiet():
            foot = texts(b.observation())[-1]
        self.assertIn("bottom of the page", foot)
        self.assertIn("↑", foot)

    def test_header_on_a_short_page_with_two_tabs(self):
        b = self.start("/fixtures/popups.html?timer=off", observation_format="header")
        self.click(b, "#open", "Open help")
        with quiet():
            obs = b.observation()
        head, foot = texts(obs)[0], texts(obs)[-1]
        self.assertIn("Open Tabs (2):", head)
        self.assertIn("Popup page (click)</untrusted_page_text> (current)", head)
        self.assertIn("Entire page visible", foot)

    def test_classic_stays_as_it_was(self):
        b = self.start("/fixtures/long.html")
        with quiet():
            obs = b.observation()
        self.assertEqual(texts(obs)[0], "<observation>\n")
        self.assertRegex(texts(obs)[-1], r"^\nURL: \S+\nOpen tabs:\n  \[0\] <untrusted_page_text>Long page"
                                         r"</untrusted_page_text> - \S+ \(current\)\n</observation>$")


class ReaderModeTests(SiteBrowserCase):
    def test_markdown(self):
        b = self.start("/fixtures/reader.html")
        result = self.act(b, "reader_mode")
        for piece in ("# Reading a long article", "## Facts", "- First point", "  - Nested point", "3. Third",
                      "4. Fourth", "| Name | Value |", "| --- | --- |", "| Alpha | 1 |",
                      f"[link to the long page]({self.site.url('/fixtures/long.html')})",
                      "```\ncode line 1\n  code line 2\n```", "FRAME-TEXT-INCLUDED", "script link"):
            self.assertIn(piece, result)
        for hidden in ("MENU-LINK", "HIDDEN-TEXT", "STYLE-HIDDEN", "SIDEBAR", "BUTTON-SHOULD", "FOOTER-SHOULD",
                       "javascript:", "GhostAgent benchmark site"):
            self.assertNotIn(hidden, result)
        self.assertRegex(result, r"\[Showing chars 0-(\d+) of \1 total\]$")

    def test_a_long_article_in_pages(self):
        b = self.start("/fixtures/reader.html?paras=49")
        start, calls, parts, total = 0, 0, [], None
        while True:
            result = self.act(b, "reader_mode", **({"start": start} if start else {}))
            calls += 1
            m = re.search(r"\[Showing chars (\d+)-(\d+) of (\d+) total\]", result)
            a, end, n = map(int, m.groups())
            self.assertEqual(a, start)
            total = total or n
            self.assertEqual(n, total)
            body = result.split("<untrusted_page_text>\n", 1)[1].rsplit("\n</untrusted_page_text>", 1)[0]
            self.assertEqual(len(body), end - a)
            parts.append(body)
            if end >= total:
                self.assertNotIn("For the next part", result)
                break
            self.assertIn(f"call reader_mode with start={end}", result)
            start = end
        self.assertTrue(48000 < total <= 56000, total)
        self.assertEqual(calls, 7)
        joined = "".join(parts)
        self.assertIn("Paragraph 1. ", joined)
        self.assertIn("Paragraph 49. ", joined)


class GeometryTests(SiteBrowserCase):
    def test_whole_tiles_in_the_bundled_browser(self):
        b = self.start("/fixtures/long.html", snap_viewport=True)
        with quiet():
            b.observation()
        self.assertEqual((b.last_shot["png_w"], b.last_shot["png_h"]), (1280, 832))

    def test_the_default_size_is_kept(self):
        b = self.start("/fixtures/long.html")
        with quiet():
            b.observation()
        self.assertEqual((b.last_shot["png_w"], b.last_shot["png_h"]), (1280, 800))

    def test_a_big_window_is_mentioned_once_per_task(self):
        b = self.start("/fixtures/long.html", viewport_width=1600, viewport_height=900)
        with quiet() as out:
            b.observation()
            b.observation()
        self.assertEqual(out.getvalue().count("a big screenshot makes every step slower"), 1)
        self.assertIn("viewport_width and viewport_height", out.getvalue())  # the bundled browser has no window to resize
        b.begin_task("again")
        with quiet() as out:
            b.observation()
        self.assertEqual(out.getvalue().count("a big screenshot makes every step slower"), 1)

    def test_refit_is_only_for_installed_browsers(self):
        b = self.start("/fixtures/long.html")
        self.assertFalse(b.refit_window())


class AdaptiveSettleTests(SiteBrowserCase):
    def test_a_quiet_page_ends_the_wait_early(self):
        b = self.start("/fixtures/navigate.html", settle_mode="adaptive", settle_seconds=1.0)
        result = self.click(b, "#b", "A button")
        self.assertIn("Executed.", result)
        settle = b.last_timings["settle"]
        self.assertGreaterEqual(settle, 0.25)
        self.assertLess(settle, 0.95)
        self.assertEqual(b.page.inner_text("#status"), "Clicked.")
        self.act(b, "scroll_web", direction="down")
        self.assertLess(b.last_timings["settle"], 0.6)
        self.act(b, "ctrl_f_web", query="button")
        self.assertLess(b.last_timings["settle"], 0.1)

    def test_fixed_waits_the_whole_time(self):
        b = self.start("/fixtures/navigate.html", settle_mode="fixed", settle_seconds=0.6)
        self.click(b, "#b", "A button")
        self.assertGreaterEqual(b.last_timings["settle"], 0.6)

    def test_a_page_still_loading_is_waited_for_up_to_the_limit(self):
        b = self.start("/fixtures/navigate.html", settle_mode="adaptive", settle_seconds=1.0)
        self.click(b, "#slow", "Open the slow page")
        self.assertGreaterEqual(b.last_timings["settle"], 0.95)  # the slow page takes 3 s
        self.assertLess(b.last_timings["settle"], 2.0)

    def test_wait_web_ends_once_the_page_is_quiet(self):
        b = self.start("/fixtures/delayed.html?ms=700", settle_mode="adaptive")
        t0 = time.monotonic()
        self.act(b, "wait_web", seconds=6)
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertIn("(loaded)", b.page.title())
        t0 = time.monotonic()
        self.act(b, "wait_web", seconds=0.4)
        self.assertGreaterEqual(time.monotonic() - t0, 0.4)  # never shorter than asked, up to 1 s

    def test_go_to_waits_for_the_page(self):
        b = self.start("/fixtures/long.html", settle_mode="adaptive")
        self.act(b, "go_to_web", url=self.site.url("/fixtures/delayed.html?ms=200"))
        self.assertLessEqual(b.last_timings["settle"], 1.6)


class LoopTrapAgentTests(SiteBrowserCase):
    """The whole loop with the model played by fake_llama: three clicks on a dead button, then the
    fourth observation the model sees starts with the alert."""

    def test_the_alert_reaches_the_model(self):
        b = self.start("/fixtures/dead-button.html")
        point = self.at(b, "#more")
        # The first click moves the focus to the button (a change); the next three change nothing.
        script = [{"tool_calls": [call("click_web", element="Load more", **point)]}] * 4
        script.append({"tool_calls": [call("answer", content="The Load more button does nothing.")]})
        fake = helpers.start_fake_llama(self, script=script)
        agent = AGENT.Agent(settings(settle_seconds=0.1), StubServer(fake.base), b)
        with quiet():
            agent.run("load more results")
        requests = fake.chat_requests()
        self.assertEqual(len(requests), 5)

        def last_observation(body):
            obs = [m for m in body["messages"] if AGENT.is_observation(m)]
            return obs[-1]["content"][0]["text"]
        for r in requests[:4]:
            self.assertFalse(last_observation(r["body"]).startswith("<system_alert>"))
        self.assertTrue(last_observation(requests[4]["body"]).startswith("<system_alert>Your last 3 actions"))
        tools = [m["content"] for m in requests[4]["body"]["messages"] if m["role"] == "tool"]
        self.assertTrue(all("The page did not change." in t for t in tools[1:]), tools)


class CheckinTests(unittest.TestCase):
    def test_a_strategy_reminder_every_n_steps(self):
        from agent_support import make_agent
        script = [{"tool_calls": [call("wait_web", seconds=0)]}] * 4 + [{"tool_calls": [call("answer", content="ok")]}]
        agent, fake, _ = make_agent(self, script=script, progress_checkin_steps=2)
        with quiet():
            agent.run("anything")
        firsts = []
        for r in fake.chat_requests():
            obs = [m for m in r["body"]["messages"] if AGENT.is_observation(m)][-1]
            firsts.append(obs["content"][0]["text"])
        self.assertEqual([f.startswith("<system_alert>") for f in firsts], [False, False, True, False, True])
        self.assertIn("2 steps so far", firsts[2])
        self.assertIn("4 steps so far", firsts[4])


if __name__ == "__main__":
    unittest.main()
