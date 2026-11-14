"""CS-29 and CS-07 in the agent's own headless browser on the benchmark site's fixture pages: the
"h" tool arguments (write's overwrite and enter, tabs by tab_id or url, scroll_web's scroll_element
and num_pixels, drag_and_drop_web) next to the classic names, which keep working; and what the
agent loop reads to skip the rest of a step's calls (Browser.location and Browser.last_ok)."""

import time
import unittest

import helpers
from agent_support import AGENT, StubServer, assert_requests_valid, call, quiet, settings
from security_support import SiteBrowserCase


def settled(read, seconds=3.0):
    """A value from the page once it has stopped changing (a scroll may be animated)."""
    last, deadline = read(), time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(0.1)
        now = read()
        if now == last:
            return now
        last = now
    return last


class HToolTests(SiteBrowserCase):
    def keys_pressed(self, b):
        return [k.get("key") for k in b.page.evaluate("window.keys") if k.get("type") == "keydown"]

    def test_write_overwrite_and_enter(self):
        b = self.start("/fixtures/keys.html")
        at = self.at(b, "#target")
        self.act(b, "write", element="Target field", text="first", **at)
        self.act(b, "write", element="Target field", text=" second", overwrite=False, **at)
        self.assertEqual(b.page.input_value("#target"), "first second")
        self.act(b, "write", element="Target field", text="third", clear=True, **at)  # the classic name
        self.assertEqual(b.page.input_value("#target"), "third")
        self.assertTrue(b.last_ok)
        b.page.evaluate("window.keys = []")
        self.act(b, "write", element="Target field", text="x", enter=True, **at)
        self.assertIn("Enter", self.keys_pressed(b))
        b.page.evaluate("window.keys = []")
        self.act(b, "write", element="Target field", text="y", press_enter=True, enter=False, **at)
        self.assertNotIn("Enter", self.keys_pressed(b))  # the "h" name wins over the classic one
        b.page.evaluate("window.keys = []")
        self.act(b, "write", element="Target field", text="z", press_enter="true", **at)
        self.assertIn("Enter", self.keys_pressed(b))

    def test_write_enter_default(self):
        b = self.start("/fixtures/keys.html", write_enter_default=True)
        at = self.at(b, "#target")
        self.act(b, "write", element="Target field", text="go", **at)
        self.assertIn("Enter", self.keys_pressed(b))
        b.page.evaluate("window.keys = []")
        self.act(b, "write", element="Target field", text="stay", enter=False, **at)
        self.assertNotIn("Enter", self.keys_pressed(b))

    def test_tabs_by_id_and_by_url(self):
        b = self.start("/fixtures/popups.html?timer=off")
        first = b.page
        where = b.location()
        self.assertEqual(where, (id(first), first.url))
        result = self.act(b, "switch_tab_web", url=self.site.url("/fixtures/keys.html"))
        self.assertTrue(result.startswith("Executed."), result)
        self.assertIn("keys.html", b.page.url)
        self.assertNotEqual(b.location(), where)
        self.act(b, "switch_tab_web", tab_id=0)
        self.assertIs(b.page, first)
        self.assertEqual(b.location(), where)
        self.act(b, "switch_tab_web", tab_id="[1]")
        self.assertIn("keys.html", b.page.url)
        self.act(b, "switch_tab_web", tab_index=0)  # the classic name
        self.assertIs(b.page, first)
        self.assertTrue(b.last_ok)
        result = self.act(b, "switch_tab_web", tab_id=5)
        self.assertEqual(result, "There is no tab 5. Open tabs: 0 to 1.")
        self.assertFalse(b.last_ok)
        result = self.act(b, "switch_tab_web", tab_id="the second one")
        self.assertIn("is not a tab id", result)
        self.assertFalse(b.last_ok)
        self.act(b, "close_tab_web", tab_id=1)
        self.assertEqual(len(b._tabs()), 1)
        self.assertIs(b.page, first)
        self.assertTrue(b.last_ok)

    def test_scroll_element(self):
        b = self.start("/panel.html")
        target = dict(element="Release notes", **self.at(b, "#notes"))
        result = self.act(b, "scroll_web", direction="down", scroll_element=target)
        self.assertNotIn("did not move", result)
        self.assertGreater(settled(lambda: b.page.evaluate("document.getElementById('notes').scrollTop")), 0)

    def test_num_pixels(self):
        b = self.start("/fixtures/long.html")
        self.act(b, "scroll_web", direction="down", num_pixels=200)
        y = settled(lambda: b.page.evaluate("window.scrollY"))
        self.assertTrue(100 <= y <= 300, y)  # not the default (most of the window, about 640 pixels)
        self.act(b, "scroll_web", direction="down")
        self.assertGreater(settled(lambda: b.page.evaluate("window.scrollY")), y + 400)

    def test_drag_a_slider_and_a_card(self):
        b = self.start("/fixtures/slider.html")
        box = b.page.locator("#volume").bounding_box()
        y = round((box["y"] + box["height"] / 2) / b.h * 1000)
        x1 = round((box["x"] + box["width"] * 0.2) / b.w * 1000)   # the knob, at 20
        x2 = round((box["x"] + box["width"] * 0.9) / b.w * 1000)
        result = self.act(b, "drag_and_drop_web", description="volume knob to the right", x1=x1, y1=y, x2=x2, y2=y)
        self.assertTrue(result.startswith("Executed."), result)
        self.assertTrue(b.last_ok)
        self.assertGreater(int(b.page.input_value("#volume")), 60)
        start, end = self.at(b, "#drag"), self.at(b, "#drop")
        self.act(b, "drag_and_drop_web", element="the card into the box",
                 x1=start["x"], y1=start["y"], x2=end["x"], y2=end["y"])
        self.assertEqual(b.page.text_content("#status"), "The card is in the box.")
        result = self.act(b, "drag_and_drop_web", description="nowhere", x1=10)
        self.assertIn("needs x1, y1, x2 and y2", result)
        self.assertFalse(b.last_ok)

    def test_last_ok_is_false_when_nothing_was_done(self):
        b = self.start("/fixtures/keys.html")
        self.act(b, "write", element="plain text", text="hello", **self.at(b, "#plain"))
        self.assertFalse(b.last_ok)
        self.act(b, "write", element="Target field", text="hello", **self.at(b, "#target"))
        self.assertTrue(b.last_ok)
        self.act(b, "no_such_tool")
        self.assertFalse(b.last_ok)

    def test_a_dropdown_click_is_not_done(self):
        b = self.start("/fixtures/select.html")
        result = self.click(b, "select", "a dropdown")
        self.assertIn("That is a dropdown menu", result)
        self.assertFalse(b.last_ok)

    def test_a_select_option_that_found_nothing_is_not_done(self):
        # Review fix: an option that is not there used to count as done, so a chained Submit ran.
        b = self.start("/fixtures/select.html")
        result = self.act(b, "select_option", element="size", option="Enormous", **self.at(b, "#size"))
        self.assertIn("Option not found", result)
        self.assertFalse(b.last_ok)
        self.act(b, "select_option", element="size", option="Large", **self.at(b, "#size"))
        self.assertTrue(b.last_ok)
        self.assertEqual(b.page.text_content("#status"), "Size: Large")


class LoopInTheBrowserTests(SiteBrowserCase):
    """The agent loop with the real (headless) browser: several calls of one step run; a call that
    opens a tab ends the step's chain."""

    def run_agent(self, b, script):
        fake = helpers.start_fake_llama(self, script=script, validate="strict")
        agent = AGENT.Agent(settings(), StubServer(fake.base), b)
        with quiet():
            answer = agent.run("fill in the fields")
        assert_requests_valid(self, fake)
        return agent, fake, answer

    def test_two_fields_in_one_step(self):
        b = self.start("/fixtures/keys.html")
        script = [{"tool_calls": [call("write", element="Target field", text="alpha", **self.at(b, "#target")),
                                  call("write", element="Notes", text="beta", **self.at(b, "#area"))]},
                  {"tool_calls": [call("answer", content="Both filled in.")]}]
        agent, fake, answer = self.run_agent(b, script)
        self.assertEqual(answer, "Both filled in.")
        self.assertEqual((b.page.input_value("#target"), b.page.input_value("#area")), ("alpha", "beta"))

    def test_a_new_tab_skips_the_rest_of_the_step(self):
        b = self.start("/fixtures/popups.html?timer=off")
        opener = b.page
        script = [{"tool_calls": [call("click_web", element="Open help", **self.at(b, "#open")),
                                  call("click_web", element="Open help", **self.at(b, "#open"))]},
                  {"tool_calls": [call("answer", content="Opened.")]}]
        agent, fake, answer = self.run_agent(b, script)
        results = [m["content"] for m in agent.messages if m["role"] == "tool"]
        self.assertIn("A new tab opened and is now the current tab.", results[0])
        self.assertEqual(results[1], AGENT.SKIPPED_AFTER_CHANGE)
        self.assertEqual(len(b._tabs()), 2)  # the second click never ran
        self.assertIsNot(b.page, opener)


if __name__ == "__main__":
    unittest.main()
