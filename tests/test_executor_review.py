"""Regression tests for problems found when batch D1 was reviewed: frames that have no document yet
must not make the whole page "busy", closing a background tab is not "no change", a tab the agent
opens itself is never reported as opened by the page, reader_mode reads display: contents wrappers,
press_keys_web takes a real list, and download names lose direction marks."""

import time
import unittest

from agent_support import AGENT, quiet
from security_support import SiteBrowserCase


class PendingFrameTests(SiteBrowserCase):
    """An iframe whose page is still on its way (or a lazy one further down) has no document to run a
    script in. Asking it anything waits until the time limit: only the frames that hold the focus, or
    the one under the point, may be asked."""

    def add_pending_frame(self, b, where="beforeend"):
        b.page.evaluate("""([url, where]) => {
            const f = document.createElement('iframe');
            f.src = url;
            f.style.cssText = 'width:600px;height:200px';
            document.querySelector('main').insertAdjacentElement(where, f);
        }""", [self.site.url("/slow?ms=20000"), where])
        b.page.wait_for_timeout(300)

    def test_keys_are_pressed_while_another_frame_is_loading(self):
        b = self.start("/fixtures/keys.html")
        self.add_pending_frame(b)
        b.page.focus("#target")
        t0 = time.monotonic()
        result = self.act(b, "press_keys_web", keys="a b")
        self.assertTrue(result.startswith("Executed."), result)
        self.assertLess(time.monotonic() - t0, 2.5)
        self.assertEqual(b.page.input_value("#target"), "ab")

    def test_writing_while_another_frame_is_loading(self):
        b = self.start("/fixtures/keys.html")
        self.add_pending_frame(b)
        result = self.act(b, "write", element="Target field", text="hi", **self.at(b, "#target"))
        self.assertIn('it now reads "hi"', result)

    def test_a_click_on_a_frame_still_loading_is_not_busy(self):
        b = self.start("/fixtures/keys.html")
        self.add_pending_frame(b, "afterbegin")
        point = self.at(b, "main > iframe")
        t0 = time.monotonic()
        result = self.act(b, "click_web", element="the frame", **point)
        self.assertTrue(result.startswith("Executed."), result)
        self.assertLess(time.monotonic() - t0, 2.5)
        result = self.act(b, "scroll_web", direction="down", **point)
        self.assertNotIn(AGENT.BUSY_RESULT, result)

    def test_keys_still_reach_a_focused_frame(self):
        # The focus chain still follows the focus into a frame: a password field there is refused.
        b = self.start("/fixtures/keys.html")
        b.page.evaluate("""() => {
            const f = document.createElement('iframe');
            f.srcdoc = '<input id=pw type=password>';
            document.querySelector('main').appendChild(f);
        }""")
        b.page.wait_for_timeout(300)
        b.page.frames[1].focus("#pw")
        self.assertIn(AGENT.PASSWORD_REFUSAL, self.act(b, "press_keys_web", keys="x"))

    def test_writing_into_a_rich_text_editor_frame(self):
        # Classic rich-text editors are an iframe whose whole document is editable.
        b = self.start("/fixtures/keys.html")
        b.page.evaluate("""() => {
            const f = document.createElement('iframe');
            f.id = 'rte';
            f.style.cssText = 'width:600px;height:120px';
            f.srcdoc = '<body contenteditable=true style="min-height:100px"></body>';
            document.querySelector('main').insertAdjacentElement('afterbegin', f);
        }""")
        b.page.wait_for_timeout(300)
        result = self.act(b, "write", element="editor", text="Dear team", **self.at(b, "#rte"))
        self.assertIn('it now reads "Dear team"', result)


class TabReviewTests(SiteBrowserCase):
    def test_closing_a_background_tab_is_not_reported_as_no_change(self):
        b = self.start("/fixtures/popups.html?timer=off")
        self.click(b, "#open", "Open help")
        self.act(b, "switch_tab_web", tab_index=0)
        result = self.act(b, "close_tab_web", tab_index=1)
        self.assertNotIn("did not change", result)
        self.assertEqual(b.stall.unchanged, 0)
        self.assertEqual(len(b._tabs()), 1)

    def test_a_tab_the_agent_opens_is_not_one_the_page_opened(self):
        b = self.start("/fixtures/long.html")
        result = self.act(b, "switch_tab_web", new_tab_url=self.site.url("/fixtures/keys.html"))
        self.assertNotIn("by itself", result)
        self.assertEqual(b._by_page, set())
        # Every tab closed under the agent's feet: the replacement is the agent's own.
        for p in list(b.ctx.pages):
            p.close()
        with quiet():
            obs = b.observation()
        text = " ".join(p.get("text", "") for p in obs["content"])
        self.assertNotIn("by itself", text)
        self.assertNotIn("(opened by the page)", text)


class ReaderReviewTests(SiteBrowserCase):
    def test_display_contents_wrappers_are_read(self):
        b = self.start("/fixtures/long.html")
        b.page.evaluate("""() => {
            const d = document.createElement('div');
            d.style.display = 'contents';
            d.innerHTML = '<p>CONTENTS-WRAPPED paragraph that must be read.</p>';
            document.body.appendChild(d);
        }""")
        self.assertIn("CONTENTS-WRAPPED", self.act(b, "reader_mode"))


class SmallReviewTests(unittest.TestCase):
    def test_keys_sent_as_a_list(self):
        self.assertEqual(AGENT.parse_keys(["ctrl", "a"]), ["Control+a"])
        self.assertEqual(AGENT.key_presses(["ctrl", "v"]), ["Control+v"])

    def test_download_names_lose_direction_marks(self):
        name = AGENT.safe_download_name("invoice\u202efdp.exe")
        self.assertEqual(name, "invoicefdp.exe")
        self.assertTrue(name.endswith(".exe"))


if __name__ == "__main__":
    unittest.main()
