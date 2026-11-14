"""Downloads in the agent's own browser (CS-28): kept in the download folder (a scratch folder in these
tests) under names that replace nothing, marked as coming from the internet, stopped when larger than
download_max_mb, and cancelled when downloads are off."""

from pathlib import Path

import helpers
from agent_support import AGENT
from security_support import SiteBrowserCase


class DownloadTests(SiteBrowserCase):
    def download(self, b, selector="#small"):
        said = self.click(b, selector, "download")
        said += " " + self.act(b, "wait_web", seconds=1)
        return said

    def files(self):
        folder = Path(AGENT.download_folder())
        return sorted(p.name for p in folder.iterdir()) if folder.exists() else []

    def test_unique_names_and_the_mark_of_the_web(self):
        b = self.start("/fixtures/download.html")
        said = self.download(b)
        self.assertIn("Downloaded ghostagent-small.txt (73 bytes) to ", said)
        self.assertIn("GhostAgent never opens downloaded files.", said)
        said = self.download(b)
        self.assertIn("Downloaded ghostagent-small (1).txt", said)
        self.assertEqual(self.files(), ["ghostagent-small (1).txt", "ghostagent-small.txt"])
        saved = Path(AGENT.download_folder()) / "ghostagent-small.txt"
        self.assertEqual(saved.read_bytes()[:30], b"GhostAgent benchmark download.")
        try:
            zone = Path(str(saved) + ":Zone.Identifier").read_text(encoding="ascii")
        except OSError:
            self.skipTest("the scratch folder is not on NTFS")
        self.assertIn("ZoneId=3", zone)
        self.assertIn("HostUrl=" + self.site.url("/download/small.txt"), zone)

    def test_an_oversize_download_is_stopped(self):
        b = self.start("/fixtures/download.html", download_max_mb=1)
        b.page.evaluate("document.getElementById('big').href = '/download/big.bin?mb=4'")
        said = self.download(b, "#big")
        self.assertIn("The download of ghostagent-big.bin was stopped: it is larger than the download_max_mb limit "
                      "(1 MB).", said)
        self.assertEqual(self.files(), [])

    def test_a_running_download_past_the_limit_is_stopped_between_steps(self):
        b = self.start("/fixtures/download.html", download_max_mb=1)
        partial = helpers.temp_dir(self) / "x.bin.crdownload"
        partial.write_bytes(bytes(2 * 2**20))

        class Running:
            cancelled = deleted = False

            def cancel(self):
                Running.cancelled = True

            def delete(self):
                Running.deleted = True

        small = helpers.temp_dir(self) / "y.bin.crdownload"
        small.write_bytes(b"x" * 10)
        big, ok = Running(), Running()
        b._downloads[big] = {"name": "x.bin", "partial": partial, "told": False}
        b._downloads[ok] = {"name": "y.bin", "partial": small, "told": False}
        b._check_downloads()
        self.assertTrue(Running.cancelled and Running.deleted)
        self.assertNotIn(big, b._downloads)
        self.assertIn("The download of x.bin was stopped", " ".join(b.events))
        self.assertIn("y.bin is still downloading (10 bytes so far)", " ".join(b.events))
        b.events = []
        b._check_downloads()
        self.assertEqual(b.events, [])  # said once

    def test_downloads_off(self):
        b = self.start("/fixtures/download.html", downloads="off")
        said = self.download(b)
        self.assertIn("A download (ghostagent-small.txt) was blocked: downloads are turned off in the settings.", said)
        self.assertEqual(self.files(), [])
