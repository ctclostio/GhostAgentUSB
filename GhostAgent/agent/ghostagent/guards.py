"""Where the agent may go, and what it may not do without the user's yes. These checks are outside
the model on purpose: text on a web page can talk the model into things, but not past these."""

import ipaddress
import re
import socket
import threading
import urllib.parse
import unicodedata
from pathlib import Path


# --------------------------------------------------------------------------------------------
# Where the agent may go, and what it may not do without the user's yes. These checks are outside
# the model on purpose: text on a web page can talk the model into things, but not past these.
# --------------------------------------------------------------------------------------------

class ToolRefused(Exception):
    """An action the agent will not take. Its message is the tool's answer to the model."""


# The browser's own pages and its extensions' pages: settings, history, saved passwords, downloads
# (where a downloaded file can be opened), an unlocked password manager's vault.
INTERNAL_SCHEMES = {"chrome", "chrome-extension", "chrome-untrusted", "chrome-search", "devtools", "edge",
                    "extension", "view-source"}
# What the browser shows when a page cannot be loaded.
ERROR_SCHEMES = {"chrome-error", "edge-error"}
SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*):")


def url_scheme(url):
    m = SCHEME_RE.match(url or "")
    return m.group(1).lower() if m else ""


def vet_url(raw, allow_file=False, allow_internal=False):
    """The address to open for a URL from the model or a setting, or ToolRefused. Only http(s) pages
    and about:blank: a file: page would let a page's instructions have the agent read this PC's
    files, javascript: runs code inside whatever site is open (signed in as the user), and the
    browser's own pages hold its settings, history and saved passwords. A URL without a scheme gets
    https://."""
    url = re.sub(r"[\t\r\n]", "", str(raw or ""))  # browsers drop these as well: 'java\nscript:' is javascript:
    url = url.strip("".join(map(chr, range(0x21))))  # control characters and spaces at either end
    if not url:
        raise ToolRefused("No address was given.")
    if url.lower() == "about:blank":
        return "about:blank"
    scheme = url_scheme(url)
    if scheme and re.match(r"^[^:/?#]+:\d+(?:[/?#]|$)", url):
        scheme = ""  # 'localhost:8080/x' is a host and a port, not a scheme
    if not scheme:
        url, scheme = "https://" + url, "https"
    if scheme in ("http", "https"):
        # Written the way the browser reads it: any number of slashes after the scheme, and backslashes
        # before the query as slashes ('http:192.168.1.1', 'http:\\host'). Otherwise the host checked
        # here and the host the browser opens could differ ('http://192.168.1.1\@example.com').
        rest = url[len(scheme) + 1:]
        cut = len(re.match(r"[^?#]*", rest).group(0))
        return f"{scheme}://" + rest[:cut].replace("\\", "/").lstrip("/") + rest[cut:]
    if (scheme == "file" and allow_file) or (scheme in INTERNAL_SCHEMES and allow_internal):
        return url
    raise ToolRefused(f"Opening {scheme}: pages is blocked. Only http(s) web pages can be opened.")


LOCAL_NETWORKS = tuple(ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12",
    "192.168.0.0/16", "::/128", "::1/128", "fc00::/7", "fe80::/10"))
LOCAL_SUFFIXES = (".local", ".internal", ".lan", ".localhost", ".home.arpa")


def is_local_ip(text):
    """True for an address on this PC or a local network: loopback, private (RFC 1918), link-local,
    carrier-grade NAT, IPv6 unique-local (also written as IPv4 inside IPv6)."""
    try:
        ip = ipaddress.ip_address(str(text).split("%", 1)[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return any(ip in net for net in LOCAL_NETWORKS)


def resolve_host(host, timeout=2.0):
    """The addresses a host name resolves to; [] for an unknown name or no answer within timeout
    (in a helper thread, so a slow DNS server cannot hold up a step)."""
    found = []

    def work():
        try:
            found.extend(entry[4][0] for entry in socket.getaddrinfo(host, None))
        except (OSError, UnicodeError):
            pass

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    return list(found)


def normalise_host(host):
    """A host as written in a URL or in allowed_hosts: lower case, without brackets, port or final dot,
    and read the way the browser reads it (%-escapes decoded, full-width characters and dots made
    plain), so '%31%39%32.168.1.1' is 192.168.1.1 here as it is in the browser."""
    h = str(host or "").strip()
    h = unicodedata.normalize("NFKC", urllib.parse.unquote(h)).lower()
    h = re.sub("[。．｡]", ".", h)
    if "://" in h:
        try:
            h = urllib.parse.urlsplit(h).hostname or ""
        except ValueError:
            h = ""
    elif h.startswith("["):
        h = h[1:].split("]", 1)[0]
    elif h.count(":") == 1:
        h = h.split(":", 1)[0]
    return h.split("/", 1)[0].rstrip(".")


def browser_ipv4(host):
    """The IPv4 address a browser reads in a host such as '127.1', '0x7f.1' or '0300.0250.1.1' (parts in
    decimal, hex or octal; the last part fills the remaining bytes), or None when it is not one."""
    parts = host.split(".")
    if len(parts) > 1 and parts[-1] == "":
        parts.pop()
    if not 1 <= len(parts) <= 4:
        return None
    nums = []
    for p in parts:
        if re.fullmatch(r"0[xX][0-9a-fA-F]*", p):
            nums.append(int(p[2:] or "0", 16))
        elif re.fullmatch(r"0[0-7]+", p):
            nums.append(int(p, 8))
        elif re.fullmatch(r"[0-9]+", p):
            nums.append(int(p))
        else:
            return None
    if any(n > 255 for n in nums[:-1]) or nums[-1] >= 256 ** (5 - len(nums)):
        return None
    value = nums[-1]
    for i, n in enumerate(nums[:-1]):
        value += n << (8 * (3 - i))
    return ipaddress.IPv4Address(value)


def host_is_local(host, resolve=resolve_host):
    """True for an address on this PC or the local network: a local IP address, a name without dots
    (an intranet or router name, localhost), a .local/.internal/.lan name, or a name that resolves to
    a local address. Best effort: a name can resolve differently for the browser a moment later,
    which is why the page's address is checked again after every action."""
    h = normalise_host(host)
    if not h:
        return False
    try:
        ipaddress.ip_address(h.split("%", 1)[0])
        return is_local_ip(h)
    except ValueError:
        pass
    v4 = browser_ipv4(h)
    if v4 is not None:
        return is_local_ip(v4)
    if "." not in h or h.endswith(LOCAL_SUFFIXES):
        return True
    return any(is_local_ip(a) for a in resolve(h))


def host_listed(host, entries):
    """True when allowed_hosts names this host: exactly, or as *.example.com (or .example.com) for
    that domain and every host under it."""
    h = normalise_host(host)
    if not h:
        return False
    for entry in entries or []:
        e = str(entry).strip().lower()
        if e.startswith("*."):
            e = e[1:]
        if e.startswith("."):
            domain = normalise_host(e[1:])
            if domain and (h == domain or h.endswith("." + domain)):
                return True
        elif normalise_host(e) == h:
            return True
    return False


UNTRUSTED_TAG_RE = re.compile(r"</?\s*untrusted_page_text[^>]*>", re.I)


def untrusted(text):
    """Text a web page controls (page text, a tab title, a dialog's message), marked for the model as
    data rather than instructions. Copies of the tags inside it are removed, so a page cannot end the
    block early and continue as if it were the agent speaking."""
    return "<untrusted_page_text>" + UNTRUSTED_TAG_RE.sub("", str(text or "")) + "</untrusted_page_text>"


# Buttons and links whose click is hard to undo: deleting, sending or publishing, paying, moving
# money, (un)subscribing, closing an account, resetting. Each goes with the words that, in the
# user's task, count as asking for exactly that. Buying and sending are one family each: a task that
# says "buy" has asked for the checkout too, and one that says "reply" for the Send button.
_BUY = r"buy(?:s|ing)?|bought|purchas(?:e|es|ed|ing)|order(?:s|ed|ing)?|pa(?:y|ys|id|ying|yment)|check(?:ing)?\s?out"
_SEND = r"send(?:s|ing)?|sent|post(?:s|ed|ing)?|publish(?:es|ed|ing)?|repl(?:y|ies|ied|ying)"
RISKY_ACTIONS = [
    (r"delete", r"delet(?:e|es|ed|ing|ion)"),
    (r"remove", r"remov(?:e|es|ed|ing|al)"),
    (r"erase", r"eras(?:e|es|ed|ing)"),
    (r"destroy", r"destroy(?:s|ed|ing)?"),
    (r"empty\s+(?:the\s+)?trash", r"empty(?:ing)?|emptie[sd]"),
    (r"send", _SEND), (r"post", _SEND), (r"publish", _SEND), (r"reply\s+all", _SEND),
    (r"pay", _BUY), (r"buy", _BUY), (r"purchase", _BUY), (r"place\s+(?:your\s+|my\s+|the\s+)?order", _BUY),
    (r"checkout", _BUY),
    (r"transfer", r"transfer(?:s|red|ring)?"),
    (r"withdraw", r"withdr(?:aw|aws|awing|awn|ew)"),
    (r"subscribe", r"subscrib(?:e|es|ed|ing)"),
    (r"unsubscribe", r"unsubscrib(?:e|es|ed|ing)"),
    (r"deactivate", r"deactivat(?:e|es|ed|ing|ion)"),
    (r"close\s+(?:your\s+|my\s+|the\s+)?account", r"clos(?:e|es|ed|ing)"),
    (r"reset", r"reset(?:s|ting)?"),
]
_RISKY = [(re.compile(rf"\b(?:{label})\b", re.I), re.compile(rf"\b(?:{task})\b", re.I)) for label, task in RISKY_ACTIONS]


def risky_words(label, task_text):
    """The words on a button or link that mark a hard-to-undo action the task does not ask for: empty
    when there are none, or when the task asks for each of them."""
    found = []
    for label_re, task_re in _RISKY:
        m = label_re.search(label or "")
        if m and not task_re.search(task_text or ""):
            found.append(m.group(0))
    return found


RISKY_REFUSAL = "The user did not allow this action."
PASSWORD_REFUSAL = "Ask the user to type passwords and codes themselves."


def playwright_feature_switches():
    """The bundled browser's --disable-features and --enable-features switches: Playwright's own lists,
    except that HTTPS-Upgrades (http:// links tried over https first, as in every normal Chrome) is
    switched back on. Chromium keeps only the last copy of each switch and lets a disabled feature
    win over an enabled one, so both lists are written out in full, read from the installed
    Playwright. [] when they cannot be read: then Playwright's defaults stay as they are."""
    try:
        import playwright
        bundle = Path(playwright.__file__).parent / "driver" / "package" / "lib" / "coreBundle.js"
        text = bundle.read_text(encoding="utf-8", errors="replace")
        block = re.search(r"disabledFeatures\s*=\s*\[(.*?)\]\.filter\(Boolean\)", text, re.S)
        switches = re.search(r"chromiumSwitches\s*=\s*\(options\)\s*=>\s*\[(.*?)\]\.filter\(Boolean\)", text, re.S)
        if not block or not switches:
            return []
        disabled = re.findall(r'"([A-Za-z0-9_]+)"', re.sub(r"//[^\n]*", "", block.group(1)))
        enabled = re.findall(r'"--enable-features=([A-Za-z0-9_,]+)"', switches.group(1))
        if "HttpsUpgrades" not in disabled or len(enabled) > 1:
            return []
        enabled = (enabled[0].split(",") if enabled else []) + ["HttpsUpgrades"]
        return ["--disable-features=" + ",".join(f for f in disabled if f != "HttpsUpgrades"),
                "--enable-features=" + ",".join(dict.fromkeys(enabled))]
    except Exception:
        return []


# Playwright's default switches that turn off protections a normal Chrome has: the pop-up blocker,
# phishing detection, and updates of the browser's security components (with the background requests
# they need). Not removed in the headless test browser, so tests stay off the internet.
UNSAFE_DEFAULT_SWITCHES = ["--disable-popup-blocking", "--disable-client-side-phishing-detection",
                           "--disable-component-update", "--disable-background-networking"]
HEADLESS_UNSAFE_DEFAULT_SWITCHES = UNSAFE_DEFAULT_SWITCHES[:2]
