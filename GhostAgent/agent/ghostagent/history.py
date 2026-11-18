"""The conversation's history: screenshots leave it, old tool results are shortened, and a history
that outgrows the context keeps one line per step it drops."""

import json
import re

from .console import one_line
from .prompts import FOLLOW_UP_PREFIX, SUMMARY_LABEL
from .tools import describe_call


# --------------------------------------------------------------------------------------------
# The conversation's history: screenshots leave it, old tool results are shortened, and a history
# that outgrows the context keeps one line per step it drops.
# --------------------------------------------------------------------------------------------

EVICTED_NOTE = "[screenshot evicted]"


def image_chunks(messages):
    """Every screenshot in the history, oldest first, as (message index, content part)."""
    return [(i, part) for i, m in enumerate(messages)
            if m.get("role") == "user" and isinstance(m.get("content"), list)
            for part in m["content"] if isinstance(part, dict) and part.get("type") == "image_url"]


def trim_images(messages, budget=3, mode="rolling", keep=2):
    """Keep at most `budget` screenshots in the history. An evicted one becomes the text
    '[screenshot evicted]'; its <observation> text stays. Returns True when this call evicted any
    (the step trace records it).

    mode "rolling": every step past the budget, the oldest goes, so the newest `budget` are always
    there. But the prompt then changes a little further back on every step, and Holo4's server can
    only reuse what comes before the change (its recurrent layers cannot be wound back), so most of
    the history is read again on each step.
    mode "block": nothing goes until there are more than `budget`; then all but the newest `keep`
    go at once. The prompt only grows at the end in between, so the server reuses it. The model sees
    keep..budget screenshots instead of always `budget` (image_eviction's note has the trade-off)."""
    budget = max(1, int(budget))
    images = image_chunks(messages)
    if len(images) <= budget:
        return False
    keep_n = budget
    if mode == "block":
        keep_n = max(1, min(int(keep), budget - 1)) if budget > 1 else 1
    for _, part in images[:len(images) - keep_n]:
        part["type"] = "text"
        part["text"] = EVICTED_NOTE
        part.pop("image_url", None)
    return True


def count_images(messages):
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for part in m["content"] if isinstance(part, dict) and part.get("type") == "image_url")


def is_observation(msg):
    """True for the user message that starts a step: its first text is the <observation>, possibly
    after a harness alert (see with_alert)."""
    if msg["role"] != "user" or not isinstance(msg["content"], list) or not msg["content"]:
        return False
    first = msg["content"][0]
    text = first.get("text", "") if isinstance(first, dict) else ""
    return text.startswith("<observation>") or (text.startswith("<system_alert>") and "<observation>" in text)


def step_starts(messages):
    """Indexes of the observations that start the steps (the system prompt and the first task, at 0
    and 1, are never part of a step)."""
    return [i for i, m in enumerate(messages) if i >= 2 and is_observation(m)]


NOTES_HEADER = "Notes from earlier steps:"
NOTES_MAX_CHARS = 6000     # the whole notes message; the newest lines are kept
NOTE_LINE_MAX = 1000       # one step's line
OLD_RESULT_STEPS = 3       # tool results of the last this many steps stay whole ...
OLD_RESULT_CHARS = 300     # ... older ones are cut to this many characters


def old_result_steps(image_budget):
    """How many recent steps keep their long tool results whole: OLD_RESULT_STEPS, or 2 with an
    image_budget of 2. Nothing before the first screenshot an eviction removes may change (see
    shorten_old_results), and with 2 screenshots that one is only 2 steps back: results 3 steps old are
    already in front of it, so with 3 nothing would ever be shortened. Never fewer than 2 (with an
    image_budget of 1 nothing is shortened until the history is trimmed): the newest result stays whole."""
    return max(2, min(OLD_RESULT_STEPS, int(image_budget)))
TRUNCATED = " [truncated]"
UNTRUSTED_OPEN, UNTRUSTED_CLOSE = "<untrusted_page_text>", "</untrusted_page_text>"


def is_notes(msg):
    return msg.get("role") == "user" and isinstance(msg.get("content"), str) and msg["content"].startswith(NOTES_HEADER)


def is_follow_up(msg):
    """A task the user gave later in the conversation (see task_message)."""
    content = msg.get("content")
    if msg.get("role") != "user" or not isinstance(content, str):
        return False
    return content.startswith(FOLLOW_UP_PREFIX) or (content.startswith(SUMMARY_LABEL)
                                                    and "\n" + FOLLOW_UP_PREFIX in content)


def follow_up_text(msg):
    """The user's own words in a follow-up message."""
    content = msg["content"]
    text = content[content.find(FOLLOW_UP_PREFIX) + len(FOLLOW_UP_PREFIX):]
    return re.sub(r"\n\n\(Local time: [^\n]*\.\)$", "", text)


def _call_args(call):
    try:
        args = json.loads((call.get("function") or {}).get("arguments") or "{}")
    except ValueError:
        return {}
    return args if isinstance(args, dict) else {}


def step_note(n, assistant, results):
    """'<n>. <what the step did> - <the model's message, verbatim>' for the notes block. results: the
    step's tool messages by call id (a call that was skipped or failed is marked so)."""
    acts = []
    for call in assistant.get("tool_calls") or []:
        name = (call.get("function") or {}).get("name") or "?"
        what = describe_call(name, _call_args(call))
        result = str(results.get(call.get("id")) or "")
        if result.startswith(("Skipped", "Not run")):
            what += " (not run)"
        elif result.startswith("Error"):
            what += " (failed)"
        acts.append(what)
    line = f"{n}. " + ("; ".join(acts) if acts else "(no action)")
    content = str(assistant.get("content") or "").strip()
    if content:
        line += " - " + content
    return one_line(line, NOTE_LINE_MAX)


def drop_oldest_steps(messages, dropped, reasoning_first=False):
    """Make room in a history that is outgrowing the context. Returns (dropped, changed): the new
    total of steps dropped from this conversation, and whether anything changed.

    The older half of the steps (each starts at an observation; whole steps only, so every tool
    call keeps its tool message) is replaced by one message, 'Notes from earlier steps:', with a line
    per step: what it did and the model's message verbatim (its reasoning, screenshots and tool
    results go). The block holds at most NOTES_MAX_CHARS, the newest lines first; the lines of an
    earlier block are carried over. The system prompt and the first task (indexes 0 and 1) stay, and
    so does the current task's message when it was among the dropped steps.

    reasoning_first (preserve_reasoning): when the older half still holds reasoning, only that is
    removed this time (H Company's agent drops past reasoning first when it compacts); the steps go
    the next time the history is too long."""
    starts = step_starts(messages)
    if len(starts) < 4:
        return dropped, False
    cut = starts[len(starts) // 2]
    if reasoning_first:
        stripped = False
        for m in messages[2:cut]:
            if m.get("role") == "assistant" and m.get("reasoning_content"):
                del m["reasoning_content"]
                stripped = True
        if stripped:
            return dropped, True
    newest_task = max((i for i, m in enumerate(messages) if i >= 2 and is_follow_up(m)), default=None)
    lines = []
    n = dropped
    for i in range(2, cut):
        m = messages[i]
        if is_notes(m):
            lines += [line for line in m["content"].split("\n")[1:] if line and not line.startswith("(")]
        elif is_observation(m):
            n += 1
        elif m.get("role") == "assistant":
            results = {}
            for later in messages[i + 1:cut]:
                if later.get("role") != "tool":
                    break
                results[later.get("tool_call_id")] = later.get("content")
            lines.append(step_note(n, m, results))
        elif is_follow_up(m) and i != newest_task:
            lines.append(one_line("The user then asked: " + follow_up_text(m), NOTE_LINE_MAX))
    footer = (f"(The screenshots and tool results of these {n} steps were removed to save memory. Continue from "
              "the current screen.)")
    room = NOTES_MAX_CHARS - len(NOTES_HEADER) - len(footer) - 2
    kept = []
    for line in reversed(lines):
        if len(line) + 1 > room:
            break
        kept.append(line)
        room -= len(line) + 1
    notes = {"role": "user", "content": "\n".join([NOTES_HEADER] + kept[::-1] + [footer])}
    keep_task = messages[newest_task] if newest_task is not None and newest_task < cut else None
    del messages[2:cut]  # also removes the block an earlier trim left
    messages[2:2] = [notes] + ([keep_task] if keep_task else [])
    return n, True


def _shorten(text, limit):
    """text cut to `limit` characters plus ' [truncated]', never in the middle of a tag, and with an
    untrusted-page-text block that the cut left open closed again."""
    part = text[:limit]
    lt = part.rfind("<")
    if lt != -1 and ">" not in part[lt:]:
        part = part[:lt]
    if part.count(UNTRUSTED_OPEN) > part.count(UNTRUSTED_CLOSE):
        part += UNTRUSTED_CLOSE
    return part + TRUNCATED


def shorten_old_results(messages, everywhere=False, steps=OLD_RESULT_STEPS, limit=OLD_RESULT_CHARS, since=0):
    """Cut tool results from `steps` or more steps ago that are longer than `limit` characters (in
    practice reader_mode pages and long errors) to `limit` plus ' [truncated]'. Returns how many.

    The agent calls this only on steps whose prompt changes anyway (a screenshot was evicted, or the
    history was trimmed), and, unless `everywhere`, only before the oldest screenshot still kept:
    between evictions the prompt then only grows at the end, which keeps image_eviction "block"
    fast. With the default image_budget of 3 that is exactly the results of 3 or more steps ago.
    Nothing before message `since` changes (the first one the eviction changed: the server reuses
    everything before it). So with "block" a long result of the step just before an evicted batch keeps
    its length, until the history is trimmed. The agent passes steps=old_result_steps(image_budget)."""
    starts = step_starts(messages)
    if len(starts) < steps:
        return 0
    bound = starts[-steps]
    if not everywhere:
        images = image_chunks(messages)
        if images:
            bound = min(bound, images[0][0])
    count = 0
    for m in messages[max(0, since):bound]:
        content = m.get("content")
        if (m.get("role") == "tool" and isinstance(content, str) and len(content) > limit + len(TRUNCATED)
                and not content.endswith(TRUNCATED)):
            m["content"] = _shorten(content, limit)
            count += 1
    return count


def answer_open_calls(messages, text):
    """Give each tool call of the last assistant message that has no tool message yet one that says
    `text`. A task stopped in the middle of a step leaves such calls behind, and llama.cpp's template
    (like any OpenAI-style server) needs every call answered before the conversation goes on.
    Returns how many were added."""
    for i in range(len(messages) - 1, 0, -1):
        if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
            break
    else:
        return 0
    j, answered = i + 1, set()
    while j < len(messages) and messages[j].get("role") == "tool":
        answered.add(messages[j].get("tool_call_id"))
        j += 1
    missing = [{"role": "tool", "tool_call_id": call.get("id"), "content": text}
               for call in messages[i]["tool_calls"] if call.get("id") not in answered]
    messages[j:j] = missing
    return len(missing)


def unique_call_ids(calls, step):
    """Make the ids of one reply's tool calls present and distinct (each tool message names one)."""
    seen = set()
    for k, call in enumerate(calls):
        if not call.get("id") or call["id"] in seen:
            call["id"] = f"call_{step}_{k}"
        seen.add(call["id"])
    return calls
