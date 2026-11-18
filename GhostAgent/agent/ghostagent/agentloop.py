"""The agent loop: one task, step by step (observe, ask the model, run its tool calls), and the
Task> prompt's input."""

import itertools
import json
import time
import uuid
from datetime import datetime

from .console import LiveLine, c, clean, first_line, one_line, set_console_title
from .history import (answer_open_calls, count_images, drop_oldest_steps, image_chunks, old_result_steps,
                      shorten_old_results, trim_images, unique_call_ids)
from .modelclient import StepCancelled, step_status_hint, stream_chat
from .pagehelpers import CHECKIN_ALERT, with_alert
from .preflight import context_size, model_alias, preserves_reasoning, takes_reasoning_effort
from .prompts import system_prompt, task_message
from .questions import _keyboard
from .runstate import STOP
from .steptrace import Trace, trace_args_summary, trace_url
from .tools import canonical_args, describe_call, tool_list


# --------------------------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------------------------

def parse_tool_calls(calls):
    """([(call, arguments dict)], None) when every call's arguments are a JSON object, else
    ([], what is wrong). Nothing unreadable may go into the history: llama.cpp re-reads every
    stored tool call on each request and would fail every later request of the task."""
    parsed = []
    for call in calls:
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
        except ValueError as e:
            return [], str(e)
        if not isinstance(args, dict):
            return [], "the arguments are not a JSON object"
        parsed.append((call, args))
    return parsed, None


CUT_OFF_NOTE = "Your last reply was cut off. Think briefly, then call a tool."
NO_TOOL_NOTE = "Use one of your tools to continue, or call answer if you are done."
MAX_FAILED_REPLIES = 3

# What the model is told about the calls of a step that were not run, and about its answer. Every
# tool call gets a tool message with its id, so the conversation can always go on.
SKIPPED_AFTER_CHANGE = "Skipped: the page changed after an earlier call in this step."
SKIPPED_AFTER_FAILURE = "Skipped: an earlier call in this step did not succeed."
SKIPPED_AFTER_ANSWER = "Skipped: the task ended with the answer."
NOT_RUN_STOPPED = "Not run: the user stopped the task."
ANSWER_DELIVERED = "Answer delivered to the user."


class Agent:
    """The agent loop. One conversation (self.messages) goes on across tasks: after an answer, the
    user's next task is added to it as a follow-up, so the agent can be told "yes, send it" or "I
    have signed in, go on" and continue with everything it found. /new starts a new conversation,
    and so does a task that ended in an error."""

    def __init__(self, settings, server, browser):
        self.s = settings
        self.server = server
        self.browser = browser
        self.history = []          # (task, answer) of this session: the summary and the risky-click check
        self.messages = []         # the conversation the model sees
        self.dropped = 0           # steps of this conversation replaced by notes (drop_oldest_steps)
        self.used = 0              # tokens in the context after the last step
        self.tasks = 0             # tasks in this conversation
        self.summary_due = False   # the next task message carries the summary of earlier tasks
        self.started = None        # when this conversation began (the date in its system prompt)
        self.note = None           # the browser's note in this conversation's system prompt
        self.trace = Trace(settings)
        # The last task: task_id, steps, outcome (answer|error|cancelled|max_steps), wall_s, and the
        # outcome the model gave with its answer (tool_schema "h"), if any.
        self.last = None

    def new_conversation(self):
        """/new: the next task starts a new conversation. Earlier tasks and answers come into it only
        as a short summary, labelled as possibly holding text from web pages."""
        self.messages = []
        self.dropped = 0
        self.used = 0
        self.tasks = 0
        self.summary_due = bool(self.history)

    def reset(self):
        """Unattended runs: the next task starts with nothing from the earlier ones."""
        self.new_conversation()
        self.history.clear()
        self.summary_due = False

    def request_body(self, messages):
        s = self.s
        body = {
            "model": model_alias(s),
            "messages": messages,
            "tools": tool_list(s),
            "tool_choice": "required",
            # H Company's sampling for Holo4 (top_k 20, top_p 0.95, no min_p) at the agent-loop
            # temperature, sent explicitly: llama.cpp would otherwise add its own min_p 0.05.
            "temperature": float(s["temperature"]),
            "top_k": int(s["top_k"]),
            "top_p": float(s["top_p"]),
            "min_p": float(s["min_p"]),
            "stream": True,
            "return_progress": True,
            "chat_template_kwargs": {"enable_thinking": bool(s["thinking"])},
        }
        if preserves_reasoning(s):
            # Holo4's template shows the reasoning of earlier turns only when asked to.
            body["chat_template_kwargs"]["preserve_thinking"] = True
        effort = str(s.get("reasoning_effort") or "").strip()
        if effort and takes_reasoning_effort(s):
            # The 27B's template reads it (how long the model thinks before it acts); the 35B's
            # template has no such switch, so it is never sent there.
            body["chat_template_kwargs"]["reasoning_effort"] = effort
        if int(s["max_tokens"]) > 0:
            # Thinking included. Without a cap, a reasoning loop runs until the context is full.
            body["max_tokens"] = int(s["max_tokens"])
        if int(s["seed"]) != -1:
            body["seed"] = int(s["seed"])  # benchmarks: repeatable sampling
        return body

    def run(self, task):
        """Work on one task until the model answers. Returns the answer text; self.last says how it
        ended. Every step and the task's end go into the step trace."""
        task_id = uuid.uuid4().hex[:8]
        self.last = {"task_id": task_id, "steps": 0, "outcome": "error", "wall_s": 0.0, "answer_outcome": None}
        t_task = time.time()
        try:
            result, self.last["outcome"] = self._run(task, task_id)
            return result
        except (StepCancelled, KeyboardInterrupt):
            self.last["outcome"] = "cancelled"
            self.history.append((task, "(the user stopped this task before an answer)"))
            raise
        except Exception:
            # A conversation that ended in an error (a request the server would not take, a model
            # that kept failing, a closed browser) is not continued: the next task starts a new one,
            # with the summary, so one bad history cannot spoil every later task.
            self.history.append((task, "(no answer: the task ended with an error)"))
            self.new_conversation()
            raise
        finally:
            self.last["wall_s"] = round(time.time() - t_task, 3)
            self.trace.task(task_id, self.last["steps"], self.last["outcome"], self.last["wall_s"])
            end = getattr(self.browser, "end_task", None)
            if end:
                try:
                    end()
                except Exception:
                    pass

    def _location(self):
        where = getattr(self.browser, "location", None)
        return where() if where else None

    def _start_task(self, task):
        """Add the task to the conversation (a new one, or a follow-up in the current one)."""
        s = self.s
        note = self.browser.prompt_note
        if self.messages and note != self.note:
            # The browser changed since the conversation began (the user's Chrome could not be
            # reached again, so the browser on the drive took over): the system prompt would describe
            # the wrong one, so a new conversation starts, with the summary.
            self.new_conversation()
        earlier = self.history[-5:] if self.summary_due else None
        self.summary_due = False
        if self.messages:
            # A task stopped in the middle of a step may have left calls without a result.
            answer_open_calls(self.messages, NOT_RUN_STOPPED)
            self.messages.append({"role": "user", "content": task_message(s, task, True, earlier, self.started)})
        else:
            self.started = datetime.now()
            self.dropped = 0
            self.used = 0
            self.tasks = 0
            self.note = note
            self.messages = [
                {"role": "system", "content": system_prompt(s, note)},
                {"role": "user", "content": task_message(s, task, False, earlier, self.started)},
            ]
        self.tasks += 1

    def _run(self, task, task_id):
        s = self.s
        begin = getattr(self.browser, "begin_task", None)
        if begin:
            # What the user asked for (this task and the earlier ones of the session, never the
            # model's answers) decides which risky clicks and local addresses need a question.
            begin("\n".join([t for t, _ in self.history[-5:]] + [task]))
        refit = getattr(self.browser, "refit_window", None)
        if refit:
            # refit_window_each_task: a window the user maximized between tasks gets its size back.
            refit()
        self._start_task(task)
        messages = self.messages
        limit = int(s["max_steps"])
        failures = 0  # replies in a row without a usable tool call
        checkin = int(s.get("progress_checkin_steps") or 0)
        preserve = preserves_reasoning(s)
        enter_default = bool(s.get("write_enter_default", False))
        for step in range(1, limit + 1) if limit > 0 else itertools.count(1):
            if STOP.is_set():
                raise StepCancelled()
            self.last["steps"] = step
            rec = Trace.new_step(task_id, step)
            try:
                trimmed = False
                if self.used > context_size(s) * 3 // 4:
                    before_drop = self.dropped
                    self.dropped, trimmed = drop_oldest_steps(messages, self.dropped, reasoning_first=preserve)
                    rec["trimmed"] = trimmed
                    if self.dropped > before_drop and self.tasks > 1:
                        # Earlier tasks of this conversation may now be only notes: the next task
                        # message brings their summary. (Not when only old reasoning was removed:
                        # every step is still there then.)
                        self.summary_due = True
                t_obs = time.perf_counter()
                messages.append(self.browser.observation(bring_to_front=step == 1))
                if checkin > 0 and step > 1 and (step - 1) % checkin == 0:
                    with_alert(messages[-1], CHECKIN_ALERT.format(n=step - 1))
                rec["t_observation"] = round(time.perf_counter() - t_obs, 3)
                images = image_chunks(messages)
                first_shot = images[0][0] if images else len(messages)  # where an eviction changes the prompt
                rec["evicted"] = trim_images(messages, int(s["image_budget"]), str(s.get("image_eviction") or "rolling"),
                                             int(s.get("image_keep_after_evict") or 2))
                if rec["evicted"] or trimmed:
                    # The prompt changes here anyway: long tool results from older steps go too, but none
                    # before the first change (after a trim that is early on anyway).
                    shorten_old_results(messages, everywhere=trimmed, steps=old_result_steps(s["image_budget"]),
                                        since=0 if trimmed else first_shot)
                rec["images_in_prompt"] = count_images(messages)
                rec["url"] = trace_url(getattr(self.browser, "last_url", None))
                rec.update(getattr(self.browser, "last_shot", None) or {})
                if STOP.is_set():
                    raise StepCancelled()
                # No task text in the page: the site's own scripts can read anything drawn into it.
                self.browser.overlay("status", f"GhostAgent - step {step}")
                set_console_title(f"GhostAgent - step {step}")
                body = self.request_body(messages)
                self.trace.save_request(task_id, step, body)
                t0 = time.time()
                status = {"phase": "reading the screen", "tokens": 0}
                hint = step_status_hint()
                live = LiveLine(every=None)  # without a console only the step's summary line below is written

                def on_event(kind, val):
                    if kind == "prompt":
                        done, total = val
                        pct = int(100 * done / total) if total else 0
                        status["phase"] = f"reading the screen {pct}%"
                    elif kind in ("reasoning", "content", "tool"):
                        status["tokens"] += 1
                        status["phase"] = "thinking" if kind == "reasoning" else "acting"
                    live.show(f"step {step}: {status['phase']} ... {time.time() - t0:4.0f}s  {status['tokens']} tok{hint}")

                try:
                    msg = stream_chat(self.server.base, body, on_event, api_key=getattr(self.server, "api_key", None))
                finally:
                    live.clear(70 + len(hint))
                    rec["t_model"] = round(time.time() - t0, 3)
                dt = time.time() - t0
                timings = msg["timings"] or {}
                self.used = timings.get("cache_n", 0) + timings.get("prompt_n", 0) + timings.get("predicted_n", 0)
                for key in ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms"):
                    rec[key] = timings.get(key)
                rec["finish_reason"] = msg.get("finish_reason")
                rec["reasoning_chars"] = len(msg["reasoning"])

                # Every call of the reply runs, in order (Holo4 may chain several in one step).
                calls = unique_call_ids(list(msg["tool_calls"]), step)
                parsed, problem = parse_tool_calls(calls)
                if calls:
                    rec["tool"] = ",".join(call["function"]["name"] for call in calls)
                gen = timings.get("predicted_per_second")
                speed = f", {gen:.0f} tok/s" if gen else ""
                print(c(f"  step {step}  ({dt:.0f}s{speed})", "90"))
                # Everything the model writes is cleaned before it is printed: it can repeat a
                # page's text, escape sequences included (see clean).
                if msg["reasoning"]:
                    print(c("    thinks: " + one_line(clean(msg["reasoning"])), "90"))
                if msg["content"].strip() and not (calls and calls[-1]["function"]["name"] == "answer"):
                    print(c("    notes:  " + one_line(clean(msg["content"]), 300), "37"))

                if not parsed:
                    # No usable tool call: keep the text, never the broken call, and say what went wrong.
                    messages.append({"role": "assistant", "content": msg["content"]})
                    if msg.get("finish_reason") == "length":
                        note, rec["error"] = CUT_OFF_NOTE, "the reply was cut off (max_tokens)"
                    elif problem:
                        note = f"Your tool call could not be parsed ({problem}). Call a tool with valid JSON arguments."
                        rec["error"] = f"the tool call could not be parsed ({problem})"
                    else:
                        note, rec["error"] = NO_TOOL_NOTE, "no tool call"
                    messages.append({"role": "user", "content": note})
                    print(c("    " + rec["error"] + "; asking the model again", "33"))
                    failures += 1
                    if failures >= MAX_FAILED_REPLIES:
                        raise RuntimeError(f"The model's last {MAX_FAILED_REPLIES} replies had no usable tool call "
                                           f"(the last one: {rec['error']}). The task was stopped.")
                    continue
                failures = 0
                assistant = {"role": "assistant", "content": msg["content"], "tool_calls": calls}
                if preserve and msg["reasoning"]:
                    # preserve_reasoning: the reasoning goes back to the model with the next requests.
                    assistant["reasoning_content"] = msg["reasoning"]
                messages.append(assistant)
                todo = [(call, call["function"]["name"]) for call, _ in parsed]
                todo = [(call, name, canonical_args(name, args, enter_default))
                        for (call, name), (_, args) in zip(todo, parsed)]
                rec["args_summary"] = one_line(" ; ".join(trace_args_summary(name, args) for _, name, args in todo), 600)
                answer = self._run_calls(todo, msg, rec)
                if answer is not None:
                    self.history.append((task, one_line(answer, 600)))
                    return answer, "answer"
                if STOP.is_set():
                    raise StepCancelled()
            except (StepCancelled, KeyboardInterrupt):
                rec["error"] = "cancelled"
                raise
            except Exception as e:
                rec["error"] = first_line(e)
                raise
            finally:
                self.trace.step(rec)

        result = f"I stopped after {s['max_steps']} steps without finishing. The browser is left as it is."
        self.history.append((task, result))
        return result, "max_steps"

    def _run_calls(self, todo, msg, rec):
        """Run a step's tool calls in order, each answered by its own tool message. After a call that
        failed, or one that changed the page's address or the current tab, the rest are skipped (they
        were aimed at what the model saw). answer counts wherever it comes, after the calls before it.
        Returns the answer text, or None when the task goes on."""
        messages = self.messages
        answer = outcome = None
        skip = None
        executed, settled = [], []
        for call, name, args in todo:
            if skip is None and answer is not None:
                skip = SKIPPED_AFTER_ANSWER
            if skip:
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": skip})
                print(c("    skipped: " + clean(describe_call(name, args)), "90"))
                continue
            if name == "answer":
                # An empty answer falls back to the message the model wrote with it.
                # (content may also come as null: never the word "None")
                answer = str(args.get("content") or "").strip() or str(msg.get("content") or "").strip()
                outcome = args.get("outcome")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": ANSWER_DELIVERED})
                continue
            print(c("    does:   " + clean(describe_call(name, args)), "36"))
            before = self._location()
            try:
                result = self.browser.execute(name, args)
                ok = bool(getattr(self.browser, "last_ok", True))
            except Exception as e:
                result = f"Error: {first_line(e)}"
                ok = False
                rec["error"] = rec["error"] or first_line(e)
                print(c("    " + clean(result), "33"))
            timed = getattr(self.browser, "last_timings", None) or {}
            if timed.get("execute") is not None:
                executed.append(timed["execute"])
            if timed.get("settle") is not None:
                settled.append(timed["settle"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            if STOP.is_set():
                answer_open_calls(messages, NOT_RUN_STOPPED)
                answer = None
                break
            if not ok:
                skip = SKIPPED_AFTER_FAILURE
            elif before is not None and self._location() != before:
                skip = SKIPPED_AFTER_CHANGE
        rec["t_execute"] = round(sum(executed), 3) if executed else None
        rec["t_settle"] = round(sum(settled), 3) if settled else None
        if answer is not None:
            self.last["answer_outcome"] = outcome
        return answer


HELP = """  Type a task in plain English and press Enter, for example:
      find the weather forecast for Denver this weekend
      search youtube for lofi music and play the first video
  Text pasted as several lines counts as one task. After an answer, your next task continues the
  same conversation, so you can reply to a question the agent asked ("yes, send it") or follow up.
  While a task runs: press Ctrl+C to stop it. The agent only works in its own browser window. You
  can use that window yourself between tasks (for example to sign in to a site); the agent
  continues from whatever page is showing there.
  Commands:  /new  start a new conversation (earlier tasks are kept only as a short summary)
             /help  this text     /quit  exit (or just close the window)
             /copy  copy the last answer to the clipboard
             /forget-pc  remove what GhostAgent keeps on this PC (model copy, browser cache, screenshots)"""

# What the agent shows inside a web page. Never the task or the answer: the site's own scripts can
# read anything drawn into its page.
ANSWER_OVERLAY = "GhostAgent finished - see the console window"
STOPPED_OVERLAY = "GhostAgent stopped - see the console window"

# A paste of several lines reaches input() one line at a time, the rest waiting in the console's
# key buffer. Lines that follow within this many seconds belong to the same paste: a person cannot
# type the next line that fast, so a single typed line is still taken (almost) at once.
PASTE_GAP_SECONDS = 0.15


def read_task(prompt, keyboard=None, clock=time.monotonic, nap=time.sleep):
    """A task typed or pasted at the Task> prompt, as one text (a multi-line paste joined with line
    breaks). keyboard: what tells whether more keys are waiting (msvcrt, or a stand-in in tests);
    None = the console's, and no waiting at all without one (input redirected)."""
    lines = [input(prompt)]
    kb = keyboard if keyboard is not None else _keyboard()
    if kb is None:
        return lines[0]
    while True:
        deadline = clock() + PASTE_GAP_SECONDS
        while not kb.kbhit():
            if clock() >= deadline:
                return "\n".join(line.rstrip() for line in lines)
            nap(0.01)
        try:
            lines.append(input(""))
        except EOFError:
            return "\n".join(line.rstrip() for line in lines)
