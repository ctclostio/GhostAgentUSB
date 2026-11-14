"""The model server at run time: which flags the installed llama.cpp build knows, the model's layout
read from its GGUF header (and the VRAM reserve worked out from it), the model profiles, the choice
of build and GPU, programs Windows refuses to run, restarts after a crash, how a load is watched,
ports, CPU threads, and the small Windows helpers around them.

Nothing here starts llama-server with a model or uses the GPU: llama-server.exe only runs with
--help, the model server is tools\\bench\\fake_llama.py, and GPU details come from stubs."""

import json
import os
import socket
import struct
import subprocess
import time
import unittest
import unittest.mock as mock
from pathlib import Path

import helpers
from agent_support import AGENT, StubBrowser, call, quiet, scratch_paths, settings

REAL_POPEN = subprocess.Popen
CPU_EXE = AGENT.BIN / "cpu" / "llama-server.exe"
# The real 35B model, read only, when it is on this PC: GHOST_TEST_35B_GGUF names the file, else the
# stick's own models folder is looked in.
REAL_35B = Path(os.environ.get("GHOST_TEST_35B_GGUF") or AGENT.ROOT / "models" / "Holo4-35B-A3B-Q4_K_M.gguf")
DEVICE = {"id": "CUDA0", "name": "NVIDIA GeForce RTX 4090", "total": 24564, "free": 22000, "integrated": False}


def os_error(winerror):
    return OSError(0, "refused", None, winerror)


class Proc:
    """Popen's stand-in: a process that runs until terminated."""
    _handle = 0
    pid = 4

    def __init__(self):
        self.code = None
        self.terminated = 0

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated += 1
        self.code = 1

    def kill(self):
        self.code = 1

    def wait(self, timeout=None):
        return self.code


def write_gguf(path, meta, tensors, alignment=32, long_array=0):
    """A small GGUF (version 3) file: metadata values (int, float, bool, str, list of int) and tensors
    given as (name, bytes). long_array adds a tokenizer-like list of that many strings."""
    def text(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def value(v):
        if isinstance(v, bool):
            return struct.pack("<I?", 7, v)
        if isinstance(v, int):
            return struct.pack("<II", 4, v)
        if isinstance(v, float):
            return struct.pack("<If", 6, v)
        if isinstance(v, str):
            return struct.pack("<I", 8) + text(v)
        return struct.pack("<IIQ", 9, 5, len(v)) + struct.pack("<" + "i" * len(v), *v)

    items = dict(meta)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(items) + (1 if long_array else 0))
    for k, v in items.items():
        head += text(k) + value(v)
    if long_array:
        head += text("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, long_array)
        head += b"".join(text(f"tok{i}") for i in range(long_array))
    offset = 0
    for name, n in tensors:
        head += text(name) + struct.pack("<IQIQ", 1, n // 4, 0, offset)
        offset += -(-n // alignment) * alignment
    pad = -len(head) % alignment
    Path(path).write_bytes(head + b"\0" * pad + b"\x11" * offset)


TEST_META = {
    "general.architecture": "testmoe", "general.alignment": 32, "testmoe.block_count": 4,
    "testmoe.embedding_length": 32, "testmoe.attention.head_count": 4, "testmoe.attention.head_count_kv": 2,
    "testmoe.attention.key_length": 8, "testmoe.attention.value_length": 8, "testmoe.full_attention_interval": 2,
    "testmoe.ssm.state_size": 4, "testmoe.ssm.conv_kernel": 4, "testmoe.ssm.inner_size": 16,
    "testmoe.ssm.group_count": 2, "testmoe.rope.freq_base": 10000.0, "testmoe.use_parallel_residual": True,
}
TEST_TENSORS = [("token_embd.weight", 128), ("blk.0.ffn_gate_exps.weight", 256), ("blk.0.ffn_up_exps.weight", 256),
                ("blk.1.attn_k.weight", 64), ("blk.3.attn_k.weight", 64), ("output.weight", 96)]


class GgufTests(unittest.TestCase):
    def setUp(self):
        self.folder = scratch_paths(self)

    def test_header_gives_sizes_and_layout(self):
        path = self.folder / "m.gguf"
        write_gguf(path, TEST_META, TEST_TENSORS, long_array=1500)
        meta, sizes = AGENT.read_gguf_header(path)
        self.assertEqual(meta["general.architecture"], "testmoe")
        self.assertIsNone(meta["tokenizer.ggml.tokens"])  # long arrays are skipped, not read
        self.assertEqual(sizes, dict(TEST_TENSORS))
        layout = AGENT.gguf_layout(path)
        self.assertEqual((layout["expert_bytes"], layout["nonexpert_bytes"]), (512, 128 + 64 + 64 + 96))
        self.assertEqual((layout["kv_layers"], layout["kv_elements_per_token"]), (2, 2 * 2 * (8 + 8)))
        # 2 recurrent layers, f32 state: (conv - 1) * (inner + 2 * groups * state) + inner * state values each
        self.assertEqual((layout["recurrent_layers"], layout["recurrent_state_bytes"]), (2, 2 * 4 * (3 * 32 + 64)))

    def test_per_layer_kv_heads_and_layers_from_the_interval(self):
        meta = dict(TEST_META, **{"testmoe.attention.head_count_kv": [0, 2, 0, 2]})
        path = self.folder / "m.gguf"
        write_gguf(path, meta, [t for t in TEST_TENSORS if "attn_k" not in t[0]])  # no attention tensors to count
        layout = AGENT.gguf_layout(path)
        self.assertEqual((layout["kv_layers"], layout["kv_elements_per_token"]), (2, 64))

    def test_broken_files_are_refused(self):
        bad = self.folder / "bad.gguf"
        bad.write_bytes(b"GGML" + b"\0" * 64)
        with self.assertRaises(ValueError):
            AGENT.read_gguf_header(bad)
        path = self.folder / "cut.gguf"
        write_gguf(path, TEST_META, TEST_TENSORS)
        path.write_bytes(path.read_bytes()[:100])
        with self.assertRaises(ValueError):
            AGENT.read_gguf_header(path)
        self.assertIsNone(AGENT.model_layout(path))
        self.assertIsNone(AGENT.model_layout(self.folder / "missing.gguf"))

    def test_layout_is_kept_next_to_the_model_until_the_file_changes(self):
        path = self.folder / "m.gguf"
        write_gguf(path, TEST_META, TEST_TENSORS)
        first = AGENT.model_layout(path)
        kept = self.folder / "m.gguf.layout.json"
        self.assertTrue(kept.is_file())
        with mock.patch.object(AGENT, "gguf_layout", side_effect=AssertionError("read again")):
            self.assertEqual(AGENT.model_layout(path), first)
        write_gguf(path, dict(TEST_META, **{"testmoe.attention.head_count_kv": 1}), TEST_TENSORS)
        os.utime(path, ns=(time.time_ns(), time.time_ns() + 5_000_000_000))
        self.assertEqual(AGENT.model_layout(path)["kv_elements_per_token"], 2 * 1 * 16)

    def test_layout_goes_to_the_logs_when_the_model_folder_is_read_only(self):
        path = self.folder / "m.gguf"
        write_gguf(path, TEST_META, TEST_TENSORS)
        real_write = Path.write_text

        def write_text(p, *a, **kw):
            if p.parent == self.folder:
                raise PermissionError("read-only")
            return real_write(p, *a, **kw)

        with mock.patch.object(Path, "write_text", write_text):
            AGENT.model_layout(path)
        self.assertTrue((self.folder / "logs" / "m.gguf.layout.json").is_file())

    def test_reserve(self):
        layout = {"nonexpert_bytes": 1000 * 2**20, "kv_elements_per_token": 1024, "recurrent_state_bytes": 2**20}
        # KV at q8_0: 34 bytes per 32 values
        self.assertEqual(AGENT.kv_cache_bytes(layout, 4096), 4096 * 1024 * 34 // 32)
        self.assertEqual(AGENT.moe_reserve_mib(layout, 500 * 2**20, 4096),
                         1000 + 500 + 4 + 1 + AGENT.COMPUTE_RESERVE_MIB + 1)  # 4.25 MiB of KV, rounded up

    @unittest.skipUnless(REAL_35B.is_file(), "the 35B model file is not on this PC")
    def test_reserve_for_the_real_35b_header(self):
        """Only the header is read (well under a second); the file is opened read-only."""
        t0 = time.time()
        layout = AGENT.gguf_layout(REAL_35B)
        self.assertLess(time.time() - t0, 10)
        self.assertEqual((layout["arch"], layout["kv_layers"], layout["recurrent_layers"]), ("qwen35moe", 10, 30))
        reserve = AGENT.moe_reserve_mib(layout, 899282976, 65536)  # with the f16 projector, 64k context
        self.assertGreater(reserve, 4000)
        self.assertLess(reserve, 5000)
        self.assertLess(abs(reserve - AGENT.DEFAULT_MOE_RESERVE_MIB), 600)  # the hand-measured reserve


SAMPLE_HELP = """----- common params -----

-h,    --help, --usage                  print usage and exit
-m,    --model FNAME                    model path to load
-dev,  --device <dev1,dev2,..>          comma-separated list of devices to use for offloading (none = don't
                                        offload); see --list-devices
--cache-idle-slots, --no-cache-idle-slots
                                        save idle slots (default: enabled, requires --cache-ram)
-cram, --cache-ram N                    set the maximum cache size in MiB
--port PORT                             port to listen
"""


class FlagTests(unittest.TestCase):
    def setUp(self):
        self.folder = scratch_paths(self)

    def test_flags_are_read_from_definitions_only(self):
        flags = AGENT.parse_help_flags(SAMPLE_HELP)
        self.assertEqual(flags, {"-h", "--help", "--usage", "-m", "--model", "-dev", "--device", "--cache-idle-slots",
                                 "--no-cache-idle-slots", "-cram", "--cache-ram", "--port"})
        self.assertNotIn("--list-devices", flags)  # only mentioned in a description

    @unittest.skipUnless(CPU_EXE.is_file(), "no llama.cpp CPU build in GhostAgent\\bin")
    def test_the_installed_build_knows_every_required_flag(self):
        flags = AGENT.server_flags(CPU_EXE)
        self.assertEqual([f for f in AGENT.REQUIRED_SERVER_FLAGS if f not in flags], [])
        self.assertIn("--checkpoint-min-step", flags)
        cache = json.loads((self.folder / "logs" / "server-flags-cache.json").read_text(encoding="utf-8"))
        self.assertEqual(cache["cpu"]["fingerprint"], AGENT.build_fingerprint(CPU_EXE))
        self.assertIn("llama-server-impl.dll", cache["cpu"]["fingerprint"])  # the exe alone is the same in every release
        with mock.patch.object(AGENT.subprocess, "run", side_effect=AssertionError("--help ran again")):
            self.assertEqual(AGENT.server_flags(CPU_EXE), flags)
        cache["cpu"]["fingerprint"] = "another build"
        (self.folder / "logs" / "server-flags-cache.json").write_text(json.dumps(cache), encoding="utf-8")
        runs = []
        real_run = subprocess.run
        with mock.patch.object(AGENT.subprocess, "run", lambda *a, **kw: runs.append(a) or real_run(*a, **kw)):
            self.assertEqual(AGENT.server_flags(CPU_EXE), flags)
        self.assertEqual(len(runs), 1)

    def test_unreadable_help_means_every_flag_is_passed(self):
        exe = self.folder / "bin" / "x" / "llama-server.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"MZ")
        done = subprocess.CompletedProcess([], 0, stdout="something else entirely", stderr="")
        with mock.patch.object(AGENT.subprocess, "run", return_value=done):
            self.assertIsNone(AGENT.server_flags(exe))
        self.assertIsNone(AGENT.server_flags(self.folder / "missing" / "llama-server.exe"))
        groups = [("-m", ["m"]), ("--checkpoint-min-step", ["1536"])]
        self.assertEqual(AGENT.keep_supported_flags(groups, None), (groups, [], []))

    def test_a_blocked_program_is_named(self):
        exe = self.folder / "bin" / "x" / "llama-server.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"MZ")
        with mock.patch.object(AGENT.subprocess, "run", side_effect=os_error(4551)), \
                self.assertRaises(SystemExit) as cm:
            AGENT.server_flags(exe)
        self.assertIn("App Control", str(cm.exception))
        with mock.patch.object(AGENT.subprocess, "run", side_effect=os_error(4551)), \
                self.assertRaises(SystemExit) as cm:
            AGENT.list_devices(exe)
        self.assertIn("llama-server.exe", str(cm.exception))

    def test_keep_supported_flags(self):
        groups = [("-m", ["m"]), ("--checkpoint-min-step", ["1536"]), ("--no-slots", []), ("--alias", ["a"])]
        kept, dropped, missing = AGENT.keep_supported_flags(groups, {"-m", "--no-slots"})
        self.assertEqual(kept, [("-m", ["m"]), ("--no-slots", []), ("--alias", ["a"])])
        self.assertEqual((dropped, missing), (["--checkpoint-min-step"], ["--alias"]))
        self.assertEqual(AGENT.flat_args(kept), ["-m", "m", "--no-slots", "--alias", "a"])

    def test_server_flags_file_matches_the_code(self):
        doc = json.loads((helpers.REPO / "tools" / "server-flags.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["required"], list(AGENT.REQUIRED_SERVER_FLAGS))
        self.assertEqual(doc["optional"], list(AGENT.OPTIONAL_SERVER_FLAGS))
        self.assertEqual(set(doc["_optional_when"]), set(AGENT.OPTIONAL_SERVER_FLAGS))
        self.assertTrue(set(AGENT.LOCKDOWN_SERVER_FLAGS) <= set(AGENT.OPTIONAL_SERVER_FLAGS))


class ServerArgsTests(unittest.TestCase):
    """LlamaServer.start with the build, the model files and Popen replaced."""

    def start(self, s, device=DEVICE, backend="cuda13", supported=None, files=None, layout=None, server=None):
        folder = scratch_paths(self)
        exe = folder / "bin" / backend / "llama-server.exe"
        exe.parent.mkdir(parents=True, exist_ok=True)
        launched = []
        server = server or AGENT.LlamaServer(s)
        files = files or [Path("m.gguf"), Path("p.gguf")]
        with mock.patch.object(AGENT, "pick_backend", return_value=(backend, exe, device)), \
                mock.patch.object(AGENT, "model_files", return_value=files), \
                mock.patch.object(AGENT, "server_flags", return_value=supported), \
                mock.patch.object(AGENT, "model_layout", return_value=layout), \
                mock.patch.object(AGENT.subprocess, "Popen", lambda args, **kw: launched.append((args, kw)) or Proc()), \
                mock.patch.object(AGENT, "port_is_free", return_value=True), \
                mock.patch.object(AGENT, "port_bindable", return_value=True), quiet() as out:
            server.start()
        args, kw = launched[-1]
        return args, kw, server, out.getvalue()

    @staticmethod
    def value(args, flag):
        return args[args.index(flag) + 1]

    def test_prompt_cache_flags(self):
        args, kw, server, out = self.start(settings())
        self.assertEqual([self.value(args, f) for f in ("--ctx-checkpoints", "--checkpoint-min-step", "--cache-ram")],
                         ["8", "1536", "0"])
        self.assertIn("--no-cache-idle-slots", args)
        args, *_ = self.start(settings(cache_ram_mib=2048, checkpoint_min_step=0, ctx_checkpoints=16))
        self.assertEqual([self.value(args, f) for f in ("--ctx-checkpoints", "--checkpoint-min-step", "--cache-ram")],
                         ["16", "0", "2048"])
        self.assertNotIn("--no-cache-idle-slots", args)  # it needs the RAM cache

    def test_an_older_build_starts_without_the_flags_it_lacks(self):
        b11669 = set(AGENT.REQUIRED_SERVER_FLAGS) | set(AGENT.OPTIONAL_SERVER_FLAGS)
        b11669 -= {"--moe-cache-cpu-share", "--checkpoint-min-step", "--no-slots"}
        server = AGENT.LlamaServer(settings(moe_cache_cpu_share=0.25))
        args, kw, server, out = self.start(settings(), supported=b11669, server=server)
        for flag in ("--moe-cache-cpu-share", "--checkpoint-min-step", "--no-slots"):
            self.assertNotIn(flag, args)
        self.assertIn("does not know --checkpoint-min-step, --no-slots, --moe-cache-cpu-share; started without them", out)
        self.assertIn("locked down less tightly", out)  # --no-slots is part of the lock-down
        args, kw, server, out = self.start(settings(), supported=b11669, server=server)
        self.assertNotIn("does not know", out)  # said once per build and session
        missing = b11669 - {"--alias"}
        args, kw, server, out = self.start(settings(), supported=missing)
        self.assertIn("--alias", args)  # a required flag is passed anyway, with a warning
        self.assertIn("does not list --alias", out)

    def test_moe_cache_cpu_share_and_op_offload(self):
        args, kw, server, out = self.start(settings(moe_cache_cpu_share=0.5, op_offload_min_batch=64))
        self.assertEqual(self.value(args, "--moe-cache-cpu-share"), "0.5")
        self.assertEqual(kw["env"]["GGML_OP_OFFLOAD_MIN_BATCH"], "64")
        args, kw, server, out = self.start(settings())
        self.assertNotIn("--moe-cache-cpu-share", args)
        self.assertNotIn("GGML_OP_OFFLOAD_MIN_BATCH", kw["env"])

    def test_fast_profile_reserve_comes_from_the_model(self):
        layout = {"nonexpert_bytes": 2000 * 2**20, "kv_elements_per_token": 0, "recurrent_state_bytes": 0}
        args, kw, server, out = self.start(settings(moe_cache_cap_mib=99999), layout=layout)
        self.assertEqual(self.value(args, "--alias"), "holo4-35b-a3b")
        self.assertEqual(self.value(args, "-c"), "65536")
        self.assertIn("--no-reasoning-preserve", args)
        self.assertEqual(args[args.index("-ngl"):args.index("-ngl") + 5], ["-ngl", "99", "--cpu-moe", "--fit", "off"])
        # 22000 free - (2000 + 1024 compute) reserve - 1024 fit target
        self.assertEqual(self.value(args, "--moe-cache-mib"), str(22000 - 3024 - 1024))
        args, *_ = self.start(settings(), layout=layout)
        self.assertEqual(self.value(args, "--moe-cache-mib"), "4096")  # moe_cache_cap_mib
        args, *_ = self.start(settings(moe_cache_cap_mib=2048), layout=None)
        self.assertEqual(self.value(args, "--moe-cache-mib"), "2048")

    def test_accurate_profile_is_dense(self):
        folder = helpers.temp_dir(self)
        model, mmproj = folder / "Holo4-27B-Q4_K_M.gguf", folder / "Holo4-27B-mmproj-f16.gguf"
        model.write_bytes(b"x")
        mmproj.write_bytes(b"x" * (3 * 2**20 - 5))  # 3 MiB, rounded up
        args, kw, server, out = self.start(settings(model_profile="accurate"), files=[model, mmproj])
        self.assertEqual(self.value(args, "--alias"), "holo4-27b")
        self.assertEqual(self.value(args, "-c"), "32768")
        self.assertNotIn("--no-reasoning-preserve", args)  # its template renders past reasoning blocks
        for flag in ("-ngl", "--cpu-moe", "--fit", "--moe-cache-mib"):
            self.assertNotIn(flag, args)
        self.assertEqual(self.value(args, "--device"), "CUDA0")
        self.assertEqual(self.value(args, "--fit-target"), str(1024 + 3 + AGENT.VISION_RESERVE_MIB))
        self.assertFalse(server.explicit_layout)
        # a model setting that names the 27B file selects the profile as well
        args, *_ = self.start(settings(model="models/Holo4-27B-Q4_K_M.gguf", ctx_size=16384), files=[model, mmproj])
        self.assertEqual((self.value(args, "--alias"), self.value(args, "-c")), ("holo4-27b", "16384"))

    def test_a_gpu_build_without_a_gpu_offloads_nothing(self):
        args, *_ = self.start(settings(backend="vulkan"), device=None, backend="vulkan")
        self.assertEqual(self.value(args, "--device"), "none")
        args, *_ = self.start(settings(), device=None, backend="cpu")
        self.assertNotIn("--device", args)

    def test_threads_follow_the_performance_cores(self):
        with mock.patch.object(AGENT, "performance_cores", return_value=6):
            args, *_ = self.start(settings())
        self.assertEqual(self.value(args, "-t"), "6")
        with mock.patch.object(AGENT, "performance_cores", return_value=None), \
                mock.patch.object(AGENT.os, "cpu_count", return_value=4):
            args, *_ = self.start(settings())
        self.assertEqual(self.value(args, "-t"), "4")
        args, *_ = self.start(settings(threads=3))
        self.assertEqual(self.value(args, "-t"), "3")

    def test_graceful_stop_shares_the_console_and_sends_ctrl_break(self):
        with mock.patch.object(AGENT, "has_console", return_value=True):
            args, kw, server, out = self.start(settings(graceful_server_stop=True))
        self.assertEqual(kw["creationflags"] & helpers.CREATE_NO_WINDOW, 0)
        self.assertTrue(kw["creationflags"] & AGENT.CREATE_NEW_PROCESS_GROUP)
        sent = []
        with mock.patch.object(AGENT.os, "kill", lambda pid, sig: sent.append((pid, sig)) or setattr(server.proc, "code", 0)):
            server.stop()
        self.assertEqual(sent, [(server.proc.pid, AGENT.signal.CTRL_BREAK_EVENT)])
        self.assertEqual(server.proc.terminated, 0)  # it stopped by itself
        # Without a console of its own a child would open a window: then it is started as before.
        with mock.patch.object(AGENT, "has_console", return_value=False):
            args, kw, server, out = self.start(settings(graceful_server_stop=True))
        self.assertTrue(kw["creationflags"] & helpers.CREATE_NO_WINDOW)

    def test_the_log_handle_is_closed_after_the_start(self):
        args, kw, server, out = self.start(settings())
        self.assertTrue(kw["stdout"].closed)

    def test_a_blocked_server_program_is_named(self):
        folder = scratch_paths(self)
        exe = folder / "bin" / "cuda13" / "llama-server.exe"
        exe.parent.mkdir(parents=True)
        server = AGENT.LlamaServer(settings())
        with mock.patch.object(AGENT, "pick_backend", return_value=("cuda13", exe, DEVICE)), \
                mock.patch.object(AGENT, "model_files", return_value=[Path("m.gguf"), Path("p.gguf")]), \
                mock.patch.object(AGENT.subprocess, "Popen", side_effect=os_error(4551)), \
                mock.patch.object(AGENT, "port_is_free", return_value=True), quiet(), \
                self.assertRaises(SystemExit) as cm:
            server.start()
        self.assertIn("App Control", str(cm.exception))
        self.assertIn("cuda13", str(cm.exception))

    def test_reserved_ports_are_skipped(self):
        held = socket.socket()
        held.bind(("127.0.0.1", 0))  # bound, not listening: a connect test calls it free
        self.addCleanup(held.close)
        port = held.getsockname()[1]
        self.assertTrue(AGENT.port_is_free(port))
        self.assertFalse(AGENT.port_bindable(port))
        folder = scratch_paths(self)
        exe = folder / "bin" / "cpu" / "llama-server.exe"
        exe.parent.mkdir(parents=True)
        server = AGENT.LlamaServer(settings(port=port))
        with mock.patch.object(AGENT, "pick_backend", return_value=("cpu", exe, None)), \
                mock.patch.object(AGENT, "model_files", return_value=[Path("m.gguf"), Path("p.gguf")]), \
                mock.patch.object(AGENT, "port_is_free", return_value=True), \
                mock.patch.object(AGENT.subprocess, "Popen", lambda args, **kw: Proc()), quiet():
            server.start()
        self.assertNotEqual(server.port, port)
        server = AGENT.LlamaServer(settings(port=port))
        with mock.patch.object(AGENT, "pick_backend", return_value=("cpu", exe, None)), \
                mock.patch.object(AGENT, "model_files", return_value=[Path("m.gguf"), Path("p.gguf")]), \
                mock.patch.object(AGENT, "port_bindable", return_value=False), quiet(), \
                self.assertRaises(SystemExit) as cm:
            server.start()
        self.assertIn("No port between", str(cm.exception))

    def test_the_port_search_does_not_wait_on_a_free_port(self):
        """A connect test to a port nobody listens on waits about two seconds on Windows; the search
        uses a short timeout, and still skips a program that listens on every address."""
        listener = socket.socket()
        listener.bind(("0.0.0.0", 0))  # binding 127.0.0.1 on the same port still works on Windows
        listener.listen()
        self.addCleanup(listener.close)
        busy = listener.getsockname()[1]
        self.assertTrue(AGENT.port_bindable(busy))
        self.assertFalse(AGENT.port_is_free(busy, timeout=AGENT.PORT_CONNECT_TIMEOUT))
        free = helpers.free_port()
        t0 = time.monotonic()
        self.assertTrue(AGENT.port_is_free(free, timeout=AGENT.PORT_CONNECT_TIMEOUT))
        self.assertLess(time.monotonic() - t0, 1.5)
        folder = scratch_paths(self)
        exe = folder / "bin" / "cpu" / "llama-server.exe"
        exe.parent.mkdir(parents=True)
        server = AGENT.LlamaServer(settings(port=busy))
        t0 = time.monotonic()
        with mock.patch.object(AGENT, "pick_backend", return_value=("cpu", exe, None)), \
                mock.patch.object(AGENT, "model_files", return_value=[Path("m.gguf"), Path("p.gguf")]), \
                mock.patch.object(AGENT, "server_flags", return_value=None), \
                mock.patch.object(AGENT.subprocess, "Popen", lambda args, **kw: Proc()), quiet():
            server.start()
        self.assertNotEqual(server.port, busy)
        self.assertLess(time.monotonic() - t0, 3.0)  # was about 2 s per free port tried, plus the start


class LayoutArgsTests(unittest.TestCase):
    def test_moe_layout_uses_the_reserve(self):
        s = settings()
        args, explicit = AGENT.gpu_layout_args(DEVICE, s, reserve_mib=4326)
        self.assertTrue(explicit)
        self.assertEqual(args[-1], "4096")
        args, explicit = AGENT.gpu_layout_args(dict(DEVICE, free=6000), s, reserve_mib=4326)
        self.assertEqual(args[-2:], ["--moe-cache-mib", str(6000 - 4326 - 1024)])
        args, explicit = AGENT.gpu_layout_args(dict(DEVICE, free=5000), s, reserve_mib=4326)
        self.assertFalse(explicit)  # less than the reserve plus 900 MiB: llama.cpp fits it
        self.assertEqual(args, ["--device", "CUDA0", "--fit-target", "1024"])

    def test_dense_layout_always_fits_with_room_for_the_projector(self):
        args, explicit = AGENT.gpu_layout_args(DEVICE, settings(), dense=True, mmproj_mib=885)
        self.assertFalse(explicit)
        self.assertEqual(args, ["--device", "CUDA0", "--fit-target", str(1024 + 885 + AGENT.VISION_RESERVE_MIB)])
        args, explicit = AGENT.gpu_layout_args(DEVICE, settings(), dense=True, mmproj_mib=885, emulate_free_mib=12000)
        self.assertEqual(args[-1], str(1024 + 885 + AGENT.VISION_RESERVE_MIB + 22000 - 12000))


class ProfileTests(unittest.TestCase):
    def test_profiles_and_their_names(self):
        self.assertEqual(AGENT.profile_names(), ["accurate", "fast", "fast-q8"])
        s = settings()
        self.assertEqual((AGENT.active_profile(s)[0], AGENT.model_alias(s), AGENT.context_size(s)),
                         ("fast", "holo4-35b-a3b", 65536))
        self.assertEqual(AGENT.model_path(s).name, "Holo4-35B-A3B-Q4_K_M.gguf")
        self.assertEqual(AGENT.model_path(s, "mmproj").name, "Holo4-35B-A3B-mmproj-f16.gguf")
        s = settings(model_profile="fast-q8")
        self.assertEqual((AGENT.model_alias(s), AGENT.model_path(s).name, AGENT.model_path(s, "mmproj").name),
                         ("holo4-35b-a3b-q8", "Holo4-35B-A3B-Q8_0.gguf", "Holo4-35B-A3B-mmproj-f16.gguf"))
        s = settings(model_profile="accurate")
        self.assertTrue(AGENT.is_dense(s) and AGENT.takes_reasoning_effort(s))
        self.assertFalse(AGENT.is_dense(settings()) or AGENT.takes_reasoning_effort(settings()))
        self.assertEqual(AGENT.model_label(s), "Holo4-27B Q4_K_M")
        self.assertEqual(AGENT.model_label(settings(model="models/my-own.gguf")), "my-own")

    def test_reasoning_effort_goes_only_to_the_27b(self):
        fake = helpers.start_fake_llama(self)
        body = AGENT.Agent(settings(reasoning_effort="low"), StubServerFor(fake), StubBrowser()).request_body([])
        self.assertEqual(body["model"], "holo4-35b-a3b")
        self.assertNotIn("reasoning_effort", body["chat_template_kwargs"])
        body = AGENT.Agent(settings(model_profile="accurate", reasoning_effort="low"), StubServerFor(fake),
                           StubBrowser()).request_body([])
        self.assertEqual(body["model"], "holo4-27b")
        # preserve_reasoning "auto": the dense 27B gets its earlier reasoning back (see test_agent_loop).
        self.assertEqual(body["chat_template_kwargs"],
                         {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "low"})
        body = AGENT.Agent(settings(model_profile="accurate"), StubServerFor(fake), StubBrowser()).request_body([])
        self.assertNotIn("reasoning_effort", body["chat_template_kwargs"])

    def test_settings_rules(self):
        c = AGENT.coerce_setting
        self.assertEqual([c("copy_model_to_pc", v, "ask") for v in ("ask", "", None, "true", False, "no")],
                         ["ask", "ask", "ask", True, False, False])
        with self.assertRaises(AGENT.BadSetting):
            c("copy_model_to_pc", "sometimes", "ask")
        self.assertEqual([c("moe_cache_cpu_share", v, "") for v in ("", " ", None, "0.5", 1)], ["", "", "", 0.5, 1.0])
        self.assertEqual([c("op_offload_min_batch", v, "") for v in ("", "64", 128)], ["", 64, 128])
        with self.assertRaises(AGENT.BadSetting):
            c("op_offload_min_batch", "1.5", "")
        with self.assertRaises(AGENT.BadSetting):
            AGENT.check_setting("moe_cache_cpu_share", 1.5)
        with self.assertRaises(AGENT.BadSetting):
            AGENT.check_setting("ctx_size", 1000)
        self.assertEqual(AGENT.check_setting("ctx_size", 0), 0)
        self.assertEqual(AGENT.check_setting("model_profile", "Accurate"), "accurate")
        with self.assertRaises(AGENT.BadSetting):
            AGENT.check_setting("model_profile", "huge")
        for mode in ("user_edge", "auto", "chrome-agent"):
            self.assertEqual(AGENT.check_setting("browser", mode), mode)


class StubServerFor:
    def __init__(self, fake):
        self.base = fake.base
        self.api_key = None

    def alive(self):
        return True


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.folder = scratch_paths(self)
        bin_dir = self.folder / "bin"
        for name in AGENT.BACKENDS:
            (bin_dir / name).mkdir(parents=True)
            (bin_dir / name / "llama-server.exe").write_bytes(b"MZ")
        patcher = mock.patch.object(AGENT, "BIN", bin_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.seen = {"cuda13": [DEVICE], "cuda12": [DEVICE], "vulkan": [dict(DEVICE, id="Vulkan0")]}
        self.listed = []
        self.gpus = [{"name": DEVICE["name"], "compute_cap": 8.9, "driver": "621.01", "driver_major": 621}]
        self.nvidia = True
        for name, value in (("list_devices", self.list_devices), ("nvidia_smi_gpus", lambda: self.gpus),
                            ("nvidia_gpu_present", lambda: self.nvidia)):
            patcher = mock.patch.object(AGENT, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def list_devices(self, exe):
        self.listed.append(exe.parent.name)
        return [dict(d) for d in self.seen.get(exe.parent.name, [])]

    def pick(self, after=None, **over):
        with quiet() as out:
            backend, exe, device = AGENT.pick_backend(settings(**over), after=after)
        return backend, device, out.getvalue()

    def test_first_build_with_a_gpu(self):
        self.assertEqual(self.pick()[0], "cuda13")
        self.assertEqual(self.listed, ["cuda13"])

    def test_cuda13_is_skipped_for_old_gpus_and_drivers(self):
        self.gpus[0]["compute_cap"] = 6.1
        backend, device, out = self.pick()
        self.assertEqual(backend, "cuda12")
        self.assertNotIn("cuda13", self.listed)
        self.assertIn("compute capability 6.1", out)
        self.gpus[0].update(compute_cap=8.9, driver="560.94", driver_major=560)
        self.listed.clear()
        backend, device, out = self.pick()
        self.assertEqual((backend, self.listed), ("cuda12", ["cuda12"]))
        self.assertIn("older than 580", out)

    def test_after_a_failed_build_the_next_one(self):
        self.assertEqual(self.pick(after="cuda13")[0], "cuda12")
        self.assertEqual(self.pick(after="cuda12")[0], "vulkan")
        self.assertEqual(self.pick(after="vulkan")[0], "cpu")

    def test_known_bad_builds_are_skipped(self):
        AGENT.remember_bad_backend(AGENT.gpu_key(DEVICE, self.gpus), "cuda13")
        AGENT.remember_bad_backend(AGENT.gpu_key(DEVICE, self.gpus), "cuda13")  # once is enough
        doc = json.loads((self.folder / "logs" / "backend-cache.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["bad"], {"NVIDIA GeForce RTX 4090 | driver 621.01": ["cuda13"]})
        backend, device, out = self.pick()
        self.assertEqual(backend, "cuda12")
        self.assertIn("could not run the model on this GPU before", out)
        self.gpus[0]["driver"] = "630.00"  # a new driver: tried again
        self.assertEqual(self.pick()[0], "cuda13")

    def test_integrated_gpus_are_not_used_and_an_nvidia_card_without_driver_is_reported(self):
        self.seen = {"vulkan": [dict(DEVICE, id="Vulkan0", name="Intel(R) UHD Graphics", integrated=True)]}
        self.gpus = []
        backend, device, out = self.pick()
        self.assertEqual((backend, device), ("cpu", None))
        self.assertIn("Update the NVIDIA driver", out)
        self.nvidia = False
        self.assertNotIn("NVIDIA", self.pick()[2])

    def test_a_forced_build_without_gpu(self):
        self.seen = {}
        backend, device, out = self.pick(backend="vulkan")
        self.assertEqual((backend, device), ("vulkan", None))
        self.assertIn("found no usable GPU", out)

    def test_integrated_by_dedicated_memory(self):
        adapters = [{"name": "Intel(R) UHD Graphics 770", "vendor": 0x8086, "dedicated_mib": 128},
                    {"name": "AMD Radeon RX 7600", "vendor": 0x1002, "dedicated_mib": 8176},
                    {"name": "AMD Radeon(TM) Graphics", "vendor": 0x1002, "dedicated_mib": 512}]
        dev = {"id": "Vulkan1", "name": "Intel(R) UHD Graphics 770", "total": 32620, "free": 32620}
        self.assertTrue(AGENT.is_integrated(dev, adapters))
        self.assertFalse(AGENT.is_integrated(dict(dev, name="AMD Radeon RX 7600", total=8176, free=7000), adapters))
        self.assertFalse(AGENT.is_integrated(dict(dev, name="AMD Radeon(TM) Graphics"), adapters))  # 512 MiB carve-out
        # not listed by Windows: all of its memory free means shared memory
        self.assertTrue(AGENT.is_integrated(dict(dev, name="Unknown iGPU"), adapters))
        self.assertFalse(AGENT.is_integrated(dict(dev, name="Unknown dGPU", free=1000), adapters))
        self.assertFalse(AGENT.is_integrated(dict(dev, id="CUDA0"), adapters))

    def test_this_pcs_adapters_can_be_read(self):
        adapters = AGENT.dxgi_adapters()
        self.assertIsInstance(adapters, list)
        for a in adapters:
            self.assertTrue(a["name"] and isinstance(a["dedicated_mib"], int))

    def test_nvidia_smi_output(self):
        gpus = AGENT.parse_nvidia_smi("NVIDIA GeForce RTX 4090, 8.9, 621.01\nQuadro K620, [N/A], 472.12\n\n")
        self.assertEqual(gpus[0], {"name": "NVIDIA GeForce RTX 4090", "compute_cap": 8.9, "driver": "621.01",
                                   "driver_major": 621})
        self.assertIsNone(gpus[1]["compute_cap"])
        self.assertIn("472.12", AGENT.cuda13_unsuitable(gpus[1:]))
        self.assertIsNone(AGENT.cuda13_unsuitable(gpus))  # the best card decides
        self.assertIsNone(AGENT.cuda13_unsuitable([]))


class FakeServer:
    """start_model_server's view of a server: each wait_ready gives the next result."""

    def __init__(self, results, builds=("cuda13", "cuda12", "vulkan", "cpu")):
        self.results, self.builds, self.i = list(results), builds, 0
        self.explicit_layout = False
        self.quit_requested = False
        self.calls = []
        self.remembered = False

    def start(self, auto_fit=False):
        self.calls.append((self.builds[self.i], auto_fit))
        self.explicit_layout = not auto_fit and self.builds[self.i] != "cpu"
        return self.builds[self.i]

    def wait_ready(self):
        return self.results.pop(0)

    def can_fall_back(self):
        return self.builds[self.i] != "cpu" and not self.quit_requested

    def next_backend(self):
        self.i += 1
        return self.builds[self.i]

    def remember_failed_backends(self):
        self.remembered = True


class StartAndRestartTests(unittest.TestCase):
    def setUp(self):
        self.folder = scratch_paths(self)

    def test_layout_retry_then_the_next_build(self):
        server = FakeServer([None, None, 12.5])
        with quiet() as out:
            self.assertEqual(AGENT.start_model_server(server), ("cuda12", 12.5))
        self.assertEqual(server.calls, [("cuda13", False), ("cuda13", True), ("cuda12", False)])
        self.assertTrue(server.remembered)
        self.assertIn("The cuda13 build could not run the model; trying the cuda12 build", out.getvalue())

    def test_all_builds_fail_and_quit_stops_the_search(self):
        server = FakeServer([None] * 7)
        with quiet():
            self.assertEqual(AGENT.start_model_server(server), ("cpu", None))
        self.assertEqual(server.calls[-1], ("cpu", False))
        self.assertFalse(server.remembered)
        server = FakeServer([None])
        server.quit_requested = True
        with quiet():
            self.assertEqual(AGENT.start_model_server(server, "cuda13"), ("cuda13", None))
        self.assertEqual(server.calls, [])  # begin() had started it already

    def test_failed_builds_are_remembered_once_another_works(self):
        dev = dict(DEVICE)
        server = AGENT.LlamaServer(settings())
        picks = {None: ("cuda13", Path("13"), dev), "cuda13": ("cuda12", Path("12"), dev)}
        with mock.patch.object(AGENT, "pick_backend", lambda s, after=None: picks[after]), \
                mock.patch.object(AGENT, "nvidia_smi_gpus", return_value=[]), quiet():
            server.prepare(interactive=False)
            self.assertTrue(server.can_fall_back())
            self.assertEqual(server.next_backend(), "cuda12")
            server.remember_failed_backends()
        self.assertEqual(AGENT.known_bad_backends(), {DEVICE["name"]: ["cuda13"]})

    def test_failed_builds_are_not_remembered_when_only_the_cpu_worked(self):
        """Every GPU build failing while the CPU build works may only mean the GPU was short of free
        memory then (a game was running): later starts must still try the GPU."""
        dev = dict(DEVICE)
        server = AGENT.LlamaServer(settings())
        picks = {None: ("cuda13", Path("13"), dev), "cuda13": ("cuda12", Path("12"), dev),
                 "cuda12": ("cpu", Path("cpu"), None)}
        with mock.patch.object(AGENT, "pick_backend", lambda s, after=None: picks[after]), \
                mock.patch.object(AGENT, "nvidia_smi_gpus", return_value=[]), quiet():
            server.prepare(interactive=False)
            server.next_backend()
            self.assertEqual(server.next_backend(), "cpu")
            server.remember_failed_backends()
        self.assertEqual(AGENT.known_bad_backends(), {})
        self.assertEqual(server.failed, [])

    def fake_llama_server(self, s):
        """A LlamaServer whose 'llama-server' is fake_llama.py (it takes the same command line)."""
        exe = self.folder / "bin" / "cpu" / "llama-server.exe"
        exe.parent.mkdir(parents=True, exist_ok=True)

        def popen(args, **kw):
            cmd = [str(helpers.PYTHON), "-I", str(helpers.BENCH / "fake_llama.py")] + [str(a) for a in args[1:]]
            proc = REAL_POPEN(cmd, stdout=kw["stdout"], stderr=kw["stderr"], env=kw["env"],
                              creationflags=kw["creationflags"])
            self.addCleanup(proc.wait, 10)
            self.addCleanup(proc.kill)
            return proc

        for name, value in (("pick_backend", lambda s, after=None: ("cpu", exe, None)),
                            ("model_files", lambda s, **kw: [Path("m.gguf"), Path("p.gguf")])):
            patcher = mock.patch.object(AGENT, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(AGENT.subprocess, "Popen", popen)
        patcher.start()
        self.addCleanup(patcher.stop)
        return AGENT.LlamaServer(s)

    def test_a_crashed_server_is_restarted_twice_at_most(self):
        server = self.fake_llama_server(settings(port=helpers.free_port()))
        with quiet():
            backend, secs = AGENT.start_model_server(server)
        self.assertIsNotNone(secs)
        for n in (1, 2):
            first_log = server.log_path
            server.proc.kill()
            server.proc.wait(10)
            self.assertFalse(server.alive())
            with quiet() as out:
                self.assertTrue(AGENT.recover_server(server))
            self.assertIn(f"crashed and was restarted (the log of the crash: {first_log.name})", out.getvalue())
            self.assertTrue(server.alive())
            self.assertEqual(server.restarts, n)
        server.proc.kill()
        server.proc.wait(10)
        with quiet(), self.assertRaises(SystemExit) as cm:
            AGENT.recover_server(server)
        self.assertIn("after 2 restarts", str(cm.exception))
        server.stop()

    def test_a_server_that_cannot_be_restarted_ends_the_session(self):
        server = self.fake_llama_server(settings(port=helpers.free_port()))
        with quiet():
            AGENT.start_model_server(server)
        server.proc.kill()
        server.proc.wait(10)
        with mock.patch.object(server, "wait_ready", return_value=None), quiet(), self.assertRaises(SystemExit) as cm:
            AGENT.recover_server(server)
        self.assertIn("could not be started again", str(cm.exception))
        server.stop()

    def test_an_external_server_is_not_restarted(self):
        server = AGENT.ExternalServer("http://127.0.0.1:1")
        with quiet() as out:
            self.assertFalse(AGENT.recover_server(server))
        self.assertIn("The model server has stopped", out.getvalue())

    def test_run_interactive_restarts_after_a_crash_and_copies_the_last_answer(self):
        fake = helpers.start_fake_llama(self, script=[{"tool_calls": [call("answer", content="first answer")]},
                                                      {"tool_calls": [call("answer", content="second")]}])

        class Crashy(AGENT.ExternalServer):
            """A server the agent started (it has a process) that crashes between two tasks."""
            down = False
            restarted = 0

            def __init__(self, url):
                super().__init__(url)
                self.proc = object()

            def alive(self):
                return not self.down

            def can_restart(self):
                return True

            def restart(self):
                self.down = False
                self.restarted += 1
                return 0.5

        server = Crashy(fake.base)
        browser = StubBrowser()
        browser.start = lambda start_url=None: None
        browser.close = lambda: None
        browser.quit_note = "Remote debugging stays allowed in your Chrome until you turn it off."
        answers = ["first task", "/copy", lambda: setattr(server, "down", True) or "second task", "/copy", "/quit"]
        copied = []

        def fake_input(prompt=""):
            value = answers.pop(0)
            return value() if callable(value) else value

        with quiet() as out, mock.patch("builtins.input", fake_input), \
                mock.patch.object(AGENT, "copy_to_clipboard", lambda text: copied.append(text) or True):
            self.assertEqual(AGENT.run_interactive(settings(), server, browser), 0)
        text = out.getvalue()
        self.assertEqual(server.restarted, 1)
        self.assertIn("crashed and was restarted", text)
        self.assertNotIn("Run the task again", text)  # it crashed between tasks: the typed task just runs
        self.assertEqual(copied, ["first answer", "second"])
        self.assertIn("The last answer is on the clipboard", text)
        self.assertIn("Remote debugging stays allowed", text)


class LoadWatchTests(unittest.TestCase):
    GB = 10**9

    def test_progress_line_speed_and_time_left(self):
        w = AGENT.LoadWatch(int(22.2 * self.GB), 0.0)
        self.assertEqual(w.line(3), "loading the model into memory ... 3s")
        w.update(0.0, 0, 100, 1.0)
        w.update(4.0, int(4 * 69e6), 100, 2.0)
        self.assertEqual(w.line(4), "loading the model: 0.3 of 22.2 GB (69 MB/s, about 5 min left)")
        w.update(8.0, int(22.2 * self.GB), 120, 3.0)
        self.assertEqual(w.line(9), "loading the model: all read, getting it ready ... 9s")
        self.assertAlmostEqual(w.average_rate(), 22.2 * self.GB / 8, delta=1)

    def test_a_slow_read_is_hinted_once(self):
        w = AGENT.LoadWatch(20 * self.GB, 0.0)
        events = []
        for t in range(0, 40):
            events += w.update(float(t), t * 30_000_000, t, float(t))  # 30 MB/s
        self.assertEqual(events, ["slow"])
        w = AGENT.LoadWatch(20 * self.GB, 0.0)
        fast = []
        for t in range(0, 40):
            fast += w.update(float(t), t * 300_000_000, t, float(t))
        self.assertEqual(fast, [])

    def test_a_stall_is_nothing_read_logged_or_computed(self):
        w = AGENT.LoadWatch(20 * self.GB, 0.0)
        w.update(1.0, 5, 10, 1.0)
        w.update(50.0, 5, 10, 1.0)
        self.assertEqual(w.stalled_for(50.0), 49.0)
        w.update(60.0, 5, 11, 1.0)  # a log line is a sign of life
        self.assertEqual(w.stalled_for(61.0), 1.0)
        w.update(90.0, 5, 11, 1.5)  # so is CPU time
        self.assertEqual(w.stalled_for(91.0), 1.0)
        self.assertIsNone(w.average_rate())  # less than 1 GiB read while watched

    def test_eta_text(self):
        self.assertEqual([AGENT.eta_text(s) for s in (3, 44, 89, 95, 600)],
                         ["about 5 s", "about 45 s", "about 90 s", "about 2 min", "about 10 min"])

    def test_load_speed_keeps_the_slowest(self):
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": str(helpers.temp_dir(self))}):
            AGENT.record_load_speed(69e6)
            AGENT.record_load_speed(3000e6)  # a second start from Windows' file cache
            self.assertEqual(AGENT.read_pc_settings()["load_mb_s"], 69)
            AGENT.record_load_speed(40e6)
            AGENT.record_load_speed(None)
            self.assertEqual(AGENT.read_pc_settings()["load_mb_s"], 40)


class StallTests(unittest.TestCase):
    def setUp(self):
        scratch_paths(self)

    def waiting_server(self, unattended):
        server = AGENT.LlamaServer(settings())
        server.proc = Proc()
        server.log_path = helpers.temp_dir(self) / "server.log"
        server.log_path.write_text("loading\n", encoding="utf-8")
        server.load_bytes = 20 * 10**9
        server.unattended = unattended
        return server

    def test_an_unattended_stall_gives_up(self):
        server = self.waiting_server(True)
        with mock.patch.object(AGENT, "UNATTENDED_STALL_SECONDS", 1), \
                mock.patch.object(AGENT, "server_healthy", return_value=False), \
                mock.patch.object(AGENT, "process_read_bytes", return_value=1000), \
                mock.patch.object(AGENT, "process_cpu_seconds", return_value=2.0), quiet() as out:
            self.assertIsNone(server.wait_ready())
        self.assertTrue(server.quit_requested)
        self.assertEqual(server.proc.terminated, 1)
        self.assertIn("no progress", out.getvalue())
        self.assertIn("loading", out.getvalue())  # the log's last lines are shown

    def test_the_user_can_restart_or_quit_a_stalled_load(self):
        server = self.waiting_server(False)
        healthy = []
        starts = []

        def start(auto_fit=False):
            starts.append(auto_fit)
            server.proc = Proc()
            healthy.append(True)  # the new start comes up

        with mock.patch.object(AGENT, "STALL_SECONDS", 1), \
                mock.patch.object(server, "_can_ask", return_value=True), \
                mock.patch.object(server, "start", start), \
                mock.patch.object(AGENT, "ask_console", return_value=("r", "answered")), \
                mock.patch.object(AGENT, "server_healthy", lambda base: bool(healthy)), \
                mock.patch.object(AGENT, "process_read_bytes", return_value=None), \
                mock.patch.object(AGENT, "process_cpu_seconds", return_value=None), quiet():
            self.assertIsNotNone(server.wait_ready())
        self.assertEqual(starts, [False])
        # A start that llama.cpp laid out itself (the auto-fit retry) is restarted the same way.
        server = self.waiting_server(False)
        server._auto_fit = True
        healthy.clear()
        with mock.patch.object(AGENT, "STALL_SECONDS", 1), \
                mock.patch.object(server, "_can_ask", return_value=True), \
                mock.patch.object(server, "start", start), \
                mock.patch.object(AGENT, "ask_console", return_value=("r", "answered")), \
                mock.patch.object(AGENT, "server_healthy", lambda base: bool(healthy)), \
                mock.patch.object(AGENT, "process_read_bytes", return_value=None), \
                mock.patch.object(AGENT, "process_cpu_seconds", return_value=None), quiet():
            self.assertIsNotNone(server.wait_ready())
        self.assertEqual(starts, [False, True])
        server = self.waiting_server(False)
        with mock.patch.object(AGENT, "STALL_SECONDS", 1), \
                mock.patch.object(server, "_can_ask", return_value=True), \
                mock.patch.object(AGENT, "ask_console", return_value=("q", "answered")), \
                mock.patch.object(AGENT, "server_healthy", return_value=False), \
                mock.patch.object(AGENT, "process_read_bytes", return_value=None), \
                mock.patch.object(AGENT, "process_cpu_seconds", return_value=None), quiet():
            self.assertIsNone(server.wait_ready())
        self.assertTrue(server.quit_requested)


class WindowsHelperTests(unittest.TestCase):
    def test_blocked_messages(self):
        what = AGENT.LLAMA_SERVER_WHAT
        self.assertIn("App Control", AGENT.blocked_message(os_error(4551), what))
        self.assertIn("group policy", AGENT.blocked_message(os_error(1260), what))
        self.assertIn("Defender", AGENT.blocked_message(os_error(225), what))
        self.assertIn("refused to start", AGENT.blocked_message(os_error(5), what))
        self.assertIsNone(AGENT.blocked_message(os_error(2), what))
        e = ImportError("DLL load failed while importing _greenlet: An Application Control policy has blocked this file.")
        self.assertIn("App Control", AGENT.blocked_message(e, "Playwright"))
        self.assertIsNone(AGENT.blocked_message(ImportError("No module named x"), "Playwright"))

    def test_a_blocked_playwright_is_named(self):
        scratch_paths(self)
        browser = AGENT.Browser(settings(browser="bundled"), headless=True)
        error = ImportError("DLL load failed while importing _greenlet: An Application Control policy has blocked this file.")
        with mock.patch("playwright.sync_api.sync_playwright", side_effect=error), \
                self.assertRaises(SystemExit) as cm:
            browser.start()
        self.assertIn("Playwright", str(cm.exception))
        self.assertIn("App Control", str(cm.exception))

    def test_performance_cores_from_the_processor_records(self):
        def record(efficiency):
            return struct.pack("<II", 0, 16) + bytes([0, efficiency]) + b"\0" * 6

        self.assertEqual(AGENT.p_cores_from(b"".join(record(c) for c in (1, 1, 0, 0, 0, 0))), 2)
        self.assertIsNone(AGENT.p_cores_from(b"".join(record(0) for _ in range(8))))  # all alike
        mixed = record(1) + struct.pack("<II", 2, 12) + b"\0" * 4 + record(0)  # another kind of record between
        self.assertEqual(AGENT.p_cores_from(mixed), 1)
        p = AGENT.performance_cores()
        self.assertTrue(p is None or 1 <= p <= os.cpu_count())
        with mock.patch.object(AGENT, "performance_cores", return_value=16):
            self.assertEqual(AGENT.default_threads(), 8)

    def test_keep_pc_awake(self):
        calls = []

        class K32:
            class SetThreadExecutionState:
                def __new__(cls, flags):
                    calls.append(flags)
                    return 1

        class Windll:
            kernel32 = K32

        with mock.patch.object(AGENT.ctypes, "windll", Windll):
            with AGENT.keep_pc_awake():
                self.assertEqual(calls, [AGENT.ES_CONTINUOUS | AGENT.ES_SYSTEM_REQUIRED])
            self.assertEqual(calls[-1], AGENT.ES_CONTINUOUS)
            with self.assertRaises(KeyError), AGENT.keep_pc_awake():
                raise KeyError("x")
            self.assertEqual(calls[-1], AGENT.ES_CONTINUOUS)  # ends with the block, however it ends

    def test_quick_edit_is_turned_off(self):
        set_modes = []

        class K32:
            class GetStdHandle:
                restype = None

                def __new__(cls, which):
                    return 7

            @staticmethod
            def GetConsoleMode(handle, ref):
                ref._obj.value = 0x01F7  # QuickEdit (0x40) and extended flags (0x80) among others
                return 1

            @staticmethod
            def SetConsoleMode(handle, mode):
                set_modes.append(mode)
                return 1

        class Windll:
            kernel32 = K32

        self.addCleanup(setattr, AGENT, "_SAVED_CONSOLE_MODE", None)
        with mock.patch.object(AGENT.ctypes, "windll", Windll):
            self.assertTrue(AGENT.disable_quick_edit())
            self.assertEqual(set_modes, [0x01F7 & ~0x40 | 0x80])
            # The console may be a command prompt the user keeps using: its mode comes back at the end.
            AGENT.restore_quick_edit()
            self.assertEqual(set_modes[-1], 0x01F7)
            AGENT.restore_quick_edit()  # only once
            self.assertEqual(len(set_modes), 2)

    def test_run_main_puts_the_console_mode_back(self):
        with mock.patch.object(AGENT, "main", side_effect=SystemExit(0)), mock.patch.object(AGENT, "HEADLESS", False), \
                mock.patch.object(AGENT, "restore_quick_edit") as restore:
            self.assertEqual(AGENT.run_main(["--task", "x"]), 0)
        restore.assert_called_once()

    def test_step_hint_only_in_a_wide_console(self):
        size = os.terminal_size
        with mock.patch.object(AGENT.shutil, "get_terminal_size", return_value=size((120, 30))):
            self.assertIn("Ctrl+C stops this task", AGENT.step_status_hint())
        with mock.patch.object(AGENT.shutil, "get_terminal_size", return_value=size((80, 30))):
            self.assertEqual(AGENT.step_status_hint(), "")

    def test_process_counters_of_a_real_child(self):
        child = helpers.start_python(["-c", "import sys; open(sys.executable, 'rb').read(); input()"],
                                     stdin=subprocess.PIPE)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        self.assertTrue(helpers.wait_until(lambda: (AGENT.process_read_bytes(child) or 0) > 10000, timeout=20))
        self.assertIsNotNone(AGENT.process_cpu_seconds(child))
        self.assertIsNone(AGENT.process_read_bytes(object()))
        child.stdin.close()
        child.wait(10)
        child.stdout.close()


if __name__ == "__main__":
    unittest.main()
