"""The model file's layout, read from its GGUF header: how much of it must sit in VRAM."""

import json
import math
import os
import re
import struct
from pathlib import Path

from . import paths


# --------------------------------------------------------------------------------------------
# The model file's layout, read from its GGUF header: how much of it must sit in VRAM
# --------------------------------------------------------------------------------------------

GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
EXPERT_TENSOR_RE = re.compile(r"ffn_\w*_exps")
LAYOUT_VERSION = 1


def read_gguf_header(path):
    """(metadata, {tensor name: bytes}) from a GGUF file's header; the tensor data is never read.
    Arrays longer than 1024 values (the tokenizer's) are skipped. Tensor sizes come from the gaps
    between their offsets, so new quantization types need no table. ValueError for anything that is
    not a readable GGUF header."""
    size = os.path.getsize(path)
    with open(path, "rb", buffering=1 << 20) as f:
        def take(fmt):
            n = struct.calcsize("<" + fmt)
            data = f.read(n)
            if len(data) != n:
                raise ValueError("the file ends inside its header")
            return struct.unpack("<" + fmt, data)

        def text():
            (n,) = take("Q")
            if n > 1 << 26:
                raise ValueError("a string in the header is too long")
            data = f.read(n)
            if len(data) != n:
                raise ValueError("the file ends inside its header")
            return data.decode("utf-8", "replace")

        def value(kind, nested=False):
            if kind in GGUF_SCALARS:
                return take(GGUF_SCALARS[kind])[0]
            if kind == 8:
                return text()
            if kind == 9 and not nested:
                inner, count = take("I")[0], take("Q")[0]
                if inner in GGUF_SCALARS:
                    width = struct.calcsize("<" + GGUF_SCALARS[inner])
                    if count > 1024:
                        f.seek(width * count, 1)
                        return None
                    return list(take(GGUF_SCALARS[inner] * count)) if count else []
                if inner == 8 and count > 1024:
                    for _ in range(count):
                        f.seek(take("Q")[0], 1)
                    return None
                items = [value(inner, True) for _ in range(count)]
                return items if count <= 1024 else None
            raise ValueError(f"unknown value type {kind} in the header")

        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        (version,) = take("I")
        if version < 2:
            raise ValueError(f"GGUF version {version} is too old")
        n_tensors, n_meta = take("QQ")
        if n_tensors > 1 << 20 or n_meta > 1 << 20:
            raise ValueError("implausible header counts")
        meta = {}
        for _ in range(n_meta):
            key = text()
            meta[key] = value(take("I")[0])
        infos = []
        for _ in range(n_tensors):
            name = text()
            (dims,) = take("I")
            if dims > 8:
                raise ValueError("implausible tensor shape")
            take("Q" * dims)
            take("I")  # the tensor type: not needed, the offsets give the size
            infos.append((take("Q")[0], name))
        header_end = f.tell()
    align = int(meta.get("general.alignment") or 32)
    data_start = -(-header_end // align) * align
    infos.sort()
    sizes = {}
    for i, (offset, name) in enumerate(infos):
        end = infos[i + 1][0] if i + 1 < len(infos) else size - data_start
        sizes[name] = max(0, end - offset)
    return meta, sizes


def gguf_layout(path):
    """What the VRAM reserve needs to know about a model: bytes outside the MoE experts, the
    KV-cache width per token and the recurrent (DeltaNet) state, from its GGUF header."""
    meta, sizes = read_gguf_header(path)
    arch = str(meta.get("general.architecture") or "")

    def key(name, default=None):
        v = meta.get(f"{arch}.{name}")
        return default if v is None else v

    expert = sum(n for t, n in sizes.items() if EXPERT_TENSOR_RE.search(t))
    blocks = int(key("block_count", 0) or 0)
    # Layers with a KV cache: the ones that have attention keys. Holo4 is hybrid: only every fourth
    # layer is attention, the other three are Gated DeltaNet with a small fixed-size state.
    kv_layers = sorted({int(m[1]) for t in sizes for m in [re.match(r"blk\.(\d+)\.attn_k\.weight$", t)] if m})
    if not kv_layers:
        interval = int(key("full_attention_interval", 0) or 0)
        kv_layers = [i for i in range(blocks) if not interval or i % interval == interval - 1]
    heads = key("attention.head_count", 1)
    heads = max(heads) if isinstance(heads, list) and heads else heads
    heads_kv = key("attention.head_count_kv", heads)

    def per_layer(v, i):
        if isinstance(v, list):
            return v[i] if i < len(v) else (max(v) if v else 0)
        return v

    head_dim = (int(key("embedding_length", 0) or 0) // int(heads or 1)) or 128
    k_len = int(key("attention.key_length", head_dim) or head_dim)
    v_len = int(key("attention.value_length", k_len) or k_len)
    kv_width = sum(int(per_layer(heads_kv, i) or 0) * (k_len + v_len) for i in kv_layers)
    recurrent = max(0, blocks - len(kv_layers)) if key("ssm.state_size") else 0
    state = 0
    if recurrent:
        conv = int(key("ssm.conv_kernel", 4) or 4)
        inner = int(key("ssm.inner_size", 0) or 0)
        d_state = int(key("ssm.state_size", 0) or 0)
        groups = int(key("ssm.group_count", 1) or 1)
        state = recurrent * 4 * ((conv - 1) * (inner + 2 * groups * d_state) + inner * d_state)  # f32
    return {"arch": arch, "nonexpert_bytes": sum(sizes.values()) - expert, "expert_bytes": expert,
            "kv_layers": len(kv_layers), "kv_elements_per_token": kv_width,
            "recurrent_layers": recurrent, "recurrent_state_bytes": state}


def model_layout(path):
    """gguf_layout(path), kept in <file>.layout.json next to the model (or in the logs folder when
    that cannot be written) while the file's size and time are unchanged. None when unreadable."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return None
    stamp = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "version": LAYOUT_VERSION}
    places = [path.with_name(path.name + ".layout.json"), paths.LOGS / (path.name + ".layout.json")]
    for p in places:
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            if all(doc.get(k) == v for k, v in stamp.items()) and isinstance(doc.get("layout"), dict):
                return doc["layout"]
        except (OSError, ValueError, AttributeError):
            pass
    try:
        layout = gguf_layout(path)
    except (OSError, ValueError, struct.error, TypeError):
        return None
    doc = {"_note": "Written by GhostAgent from the model's GGUF header (sizes for the VRAM reserve). "
                    "Rebuilt by itself when the model file changes.", **stamp, "layout": layout}
    for p in places:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
            break
        except OSError:
            continue
    return layout


# The fixed reserve before the model-aware one (measured for the 35B Q4_K_M at a 64k context): used
# when the model's header cannot be read.
DEFAULT_MOE_RESERVE_MIB = 4600
# Compute buffers (prompt micro-batches of 2048 tokens, the vision encoder) and the CUDA context.
COMPUTE_RESERVE_MIB = 1024
# The vision projector is not counted by llama.cpp's --fit: measured on the 27B, the process ends
# about the projector's size plus this much above the fit budget. Added on every --fit-target path
# (the dense profile, and the MoE profiles' low-VRAM and auto-fit fallback).
VISION_RESERVE_MIB = 384


def kv_cache_bytes(layout, ctx):
    """KV cache for ctx tokens at q8_0 (34 bytes per 32 values), as the agent runs it (-ctk/-ctv q8_0)."""
    return int(ctx) * int(layout["kv_elements_per_token"]) * 34 // 32


def moe_reserve_mib(layout, mmproj_bytes, ctx):
    """VRAM (MiB) a mixture-of-experts model needs besides its expert cache when every expert stays in
    system RAM (--cpu-moe): everything that is not an expert, the vision projector, the KV cache, the
    recurrent state and the compute buffers."""
    total = int(layout["nonexpert_bytes"]) + int(mmproj_bytes) + kv_cache_bytes(layout, ctx) \
        + int(layout.get("recurrent_state_bytes") or 0)
    return math.ceil(total / 2**20) + COMPUTE_RESERVE_MIB
