"""Screenshot size: the 32-pixel tiles Holo4 reads, shared by the model server's image flags and
the browser's viewport."""



# --------------------------------------------------------------------------------------------
# Screenshot size
# --------------------------------------------------------------------------------------------

TILE_PX = 32               # Holo4 reads a screenshot in 32x32-pixel tiles (16-pixel patches, merged 2x2)
MIN_IMAGE_TILES = 1024     # llama-server's --image-min-tokens: smaller screenshots are enlarged to this
# llama-server's --image-max-tokens: bigger screenshots are scaled down to this. Just above a normal
# screenshot (a 1280x800 one is 1089 prompt tokens), so those are untouched; it matches
# BIG_SCREENSHOT_PIXELS (1.3 MP, about 1270 tiles). Clicks are in 0-1000 coordinates relative to the
# screenshot, so a scaled-down one still clicks the right place. Measured on 13 November 2026: the
# cap shrinks the vision warm-up buffer by about 106 MiB of VRAM on both models, with the same answers.
MAX_IMAGE_TILES = 1280
BIG_SCREENSHOT_PIXELS = 1_300_000


def snap_viewport_size(w, h, max_w=None, max_h=None):
    """snap_viewport: a page area of whole 32-pixel tiles, at least MIN_IMAGE_TILES of them, close to
    w x h and within max_w x max_h (the screen), so the model server uses the screenshot as it is
    instead of resampling it. Returns (width, height, ok); ok is False when the screen is too small
    for that, and then the size is just w x h within the screen (the screenshot is resampled).
    The height grows first (1280x800 becomes 1280x832), then the width, by the fewest tiles needed:
    a 1536x730 page area on a 125%-scaled laptop screen gives 1536x704 for a 1536-wide request and
    1504x704 for a 1280-wide one."""
    max_w = int(max_w) if max_w else 1 << 16
    max_h = int(max_h) if max_h else 1 << 16
    cols = max(1, min(int(w), max_w) // TILE_PX)
    rows = max(1, min(int(h), max_h) // TILE_PX)
    max_cols, max_rows = max_w // TILE_PX, max_h // TILE_PX
    if cols * rows < MIN_IMAGE_TILES:
        need_rows = -(-MIN_IMAGE_TILES // cols)
        if need_rows <= max_rows:
            rows = need_rows
        else:
            rows = max_rows
            cols = -(-MIN_IMAGE_TILES // max(1, rows))
            if rows < 1 or cols > max_cols:
                return min(int(w), max_w), min(int(h), max_h), False
    return cols * TILE_PX, rows * TILE_PX, True
