"""JavaScript the browser runs in pages: reader_mode, page signatures, field values and labels,
secret fields, scroll probes and the agent's status box."""



# reader_mode: the page's main content as markdown. Headings, paragraphs, lists (nested ones indented),
# tables as pipe rows and links as [text](address); navigation, side bars, the site's own header and
# footer, hidden elements and the agent's status box are left out. Same-origin frames and open shadow
# roots are read in place. The root is <main> (or [role=main]), else the page's only <article>, else
# the body; a root that holds almost nothing falls back to the body.
READER_JS = r"""() => {
  const SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'SVG', 'CANVAS', 'NAV', 'ASIDE', 'BUTTON',
    'SELECT', 'INPUT', 'TEXTAREA', 'OPTION', 'OBJECT', 'EMBED', 'VIDEO', 'AUDIO', 'MAP', 'HEAD', 'IMG', 'PICTURE']);
  const SKIP_ROLES = new Set(['navigation', 'banner', 'contentinfo', 'complementary']);
  const BLOCK = new Set(['P', 'DIV', 'SECTION', 'ARTICLE', 'MAIN', 'HEADER', 'FOOTER', 'BLOCKQUOTE', 'FIGURE',
    'FIGCAPTION', 'DL', 'DT', 'DD', 'ADDRESS', 'FORM', 'FIELDSET', 'LEGEND', 'DETAILS', 'SUMMARY', 'CAPTION',
    'CENTER', 'HGROUP', 'DIALOG', 'LI', 'TR', 'BODY']);
  const INLINE = new Set(['SPAN', 'B', 'STRONG', 'I', 'EM', 'CODE', 'SMALL', 'SUP', 'SUB', 'MARK', 'ABBR', 'TIME',
    'LABEL', 'CITE', 'Q', 'U', 'S', 'KBD', 'VAR', 'DFN', 'DATA', 'BDI', 'BDO', 'WBR', 'FONT', 'TD', 'TH']);
  const pre = [];
  const one = (s) => s.replace(/\s+/g, ' ').trim();
  const hidden = (el) => {
    if (el.hidden) return true;
    // display: contents has no box of its own, so checkVisibility calls it hidden, but its children show.
    if (getComputedStyle(el).display === 'contents') return false;
    if (el.checkVisibility) return !el.checkVisibility({visibilityProperty: true, checkVisibilityCSS: true});
    const cs = getComputedStyle(el);
    return cs.display === 'none' || cs.visibility === 'hidden';
  };
  const pageLevel = (el) => !(el.parentElement && el.parentElement.closest('article, main, section, [role=main]'));
  const kids = (el) => {
    let s = '';
    const from = el.shadowRoot || el;
    for (const c of from.childNodes) s += walk(c);
    return s;
  };
  const walk = (node) => {
    if (node.nodeType === 3) return node.textContent.replace(/\s+/g, ' ');
    if (node.nodeType !== 1) return '';
    const el = node, tag = el.tagName.toUpperCase();
    if (tag === 'SLOT') {
      const assigned = el.assignedNodes({flatten: true});
      let s = '';
      for (const c of (assigned.length ? assigned : el.childNodes)) s += walk(c);
      return s;
    }
    if (SKIP.has(tag) || el.id === '__ghost_agent_overlay__') return '';
    if (SKIP_ROLES.has((el.getAttribute('role') || '').toLowerCase())) return '';
    if ((tag === 'HEADER' || tag === 'FOOTER') && pageLevel(el)) return '';
    if (el.getAttribute('aria-hidden') === 'true' || hidden(el)) return '';
    if (/^H[1-6]$/.test(tag)) {
      const t = one(kids(el));
      return t ? '\n\n' + '#'.repeat(+tag[1]) + ' ' + t + '\n\n' : '';
    }
    if (tag === 'BR') return '\n';
    if (tag === 'HR') return '\n\n---\n\n';
    if (tag === 'PRE') {
      const t = (el.innerText || '').replace(/\s+$/, '');
      if (!t.trim()) return '';
      pre.push('```\n' + t + '\n```');
      return '\n\n\u0002' + (pre.length - 1) + '\u0002\n\n';
    }
    if (tag === 'A') {
      const img = el.querySelector('img[alt]');
      const t = one(kids(el)) || one(el.getAttribute('aria-label') || el.getAttribute('title') || (img ? img.alt : '') || '');
      const href = typeof el.href === 'string' ? el.href : '';
      return t && /^https?:/i.test(href) ? '[' + t.replace(/[\[\]]/g, '') + '](' + href + ')' : t;
    }
    if (tag === 'UL' || tag === 'OL') {
      const items = [];
      let n = parseInt(el.getAttribute('start') || '1', 10) || 1;
      for (const li of el.children) {
        if (li.tagName !== 'LI' || hidden(li)) continue;
        const lines = kids(li).split('\n').map(l => l.trim()).filter(Boolean);
        if (!lines.length) continue;
        const marker = tag === 'OL' ? (n++) + '. ' : '- ';
        items.push(marker + lines[0] + lines.slice(1).map(l => '\n\u0001' + l).join(''));
      }
      return items.length ? '\n\n' + items.join('\n') + '\n\n' : '';
    }
    if (tag === 'TABLE') {
      const rows = [];
      for (const tr of el.rows) {
        if (hidden(tr)) continue;
        const cells = Array.from(tr.cells).map(c => one(kids(c)).replace(/\|/g, '\\|'));
        if (!cells.some(Boolean)) continue;
        rows.push('| ' + cells.join(' | ') + ' |');
        if (rows.length === 1) rows.push('|' + cells.map(() => ' --- |').join(''));
      }
      return rows.length ? '\n\n' + rows.join('\n') + '\n\n' : '';
    }
    if (tag === 'IFRAME' || tag === 'FRAME') {
      let doc = null;
      try { doc = el.contentDocument; } catch (e) { doc = null; }
      return doc && doc.body ? '\n\n' + walk(doc.body) + '\n\n' : '';
    }
    const inner = kids(el);
    if (BLOCK.has(tag)) return '\n\n' + inner + '\n\n';
    if (INLINE.has(tag)) return tag === 'TD' || tag === 'TH' ? ' ' + inner + ' ' : inner;
    const d = getComputedStyle(el).display;
    return /^(block|flex|grid|table|list-item|flow-root)/.test(d) ? '\n\n' + inner + '\n\n' : inner;
  };
  const read = (root) => walk(root)
    .replace(/[ \t]+\n/g, '\n').replace(/\n[ \t]+/g, '\n').replace(/[ \t]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n').replace(/\u0001/g, '  ')
    .replace(/\u0002(\d+)\u0002/g, (m, i) => pre[+i]).trim();
  const articles = document.querySelectorAll('article');
  const root = document.querySelector('main, [role=main]') || (articles.length === 1 ? articles[0] : null) || document.body;
  if (!root) return '';
  let text = read(root);
  if (text.length < 200 && root !== document.body && document.body) text = read(document.body);
  return text;
}"""

READER_CHUNK = 8000      # characters reader_mode returns per call unless asked for another range
READER_MAX_CHUNK = 16000

# The state of a page that an action can change, for "The page did not change." and the stall
# check: the address, the scroll position and page height, the text (including same-origin frames),
# the focused element and the form fields (password fields only by their length).
SIGNATURE_JS = r"""() => {
  const se = document.scrollingElement || document.documentElement;
  let a = document.activeElement;
  while (a && a.shadowRoot && a.shadowRoot.activeElement) a = a.shadowRoot.activeElement;
  const textOf = (doc) => (doc && doc.body ? doc.body.innerText || '' : '');
  let text = textOf(document);
  for (const f of document.querySelectorAll('iframe, frame')) {
    if (text.length >= 20000) break;
    try { text += '\n' + textOf(f.contentDocument); } catch (e) { /* another origin */ }
  }
  const fields = [];
  for (const e of document.querySelectorAll('input, select, textarea')) {
    if (fields.length >= 500) break;
    const t = (e.type || '').toLowerCase();
    const ac = (e.getAttribute('autocomplete') || '').toLowerCase();
    if (t === 'checkbox' || t === 'radio') fields.push(e.checked ? '1' : '0');
    else if (t === 'password' || /current-password|new-password|one-time-code/.test(ac)) fields.push('#' + e.value.length);
    else fields.push(String(e.value).slice(0, 200));
  }
  return [location.href, Math.round(scrollX), Math.round(scrollY), se ? se.scrollHeight : 0, text.slice(0, 20000),
          a ? a.tagName + ' ' + (a.getAttribute('name') || a.id || '') : '', fields.join('\u0001')];
}"""

# What the focused field in this frame holds now (the first 200 characters).
FIELD_VALUE_JS = r"""() => {
  let a = document.activeElement;
  while (a && a.shadowRoot && a.shadowRoot.activeElement) a = a.shadowRoot.activeElement;
  if (!a) return null;
  if (a.isContentEditable) return (a.innerText || '').slice(0, 200);
  return 'value' in a ? String(a.value).slice(0, 200) : null;
}"""

# settle_mode adaptive: milliseconds since the page's DOM last changed. The first call (or the first
# on a new document) starts watching and returns 0; 'stop' ends the watching.
QUIET_JS = r"""(op) => {
  const K = Symbol.for('ghostagent.quiet');
  let q = window[K];
  if (op === 'stop') { if (q && q.mo) q.mo.disconnect(); window[K] = undefined; return 0; }
  if (!q || op === 'start') {
    if (q && q.mo) q.mo.disconnect();
    q = window[K] = {last: performance.now(), mo: null};
    try {
      q.mo = new MutationObserver(() => { q.last = performance.now(); });
      q.mo.observe(document, {subtree: true, childList: true, attributes: true, characterData: true});
    } catch (e) { /* no document yet */ }
    return 0;
  }
  return performance.now() - q.last;
}"""

# observation_format "header": scroll position and size of the page and of the visible part of it.
METRICS_JS = r"""() => {
  const se = document.scrollingElement || document.documentElement;
  return {x: Math.round(scrollX), y: Math.round(scrollY), iw: innerWidth, ih: innerHeight,
          cw: se ? se.clientWidth || innerWidth : innerWidth, ch: se ? se.clientHeight || innerHeight : innerHeight,
          sw: se ? se.scrollWidth : innerWidth, sh: se ? se.scrollHeight : innerHeight};
}"""

SELECT_JS = """([x, y, wanted]) => {
  let el = document.elementFromPoint(x, y);
  if (el && el.tagName === 'LABEL' && el.control) el = el.control;
  while (el && el.tagName !== 'SELECT') el = el.parentElement;
  if (!el) return {ok: false, msg: 'No <select> dropdown at that point.'};
  const w = wanted.trim().toLowerCase();
  const opts = Array.from(el.options);
  const hit = opts.find(o => o.text.trim().toLowerCase() === w)
           || opts.find(o => o.value.toLowerCase() === w)
           || opts.find(o => o.text.trim().toLowerCase().includes(w));
  if (!hit) return {ok: false, msg: 'Option not found. Options: ' + opts.map(o => o.text.trim()).slice(0, 40).join(' | ')};
  el.value = hit.value;
  el.dispatchEvent(new Event('input', {bubbles: true}));
  el.dispatchEvent(new Event('change', {bubbles: true}));
  return {ok: true, msg: 'Selected "' + hit.text.trim() + '".'};
}"""


SELECT_OPTIONS_JS = """([x, y]) => {
  let el = document.elementFromPoint(x, y);
  if (el && el.tagName === 'LABEL' && el.control) el = el.control;
  while (el && el.tagName !== 'SELECT') el = el.parentElement;
  if (!el || el.multiple) return null;
  return Array.from(el.options).slice(0, 60).map(o => o.text.trim() + (o.selected ? ' (selected)' : ''));
}"""

# Shared by the checks below: the element at a point (through open shadow roots, and from a label to
# its field), the next element up (leaving a shadow root through its host), and whether a field is
# for a password or a one-time code.
_ELEMENT_JS = """
  const deepAt = (x, y) => {
    let el = document.elementFromPoint(x, y);
    while (el && el.shadowRoot) {
      const inner = el.shadowRoot.elementFromPoint(x, y);
      if (!inner || inner === el) break;
      el = inner;
    }
    if (el && el.tagName === 'LABEL' && el.control) el = el.control;
    return el;
  };
  const up = (e) => e.parentElement || ((e.getRootNode && e.getRootNode().host) || null);
  const secret = (e) => {
    if (!e || e.tagName !== 'INPUT') return false;
    if ((e.getAttribute('type') || '').toLowerCase() === 'password') return true;
    const ac = (e.getAttribute('autocomplete') || '').toLowerCase().split(/\\s+/);
    return ac.includes('current-password') || ac.includes('new-password') || ac.includes('one-time-code');
  };
"""

# The words on a button or link (clickable: the element or the closest clickable one above it), and on
# the button that Enter in a form field would press (submitLabel).
_LABEL_FNS_JS = """
  const words = (e) => [e.innerText, e.value, e.getAttribute('aria-label'), e.getAttribute('title'), e.getAttribute('alt')]
      .filter(v => typeof v === 'string' && v.trim()).join(' ').replace(/\\s+/g, ' ').trim().slice(0, 150);
  const SEL = 'button, a, [role=button], [role=menuitem], input[type=submit], input[type=button], input[type=reset], input[type=image]';
  const clickable = (e) => {
    let t = e;
    while (t && !(t.matches && t.matches(SEL))) t = up(t);
    return t;
  };
  const submitLabel = (e) => {
    const form = e.form || (e.closest && e.closest('form'));
    if (!form) return null;
    const b = form.querySelector('button[type=submit], button:not([type]), input[type=submit], input[type=image]');
    return b ? words(b) || null : null;
  };
"""

# The words on the button or link at a point ('click'), or on the button that Enter in the field at a
# point would press ('submit'). null when there is none.
LABEL_JS = "([x, y, mode]) => {" + _ELEMENT_JS + _LABEL_FNS_JS + """
  const el = deepAt(x, y);
  if (!el) return null;
  if (mode === 'submit') return submitLabel(el);
  const t = clickable(el);
  return t ? words(t) || null : null;
}"""

# What a key press on the focused element would set off: ['click', label] for a focused button or link
# (Enter or Space presses it), ['submit', label] for a field in a form (Enter submits it), else null.
# Child frames count only while they have the focus; in the main frame a focused iframe is skipped,
# since the frame inside it reports for itself.
FOCUSED_LABEL_JS = "(main) => {" + _ELEMENT_JS + _LABEL_FNS_JS + """
  if (!main && !document.hasFocus()) return null;
  let a = document.activeElement;
  while (a && a.shadowRoot && a.shadowRoot.activeElement) a = a.shadowRoot.activeElement;
  if (!a || a === document.body || a === document.documentElement || a.tagName === 'IFRAME' || a.tagName === 'FRAME') return null;
  const t = clickable(a);
  if (t) return ['click', words(t) || null];
  return ['submit', submitLabel(a)];
}"""

# True when the field at a point is a password or one-time-code field.
SECRET_AT_JS = "([x, y]) => {" + _ELEMENT_JS + """
  return secret(deepAt(x, y));
}"""

# True when this frame's focused field is a password or one-time-code field. Child frames count only
# while they have the focus; the main frame always reports its focused element.
FOCUSED_SECRET_JS = "(main) => {" + _ELEMENT_JS + """
  if (!main && !document.hasFocus()) return false;
  let a = document.activeElement;
  while (a && a.shadowRoot && a.shadowRoot.activeElement) a = a.shadowRoot.activeElement;
  return secret(a);
}"""

# Scroll offsets that a mouse wheel at a point can change: the window's, and those of every scrollable
# element from the point upwards. With store true the elements are remembered, so the second call
# (store false) reads the same ones even after the content under the point moved. null when they were
# not remembered (the page was replaced meanwhile).
SCROLL_PROBE_JS = "([x, y, store]) => {" + _ELEMENT_JS + r"""
  const K = Symbol.for('ghostagent.scroll');
  let list = [];
  if (store) {
    let el = x === null ? null : deepAt(x, y);
    while (el && el.nodeType === 1) {
      if (el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1) list.push(el);
      el = up(el);
    }
    window[K] = list;
  } else {
    if (!window[K]) return null;
    list = window[K];
  }
  return [Math.round(scrollX), Math.round(scrollY)].concat(list.map(e => Math.round(e.scrollLeft) + ',' + Math.round(e.scrollTop)));
}"""

# The frame element (iframe or frame) that has the keyboard focus in this frame, or null: the focus is
# then inside that frame's document.
FOCUSED_FRAME_JS = r"""() => {
  let a = document.activeElement;
  while (a && a.shadowRoot && a.shadowRoot.activeElement) a = a.shadowRoot.activeElement;
  return a && (a.tagName === 'IFRAME' || a.tagName === 'FRAME') ? a : null;
}"""

# The element with the keyboard focus in this frame (through open shadow roots): what kind of target it
# is for typing. Child frames answer only while they have the focus; a focused frame element in the
# main frame means the frame inside it has to be asked. Kinds: field (an editable input, textarea or
# contenteditable element), readonly, editor (a canvas or a custom text box that cannot be read back),
# other (a button, a link...), none (nothing focused), frame.
FOCUSED_FIELD_JS = "(main) => {" + _ELEMENT_JS + r"""
  if (!main && !document.hasFocus()) return null;
  let a = document.activeElement;
  while (a && a.shadowRoot && a.shadowRoot.activeElement) a = a.shadowRoot.activeElement;
  // A whole document can be the text box (the iframe of a classic rich-text editor).
  const docEditable = document.designMode === 'on' || !!(document.body && document.body.isContentEditable);
  if (docEditable && (!a || a === document.documentElement)) a = document.body;
  if (!a || (a === document.body && !docEditable) || a === document.documentElement) return main ? {kind: 'none'} : null;
  if (a.tagName === 'IFRAME' || a.tagName === 'FRAME') return main ? {kind: 'frame'} : null;
  const tag = a.tagName.toLowerCase();
  const type = (a.getAttribute('type') || '').toLowerCase();
  const TEXT = ['', 'text', 'search', 'email', 'url', 'tel', 'password', 'number', 'date', 'time',
                'datetime-local', 'month', 'week'];
  let kind = 'other';
  if (tag === 'textarea' || (tag === 'input' && TEXT.includes(type))) kind = a.readOnly || a.disabled ? 'readonly' : 'field';
  else if (a.isContentEditable || (a === document.body && docEditable)) kind = 'field';
  else if (tag === 'canvas' || ['textbox', 'searchbox', 'combobox'].includes((a.getAttribute('role') || '').toLowerCase())) kind = 'editor';
  const name = a.getAttribute('aria-label') || a.getAttribute('placeholder') || a.getAttribute('name') || a.id || '';
  return {kind, tag, type: tag === 'input' ? type || 'text' : '', name: String(name).slice(0, 60), secret: secret(a)};
}"""

# A small status / answer box drawn inside the page. It is removed before every screenshot so the
# model never sees it, ignores the mouse while a task runs, and uses DOM calls only (no innerHTML)
# so pages that enforce Trusted Types accept it.
OVERLAY_JS = """([kind, title, text]) => {
  const ID = '__ghost_agent_overlay__';
  let host = document.getElementById(ID);
  if (host) host.remove();
  if (kind === 'remove') return;
  host = document.createElement('div');
  host.id = ID;
  host.style.cssText = 'all:initial;position:fixed;z-index:2147483647;right:16px;bottom:16px;'
    + (kind === 'answer' ? 'pointer-events:auto;' : 'pointer-events:none;');
  const box = document.createElement('div');
  box.style.cssText = 'font:13px/1.45 Segoe UI,system-ui,sans-serif;color:#f4f4f5;background:rgba(24,24,27,.94);'
    + 'border:1px solid #3f3f46;border-radius:10px;padding:10px 14px;box-shadow:0 6px 24px rgba(0,0,0,.35);'
    + (kind === 'answer' ? 'max-width:460px;max-height:60vh;overflow:auto;white-space:pre-wrap;' : 'max-width:360px;');
  const head = document.createElement('div');
  head.style.cssText = 'font-weight:600;color:' + (kind === 'answer' ? '#86efac' : '#93c5fd') + ';margin-bottom:' + (text ? '4px' : '0');
  head.textContent = title;
  box.appendChild(head);
  if (text) {
    const body = document.createElement('div');
    body.textContent = text;
    box.appendChild(body);
  }
  if (kind === 'answer') {
    const close = document.createElement('div');
    close.textContent = 'click to dismiss';
    close.style.cssText = 'margin-top:8px;color:#a1a1aa;font-size:11px;cursor:pointer;';
    box.appendChild(close);
    host.addEventListener('click', () => host.remove());
  }
  host.appendChild(box);
  (document.body || document.documentElement).appendChild(host);
}"""
