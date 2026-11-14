// Reports what gets clicked on a benchmark page to the bench server log (kind "log", type "click"),
// so a run can be reviewed afterwards. It never changes the page. Values typed into fields are not
// sent, only the field's name or placeholder.
(function () {
  function describe(el) {
    if (el.matches('input, textarea, select')) {
      return el.name || el.placeholder || el.id || el.type || '';
    }
    return (el.innerText || el.getAttribute('aria-label') || el.title || '').trim().slice(0, 80);
  }
  document.addEventListener('click', function (e) {
    var el = e.target.closest('a, button, input, select, textarea, label, summary, [role=button], li, td') || e.target;
    var data = {
      type: 'click', page: location.pathname + location.search, tag: el.tagName.toLowerCase(),
      text: describe(el), x: Math.round(e.clientX), y: Math.round(e.clientY)
    };
    try {
      navigator.sendBeacon('/log', new Blob([JSON.stringify(data)], { type: 'application/json' }));
    } catch (err) { /* logging must never break the page */ }
  }, true);
})();
