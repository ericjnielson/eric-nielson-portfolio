// Shared theming: applied on every page.
// Loaded synchronously in <head> so the theme is set before first paint (no flash).
(function () {
  try {
    var t = localStorage.getItem('theme');
    if (!t) t = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', t);
  } catch (e) {}
})();

// Wire the navbar toggle once the DOM is ready.
(function () {
  function sync() {
    var b = document.getElementById('themeToggle');
    if (!b) return;
    var dark = document.documentElement.getAttribute('data-theme') === 'dark';
    var icon = b.querySelector('.theme-toggle__icon');
    if (icon) icon.textContent = dark ? '☀️' : '🌙';
    b.setAttribute('aria-pressed', String(dark));
  }
  document.addEventListener('DOMContentLoaded', function () {
    var b = document.getElementById('themeToggle');
    if (!b) return;
    sync();
    b.addEventListener('click', function () {
      var next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) {}
      sync();
    });
  });
})();
