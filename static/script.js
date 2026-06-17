// ---- Homepage: scroll-spy nav + reveal-on-scroll (theme toggle lives in theme.js) ----
(function () {
  document.addEventListener('DOMContentLoaded', function () {
    if (!document.body.classList.contains('home')) return;

    // Reveal-on-scroll
    var reveals = document.querySelectorAll('.highlights-section, .portfolio, .timeline, .connect-section');
    if ('IntersectionObserver' in window && reveals.length) {
      reveals.forEach(function (el) { el.classList.add('reveal'); });
      var revObs = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) { e.target.classList.add('in-view'); revObs.unobserve(e.target); }
        });
      }, { threshold: 0.08 });
      reveals.forEach(function (el) { revObs.observe(el); });
      // Safety net: if anything is still hidden after 1.5s, reveal it.
      setTimeout(function () {
        reveals.forEach(function (el) { el.classList.add('in-view'); });
      }, 1500);
    }

    // Scroll-spy nav
    var map = {};
    Array.prototype.forEach.call(document.querySelectorAll('.nav-links a[href^="#"]'), function (a) {
      var sec = document.getElementById(a.getAttribute('href').slice(1));
      if (sec) map[sec.id] = a;
    });
    var ids = Object.keys(map);
    if (ids.length && 'IntersectionObserver' in window) {
      var spy = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            ids.forEach(function (id) { map[id].classList.remove('active'); });
            map[e.target.id].classList.add('active');
          }
        });
      }, { rootMargin: '-45% 0px -50% 0px', threshold: 0 });
      ids.forEach(function (id) { spy.observe(document.getElementById(id)); });
    }
  });
})();
