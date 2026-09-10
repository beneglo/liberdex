/* Stamped before the stylesheet paints, so a light-theme reader never sees a
   frame of dark. Kept out of app.js because that one is deferred, and deferred
   is too late for this. */
try {
  var t = localStorage.getItem("liberdex.theme");
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
} catch (e) { /* private window, blocked storage: dark is the default anyway */ }
