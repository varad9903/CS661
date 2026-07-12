// Clientside handler for the Vegetation tab's High/Low vegetation-layer toggle.
// Defined in an assets/ file (auto-served on every page load) so the namespace is
// always present client-side — inline-string clientside callbacks registered from
// an imported module are not reliably injected into the page.
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    vegToggle: {
        swap: function (vegType, store) {
            if (!store || !store[vegType]) {
                var n = window.dash_clientside.no_update;
                return [n, n, n, n];
            }
            var s = store[vegType];
            return [s.anim, s.hyst, s.rootzone, s.niche];
        }
    }
});
