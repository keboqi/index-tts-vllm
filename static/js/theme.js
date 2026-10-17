"use strict";

/* ---------- Color theme (auto / light / dark) ---------- */
(function () {
    const STORAGE_KEY = 'audioStudio.theme.v1';
    const MODES = ['auto', 'light', 'dark'];
    const LABELS = { auto: 'Auto', light: 'Light', dark: 'Dark' };
    const ICONS = { auto: '◐', light: '☀', dark: '☾' };

    function readMode() {
        try {
            const stored = window.localStorage.getItem(STORAGE_KEY);
            return MODES.includes(stored) ? stored : 'auto';
        } catch (_err) {
            return 'auto';
        }
    }

    function writeMode(mode) {
        try {
            window.localStorage.setItem(STORAGE_KEY, mode);
        } catch (_err) {
            /* storage unavailable: theme still applies for this page view */
        }
    }

    function applyMode(mode) {
        const root = document.documentElement;
        if (mode === 'auto') {
            root.removeAttribute('data-theme');
        } else {
            root.setAttribute('data-theme', mode);
        }
        const button = document.getElementById('themeToggle');
        const label = document.getElementById('themeToggleLabel');
        const icon = document.getElementById('themeToggleIcon');
        if (label) label.textContent = LABELS[mode];
        if (icon) icon.textContent = ICONS[mode];
        if (button) button.setAttribute('aria-label', `Color theme: ${LABELS[mode]}`);
    }

    applyMode(readMode());

    function bindToggle() {
        const button = document.getElementById('themeToggle');
        if (!button) return;
        applyMode(readMode());
        button.addEventListener('click', () => {
            const next = MODES[(MODES.indexOf(readMode()) + 1) % MODES.length];
            writeMode(next);
            applyMode(next);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bindToggle);
    } else {
        bindToggle();
    }
})();
