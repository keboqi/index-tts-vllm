"use strict";

/* ---------- API reference filter ---------- */
(function () {
    const input = document.getElementById('apiFilter');
    const countEl = document.getElementById('apiFilterCount');
    if (!input) return;

    const sections = Array.from(document.querySelectorAll('#api .api-section'));
    const staticBlocks = ['apiQuickRef', 'apiExamples']
        .map(id => document.getElementById(id))
        .filter(Boolean);
    const total = sections.reduce((sum, section) => sum + section.querySelectorAll('.api-endpoint').length, 0);

    function applyFilter() {
        const query = input.value.trim().toLowerCase();
        let shown = 0;
        sections.forEach(section => {
            const title = (section.querySelector('.api-section-header')?.textContent || '').toLowerCase();
            const titleMatch = query !== '' && title.includes(query);
            let visibleInSection = 0;
            section.querySelectorAll('.api-endpoint').forEach(endpoint => {
                const match = !query || titleMatch || endpoint.textContent.toLowerCase().includes(query);
                endpoint.hidden = !match;
                if (match) visibleInSection += 1;
            });
            section.hidden = visibleInSection === 0;
            shown += visibleInSection;
        });
        staticBlocks.forEach(block => {
            block.hidden = query !== '';
        });
        if (countEl) {
            countEl.textContent = query ? `${shown} of ${total} endpoints` : `${total} endpoints`;
        }
    }

    input.addEventListener('input', applyFilter);
    applyFilter();
})();
