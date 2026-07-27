"use strict";

function init() {
            bindRangeOutputs();
            bindDelegatedActions();
            initDurationControlControls();
            const voicePresetSelect = document.getElementById('voiceDesignPreset');
            if (voicePresetSelect) {
                voicePresetSelect.addEventListener('change', applyVoiceDesignPreset);
            }
            const streamingModeEl = document.getElementById('streamingMode');
            if (streamingModeEl) {
                streamingModeEl.addEventListener('change', function () {
                    const streamingSettings = document.getElementById('streamingSettings');
                    if (!streamingSettings) {
                        return;
                    }
                    streamingSettings.style.display = this.checked ? 'block' : 'none';
                });
            }
            const ttsBackendEl = document.getElementById('ttsBackend');
            const emotionTextEl = document.getElementById('emotionText');
            const emotionWeightEl = document.getElementById('emotionWeight');
            function updateTtsBackendControls() {
                const usesIndexEmotionControls = !ttsBackendEl || ttsBackendEl.value === 'index';
                if (emotionTextEl) {
                    emotionTextEl.disabled = !usesIndexEmotionControls;
                }
                if (emotionWeightEl) {
                    emotionWeightEl.disabled = !usesIndexEmotionControls;
                }
                updateHiggsEnhancerVisibility();
            }
            if (ttsBackendEl) {
                ttsBackendEl.addEventListener('change', updateTtsBackendControls);
                updateTtsBackendControls();
            } else {
                updateHiggsEnhancerVisibility();
            }
            const stableAudioForm = document.getElementById('stableAudioForm');
            if (stableAudioForm) {
                stableAudioForm.addEventListener('submit', function (event) {
                    event.preventDefault();
                    generateStableAudio();
                });
            }
            const loadInitialData = function () {
                loadSpeakers();
                loadSpeakerEffects();
                loadDownloadedVideos();
            };
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', loadInitialData);
            } else {
                loadInitialData();
            }
        }

        init();
