"use strict";

function init() {
            bindRangeOutputs();
            bindDelegatedActions();
            updateFfmpegCommands();
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
            const ttsLanguageEl = document.getElementById('ttsLanguage');
            const ttsLanguagesByBackend = {
                index: [['auto', 'Auto'], ['en', 'English'], ['zh', 'Chinese']],
                index25: [
                    ['auto', 'Auto'], ['en', 'English'], ['zh', 'Chinese'],
                    ['ja', 'Japanese'], ['es', 'Spanish'], ['ar', 'Arabic']
                ],
                confucius: [
                    ['auto', 'Auto'], ['en', 'English'], ['zh', 'Chinese'], ['ja', 'Japanese'],
                    ['ko', 'Korean'], ['de', 'German'], ['fr', 'French'], ['es', 'Spanish'],
                    ['id', 'Indonesian'], ['it', 'Italian'], ['th', 'Thai'], ['pt', 'Portuguese'],
                    ['ru', 'Russian'], ['ms', 'Malay'], ['vi', 'Vietnamese']
                ]
            };
            function updateTtsBackendControls() {
                const backend = ttsBackendEl ? ttsBackendEl.value : 'index';
                const usesIndexEmotionControls = backend === 'index' || backend === 'index25';
                if (emotionTextEl) {
                    emotionTextEl.disabled = !usesIndexEmotionControls;
                }
                if (emotionWeightEl) {
                    emotionWeightEl.disabled = !usesIndexEmotionControls;
                }
                if (ttsLanguageEl) {
                    const previous = ttsLanguageEl.value;
                    const options = ttsLanguagesByBackend[backend] || ttsLanguagesByBackend.index;
                    ttsLanguageEl.replaceChildren(...options.map(([value, label]) => {
                        const option = document.createElement('option');
                        option.value = value;
                        option.textContent = label;
                        return option;
                    }));
                    ttsLanguageEl.value = options.some(([value]) => value === previous) ? previous : 'auto';
                }
            }
            if (ttsBackendEl) {
                ttsBackendEl.addEventListener('change', updateTtsBackendControls);
                updateTtsBackendControls();
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
