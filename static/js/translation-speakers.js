"use strict";

let speakerColorMap = {};
        const SPEAKER_COLOR_COUNT = 20;

        function getSpeakerColorIndex(speakerId) {
            if (!speakerId) return 0;
            const id = String(speakerId);
            if (speakerColorMap[id] !== undefined) {
                return speakerColorMap[id];
            }
            // Assign next available color
            const usedColors = Object.values(speakerColorMap);
            for (let i = 0; i < SPEAKER_COLOR_COUNT; i++) {
                if (!usedColors.includes(i)) {
                    speakerColorMap[id] = i;
                    return i;
                }
            }
            // Fallback: cycle through colors
            speakerColorMap[id] = usedColors.length % SPEAKER_COLOR_COUNT;
            return speakerColorMap[id];
        }

        function setSpeakerProfiles(profiles) {
            translateSpeakerProfiles = Array.isArray(profiles) ? profiles : [];
            translateSpeakerProfileMap = {};
            speakerColorMap = {}; // Reset color assignments
            translateSpeakerProfiles.forEach((profile, index) => {
                if (profile && profile.id) {
                    translateSpeakerProfileMap[String(profile.id)] = profile;
                    // Assign color based on order
                    speakerColorMap[String(profile.id)] = index % SPEAKER_COLOR_COUNT;
                }
            });
        }

        function setSpeakerOverrides(overrides) {
            translateSpeakerOverrides = {};
            if (overrides && typeof overrides === 'object') {
                Object.entries(overrides).forEach(([key, value]) => {
                    if (!value || typeof value !== 'object') {
                        return;
                    }
                    const normalizedId = String(key || '').toLowerCase();
                    const normalizedVolume =
                        typeof value.volume_percent === 'number'
                            ? Math.min(MAX_VOLUME_PERCENT, Math.max(MIN_VOLUME_PERCENT, value.volume_percent))
                            : undefined;
                    const overrideEntry = {
                        preset_name: value.preset_name || '',
                        use_emotion_prompt: Boolean(value.use_emotion_prompt),
                        emotion_weight:
                            typeof value.emotion_weight === 'number'
                                ? Math.min(1, Math.max(0, value.emotion_weight))
                                : DEFAULT_EMOTION_WEIGHT,
                    };
                    if (normalizedVolume !== undefined) {
                        overrideEntry.volume_percent = normalizedVolume;
                    }
                    translateSpeakerOverrides[normalizedId] = overrideEntry;
                });
            }
        }

        function renderSpeakerAssignments() {
            if (!translateSpeakerAssignments) {
                return;
            }
            if (!translateSpeakerProfiles.length) {
                translateSpeakerAssignments.style.display = 'none';
                translateSpeakerAssignments.innerHTML = '';
                return;
            }
            let html = '<h4>Detected Speakers</h4>';
            translateSpeakerProfiles.forEach((profile, idx) => {
                if (!profile) {
                    return;
                }
                const fallbackId = profile.id ? String(profile.id) : `speaker${idx + 1}`;
                const speakerId = fallbackId.toLowerCase();
                const override = translateSpeakerOverrides[speakerId] || {};
                const selectedPreset = override.preset_name || '';
                const useEmotionPrompt = Boolean(override.use_emotion_prompt) && Boolean(selectedPreset);
                const checkboxDisabled = !selectedPreset;
                const weightValue =
                    typeof override.emotion_weight === 'number'
                        ? override.emotion_weight
                        : DEFAULT_EMOTION_WEIGHT;
                const volumeValue =
                    typeof override.volume_percent === 'number'
                        ? override.volume_percent
                        : '';
                let optionsHtml = '<option value="">Auto (clone original voice)</option>';
                availableSpeakerPresets.forEach(name => {
                    const safeName = String(name || '');
                    const selectedAttr = safeName === selectedPreset ? 'selected' : '';
                    optionsHtml += `<option value="${safeName}" ${selectedAttr}>${safeName}</option>`;
                });
                const displayName = profile.label || fallbackId.toUpperCase();
                const description = profile.description || 'No description';
                const colorIndex = getSpeakerColorIndex(speakerId);
                html += `
                        <div class="speaker-assignment-item speaker-color-${colorIndex}" data-speaker-color="${colorIndex}" style="padding:10px;">
                            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;">
                                <div style="flex:1;min-width:150px;">
                                    <strong style="font-size:0.9rem;color:var(--speaker-color);">${displayName}</strong>
                                    <div style="font-size:0.75rem;color:var(--text-muted);margin-top:2px;">${description}</div>
                                </div>
                                <select class="speaker-override-select" data-speaker-id="${speakerId}" style="flex:1;min-width:150px;max-width:250px;">
                                    ${optionsHtml}
                                </select>
                                <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                                    <label style="display:flex;align-items:center;gap:4px;font-size:0.8rem;margin:0;">
                                        <input type="checkbox" class="speaker-emo-checkbox" data-speaker-id="${speakerId}" ${checkboxDisabled ? 'disabled' : ''} ${useEmotionPrompt ? 'checked' : ''}>
                                        Emo prompt
                                    </label>
                                    <input type="number" min="0" max="1" step="0.05" value="${weightValue}" class="speaker-emo-weight-input" data-speaker-id="${speakerId}" ${useEmotionPrompt ? '' : 'disabled'} style="width:60px;padding:4px 6px;font-size:0.85rem;" title="Emotion weight">
                                    <input type="number" class="speaker-volume-input" data-speaker-id="${speakerId}" min="${MIN_VOLUME_PERCENT}" max="${MAX_VOLUME_PERCENT}" step="5" value="${volumeValue}" style="width:60px;padding:4px 6px;font-size:0.85rem;" placeholder="Vol%" title="Volume %">
                                </div>
                            </div>
                            <div class="speaker-assignment-preview" data-speaker-id="${speakerId}" style="margin-top:6px;">
                                <small class="speaker-preview-message" style="font-size:0.75rem;"></small>
                                <audio controls preload="none" class="speaker-preview-audio" style="display:none;height:30px;"></audio>
                            </div>
                        </div>
                    `;
            });
            translateSpeakerAssignments.innerHTML = html;
            translateSpeakerAssignments.style.display = 'block';
            translateSpeakerAssignments.querySelectorAll('.speaker-override-select').forEach(select => {
                select.addEventListener('change', onSpeakerPresetChange);
            });
            translateSpeakerAssignments.querySelectorAll('.speaker-emo-checkbox').forEach(checkbox => {
                checkbox.addEventListener('change', onSpeakerEmotionToggle);
            });
            translateSpeakerAssignments.querySelectorAll('.speaker-emo-weight-input').forEach(input => {
                input.addEventListener('input', onSpeakerEmotionWeightChange);
                input.addEventListener('change', onSpeakerEmotionWeightChange);
            });
            translateSpeakerAssignments.querySelectorAll('.speaker-volume-input').forEach(input => {
                input.addEventListener('input', onSpeakerVolumeChange);
                input.addEventListener('change', onSpeakerVolumeChange);
            });
            translateSpeakerProfiles.forEach((profile, idx) => {
                const fallbackId = profile.id ? String(profile.id) : `speaker${idx + 1}`;
                const speakerId = fallbackId.toLowerCase();
                updateSpeakerPreviewForId(speakerId);
                updateSpeakerEmotionWeightInput(speakerId);
                updateSpeakerVolumeInput(speakerId);
            });
        }

        function onSpeakerPresetChange(event) {
            const select = event.target;
            const speakerId = select.dataset.speakerId;
            if (!speakerId) {
                return;
            }
            const newPreset = select.value;
            const existing = translateSpeakerOverrides[speakerId] || {};
            if (!newPreset) {
                if (typeof existing.volume_percent === 'number') {
                    translateSpeakerOverrides[speakerId] = {
                        preset_name: '',
                        use_emotion_prompt: false,
                        emotion_weight: DEFAULT_EMOTION_WEIGHT,
                        volume_percent: existing.volume_percent,
                    };
                } else {
                    delete translateSpeakerOverrides[speakerId];
                }
            } else {
                const nextOverride = {
                    preset_name: newPreset,
                    use_emotion_prompt: Boolean(existing.use_emotion_prompt),
                    emotion_weight:
                        typeof existing.emotion_weight === 'number'
                            ? existing.emotion_weight
                            : DEFAULT_EMOTION_WEIGHT,
                };
                if (typeof existing.volume_percent === 'number') {
                    nextOverride.volume_percent = existing.volume_percent;
                }
                translateSpeakerOverrides[speakerId] = nextOverride;
            }
            cleanupSpeakerOverrideIfEmpty(speakerId);
            const emoToggle = translateSpeakerAssignments.querySelector(`.speaker-emo-checkbox[data-speaker-id="${speakerId}"]`);
            if (emoToggle) {
                if (newPreset) {
                    emoToggle.disabled = false;
                    emoToggle.checked = Boolean(translateSpeakerOverrides[speakerId]?.use_emotion_prompt);
                } else {
                    emoToggle.disabled = true;
                    emoToggle.checked = false;
                }
            }
            speakerOverridesDirty = true;
            updateSpeakerPreviewForId(speakerId);
            updateSpeakerEmotionWeightInput(speakerId);
            updateSpeakerVolumeInput(speakerId);
        }

        function onSpeakerEmotionToggle(event) {
            const checkbox = event.target;
            const speakerId = checkbox.dataset.speakerId;
            if (!speakerId) {
                return;
            }
            const override = translateSpeakerOverrides[speakerId];
            if (!override) {
                checkbox.checked = false;
                return;
            }
            override.use_emotion_prompt = checkbox.checked;
            speakerOverridesDirty = true;
            updateSpeakerEmotionWeightInput(speakerId);
        }

        function buildSpeakerOverridesPayload() {
            const payload = {};
            Object.entries(translateSpeakerOverrides).forEach(([speakerId, config]) => {
                if (!config) {
                    return;
                }
                const hasPreset = Boolean(config.preset_name);
                const hasVolume = typeof config.volume_percent === 'number' && !Number.isNaN(config.volume_percent);
                if (!hasPreset && !hasVolume) {
                    return;
                }
                const entry = {};
                if (hasPreset) {
                    entry.preset_name = config.preset_name;
                    entry.use_emotion_prompt = Boolean(config.use_emotion_prompt);
                    entry.emotion_weight =
                        typeof config.emotion_weight === 'number'
                            ? Math.min(1, Math.max(0, config.emotion_weight))
                            : DEFAULT_EMOTION_WEIGHT;
                }
                if (hasVolume) {
                    entry.volume_percent = Math.min(
                        MAX_VOLUME_PERCENT,
                        Math.max(MIN_VOLUME_PERCENT, config.volume_percent)
                    );
                }
                payload[speakerId] = entry;
            });
            return payload;
        }

        function getPresetPreviewUrl(presetName) {
            if (!presetName) {
                return null;
            }
            const meta = speakerPresetMeta[presetName];
            if (meta && meta.preview_url) {
                return meta.preview_url;
            }
            return null;
        }

        function updateSpeakerPreviewForId(speakerId) {
            if (!translateSpeakerAssignments) {
                return;
            }
            const select = translateSpeakerAssignments.querySelector(`.speaker-override-select[data-speaker-id="${speakerId}"]`);
            const previewContainer = translateSpeakerAssignments.querySelector(`.speaker-assignment-preview[data-speaker-id="${speakerId}"]`);
            if (!select || !previewContainer) {
                return;
            }
            const messageEl = previewContainer.querySelector('.speaker-preview-message');
            const audioEl = previewContainer.querySelector('.speaker-preview-audio');
            const override = translateSpeakerOverrides[speakerId];
            const presetName = override && override.preset_name ? override.preset_name : '';
            const previewUrl = getPresetPreviewUrl(presetName);

            if (presetName && previewUrl) {
                const cacheBustedUrl = `${previewUrl}?t=${Date.now()}`;
                audioEl.src = cacheBustedUrl;
                audioEl.style.display = 'block';
                if (messageEl) {
                    messageEl.textContent = `Preview: ${presetName}`;
                }
            } else {
                audioEl.removeAttribute('src');
                audioEl.style.display = 'none';
                if (messageEl) {
                    if (!presetName) {
                        messageEl.textContent = 'Select a preset to preview.';
                    } else {
                        messageEl.textContent = 'No preview available for this preset.';
                    }
                }
            }
        }

        function updateSpeakerEmotionWeightInput(speakerId) {
            if (!translateSpeakerAssignments) {
                return;
            }
            const weightInput = translateSpeakerAssignments.querySelector(`.speaker-emo-weight-input[data-speaker-id="${speakerId}"]`);
            if (!weightInput) {
                return;
            }
            const override = translateSpeakerOverrides[speakerId];
            const weightValue =
                typeof override?.emotion_weight === 'number'
                    ? Math.min(1, Math.max(0, override.emotion_weight))
                    : DEFAULT_EMOTION_WEIGHT;
            weightInput.value = weightValue;
            const canUseEmotion = Boolean(override && override.preset_name && override.use_emotion_prompt);
            weightInput.disabled = !canUseEmotion;
        }

        function updateSpeakerVolumeInput(speakerId) {
            if (!translateSpeakerAssignments) {
                return;
            }
            const volumeInput = translateSpeakerAssignments.querySelector(`.speaker-volume-input[data-speaker-id="${speakerId}"]`);
            if (!volumeInput) {
                return;
            }
            const override = translateSpeakerOverrides[speakerId];
            if (override && typeof override.volume_percent === 'number') {
                volumeInput.value = override.volume_percent;
            } else {
                volumeInput.value = '';
            }
        }

        function onSpeakerEmotionWeightChange(event) {
            const input = event.target;
            const speakerId = input.dataset.speakerId;
            if (!speakerId) {
                return;
            }
            if (!translateSpeakerOverrides[speakerId]) {
                translateSpeakerOverrides[speakerId] = {
                    preset_name: '',
                    use_emotion_prompt: false,
                    emotion_weight: DEFAULT_EMOTION_WEIGHT,
                };
            }
            const value = parseFloat(input.value);
            const normalized = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : DEFAULT_EMOTION_WEIGHT;
            translateSpeakerOverrides[speakerId].emotion_weight = normalized;
            speakerOverridesDirty = true;
            input.value = normalized;
        }

        function cleanupSpeakerOverrideIfEmpty(speakerId) {
            const override = translateSpeakerOverrides[speakerId];
            if (!override) {
                return;
            }
            const hasPreset = Boolean(override.preset_name);
            const hasVolume = typeof override.volume_percent === 'number' && !Number.isNaN(override.volume_percent);
            if (!hasPreset && !hasVolume) {
                delete translateSpeakerOverrides[speakerId];
            }
        }

        function onSpeakerVolumeChange(event) {
            const input = event.target;
            const speakerId = input.dataset.speakerId;
            if (!speakerId) {
                return;
            }
            const rawValue = (input.value || '').trim();
            if (!rawValue) {
                if (translateSpeakerOverrides[speakerId]) {
                    delete translateSpeakerOverrides[speakerId].volume_percent;
                    cleanupSpeakerOverrideIfEmpty(speakerId);
                }
                speakerOverridesDirty = true;
                return;
            }
            const parsed = parseFloat(rawValue);
            if (Number.isNaN(parsed)) {
                return;
            }
            const normalized = Math.min(MAX_VOLUME_PERCENT, Math.max(MIN_VOLUME_PERCENT, parsed));
            if (!translateSpeakerOverrides[speakerId]) {
                translateSpeakerOverrides[speakerId] = {
                    preset_name: '',
                    use_emotion_prompt: false,
                    emotion_weight: DEFAULT_EMOTION_WEIGHT,
                };
            }
            translateSpeakerOverrides[speakerId].volume_percent = normalized;
            speakerOverridesDirty = true;
            input.value = normalized;
        }

        function autoApplyTranslateMetadata(metadata, sessionIdOverride = null) {
            if (!metadata || typeof metadata !== 'object') {
                return;
            }
            if (metadata.output_base_name && translateBaseFilenameInput) {
                translateBaseFilenameInput.value = metadata.output_base_name;
                translateBaseFilenameInput.dataset.userEdited = 'false';
            }
            if (metadata.language_code) {
                translateLanguageCodeHint = metadata.language_code;
            }
            updateFfmpegCommands({
                baseName: metadata.output_base_name,
                languageCode: metadata.language_code,
                languageLabel: metadata.dest_language,
            });
            if (
                translateForceGeminiRefresh &&
                typeof metadata.force_gemini_regenerate === 'boolean'
            ) {
                translateForceGeminiRefresh.checked = !!metadata.force_gemini_regenerate;
            }
            const backingMeta = metadata.backing_track || {};
            if (typeof backingMeta.available === 'boolean') {
                translateBackingAvailableFromSession = Boolean(backingMeta.available);
                updateCustomBackingSummary();
                syncTranslateMergeBackState();
            }
            if (
                translateVolumeInput &&
                typeof metadata.generated_volume_percent === 'number'
            ) {
                translateVolumeInput.value = metadata.generated_volume_percent;
            }
            if (translateBackingVolumeInput) {
                let backingValue = null;
                if (typeof metadata.backing_volume_percent === 'number') {
                    backingValue = metadata.backing_volume_percent;
                } else if (
                    metadata.backing_track &&
                    typeof metadata.backing_track.volume_percent === 'number'
                ) {
                    backingValue = metadata.backing_track.volume_percent;
                }
                if (backingValue !== null) {
                    translateBackingVolumeInput.value = backingValue;
                }
            }
            if (
                translateSilenceVolumeInput &&
                typeof metadata.silence_volume_percent === 'number'
            ) {
                translateSilenceVolumeInput.value = metadata.silence_volume_percent;
                syncSilenceVolumeUI();
            }
            if (translateDefaultSpeakerSelect) {
                const metadataSpeaker = typeof metadata.default_speaker_preset === 'string' ? metadata.default_speaker_preset.trim() : '';
                if (metadataSpeaker) {
                    const hasOption = Array.from(translateDefaultSpeakerSelect.options || []).some(
                        option => option.value === metadataSpeaker
                    );
                    if (!hasOption) {
                        const option = document.createElement('option');
                        option.value = metadataSpeaker;
                        option.textContent = metadataSpeaker;
                        translateDefaultSpeakerSelect.appendChild(option);
                    }
                    translateDefaultSpeakerSelect.value = metadataSpeaker;
                } else {
                    translateDefaultSpeakerSelect.value = '';
                }
            }
            if (translateTtsBackendEl && typeof metadata.tts_backend === 'string') {
                const metadataBackend = metadata.tts_backend.trim();
                if (metadataBackend && Array.from(translateTtsBackendEl.options || []).some(option => option.value === metadataBackend)) {
                    translateTtsBackendEl.value = metadataBackend;
                    syncTranslateTtsBackendControls();
                }
            }
            if (translateDestLanguageSelect && typeof metadata.dest_language === 'string') {
                syncTranslateDestinationLanguageOptions(metadata.dest_language.trim());
                updateAiConfigSummary();
            }
            if (
                translateDefaultEmotionWeightInput &&
                typeof metadata.default_emotion_weight === 'number' &&
                !Number.isNaN(metadata.default_emotion_weight)
            ) {
                translateDefaultEmotionWeightInput.value = metadata.default_emotion_weight.toFixed(2);
                syncDefaultEmotionWeightDisplay();
            }
            if (translateManualSegmentsToggle && translateManualSegmentsInput) {
                let manualText = '';
                if (typeof metadata.gemini_raw_text === 'string' && metadata.gemini_raw_text.trim()) {
                    manualText = metadata.gemini_raw_text.trim();
                } else if (Array.isArray(metadata.gemini_raw_segments)) {
                    manualText = JSON.stringify(metadata.gemini_raw_segments, null, 2);
                }
                if (manualText) {
                    translateManualSegmentsToggle.checked = true;
                    if (typeof updateManualSegmentsVisibility === 'function') {
                        updateManualSegmentsVisibility();
                    }
                    translateManualSegmentsInput.value = manualText;
                }
            }
            const derivedSessionId =
                sessionIdOverride ||
                metadata.session_id ||
                metadata.reuse_session_id ||
                (metadata.separation && metadata.separation.session_id);
            if (derivedSessionId) {
                currentTranslateSessionId = derivedSessionId;
            }
            renderSeparationPreview(currentTranslateSessionId, metadata);
            if (Array.isArray(metadata.speaker_profiles)) {
                setSpeakerProfiles(metadata.speaker_profiles);
            }
            if (metadata.speaker_overrides !== undefined) {
                if (metadata.speaker_overrides && typeof metadata.speaker_overrides === 'object') {
                    setSpeakerOverrides(metadata.speaker_overrides);
                } else {
                    setSpeakerOverrides({});
                }
                speakerOverridesDirty = false;
            }
            if (metadata.chunk && metadata.chunk.session_id) {
                currentChunkSessionId = metadata.chunk.session_id;
                translateSelectedChunkId = metadata.chunk.session_id;
            }
            renderSpeakerAssignments();
            updateChunkSelectionUI();
            updateTranslateStepSummaries();
        }
