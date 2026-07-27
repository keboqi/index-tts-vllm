"use strict";

function toggleSegmentExpand(card) {
            if (!card) return;
            const isCompact = card.classList.contains('compact');
            if (isCompact) {
                // Expand this card, collapse others
                const allCards = translateSegmentsList.querySelectorAll('.segment-card');
                allCards.forEach(c => {
                    if (c !== card) {
                        c.classList.add('compact');
                        c.classList.remove('expanded');
                    }
                });
                card.classList.remove('compact');
                card.classList.add('expanded');
            } else {
                // Collapse this card
                card.classList.add('compact');
                card.classList.remove('expanded');
            }
        }

        // Global audio element for compact playback
        let compactAudioPlayer = null;
        let currentPlayingBtn = null;

        function stopCompactAudio() {
            if (compactAudioPlayer) {
                compactAudioPlayer.pause();
                compactAudioPlayer.currentTime = 0;
            }
            if (currentPlayingBtn) {
                currentPlayingBtn.classList.remove('playing');
                currentPlayingBtn.textContent = '▶';
                currentPlayingBtn = null;
            }
        }

        function playCompactOriginalAudio(card, btn) {
            const audioUrl = card.dataset.originalAudioUrl;
            if (!audioUrl) return;

            // If this button is already playing, stop it
            if (currentPlayingBtn === btn) {
                stopCompactAudio();
                return;
            }

            stopCompactAudio();

            // Show loading state while fetching audio
            btn.textContent = '⏳';
            btn.classList.add('loading');
            currentPlayingBtn = btn;

            if (!compactAudioPlayer) {
                compactAudioPlayer = new Audio();
                compactAudioPlayer.addEventListener('ended', stopCompactAudio);
                compactAudioPlayer.addEventListener('error', (e) => {
                    console.error('Error playing audio:', e);
                    btn.textContent = '❌';
                    btn.title = 'Failed to load audio - click to retry';
                    setTimeout(() => {
                        btn.textContent = '▶';
                        btn.title = 'Play original';
                    }, 2000);
                    stopCompactAudio();
                });
                // When audio is ready to play, update button
                const markPlaying = () => {
                    if (currentPlayingBtn === btn) {
                        btn.textContent = '⏹';
                        btn.classList.remove('loading');
                        btn.classList.add('playing');
                    }
                };
                compactAudioPlayer.addEventListener('canplay', markPlaying);
                compactAudioPlayer.addEventListener('loadeddata', markPlaying);
                compactAudioPlayer.addEventListener('playing', markPlaying);
            }

            compactAudioPlayer.src = audioUrl;
            compactAudioPlayer.load(); // Explicitly load to trigger canplay event
            compactAudioPlayer.play().then(() => {
                // If play resolves before canplay/playing fired (cached file), ensure state updates
                if (currentPlayingBtn === btn && btn.classList.contains('loading')) {
                    btn.textContent = '⏹';
                    btn.classList.remove('loading');
                    btn.classList.add('playing');
                }
            }).catch(err => {
                console.error('Play error:', err);
                btn.textContent = '❌';
                setTimeout(() => {
                    btn.textContent = '▶';
                }, 2000);
                stopCompactAudio();
            });
        }

        async function playCompactGeneratedAudio(card, btn) {
            // Check if generated audio already exists in the expanded view
            const previewAudio = card.querySelector('.segment-preview-audio');
            let audioUrl = previewAudio && previewAudio.src && previewAudio.style.display !== 'none' ? previewAudio.src : null;

            // If this button is already playing, stop it
            if (currentPlayingBtn === btn && audioUrl) {
                stopCompactAudio();
                return;
            }

            stopCompactAudio();

            // If no generated audio yet, trigger generation
            if (!audioUrl || audioUrl === '' || audioUrl === window.location.href) {
                btn.textContent = '⏳';
                btn.classList.add('loading');

                try {
                    // Trigger the preview generation
                    const previewBtn = card.querySelector('.segment-preview-btn');
                    if (previewBtn) {
                        // Call handleSegmentPreview and wait for it to complete
                        await handleSegmentPreview(card, previewBtn);

                        // Wait a bit for audio to be set
                        await new Promise(resolve => setTimeout(resolve, 500));

                        // Check again for audio
                        audioUrl = previewAudio && previewAudio.src && previewAudio.style.display !== 'none' ? previewAudio.src : null;
                    }
                } catch (err) {
                    console.error('Error generating preview:', err);
                    btn.classList.remove('loading');
                    btn.textContent = '▶';
                    return;
                }

                btn.classList.remove('loading');

                if (!audioUrl || audioUrl === '' || audioUrl === window.location.href) {
                    btn.textContent = '▶';
                    return;
                }

                // Mark as having audio
                btn.classList.add('has-audio');
            }

            // Play the audio
            btn.textContent = '⏹';
            btn.classList.add('playing');
            currentPlayingBtn = btn;

            if (!compactAudioPlayer) {
                compactAudioPlayer = new Audio();
                compactAudioPlayer.addEventListener('ended', stopCompactAudio);
                compactAudioPlayer.addEventListener('error', () => {
                    stopCompactAudio();
                    console.error('Error playing audio');
                });
            }

            compactAudioPlayer.src = audioUrl;
            compactAudioPlayer.play().catch(err => {
                console.error('Play error:', err);
                stopCompactAudio();
            });
        }

        function hasCustomBackingSelection() {
            if (!translateCustomBackingInput || !translateCustomBackingInput.files) {
                return false;
            }
            return translateCustomBackingInput.files.length > 0;
        }

        function updateCustomBackingSummary() {
            if (translateCustomBackingSummary) {
                if (hasCustomBackingSelection()) {
                    const file = translateCustomBackingInput.files[0];
                    translateCustomBackingSummary.textContent = `Selected: ${file ? file.name : ''}`;
                    translateCustomBackingSummary.style.color = '#0a7c4a';
                } else if (translateBackingAvailableFromSession) {
                    translateCustomBackingSummary.textContent = 'Using stored backing track from current session.';
                    translateCustomBackingSummary.style.color = '#0a7c4a';
                } else {
                    translateCustomBackingSummary.textContent =
                        'No custom backing selected. Upload audio here to override the extracted instrumental when mix-back is enabled.';
                    translateCustomBackingSummary.style.color = '#666';
                }
            }
            updateAdditionalSettingsSummary();
        }

        function setTranslateButtonLabel() {
            if (!translateBtn) return;
            if (translateAdvancedToggle && translateAdvancedToggle.checked) {
                translateBtn.textContent = '🧠 Analyze Segments';
            } else {
                translateBtn.textContent = '🌐 Translate Speech';
            }
        }

        function syncTranslateMergeBackState() {
            if (!translateMergeBackEl) {
                return;
            }
            // Merge back requires Audio-Separator, custom backing, or stored backing from session
            const audioSeparatorEnabled = translateAudioSeparatorEl && translateAudioSeparatorEl.checked;
            const customSelected = hasCustomBackingSelection();
            const storedBacking = translateBackingAvailableFromSession;
            const canEnableMerge = audioSeparatorEnabled || customSelected || storedBacking;
            translateMergeBackEl.disabled = !canEnableMerge;
            if (!canEnableMerge) {
                translateMergeBackEl.checked = false;
            }
            // Update label styling
            if (translateMergeBackLabelEl) {
                translateMergeBackLabelEl.style.opacity = canEnableMerge ? '1' : '0.5';
            }
            updateAdditionalSettingsSummary();
        }

        function enforceEnhancementForSuperRes() {
            if (!translateEnhanceEl) {
                return;
            }
            const superEnabled = translateSuperEl && translateSuperEl.checked;
            if (superEnabled && !translateEnhanceEl.checked) {
                translateEnhanceEl.checked = true;
            }
            translateEnhanceEl.disabled = Boolean(superEnabled);
        }

        // Super resolution requires enhancement
        if (translateSuperEl) {
            translateSuperEl.addEventListener('change', enforceEnhancementForSuperRes);
            enforceEnhancementForSuperRes();
        } else {
            enforceEnhancementForSuperRes();
        }
        // Initialize merge back state
        syncTranslateMergeBackState();
        if (translateCustomBackingInput) {
            translateCustomBackingInput.addEventListener('change', () => {
                updateCustomBackingSummary();
                syncTranslateMergeBackState();
            });
        }
        if (translateCustomBackingClearBtn) {
            translateCustomBackingClearBtn.addEventListener('click', () => {
                if (translateCustomBackingInput) {
                    translateCustomBackingInput.value = '';
                }
                updateCustomBackingSummary();
                syncTranslateMergeBackState();
            });
        }
        updateCustomBackingSummary();
        if (translateDestLanguageSelect) {
            translateDestLanguageSelect.addEventListener('change', () => {
                saveTranslatePersistentSettings();
                updateAiConfigSummary();
                refreshPromptTemplates();
                updateFfmpegCommands();
            });
        }
        function syncSilenceVolumeUI() {
            if (!translateSilenceVolumeGroup) {
                return;
            }
            const enabled = translatePreserveSilenceEl && translatePreserveSilenceEl.checked;
            translateSilenceVolumeGroup.style.display = enabled ? 'block' : 'none';
        }
        if (translatePreserveSilenceEl) {
            translatePreserveSilenceEl.addEventListener('change', () => {
                syncSilenceVolumeUI();
            });
            syncSilenceVolumeUI();
        }
        if (translateIgnoreNonSpeechEl) {
            translateIgnoreNonSpeechEl.addEventListener('change', () => {
                saveTranslatePersistentSettings();
                updateAiConfigSummary();
                refreshPromptTemplates();
            });
        }

        // Audio-Separator: toggle model selector and sync merge back state
        function syncAudioSeparatorUI() {
            const audioSepEnabled = translateAudioSeparatorEl && translateAudioSeparatorEl.checked;
            if (translateAudioSeparatorModelEl) {
                translateAudioSeparatorModelEl.disabled = !audioSepEnabled;
            }
            if (translateAudioSeparatorUseSoundfileEl) {
                translateAudioSeparatorUseSoundfileEl.disabled = !audioSepEnabled;
            }
            // Also sync the merge back state since it depends on audio-separator
            syncTranslateMergeBackState();
        }
        if (translateAudioSeparatorEl) {
            translateAudioSeparatorEl.addEventListener('change', syncAudioSeparatorUI);
            syncAudioSeparatorUI();  // Initial state
        }

        function appendSegmentParameters(formData) {
            if (!formData) {
                return;
            }
            if (translateTtsBackendEl && translateTtsBackendEl.value) {
                formData.append('tts_backend', translateTtsBackendEl.value);
            }
            formData.append('duration_control', getDurationControlMode());
            if (translateMinSpeechInput) {
                const minValue = (translateMinSpeechInput.value || '').trim();
                if (minValue) {
                    formData.append('min_speech_ms', minValue);
                }
            }
            if (translateMaxMergeInput) {
                const maxValue = (translateMaxMergeInput.value || '').trim();
                if (maxValue) {
                    formData.append('max_merge_ms', maxValue);
                }
            }
            if (translateVolumeInput) {
                const volumeValue = (translateVolumeInput.value || '').trim();
                if (volumeValue) {
                    formData.append('generated_volume_percent', volumeValue);
                }
            }
            if (translateBackingVolumeInput) {
                const backingValue = (translateBackingVolumeInput.value || '').trim();
                if (backingValue) {
                    formData.append('backing_volume_percent', backingValue);
                }
            }
            if (
                translatePreserveSilenceEl &&
                translatePreserveSilenceEl.checked &&
                translateSilenceVolumeInput
            ) {
                const silenceValue = (translateSilenceVolumeInput.value || '').trim();
                if (silenceValue) {
                    formData.append('silence_volume_percent', silenceValue);
                }
            }
            if (translateDefaultSpeakerSelect) {
                const defaultSpeaker = (translateDefaultSpeakerSelect.value || '').trim();
                if (defaultSpeaker) {
                    formData.append('default_speaker_preset', defaultSpeaker);
                }
            }
            if (translateDefaultEmotionWeightInput) {
                const weightValue = (translateDefaultEmotionWeightInput.value || '').trim();
                if (weightValue) {
                    formData.append('default_emotion_weight', weightValue);
                }
            }
        }

        function appendManualSegments(formData) {
            if (
                !formData ||
                !translateManualSegmentsToggle ||
                !translateManualSegmentsInput ||
                !translateManualSegmentsToggle.checked
            ) {
                return;
            }
            const manualText = translateManualSegmentsInput.value.trim();
            if (manualText) {
                formData.append('segments_json', manualText);
            }
        }

        function appendSrtSubtitleFiles(formData) {
            // If SRT mode is not enabled, do nothing
            if (
                !formData ||
                !translateSrtSubtitleToggle ||
                !translateSrtSubtitleToggle.checked
            ) {
                return false;
            }

            let hasFiles = false;

            // Append original SRT file if present
            if (translateOriginalSrtFile && translateOriginalSrtFile.files && translateOriginalSrtFile.files[0]) {
                formData.append('original_srt_file', translateOriginalSrtFile.files[0]);
                hasFiles = true;
            }

            // Append translated SRT file if present
            if (translateTranslatedSrtFile && translateTranslatedSrtFile.files && translateTranslatedSrtFile.files[0]) {
                formData.append('translated_srt_file', translateTranslatedSrtFile.files[0]);
                hasFiles = true;
            }

            return hasFiles;
        }

        function refreshPromptTemplates() {
            const destLang = translateDestLanguageSelect ? (translateDestLanguageSelect.value || '').trim() : '';
            const replacement = destLang || '{dest_language}';
            const instructionSegment =
                translateIgnoreNonSpeechEl &&
                    translateIgnoreNonSpeechEl.checked &&
                    typeof promptTemplates.ignoreNonSpeech === 'string' &&
                    promptTemplates.ignoreNonSpeech.trim().length > 0
                    ? `${promptTemplates.ignoreNonSpeech.trim()} `
                    : '';
            if (translatePromptTranslation) {
                const value = promptTemplates.translation
                    ? promptTemplates.translation
                        .split('{dest_language}')
                        .join(replacement)
                        .split(NON_SPEECH_PLACEHOLDER)
                        .join(instructionSegment)
                    : '';
                translatePromptTranslation.value = value.trim();
            }
            if (translatePromptTranscription) {
                const value = promptTemplates.transcription
                    ? promptTemplates.transcription
                        .split(NON_SPEECH_PLACEHOLDER)
                        .join(instructionSegment)
                    : '';
                translatePromptTranscription.value = value.trim();
            }
        }
        let updateManualSegmentsVisibility = () => { };
        if (translateManualSegmentsToggle && translateManualSegmentsPanel) {
            updateManualSegmentsVisibility = () => {
                const enabled = translateManualSegmentsToggle.checked;
                translateManualSegmentsPanel.style.display = enabled ? 'block' : 'none';
                if (translatePromptTemplates) {
                    translatePromptTemplates.style.display = enabled ? 'block' : 'none';
                }
                // Disable SRT toggle when manual segments is enabled (mutually exclusive)
                if (enabled && translateSrtSubtitleToggle) {
                    translateSrtSubtitleToggle.checked = false;
                    if (typeof updateSrtSubtitleVisibility === 'function') {
                        updateSrtSubtitleVisibility();
                    }
                }
            };
            translateManualSegmentsToggle.addEventListener('change', updateManualSegmentsVisibility);
            updateManualSegmentsVisibility();
        }

        // SRT Subtitle toggle and file handlers
        let updateSrtSubtitleVisibility = () => { };
        function updateSrtSummary() {
            if (!translateSrtSummary) return;
            const origFile = translateOriginalSrtFile?.files?.[0];
            const transFile = translateTranslatedSrtFile?.files?.[0];
            if (!origFile && !transFile) {
                translateSrtSummary.style.display = 'none';
                return;
            }
            const parts = [];
            if (origFile) {
                parts.push(`📄 Original: ${origFile.name}`);
            }
            if (transFile) {
                parts.push(`📄 Translated: ${transFile.name}`);
            }
            translateSrtSummary.querySelector('span').textContent = parts.join(' • ');
            translateSrtSummary.style.display = 'block';
        }

        if (translateSrtSubtitleToggle && translateSrtSubtitlePanel) {
            updateSrtSubtitleVisibility = () => {
                const enabled = translateSrtSubtitleToggle.checked;
                translateSrtSubtitlePanel.style.display = enabled ? 'block' : 'none';
                // Disable manual segments when SRT is enabled (mutually exclusive)
                if (enabled && translateManualSegmentsToggle) {
                    translateManualSegmentsToggle.checked = false;
                    if (typeof updateManualSegmentsVisibility === 'function') {
                        updateManualSegmentsVisibility();
                    }
                }
            };
            translateSrtSubtitleToggle.addEventListener('change', updateSrtSubtitleVisibility);
            updateSrtSubtitleVisibility();
        }

        // SRT file change handlers
        if (translateOriginalSrtFile) {
            translateOriginalSrtFile.addEventListener('change', updateSrtSummary);
        }
        if (translateTranslatedSrtFile) {
            translateTranslatedSrtFile.addEventListener('change', updateSrtSummary);
        }

        // SRT clear buttons
        if (translateOriginalSrtClear) {
            translateOriginalSrtClear.addEventListener('click', () => {
                if (translateOriginalSrtFile) {
                    translateOriginalSrtFile.value = '';
                }
                updateSrtSummary();
            });
        }
        if (translateTranslatedSrtClear) {
            translateTranslatedSrtClear.addEventListener('click', () => {
                if (translateTranslatedSrtFile) {
                    translateTranslatedSrtFile.value = '';
                }
                updateSrtSummary();
            });
        }

        async function loadPromptTemplates() {
            if (!translatePromptTranslation && !translatePromptTranscription) {
                return;
            }
            try {
                const response = await fetch(ENDPOINTS.PROMPT_TEMPLATES);
                if (!response.ok) {
                    return;
                }
                const data = await response.json();
                if (typeof data.translation === 'string') {
                    promptTemplates.translation = data.translation;
                }
                if (typeof data.transcription === 'string') {
                    promptTemplates.transcription = data.transcription;
                }
                if (typeof data.ignore_non_speech_instruction === 'string') {
                    promptTemplates.ignoreNonSpeech = data.ignore_non_speech_instruction;
                }
                refreshPromptTemplates();
            } catch (error) {
                console.warn('Failed to load prompt templates', error);
            }
        }

        loadPromptTemplates();

        function resetAdvancedPanel(clearSession = true) {
            if (clearSession) {
                currentTranslateSessionId = null;
            }
            currentTranslateSegments = [];
            translateSpeakerProfiles = [];
            translateSpeakerProfileMap = {};
            translateSpeakerOverrides = {};
            speakerOverridesDirty = false;
            if (translateAdvancedPanel) {
                translateAdvancedPanel.style.display = 'none';
            }
            if (translateSegmentsList) {
                translateSegmentsList.innerHTML = '';
            }
            if (translateSpeakerAssignments) {
                translateSpeakerAssignments.style.display = 'none';
                translateSpeakerAssignments.innerHTML = '';
            }
            if (translateSeparationPreview) {
                translateSeparationPreview.style.display = 'none';
                translateSeparationPreview.innerHTML = '';
            }
            if (translateSegmentsStatus) {
                hideStatus('translateSegmentsStatus');
            }
            if (translateSegmentsSelectAll) {
                translateSegmentsSelectAll.checked = true;
            }
            translateBackingAvailableFromSession = false;
            updateCustomBackingSummary();
            syncTranslateMergeBackState();
        }

        function updateTranslateSegmentsSummary() {
            if (!translateSegmentsStatus) {
                return;
            }
            if (!translateSegmentsList) {
                hideStatus('translateSegmentsStatus');
                return;
            }
            const speechCards = translateSegmentsList.querySelectorAll('.segment-card.speech');
            if (!speechCards.length) {
                hideStatus('translateSegmentsStatus');
                return;
            }
            let selected = 0;
            speechCards.forEach(card => {
                const checkbox = card.querySelector('input.segment-generate');
                if (checkbox && checkbox.checked) {
                    selected += 1;
                }
            });
            const preserved = speechCards.length - selected;
            showStatus(
                `Selected ${selected}/${speechCards.length} speech segments • Preserving ${preserved}`,
                'success',
                'translateSegmentsStatus'
            );
            if (translateSegmentsSelectAll) {
                translateSegmentsSelectAll.checked = selected === speechCards.length;
            }
        }

        function renderTranslateSegments(segments = []) {
            if (!translateSegmentsList) {
                return;
            }
            translateSegmentsList.innerHTML = '';
            const hasSpeech = segments.some(seg => seg.type === 'speech');
            if (!segments.length) {
                translateSegmentsList.innerHTML = '<div class="segment-empty">No segments returned from Gemini.</div>';
                updateTranslateSegmentsSummary();
                return;
            }
            if (translateSegmentsSelectAll) {
                const allSelected = segments.filter(seg => seg.type === 'speech').every(seg => seg.generate !== false);
                translateSegmentsSelectAll.checked = allSelected;
            }
            segments.forEach(segment => {
                const startMsVal = Number.isFinite(segment.start_ms) ? segment.start_ms : 0;
                const endMsVal = Number.isFinite(segment.end_ms) ? segment.end_ms : startMsVal;
                const durationVal = Number.isFinite(segment.duration_ms)
                    ? segment.duration_ms
                    : Math.max(0, endMsVal - startMsVal);

                // Calculate speaker info and color early
                const speakerInfo = segment.speaker ? (translateSpeakerProfileMap[segment.speaker] || {}) : {};
                const speakerLabel = segment.speaker ? (speakerInfo.label || segment.speaker) : (segment.type === 'silence' ? '—' : '—');
                const speakerColorIdx = segment.speaker ? getSpeakerColorIndex(segment.speaker.toLowerCase()) : -1;
                const speakerColorClass = speakerColorIdx >= 0 ? `speaker-color-${speakerColorIdx}` : '';

                const card = document.createElement('div');
                // Start in compact mode by default
                card.className = `segment-card ${segment.type} compact ${speakerColorClass}`;
                card.dataset.index = segment.index;
                card.dataset.type = segment.type;
                if (segment.speaker) {
                    card.dataset.speaker = segment.speaker;
                }
                if (speakerColorIdx >= 0) {
                    card.setAttribute('data-speaker-color', speakerColorIdx);
                }

                // Compact row - shown when card is in compact mode
                const compactRow = document.createElement('div');
                compactRow.className = 'segment-compact-row';
                const sourcePreview = segment.source_text ? truncateText(segment.source_text, 40) : (segment.type === 'silence' ? 'Silence' : '—');
                const translationPreview = segment.translated_text ? truncateText(segment.translated_text, 40) : (segment.type === 'silence' ? '' : '—');
                const startTimeFormatted = formatTimestampHHMMSS(startMsVal);
                const endTimeFormatted = formatTimestampHHMMSS(endMsVal);

                const hasOriginalAudio = !!segment.audio_preview_url;
                compactRow.innerHTML = `
                        <span class="compact-index">#${segment.index}</span>
                        <span class="compact-time">${startTimeFormatted} → ${endTimeFormatted}</span>
                        <span class="compact-speaker ${speakerColorClass}" ${speakerColorIdx >= 0 ? `data-speaker-color="${speakerColorIdx}"` : ''}>${speakerLabel}</span>
                        <span class="compact-source" title="${escapeHtml(segment.source_text || '')}">${escapeHtml(sourcePreview)}</span>
                        ${segment.type === 'speech' ? `<button type="button" class="compact-play-btn orig-btn" title="Play original" ${!hasOriginalAudio ? 'disabled style="opacity:0.4;cursor:not-allowed;"' : ''}>▶</button>` : ''}
                        <span class="compact-translation" title="${escapeHtml(segment.translated_text || '')}">${escapeHtml(translationPreview)}</span>
                        ${segment.type === 'speech' ? `
                            <button type="button" class="compact-play-btn gen-btn" title="Play generated (will generate if needed)">▶</button>
                            <span class="compact-checkbox">
                                <input type="checkbox" class="segment-generate-compact" ${segment.generate !== false ? 'checked' : ''}>
                            </span>
                        ` : ''}
                        <span class="compact-expand-icon">▼</span>
                    `;
                card.appendChild(compactRow);

                // Store original audio URL in dataset for compact play
                if (hasOriginalAudio) {
                    card.dataset.originalAudioUrl = segment.audio_preview_url;
                }

                // Add click handler for expand/collapse
                compactRow.addEventListener('click', (e) => {
                    // Don't toggle if clicking on checkbox or play buttons
                    if (e.target.classList.contains('segment-generate-compact') ||
                        e.target.classList.contains('compact-play-btn')) {
                        return;
                    }
                    toggleSegmentExpand(card);
                });

                // Add compact play button handlers
                const origPlayBtn = compactRow.querySelector('.orig-btn');
                const genPlayBtn = compactRow.querySelector('.gen-btn');

                if (origPlayBtn && hasOriginalAudio) {
                    origPlayBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        playCompactOriginalAudio(card, origPlayBtn);
                    });
                }

                if (genPlayBtn) {
                    genPlayBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        playCompactGeneratedAudio(card, genPlayBtn);
                    });
                }

                // Sync compact checkbox with main checkbox
                const compactCheckbox = compactRow.querySelector('.segment-generate-compact');
                if (compactCheckbox) {
                    compactCheckbox.addEventListener('change', (e) => {
                        e.stopPropagation();
                        const mainCheckbox = card.querySelector('input.segment-generate');
                        if (mainCheckbox) {
                            mainCheckbox.checked = compactCheckbox.checked;
                        }
                        updateTranslateSegmentsSummary();
                    });
                }

                const header = document.createElement('div');
                header.className = 'segment-header';

                const title = document.createElement('div');
                title.innerHTML = `<strong>#${segment.index}</strong> ${segment.type === 'speech' ? 'Speech Segment' : 'Silence Segment'}`;
                header.appendChild(title);
                if (segment.speaker) {
                    const speakerPill = document.createElement('span');
                    speakerPill.className = `segment-speaker-pill ${speakerColorClass}`;
                    if (speakerColorIdx >= 0) {
                        speakerPill.setAttribute('data-speaker-color', speakerColorIdx);
                    }
                    speakerPill.textContent = speakerLabel;
                    header.appendChild(speakerPill);
                }

                if (segment.type === 'speech') {
                    const checkboxLabel = document.createElement('label');
                    checkboxLabel.className = 'segment-checkbox';
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.className = 'segment-generate';
                    checkbox.checked = segment.generate !== false;
                    checkbox.addEventListener('change', () => {
                        // Sync with compact checkbox
                        if (compactCheckbox) {
                            compactCheckbox.checked = checkbox.checked;
                        }
                        updateTranslateSegmentsSummary();
                    });
                    checkboxLabel.appendChild(checkbox);
                    const span = document.createElement('span');
                    span.textContent = 'Generate';
                    checkboxLabel.appendChild(span);
                    header.appendChild(checkboxLabel);
                } else {
                    const meta = document.createElement('span');
                    meta.className = 'segment-meta';
                    meta.textContent = 'Preserved silence';
                    header.appendChild(meta);
                }

                // Add collapse button to header
                const collapseBtn = document.createElement('button');
                collapseBtn.type = 'button';
                collapseBtn.className = 'segment-collapse-btn';
                collapseBtn.textContent = '▲ Collapse';
                collapseBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    toggleSegmentExpand(card);
                });
                header.appendChild(collapseBtn);

                card.appendChild(header);

                const metaInfo = document.createElement('div');
                metaInfo.className = 'segment-meta';
                metaInfo.textContent = `${segment.start || formatTimestamp(startMsVal)} → ${segment.end || formatTimestamp(endMsVal)} (${durationVal} ms)`;
                card.appendChild(metaInfo);

                const body = document.createElement('div');
                body.className = 'segment-body';

                const timing = document.createElement('div');
                timing.className = 'segment-timing';
                timing.innerHTML = `
                        <label>Start<input type="number" class="segment-start" value="${startMsVal}" min="0"></label>
                        <label>End<input type="number" class="segment-end" value="${endMsVal}" min="0"></label>
                        <span class="segment-duration-label">${durationVal} ms</span>
                    `;
                body.appendChild(timing);

                const startInput = timing.querySelector('.segment-start');
                const endInput = timing.querySelector('.segment-end');
                const durationLabel = timing.querySelector('.segment-duration');
                const updateDuration = () => {
                    const startVal = parseInt(startInput.value || '0', 10);
                    const endVal = parseInt(endInput.value || '0', 10);
                    const diff = Math.max(0, endVal - startVal);
                    durationLabel.textContent = diff;
                };
                startInput.addEventListener('input', updateDuration);
                endInput.addEventListener('input', updateDuration);

                if (segment.type === 'speech') {
                    const textGrid = document.createElement('div');
                    textGrid.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr; gap: 10px;';
                    textGrid.innerHTML = `
                            <div><label style="font-size:0.8rem;">Source</label><textarea class="segment-source" style="min-height:50px;">${segment.source_text || ''}</textarea></div>
                            <div><label style="font-size:0.8rem;">Translation</label><textarea class="segment-translation" style="min-height:50px;">${segment.translated_text || ''}</textarea></div>
                        `;
                    body.appendChild(textGrid);

                    // Controls row: Preview button + Volume + Emotion
                    const volumeValue =
                        typeof segment.volume_percent === 'number' ? segment.volume_percent : '';
                    const emotionValue =
                        typeof segment.emotion_weight === 'number' ? segment.emotion_weight : '';
                    const controlsRow = document.createElement('div');
                    controlsRow.style.cssText = 'display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:8px;';
                    controlsRow.innerHTML = `
                            <button type="button" class="btn segment-preview-btn" style="padding:6px 12px;font-size:0.85rem;">⚡ Preview</button>
                            <small class="segment-preview-status" style="color:var(--brand-emerald);"></small>
                            <div style="display:flex;gap:8px;align-items:center;margin-left:auto;">
                                <label style="display:flex;align-items:center;gap:4px;font-size:0.8rem;margin:0;">Vol%<input type="number" class="segment-volume" min="${MIN_VOLUME_PERCENT}" max="${MAX_VOLUME_PERCENT}" step="5" value="${volumeValue}" style="width:60px;padding:4px 6px;"></label>
                                <label style="display:flex;align-items:center;gap:4px;font-size:0.8rem;margin:0;">Emo<input type="number" class="segment-emotion" min="0" max="1" step="0.05" value="${emotionValue}" style="width:55px;padding:4px 6px;"></label>
                            </div>
                        `;
                    body.appendChild(controlsRow);

                    // Audio row: Original + Generated side by side in same line
                    const audioRow = document.createElement('div');
                    audioRow.style.cssText = 'display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px;';
                    const originalAudioHtml = segment.audio_preview_url
                        ? `<div class="audio-cell"><span class="audio-label">Orig:</span><audio class="segment-audio" controls preload="none" src="${segment.audio_preview_url}"></audio></div>`
                        : `<div class="audio-cell"><span class="audio-label">Orig:</span><span class="audio-placeholder">N/A</span></div>`;
                    audioRow.innerHTML = `
                            ${originalAudioHtml}
                            <div class="audio-cell"><span class="audio-label">Gen:</span><audio class="segment-preview-audio" controls preload="none" style="display:none;"></audio><span class="segment-preview-placeholder audio-placeholder">Click Preview</span></div>
                        `;
                    body.appendChild(audioRow);

                    const previewControls = controlsRow;
                    const previewButton = previewControls.querySelector('.segment-preview-btn');
                    previewButton.addEventListener('click', () => handleSegmentPreview(card, previewButton));
                }

                card.appendChild(body);
                translateSegmentsList.appendChild(card);
            });
            if (!hasSpeech) {
                translateSegmentsList.insertAdjacentHTML('beforeend', '<div class="segment-empty">No speech segments detected.</div>');
            }
            updateTranslateSegmentsSummary();
        }

        function readSegmentCardValues(card, options = {}) {
            const { forceGenerate = false } = options;
            if (!card) {
                throw new Error('Segment card not found.');
            }
            const index = parseInt(card.dataset.index, 10);
            if (Number.isNaN(index)) {
                throw new Error('Segment metadata missing index.');
            }
            const type = card.dataset.type || 'speech';
            const startInput = card.querySelector('.segment-start');
            const endInput = card.querySelector('.segment-end');
            const startMs = parseInt(startInput ? startInput.value : '0', 10);
            const endMs = parseInt(endInput ? endInput.value : '0', 10);
            if (Number.isNaN(startMs) || Number.isNaN(endMs)) {
                throw new Error(`Segment #${index}: invalid timing.`);
            }
            if (endMs <= startMs) {
                throw new Error(`Segment #${index}: end time must be greater than start time.`);
            }
            const durationMs = endMs - startMs;
            const payload = {
                index,
                type,
                start_ms: startMs,
                end_ms: endMs,
                duration_ms: durationMs,
                start: formatTimestamp(startMs),
                end: formatTimestamp(endMs),
                source_text: '',
                translated_text: '',
                generate: false,
                keep_original: true,
                speaker: card.dataset.speaker || null,
            };
            if (type === 'speech') {
                const checkbox = card.querySelector('input.segment-generate');
                const shouldGenerate = forceGenerate ? true : checkbox ? checkbox.checked : true;
                payload.generate = shouldGenerate;
                payload.keep_original = !shouldGenerate;
                const sourceInput = card.querySelector('.segment-source');
                payload.source_text = sourceInput ? sourceInput.value : '';
                const translationInput = card.querySelector('.segment-translation');
                payload.translated_text = translationInput ? translationInput.value : '';
                const volumeInput = card.querySelector('.segment-volume');
                if (volumeInput) {
                    const rawVolume = (volumeInput.value || '').trim();
                    if (rawVolume) {
                        const parsedVolume = parseFloat(rawVolume);
                        if (Number.isNaN(parsedVolume)) {
                            throw new Error(`Segment #${index}: invalid volume override.`);
                        }
                        payload.volume_percent = parsedVolume;
                    }
                }
                const emotionInput = card.querySelector('.segment-emotion');
                if (emotionInput) {
                    const rawEmotion = (emotionInput.value || '').trim();
                    if (rawEmotion) {
                        const parsedEmotion = parseFloat(rawEmotion);
                        if (Number.isNaN(parsedEmotion)) {
                            throw new Error(`Segment #${index}: invalid emotion weight.`);
                        }
                        payload.emotion_weight = parsedEmotion;
                    }
                }
            } else {
                payload.generate = false;
                payload.keep_original = true;
            }
            return payload;
        }

        async function handleSegmentPreview(card, triggerButton) {
            if (!currentTranslateSessionId) {
                showStatus('Analyze audio first to enable previews.', 'error', 'translateSegmentsStatus');
                return;
            }
            if (!card) {
                return;
            }
            const statusEl = card.querySelector('.segment-preview-status');
            const audioEl = card.querySelector('.segment-preview-audio');
            try {
                const segmentPayload = readSegmentCardValues(card, { forceGenerate: true });
                if (segmentPayload.type !== 'speech') {
                    if (statusEl) {
                        statusEl.textContent = 'Only speech segments can be previewed.';
                        statusEl.style.color = '#d93025';
                    }
                    return;
                }
                const requestPayload = {
                    session_id: currentTranslateSessionId,
                    segment: segmentPayload,
                    duration_control: getDurationControlMode(),
                };
                if (translateVolumeInput && translateVolumeInput.value) {
                    const parsedVolume = parseFloat(translateVolumeInput.value);
                    if (!Number.isNaN(parsedVolume)) {
                        requestPayload.generated_volume_percent = parsedVolume;
                    }
                }
                if (translateBackingVolumeInput && translateBackingVolumeInput.value) {
                    const parsedBacking = parseFloat(translateBackingVolumeInput.value);
                    if (!Number.isNaN(parsedBacking)) {
                        requestPayload.backing_volume_percent = parsedBacking;
                    }
                }
                if (speakerOverridesDirty) {
                    requestPayload.speaker_overrides = buildSpeakerOverridesPayload();
                }
                if (triggerButton) {
                    triggerButton.disabled = true;
                }
                if (statusEl) {
                    statusEl.textContent = 'Generating preview...';
                    statusEl.style.color = '#666';
                }
                const response = await fetch(ENDPOINTS.TRANSLATE_SEGMENT_PREVIEW, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestPayload),
                });
                if (!response.ok) {
                    let errorMessage = `Preview failed (${response.status})`;
                    try {
                        const errorData = await response.json();
                        if (errorData && errorData.message) {
                            errorMessage = errorData.message;
                        }
                    } catch (jsonError) {
                        console.warn('Failed to parse preview error response', jsonError);
                    }
                    if (statusEl) {
                        statusEl.textContent = errorMessage;
                        statusEl.style.color = '#d93025';
                    }
                    showStatus(errorMessage, 'error', 'translateSegmentsStatus');
                    return;
                }
                const data = await response.json();
                if (!data || !data.audio_preview) {
                    const message = 'Preview failed: missing audio.';
                    if (statusEl) {
                        statusEl.textContent = message;
                        statusEl.style.color = '#d93025';
                    }
                    showStatus(message, 'error', 'translateSegmentsStatus');
                    return;
                }
                if (audioEl) {
                    audioEl.src = data.audio_preview;
                    audioEl.style.display = 'block';
                    audioEl.load();
                    // Hide placeholder if exists
                    const placeholder = card.querySelector('.segment-preview-placeholder');
                    if (placeholder) placeholder.style.display = 'none';
                }
                if (statusEl) {
                    const label = data.media_type || 'audio';
                    statusEl.textContent = `✓ ${label}`;
                    statusEl.style.color = 'var(--brand-emerald)';
                }
                // Mark compact gen button as having audio
                const compactGenBtn = card.querySelector('.gen-btn');
                if (compactGenBtn) {
                    compactGenBtn.classList.add('has-audio');
                }
            } catch (error) {
                const message = error && error.message ? error.message : 'Preview failed.';
                if (statusEl) {
                    statusEl.textContent = message;
                    statusEl.style.color = '#d93025';
                }
                showStatus(message, 'error', 'translateSegmentsStatus');
            } finally {
                if (triggerButton) {
                    triggerButton.disabled = false;
                }
            }
        }

        function renderSeparationPreview(sessionId, metadata) {
            if (!translateSeparationPreview) {
                return;
            }
            const separationMeta = metadata && metadata.separation;
            if (!separationMeta || !separationMeta.vocals_available || !separationMeta.vocals_url) {
                translateSeparationPreview.style.display = 'none';
                translateSeparationPreview.innerHTML = '';
                return;
            }
            const cacheKey = Date.now();
            const vocalsUrl = `${separationMeta.vocals_url}?session=${sessionId}&t=${cacheKey}`;
            let backingMarkup = '';
            if (separationMeta.backing_available && separationMeta.backing_url) {
                const backingUrl = `${separationMeta.backing_url}?session=${sessionId}&t=${cacheKey}`;
                let backingLabel = '🎼 Instrumental Backing';
                if (separationMeta.backing_source === 'custom') {
                    backingLabel += ' (Custom)';
                } else if (separationMeta.backing_source === 'reuse') {
                    backingLabel += ' (Reused)';
                } else if (separationMeta.backing_source === 'extracted') {
                    backingLabel += ' (ClearVoice)';
                }
                backingMarkup = `
                        <div style="margin-top: 12px;">
                            <div class="segment-header" style="margin-bottom:4px;">${backingLabel}</div>
                            <audio controls style="width: 100%;">
                                <source src="${backingUrl}" type="audio/mpeg">
                            </audio>
                        </div>
                    `;
            }
            translateSeparationPreview.innerHTML = `
                    <div class="segment-card">
                        <div class="segment-header">🎙️ Separated Vocals</div>
                        <audio controls style="width: 100%; margin-top: 6px;">
                            <source src="${vocalsUrl}" type="audio/mpeg">
                        </audio>
                        ${backingMarkup}
                        <label class="segment-checkbox" style="margin-top: 14px;">
                            <input type="checkbox" id="translateReuseSeparation" checked>
                            <span>Reuse this separation for future analyses</span>
                        </label>
                        <small style="display: block; color: var(--text-muted); margin-top: 4px;">
                            Uncheck to re-run separation on the original upload.
                        </small>
                    </div>
                `;
            translateSeparationPreview.style.display = 'block';
        }

        // Speaker color index mapping
