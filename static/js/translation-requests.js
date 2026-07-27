"use strict";

function syncSegmentRulesFromMetadata(rules) {
            if (!rules) {
                return;
            }
            if (translateMinSpeechInput) {
                if (rules.min_speech_ms !== undefined && rules.min_speech_ms !== null) {
                    translateMinSpeechInput.value = rules.min_speech_ms;
                } else {
                    translateMinSpeechInput.value = '';
                }
            }
            if (translateMaxMergeInput) {
                if (rules.max_merge_ms !== undefined && rules.max_merge_ms !== null) {
                    translateMaxMergeInput.value = rules.max_merge_ms;
                } else {
                    translateMaxMergeInput.value = '';
                }
            }
            updateAdditionalSettingsSummary();
        }

        if (translateAdvancedToggle) {
            translateAdvancedToggle.addEventListener('change', () => {
                if (translateAdvancedSettings) {
                    translateAdvancedSettings.style.display = translateAdvancedToggle.checked ? 'block' : 'none';
                }
                if (!translateAdvancedToggle.checked) {
                    resetAdvancedPanel();
                }
                setTranslateButtonLabel();
            });
            setTranslateButtonLabel();
        }

        if (translateSegmentsSelectAll && translateSegmentsList) {
            translateSegmentsSelectAll.addEventListener('change', () => {
                const speechCheckboxes = translateSegmentsList.querySelectorAll('.segment-card.speech input.segment-generate');
                const compactCheckboxes = translateSegmentsList.querySelectorAll('.segment-card.speech input.segment-generate-compact');
                speechCheckboxes.forEach(cb => {
                    cb.checked = translateSegmentsSelectAll.checked;
                });
                compactCheckboxes.forEach(cb => {
                    cb.checked = translateSegmentsSelectAll.checked;
                });
                updateTranslateSegmentsSummary();
            });
        } else {
            setTranslateButtonLabel();
        }

        // Hide/show silence segments toggle
        if (translateHideSilence && translateSegmentsList) {
            translateHideSilence.addEventListener('change', () => {
                if (translateHideSilence.checked) {
                    translateSegmentsList.classList.add('hide-silence');
                } else {
                    translateSegmentsList.classList.remove('hide-silence');
                }
            });
        }

        const translateForm = document.getElementById('translateForm');
        if (translateForm) {
            translateForm.addEventListener('submit', async function (e) {
                e.preventDefault();

                const statusId = 'translateStatus';
                const resultDiv = document.getElementById('translateResult');
                const audioInput = document.getElementById('translateAudioFile');
                const destInput = document.getElementById('translateDestLanguage');
                const formatSelect = document.getElementById('translateOutputFormat');

                hideStatus(statusId);
                hideStatus('translateSegmentsStatus');
                resultDiv.innerHTML = '';

                const destLanguage = destInput.value.trim();
                if (!destLanguage) {
                    showStatus('Please select a destination language.', 'error', statusId);
                    return;
                }

                const hasChunkSelection = Boolean(currentChunkSessionId);
                const downloadedVideoId = getSelectedDownloadedVideoId();
                const selectedFormat = (formatSelect.value || 'mp3').toLowerCase();

                const advancedEnabled = translateAdvancedToggle && translateAdvancedToggle.checked;
                const reuseSeparationCheckbox = document.getElementById('translateReuseSeparation');
                const reuseCandidateSessionId = currentChunkSessionId || currentTranslateSessionId;
                if (
                    advancedEnabled &&
                    reuseSeparationCheckbox &&
                    reuseSeparationCheckbox.checked &&
                    !reuseCandidateSessionId
                ) {
                    showStatus('Analyze audio once before reusing separated tracks.', 'error', statusId);
                    return;
                }
                const reuseSeparationEnabled = Boolean(
                    advancedEnabled &&
                    reuseSeparationCheckbox &&
                    reuseSeparationCheckbox.checked &&
                    reuseCandidateSessionId
                );

                if (
                    (!audioInput.files || audioInput.files.length === 0) &&
                    !downloadedVideoId &&
                    !reuseSeparationEnabled &&
                    !hasChunkSelection
                ) {
                    showStatus('Please select a source audio file or downloaded video.', 'error', statusId);
                    return;
                }

                if (advancedEnabled) {
                    resetAdvancedPanel(false);
                    const formData = new FormData();
                    if (reuseSeparationEnabled) {
                        formData.append('reuse_session_id', reuseCandidateSessionId);
                    } else if (downloadedVideoId) {
                        formData.append('downloaded_video_id', downloadedVideoId);
                    } else if (audioInput.files && audioInput.files[0]) {
                        formData.append('audio_file', audioInput.files[0]);
                    } else if (hasChunkSelection) {
                        formData.append('reuse_session_id', currentChunkSessionId);
                    }
                    if (translateCustomBackingInput && translateCustomBackingInput.files.length > 0) {
                        formData.append('custom_backing_audio_file', translateCustomBackingInput.files[0]);
                    }
                    formData.append('dest_language', destLanguage);
                    formData.append('response_format', selectedFormat);
                    formData.append('enhance_voice', translateEnhanceEl && translateEnhanceEl.checked ? 'true' : 'false');
                    formData.append('enhancement_model', translateEnhancementModelEl ? translateEnhancementModelEl.value : 'MossFormerGAN_SE_16K');
                    formData.append('super_resolution_voice', translateSuperEl && translateSuperEl.checked ? 'true' : 'false');
                    formData.append('audio_separator_enabled', translateAudioSeparatorEl && translateAudioSeparatorEl.checked ? 'true' : 'false');
                    formData.append('audio_separator_model', translateAudioSeparatorModelEl ? translateAudioSeparatorModelEl.value : 'balance');
                    formData.append('audio_separator_use_soundfile', translateAudioSeparatorUseSoundfileEl && translateAudioSeparatorUseSoundfileEl.checked ? 'true' : 'false');
                    formData.append('merge_backing_track', translateMergeBackEl && translateMergeBackEl.checked ? 'true' : 'false');
                    formData.append('ignore_non_speech', translateIgnoreNonSpeechEl && translateIgnoreNonSpeechEl.checked ? 'true' : 'false');
                    formData.append('preserve_silence_audio', translatePreserveSilenceEl && translatePreserveSilenceEl.checked ? 'true' : 'false');
                    appendClearVoiceParallelSettings(formData);
                    if (translateBaseFilenameInput && translateBaseFilenameInput.value.trim()) {
                        formData.append('base_filename', translateBaseFilenameInput.value.trim());
                    }
                    if (translatePreserveSilenceEl && translatePreserveSilenceEl.checked && translateSilenceVolumeInput) {
                        formData.append('silence_volume_percent', translateSilenceVolumeInput.value || '100');
                    }
                    if (translateGeminiModel && translateGeminiModel.value) {
                        formData.append('gemini_model', translateGeminiModel.value);
                    }
                    if (translateGeminiApiKey && translateGeminiApiKey.value.trim()) {
                        formData.append('gemini_api_key', translateGeminiApiKey.value.trim());
                    }
                    if (translateTranscriptionPipeline && translateTranscriptionPipeline.value) {
                        formData.append('transcription_pipeline', translateTranscriptionPipeline.value);
                    }
                    if (translateTranscriptionPipeline && translateTranscriptionPipeline.value !== 'gemini' && translateTranslationLlmModel && translateTranslationLlmModel.value) {
                        formData.append('translation_llm_model', translateTranslationLlmModel.value);
                    }
                    if (translateTranscriptionPipeline && translateTranscriptionPipeline.value === 'whisperx') {
                        formData.append('whisperx_proxy_refiner', translateWhisperXProxyRefiner && translateWhisperXProxyRefiner.checked ? 'true' : 'false');
                    }
                    if (translateTranscriptionPipeline && translateTranscriptionPipeline.value === 'qwen_omnivad') {
                        formData.append('qwen_omnivad_enable_diarization', translateQwenOmniVadEnableDiarization && translateQwenOmniVadEnableDiarization.checked ? 'true' : 'false');
                        formData.append('qwen_omnivad_diarization_backend', translateQwenOmniVadDiarizationBackend ? translateQwenOmniVadDiarizationBackend.value : 'auto');
                        formData.append('qwen_omnivad_enable_forced_aligner', translateQwenOmniVadEnableForcedAligner && translateQwenOmniVadEnableForcedAligner.checked ? 'true' : 'false');
                        formData.append('qwen_omnivad_diarization_min_seconds', String(getQwenOmniVadDiarizationMinSeconds()));
                        formData.append('qwen_omnivad_merge_gap_seconds', String(getQwenOmniVadMergeGapSeconds()));
                    }
                    if (translateForceGeminiRefresh && translateForceGeminiRefresh.checked) {
                        formData.append('force_gemini_regenerate', 'true');
                    }
                    if (translateWhileTranscribing) {
                        formData.append('translate_text', translateWhileTranscribing.checked ? 'true' : 'false');
                    }
                    const customPromptValue = translateCustomPrompt ? translateCustomPrompt.value.trim() : '';
                    if (customPromptValue) {
                        formData.append('prompt', customPromptValue);
                    }
                    appendSegmentParameters(formData);
                    appendManualSegments(formData);
                    appendSrtSubtitleFiles(formData);

                    try {
                        if (translateBtn) {
                            translateBtn.disabled = true;
                        }
                        const translateEnabledNow = translateWhileTranscribing ? translateWhileTranscribing.checked : true;
                        await streamTranslateSegmentsRequest(formData, {
                            statusId,
                            translateEnabledNow,
                        });
                    } catch (error) {
                        console.error('Segment preparation error:', error);
                        const message = error && error.message ? error.message : 'Segment preparation error.';
                        showStatus(message, 'error', statusId);
                    } finally {
                        if (translateBtn) {
                            translateBtn.disabled = false;
                        }
                    }
                    return;
                } else {
                    resetAdvancedPanel();
                }

                const formData = new FormData();
                if (hasChunkSelection) {
                    formData.append('reuse_session_id', currentChunkSessionId);
                } else if (downloadedVideoId) {
                    formData.append('downloaded_video_id', downloadedVideoId);
                } else {
                    formData.append('audio_file', audioInput.files[0]);
                }
                if (translateCustomBackingInput && translateCustomBackingInput.files.length > 0) {
                    formData.append('custom_backing_audio_file', translateCustomBackingInput.files[0]);
                }
                formData.append('dest_language', destLanguage);
                formData.append('response_format', selectedFormat);
                formData.append('enhance_voice', translateEnhanceEl && translateEnhanceEl.checked ? 'true' : 'false');
                formData.append('enhancement_model', translateEnhancementModelEl ? translateEnhancementModelEl.value : 'MossFormerGAN_SE_16K');
                formData.append('super_resolution_voice', translateSuperEl && translateSuperEl.checked ? 'true' : 'false');
                formData.append('audio_separator_enabled', translateAudioSeparatorEl && translateAudioSeparatorEl.checked ? 'true' : 'false');
                formData.append('audio_separator_model', translateAudioSeparatorModelEl ? translateAudioSeparatorModelEl.value : 'balance');
                formData.append('audio_separator_use_soundfile', translateAudioSeparatorUseSoundfileEl && translateAudioSeparatorUseSoundfileEl.checked ? 'true' : 'false');
                formData.append('merge_backing_track', translateMergeBackEl && translateMergeBackEl.checked ? 'true' : 'false');
                formData.append('ignore_non_speech', translateIgnoreNonSpeechEl && translateIgnoreNonSpeechEl.checked ? 'true' : 'false');
                formData.append('preserve_silence_audio', translatePreserveSilenceEl && translatePreserveSilenceEl.checked ? 'true' : 'false');
                appendClearVoiceParallelSettings(formData);
                if (translateBaseFilenameInput && translateBaseFilenameInput.value.trim()) {
                    formData.append('base_filename', translateBaseFilenameInput.value.trim());
                }
                if (translatePreserveSilenceEl && translatePreserveSilenceEl.checked && translateSilenceVolumeInput) {
                    formData.append('silence_volume_percent', translateSilenceVolumeInput.value || '100');
                }
                if (translateGeminiModel && translateGeminiModel.value) {
                    formData.append('gemini_model', translateGeminiModel.value);
                }
                if (translateGeminiApiKey && translateGeminiApiKey.value.trim()) {
                    formData.append('gemini_api_key', translateGeminiApiKey.value.trim());
                }
                if (translateTranscriptionPipeline && translateTranscriptionPipeline.value) {
                    formData.append('transcription_pipeline', translateTranscriptionPipeline.value);
                }
                if (translateTranscriptionPipeline && translateTranscriptionPipeline.value !== 'gemini' && translateTranslationLlmModel && translateTranslationLlmModel.value) {
                    formData.append('translation_llm_model', translateTranslationLlmModel.value);
                }
                if (translateTranscriptionPipeline && translateTranscriptionPipeline.value === 'whisperx') {
                    formData.append('whisperx_proxy_refiner', translateWhisperXProxyRefiner && translateWhisperXProxyRefiner.checked ? 'true' : 'false');
                }
                if (translateTranscriptionPipeline && translateTranscriptionPipeline.value === 'qwen_omnivad') {
                    formData.append('qwen_omnivad_enable_diarization', translateQwenOmniVadEnableDiarization && translateQwenOmniVadEnableDiarization.checked ? 'true' : 'false');
                    formData.append('qwen_omnivad_diarization_backend', translateQwenOmniVadDiarizationBackend ? translateQwenOmniVadDiarizationBackend.value : 'auto');
                    formData.append('qwen_omnivad_enable_forced_aligner', translateQwenOmniVadEnableForcedAligner && translateQwenOmniVadEnableForcedAligner.checked ? 'true' : 'false');
                    formData.append('qwen_omnivad_diarization_min_seconds', String(getQwenOmniVadDiarizationMinSeconds()));
                    formData.append('qwen_omnivad_merge_gap_seconds', String(getQwenOmniVadMergeGapSeconds()));
                }
                if (translateForceGeminiRefresh && translateForceGeminiRefresh.checked) {
                    formData.append('force_gemini_regenerate', 'true');
                }
                appendSegmentParameters(formData);
                appendManualSegments(formData);
                appendSrtSubtitleFiles(formData);

                try {
                    if (translateBtn) {
                        translateBtn.disabled = true;
                    }
                    await streamDirectTranslate(formData, selectedFormat, statusId, resultDiv);
                } catch (error) {
                    console.error('Translation error:', error);
                    showStatus(`Translation error: ${error.message}`, 'error', statusId);
                } finally {
                    if (translateBtn) {
                        translateBtn.disabled = false;
                    }
                }
            });
        }

        if (translateGenerateBtn) {
            translateGenerateBtn.addEventListener('click', async () => {
                const statusId = 'translateStatus';
                const resultDiv = document.getElementById('translateResult');
                hideStatus('translateSegmentsStatus');

                if (!currentTranslateSessionId) {
                    showStatus('Analyze audio first to load segments.', 'error', 'translateSegmentsStatus');
                    return;
                }
                if (!translateSegmentsList) {
                    showStatus('Segment list unavailable.', 'error', 'translateSegmentsStatus');
                    return;
                }
                const segmentCards = translateSegmentsList.querySelectorAll('.segment-card');
                if (!segmentCards.length) {
                    showStatus('No segments to generate.', 'error', 'translateSegmentsStatus');
                    return;
                }

                const segmentsPayload = [];
                let hasError = false;

                segmentCards.forEach(card => {
                    if (hasError) {
                        return;
                    }
                    try {
                        const segmentData = readSegmentCardValues(card);
                        segmentsPayload.push(segmentData);
                    } catch (segmentError) {
                        const message =
                            segmentError && segmentError.message
                                ? segmentError.message
                                : 'Segment validation failed.';
                        showStatus(message, 'error', 'translateSegmentsStatus');
                        hasError = true;
                    }
                });

                if (hasError || !segmentsPayload.length) {
                    return;
                }

                const formatSelect = document.getElementById('translateOutputFormat');
                const selectedFormat = (formatSelect && formatSelect.value ? formatSelect.value : 'mp3').toLowerCase();

                const payload = {
                    session_id: currentTranslateSessionId,
                    segments: segmentsPayload,
                    response_format: selectedFormat,
                    tts_backend: translateTtsBackendEl && translateTtsBackendEl.value ? translateTtsBackendEl.value : 'index',
                    duration_control: getDurationControlMode(),
                    merge_backing_track: translateMergeBackEl && translateMergeBackEl.checked ? true : false,
                };
                if (translateVolumeInput && translateVolumeInput.value) {
                    const volumeValue = parseFloat(translateVolumeInput.value);
                    if (!Number.isNaN(volumeValue)) {
                        payload.generated_volume_percent = volumeValue;
                    }
                }
                if (translateBackingVolumeInput && translateBackingVolumeInput.value) {
                    const backingValue = parseFloat(translateBackingVolumeInput.value);
                    if (!Number.isNaN(backingValue)) {
                        payload.backing_volume_percent = backingValue;
                    }
                }
                if (
                    translatePreserveSilenceEl &&
                    translatePreserveSilenceEl.checked &&
                    translateSilenceVolumeInput &&
                    translateSilenceVolumeInput.value
                ) {
                    const silenceValue = parseFloat(translateSilenceVolumeInput.value);
                    if (!Number.isNaN(silenceValue)) {
                        payload.silence_volume_percent = silenceValue;
                    }
                }
                if (speakerOverridesDirty) {
                    payload.speaker_overrides = buildSpeakerOverridesPayload();
                }

                try {
                    translateGenerateBtn.disabled = true;
                    await streamTranslateGenerateSegmentsRequest(payload, {
                        statusId,
                        segmentsStatusId: 'translateSegmentsStatus',
                        resultDiv,
                        selectedFormat,
                        segmentsPayload,
                    });
                } catch (error) {
                    console.error('Segment generation error:', error);
                    const message = error && error.message ? error.message : 'Segment generation error.';
                    showStatus(message, 'error', 'translateSegmentsStatus');
                    showStatus(message, 'error', statusId);
                } finally {
                    translateGenerateBtn.disabled = false;
                }
            });
        }

        async function streamDirectTranslate(formData, selectedFormat, statusId, resultDiv) {
            showStatus('Translating speech... this may take a moment ⏳', 'success', statusId);

            const response = await fetch(ENDPOINTS.TRANSLATE_AUDIO, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let errorMessage = `Translation failed (${response.status})`;
                const contentType = response.headers.get('Content-Type') || '';
                if (contentType.includes('application/json')) {
                    try {
                        const errorData = await response.json();
                        errorMessage = errorData.message || errorData.error || errorMessage;
                    } catch (jsonError) {
                        console.warn('Failed to parse error response:', jsonError);
                    }
                } else {
                    try {
                        errorMessage = await response.text();
                    } catch (textError) {
                        console.warn('Failed to read error response:', textError);
                    }
                }
                showStatus(errorMessage, 'error', statusId);
                return;
            }

            if (!response.body) {
                showStatus('Streaming is not supported in this browser.', 'error', statusId);
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            const newlineDelimiter = String.fromCharCode(10);
            let buffer = '';
            let translationCompleted = false;
            let lastStatusMessage = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) {
                    break;
                }
                buffer += decoder.decode(value, { stream: true });

                let newlineIndex = buffer.indexOf(newlineDelimiter);
                while (newlineIndex !== -1) {
                    const line = buffer.slice(0, newlineIndex).trim();
                    buffer = buffer.slice(newlineIndex + 1);
                    newlineIndex = buffer.indexOf(newlineDelimiter);

                    if (!line) {
                        continue;
                    }

                    let eventData;
                    try {
                        eventData = parseJsonStreamEventLine(line);
                    } catch (parseError) {
                        console.warn('Failed to parse translate event:', parseError, line);
                        continue;
                    }
                    if (!eventData) {
                        continue;
                    }

                    const eventType = eventData.event || 'status';
                    if (eventType === 'status') {
                        const message = eventData.message || 'Processing...';
                        lastStatusMessage = message;
                        showStatus(message, 'success', statusId);
                    } else if (eventType === 'heartbeat') {
                        const heartbeatMessage = lastStatusMessage
                            ? `Still processing... ⏳ (Last step: ${lastStatusMessage})`
                            : 'Still processing... ⏳';
                        showStatus(heartbeatMessage, 'success', statusId);
                    } else if (eventType === 'error') {
                        const message = eventData.message || 'Translation failed.';
                        showStatus(message, 'error', statusId);
                        return;
                    } else if (eventType === 'complete') {
                        translationCompleted = true;
                        const audioUrl = eventData.audio_url;
                        if (!audioUrl) {
                            showStatus('Translation succeeded but audio URL is missing.', 'error', statusId);
                            return;
                        }
                        const downloadName = eventData.file_name || `translated_speech.${selectedFormat}`;
                        const subtitleUrl =
                            eventData.subtitle_url ||
                            (eventData.metadata && eventData.metadata.subtitle && eventData.metadata.subtitle.url) ||
                            null;
                        const subtitleFileName =
                            eventData.subtitle_file_name ||
                            (eventData.metadata && eventData.metadata.subtitle && eventData.metadata.subtitle.filename) ||
                            'translated_speech.srt';
                        const originalSubtitleUrl =
                            eventData.original_subtitle_url ||
                            (eventData.metadata &&
                                eventData.metadata.subtitle_original &&
                                eventData.metadata.subtitle_original.url) ||
                            null;
                        const originalSubtitleFileName =
                            eventData.original_subtitle_file_name ||
                            (eventData.metadata &&
                                eventData.metadata.subtitle_original &&
                                eventData.metadata.subtitle_original.filename) ||
                            'translated_speech_original.srt';
                        const metadata = eventData.metadata || {};
                        renderTranslatedAudioPlayer(resultDiv, {
                            audioUrl,
                            downloadName,
                            subtitleUrl,
                            subtitleFileName,
                            originalSubtitleUrl,
                            originalSubtitleFileName,
                            metadata,
                            segments: currentTranslateSegments,
                        });
                        const statusMessage = buildTranslationCompleteMessage(metadata);
                        showStatus(statusMessage, 'success', statusId);
                        applyChunkGenerationMetadata(metadata, audioUrl);
                        autoApplyTranslateMetadata(metadata, metadata.session_id || null);
                    }
                }
            }

            if (!translationCompleted) {
                showStatus('Translation stream ended unexpectedly.', 'error', statusId);
            }
        }

        async function streamTranslateSegmentsRequest(formData, { statusId, translateEnabledNow }) {
            showStatus('Analyzing audio and preparing editable segments... ⏳', 'success', statusId);
            const response = await fetch(ENDPOINTS.TRANSLATE_SEGMENTS, {
                method: 'POST',
                body: formData,
            });

            const errorFromResponse = async () => {
                const contentType = response.headers.get('Content-Type') || '';
                if (contentType.includes('application/json')) {
                    try {
                        const errorData = await response.json();
                        return errorData.message || errorData.error;
                    } catch (jsonError) {
                        console.warn('Failed to parse error response:', jsonError);
                    }
                }
                try {
                    return await response.text();
                } catch (textError) {
                    console.warn('Failed to read error response:', textError);
                }
                return null;
            };

            if (!response.ok) {
                const message = (await errorFromResponse()) || `Segment preparation failed (${response.status})`;
                showStatus(message, 'error', statusId);
                throw new Error(message);
            }

            if (!response.body) {
                const message = 'Segment preparation failed: streaming not supported in this browser.';
                showStatus(message, 'error', statusId);
                throw new Error(message);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            const newline = '\n';
            let buffer = '';
            let lastStatusMessage = '';
            let completed = false;

            const applyCompletion = data => {
                if (!data || !data.session_id) {
                    const message = 'Segment preparation failed: missing session data.';
                    showStatus(message, 'error', statusId);
                    throw new Error(message);
                }

                currentTranslateSessionId = data.session_id;

                if (data.metadata && data.metadata.gemini_model && translateGeminiModel) {
                    translateGeminiModel.value = data.metadata.gemini_model;
                }
                if (
                    translateForceGeminiRefresh &&
                    data.metadata &&
                    typeof data.metadata.force_gemini_regenerate === 'boolean'
                ) {
                    translateForceGeminiRefresh.checked = !!data.metadata.force_gemini_regenerate;
                }
                if (translateIgnoreNonSpeechEl && data.metadata && typeof data.metadata.ignore_non_speech === 'boolean') {
                    translateIgnoreNonSpeechEl.checked = !!data.metadata.ignore_non_speech;
                    refreshPromptTemplates();
                }
                if (
                    data.metadata &&
                    (data.metadata.gemini_model ||
                        typeof data.metadata.ignore_non_speech === 'boolean')
                ) {
                    saveTranslatePersistentSettings();
                    updateAiConfigSummary();
                }
                if (translatePreserveSilenceEl && data.metadata && typeof data.metadata.preserve_silence_audio === 'boolean') {
                    translatePreserveSilenceEl.checked = !!data.metadata.preserve_silence_audio;
                    syncSilenceVolumeUI();
                }

                const metadata = data.metadata || {};
                autoApplyTranslateMetadata(metadata, data.session_id);
                if (metadata.segment_rules) {
                    syncSegmentRulesFromMetadata(metadata.segment_rules);
                }

                currentTranslateSegments = Array.isArray(data.segments)
                    ? data.segments.map(seg => ({
                        ...seg,
                        generate: translateEnabledNow ? seg.generate !== false : false,
                    }))
                    : [];
                renderTranslateSegments(currentTranslateSegments);
                if (translateAdvancedPanel) {
                    translateAdvancedPanel.style.display = 'block';
                }

                const speechCount =
                    (metadata && metadata.speech_segment_count) ||
                    currentTranslateSegments.filter(seg => seg.type === 'speech').length;
                const totalCount = currentTranslateSegments.length;
                let statusMessage = `✅ Segments ready: ${totalCount}`;
                if (typeof speechCount === 'number') {
                    statusMessage += ` total • ${speechCount} speech`;
                }
                showStatus(`${statusMessage}. Review below and choose segments to regenerate.`, 'success', statusId);
                if (!currentTranslateSegments.length) {
                    showStatus('No segments detected. Try adjusting the audio or prompt.', 'error', 'translateSegmentsStatus');
                }

                completed = true;
            };

            while (true) {
                const { value, done } = await reader.read();
                if (done) {
                    break;
                }
                buffer += decoder.decode(value, { stream: true });

                let newlineIndex = buffer.indexOf(newline);
                while (newlineIndex !== -1) {
                    const line = buffer.slice(0, newlineIndex).trim();
                    buffer = buffer.slice(newlineIndex + 1);
                    newlineIndex = buffer.indexOf(newline);

                    if (!line) {
                        continue;
                    }

                    let eventData;
                    try {
                        eventData = parseJsonStreamEventLine(line);
                    } catch (parseError) {
                        console.warn('Failed to parse segment stream event:', parseError, line);
                        continue;
                    }
                    if (!eventData) {
                        continue;
                    }

                    const eventType = eventData.event || 'status';
                    if (eventType === 'status') {
                        lastStatusMessage = eventData.message || 'Processing...';
                        showStatus(lastStatusMessage, 'success', statusId);
                    } else if (eventType === 'heartbeat') {
                        const heartbeatMessage = lastStatusMessage
                            ? `Still processing... ⏳ (Last step: ${lastStatusMessage})`
                            : 'Still processing... ⏳';
                        showStatus(heartbeatMessage, 'success', statusId);
                    } else if (eventType === 'error') {
                        const message = eventData.message || 'Failed to prepare segments.';
                        showStatus(message, 'error', statusId);
                        throw new Error(message);
                    } else if (eventType === 'complete') {
                        applyCompletion(eventData);
                    }
                }
            }

            if (!completed) {
                const message = 'Segment preparation stream ended unexpectedly.';
                showStatus(message, 'error', statusId);
                throw new Error(message);
            }
        }

        async function streamTranslateGenerateSegmentsRequest(requestPayload, options) {
            const {
                statusId,
                segmentsStatusId = 'translateSegmentsStatus',
                resultDiv,
                selectedFormat,
                segmentsPayload,
            } = options;

            showStatus('Generating selected segments... 🎧', 'success', statusId);
            const response = await fetch(ENDPOINTS.TRANSLATE_GENERATE_SEGMENTS, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestPayload),
            });

            const readError = async () => {
                const contentType = response.headers.get('Content-Type') || '';
                if (contentType.includes('application/json')) {
                    try {
                        const errorData = await response.json();
                        return errorData.message || errorData.error;
                    } catch (jsonError) {
                        console.warn('Failed to parse error response:', jsonError);
                    }
                }
                try {
                    return await response.text();
                } catch (textError) {
                    console.warn('Failed to read error response:', textError);
                }
                return null;
            };

            if (!response.ok) {
                const message = (await readError()) || `Generation failed (${response.status})`;
                showStatus(message, 'error', segmentsStatusId);
                showStatus(message, 'error', statusId);
                throw new Error(message);
            }

            if (!response.body) {
                const message = 'Segment generation failed: streaming not supported in this browser.';
                showStatus(message, 'error', segmentsStatusId);
                showStatus(message, 'error', statusId);
                throw new Error(message);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            const newline = '\n';
            let buffer = '';
            let lastStatusMessage = '';
            let completed = false;

            const finalizeGeneration = data => {
                const audioUrl = data && data.audio_url;
                if (!audioUrl) {
                    const message = 'Segment generation succeeded but audio URL is missing.';
                    showStatus(message, 'error', segmentsStatusId);
                    showStatus(message, 'error', statusId);
                    throw new Error(message);
                }

                const downloadName = data.file_name || `translated_speech.${selectedFormat}`;
                const metadata = data.metadata || {};
                const subtitleUrl =
                    data.subtitle_url ||
                    (metadata.subtitle && metadata.subtitle.url) ||
                    null;
                const subtitleFileName =
                    data.subtitle_file_name ||
                    (metadata.subtitle && metadata.subtitle.filename) ||
                    'translated_speech.srt';
                const originalSubtitleUrl =
                    data.original_subtitle_url ||
                    (metadata.subtitle_original && metadata.subtitle_original.url) ||
                    null;
                const originalSubtitleFileName =
                    data.original_subtitle_file_name ||
                    (metadata.subtitle_original && metadata.subtitle_original.filename) ||
                    'translated_speech_original.srt';
                if (resultDiv) {
                    renderTranslatedAudioPlayer(resultDiv, {
                        audioUrl,
                        downloadName,
                        subtitleUrl,
                        subtitleFileName,
                        originalSubtitleUrl,
                        originalSubtitleFileName,
                        metadata,
                        segments: segmentsPayload,
                    });
                }

                let statusMessage = '✅ Advanced translation complete!';
                if (typeof metadata.segment_count === 'number') {
                    statusMessage += ` (${metadata.segment_count} segments)`;
                }
                if (typeof metadata.selected_generated_count === 'number' && typeof metadata.selected_preserved_count === 'number') {
                    const detailMessage = `Generated ${metadata.selected_generated_count}, preserved ${metadata.selected_preserved_count}`;
                    showStatus(detailMessage, 'success', segmentsStatusId);
                    statusMessage += ` • Generated ${metadata.selected_generated_count}, preserved ${metadata.selected_preserved_count}`;
                }
                showStatus(statusMessage, 'success', statusId);
                applyChunkGenerationMetadata(metadata, audioUrl);

                const segmentMap = new Map(segmentsPayload.map(seg => [seg.index, seg]));
                currentTranslateSegments = currentTranslateSegments.map(seg => {
                    const updated = segmentMap.get(seg.index);
                    if (!updated) {
                        return seg;
                    }
                    const duration = Math.max(0, updated.end_ms - updated.start_ms);
                    return {
                        ...seg,
                        start_ms: updated.start_ms,
                        end_ms: updated.end_ms,
                        duration_ms: duration,
                        start: formatTimestamp(updated.start_ms),
                        end: formatTimestamp(updated.end_ms),
                        source_text: updated.source_text || '',
                        translated_text: updated.translated_text || '',
                        generate: updated.generate,
                        volume_percent:
                            Object.prototype.hasOwnProperty.call(updated, 'volume_percent')
                                ? updated.volume_percent
                                : seg.volume_percent,
                        emotion_weight:
                            Object.prototype.hasOwnProperty.call(updated, 'emotion_weight')
                                ? updated.emotion_weight
                                : seg.emotion_weight,
                    };
                });
                renderTranslateSegments(currentTranslateSegments);

                completed = true;
            };

            while (true) {
                const { value, done } = await reader.read();
                if (done) {
                    break;
                }
                buffer += decoder.decode(value, { stream: true });

                let newlineIndex = buffer.indexOf(newline);
                while (newlineIndex !== -1) {
                    const line = buffer.slice(0, newlineIndex).trim();
                    buffer = buffer.slice(newlineIndex + 1);
                    newlineIndex = buffer.indexOf(newline);

                    if (!line) {
                        continue;
                    }

                    let eventData;
                    try {
                        eventData = parseJsonStreamEventLine(line);
                    } catch (parseError) {
                        console.warn('Failed to parse generate event:', parseError, line);
                        continue;
                    }
                    if (!eventData) {
                        continue;
                    }

                    const eventType = eventData.event || 'status';
                    if (eventType === 'status') {
                        lastStatusMessage = eventData.message || 'Processing...';
                        showStatus(lastStatusMessage, 'success', statusId);
                    } else if (eventType === 'heartbeat') {
                        const heartbeatMessage = lastStatusMessage
                            ? `Still processing... ⏳ (Last step: ${lastStatusMessage})`
                            : 'Still processing... ⏳';
                        showStatus(heartbeatMessage, 'success', statusId);
                    } else if (eventType === 'error') {
                        const message = eventData.message || 'Generation failed.';
                        showStatus(message, 'error', segmentsStatusId);
                        showStatus(message, 'error', statusId);
                        throw new Error(message);
                    } else if (eventType === 'complete') {
                        finalizeGeneration(eventData);
                    }
                }
            }

            if (!completed) {
                const message = 'Segment generation stream ended unexpectedly.';
                showStatus(message, 'error', segmentsStatusId);
                showStatus(message, 'error', statusId);
                throw new Error(message);
            }
        }
