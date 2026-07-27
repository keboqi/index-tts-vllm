"use strict";

function updateChunkSelectionUI() {
            if (!translateChunkSelectionBanner || !translateChunkSelectionText) {
                return;
            }
            if (!translateSelectedChunkId) {
                translateChunkSelectionBanner.style.display = 'none';
                translateChunkSelectionText.textContent = '';
                return;
            }
            const chunk = translateChunkSessions.find(entry => entry.session_id === translateSelectedChunkId);
            if (!chunk) {
                translateChunkSelectionBanner.style.display = 'none';
                translateChunkSelectionText.textContent = '';
                return;
            }
            translateChunkSelectionBanner.style.display = 'block';
            translateChunkSelectionText.textContent = `Using chunk ${chunk.chunk_index ?? ''} (${chunk.start_label || '00:00'} → ${chunk.end_label || '??'})`;
        }

        function syncPendingSelectToggle() {
            if (!translateChunkSelectPending) {
                return;
            }
            const pendingChunks = translateChunkSessions.filter(chunk => !chunk.generated);
            if (!pendingChunks.length) {
                translateChunkSelectPending.checked = false;
                translateChunkSelectPending.disabled = true;
                return;
            }
            translateChunkSelectPending.disabled = false;
            const allSelected = pendingChunks.every(chunk => translateChunkSelections.has(chunk.session_id));
            translateChunkSelectPending.checked = allSelected;
        }

        function updateChunkBatchControlsVisibility() {
            if (!translateChunkBatchControls || !translateGenerateChunksBtn) {
                return;
            }
            const shouldShow =
                translateChunkSessions.length > 0 &&
                translateEnableChunkSplit &&
                translateEnableChunkSplit.checked;
            translateChunkBatchControls.style.display = shouldShow ? 'flex' : 'none';
            translateGenerateChunksBtn.style.display = shouldShow ? 'inline-flex' : 'none';
            translateGenerateChunksBtn.disabled = translateChunkSelections.size === 0;
            translateGenerateChunksBtn.textContent = translateChunkSelections.size
                ? `⚡ Generate Selected Chunks (${translateChunkSelections.size})`
                : '⚡ Generate Selected Chunks';
            // Show download/upload buttons when chunks are available
            if (translateDownloadChunksBtn) {
                translateDownloadChunksBtn.style.display = shouldShow && translateChunkBatchId ? 'inline-flex' : 'none';
            }
            if (translateUploadTranscriptionsBtn) {
                translateUploadTranscriptionsBtn.style.display = shouldShow && translateChunkBatchId ? 'inline-flex' : 'none';
            }
            syncPendingSelectToggle();
        }

        function applyChunkGenerationMetadata(metadata, audioUrl, options = {}) {
            if (!metadata || !metadata.chunk || !metadata.chunk.session_id) {
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
            });
            const { autoSelect = true } = options;
            const chunkSessionId = metadata.chunk.session_id;
            if (autoSelect) {
                currentChunkSessionId = chunkSessionId;
                translateSelectedChunkId = chunkSessionId;
            }
            if (!translateChunkBatchId && metadata.chunk.batch_id) {
                translateChunkBatchId = metadata.chunk.batch_id;
            }
            let chunkEntry = translateChunkSessions.find(entry => entry.session_id === chunkSessionId);
            if (!chunkEntry) {
                const startMs = typeof metadata.chunk.start_ms === 'number' ? metadata.chunk.start_ms : 0;
                const endMs = typeof metadata.chunk.end_ms === 'number' ? metadata.chunk.end_ms : startMs;
                const durationMs = Math.max(0, endMs - startMs);
                chunkEntry = {
                    chunk_index: metadata.chunk.chunk_index ?? metadata.chunk.index ?? translateChunkSessions.length + 1,
                    session_id: chunkSessionId,
                    reuse_session_id: chunkSessionId,
                    start_ms: startMs,
                    end_ms: endMs,
                    duration_ms: durationMs,
                    start_label: metadata.chunk.start_label || formatTimestamp(startMs),
                    end_label: metadata.chunk.end_label || formatTimestamp(endMs),
                    duration_label: metadata.chunk.duration_label || formatTimestamp(durationMs),
                    generated: false,
                    generated_at: null,
                    audio_url: metadata.chunk.audio_url || null,
                    output_format: metadata.chunk.output_format || null,
                    output_filename: metadata.chunk.output_filename || null,
                    backing_available:
                        metadata.chunk.backing_available !== undefined
                            ? Boolean(metadata.chunk.backing_available)
                            : true,
                    backing_source: metadata.chunk.backing_source || 'none',
                    vocals_url: metadata.chunk.vocals_url || `/api/translate_vocals/${chunkSessionId}`,
                    backing_url:
                        metadata.chunk.backing_available === false
                            ? null
                            : metadata.chunk.backing_url || `/api/translate_backing_track/${chunkSessionId}`,
                    batch_id: metadata.chunk.batch_id || translateChunkBatchId || null,
                };
                translateChunkSessions.push(chunkEntry);
            }
            chunkEntry.generated = true;
            chunkEntry.generated_at = Date.now();
            if (audioUrl) {
                chunkEntry.audio_url = audioUrl;
            }
            if (metadata.chunk.output_format) {
                chunkEntry.output_format = metadata.chunk.output_format;
            }
            if (metadata.chunk.output_filename) {
                chunkEntry.output_filename = metadata.chunk.output_filename;
            }
            renderChunkResultsFromResponse();
            updateChunkSelectionUI();
            updateChunkBatchControlsVisibility();
        }

        async function handleMergeChunks() {
            const statusId = translateMergeStatus ? 'translateMergeStatus' : 'translateStatus';
            if (!translateChunkSessions.length) {
                showStatus('Split audio into chunks before merging.', 'error', statusId);
                return;
            }
            const formatSelect = document.getElementById('translateOutputFormat');
            const selectedFormat = (formatSelect && formatSelect.value ? formatSelect.value : 'mp3').toLowerCase();
            const payload = {
                chunk_session_ids: translateChunkSessions.map(chunk => chunk.session_id),
                chunk_batch_id: translateChunkBatchId || null,
                response_format: selectedFormat,
                merge_backing_track: translateMergeBackEl && translateMergeBackEl.checked ? true : false,
            };
            try {
                if (translateMergeChunksBtn) {
                    translateMergeChunksBtn.disabled = true;
                }
                showStatus('Merging chunk outputs... ⏳', 'success', statusId);
                const response = await fetch(ENDPOINTS.TRANSLATE_MERGE_CHUNKS, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await response.json();
                if (!response.ok || data.status !== 'ok') {
                    const message = data.message || `Merge failed (${response.status})`;
                    showStatus(message, 'error', statusId);
                    return;
                }
                renderMergedAudioResult(data, selectedFormat);
                showStatus(data.message || 'Chunks merged successfully.', 'success', statusId);
            } catch (error) {
                showStatus(`Merge failed: ${error.message}`, 'error', statusId);
            } finally {
                if (translateMergeChunksBtn) {
                    translateMergeChunksBtn.disabled = false;
                }
            }
        }

        async function handleGenerateSelectedChunks() {
            const statusId = 'translateChunkBatchStatus';
            hideStatus(statusId);
            if (!translateChunkSelections.size) {
                showStatus('Select at least one pending chunk to generate.', 'error', statusId);
                return;
            }
            if (!translateDestLanguageSelect || !translateDestLanguageSelect.value.trim()) {
                showStatus('Select a destination language before generating chunks.', 'error', statusId);
                return;
            }
            const destLanguage = translateDestLanguageSelect.value.trim();
            const formatSelect = document.getElementById('translateOutputFormat');
            const selectedFormat = (formatSelect && formatSelect.value ? formatSelect.value : 'mp3').toLowerCase();
            const payload = {
                chunk_session_ids: Array.from(translateChunkSelections),
                dest_language: destLanguage,
                tts_backend: translateTtsBackendEl && translateTtsBackendEl.value ? translateTtsBackendEl.value : 'index',
                duration_control: getDurationControlMode(),
                response_format: selectedFormat,
            };
            if (translateGeminiModel && translateGeminiModel.value) {
                payload.gemini_model = translateGeminiModel.value;
            }
            if (translateGeminiApiKey && translateGeminiApiKey.value.trim()) {
                payload.gemini_api_key = translateGeminiApiKey.value.trim();
            }
            if (translateTranscriptionPipeline && translateTranscriptionPipeline.value) {
                payload.transcription_pipeline = translateTranscriptionPipeline.value;
            }
            if (
                translateTranscriptionPipeline &&
                translateTranscriptionPipeline.value !== 'gemini' &&
                translateTranslationLlmModel &&
                translateTranslationLlmModel.value
            ) {
                payload.translation_llm_model = translateTranslationLlmModel.value;
            }
            payload.whisperx_proxy_refiner = !!(
                translateTranscriptionPipeline &&
                translateTranscriptionPipeline.value === 'whisperx' &&
                translateWhisperXProxyRefiner &&
                translateWhisperXProxyRefiner.checked
            );
            payload.qwen_omnivad_enable_diarization = !!(
                translateTranscriptionPipeline &&
                translateTranscriptionPipeline.value === 'qwen_omnivad' &&
                translateQwenOmniVadEnableDiarization &&
                translateQwenOmniVadEnableDiarization.checked
            );
            payload.qwen_omnivad_diarization_backend =
                translateTranscriptionPipeline &&
                    translateTranscriptionPipeline.value === 'qwen_omnivad' &&
                    translateQwenOmniVadDiarizationBackend
                    ? translateQwenOmniVadDiarizationBackend.value
                    : 'auto';
            payload.qwen_omnivad_enable_forced_aligner = !!(
                translateTranscriptionPipeline &&
                translateTranscriptionPipeline.value === 'qwen_omnivad' &&
                translateQwenOmniVadEnableForcedAligner &&
                translateQwenOmniVadEnableForcedAligner.checked
            );
            payload.qwen_omnivad_diarization_min_seconds = getQwenOmniVadDiarizationMinSeconds();
            payload.qwen_omnivad_merge_gap_seconds = getQwenOmniVadMergeGapSeconds();
            payload.merge_backing_track = translateMergeBackEl && translateMergeBackEl.checked ? true : false;
            payload.ignore_non_speech = translateIgnoreNonSpeechEl && translateIgnoreNonSpeechEl.checked ? true : false;
            payload.preserve_silence_audio = translatePreserveSilenceEl && translatePreserveSilenceEl.checked ? true : false;
            payload.force_gemini_regenerate = !!(translateForceGeminiRefresh && translateForceGeminiRefresh.checked);
            if (translateDefaultSpeakerSelect && translateDefaultSpeakerSelect.value.trim()) {
                payload.default_speaker_preset = translateDefaultSpeakerSelect.value.trim();
            }
            if (translateDefaultEmotionWeightInput && translateDefaultEmotionWeightInput.value) {
                const defaultEmotion = parseFloat(translateDefaultEmotionWeightInput.value);
                if (!Number.isNaN(defaultEmotion)) {
                    payload.default_emotion_weight = defaultEmotion;
                }
            }
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
            try {
                if (translateGenerateChunksBtn) {
                    translateGenerateChunksBtn.disabled = true;
                }
                await streamChunkBatchGenerationRequest(payload);
            } catch (error) {
                const message = error && error.message ? error.message : 'Chunk batch generation failed.';
                showStatus(message, 'error', statusId);
            } finally {
                if (translateGenerateChunksBtn) {
                    translateGenerateChunksBtn.disabled = false;
                }
                updateChunkBatchControlsVisibility();
            }
        }

        async function streamChunkBatchGenerationRequest(payload) {
            const statusId = 'translateChunkBatchStatus';
            showStatus('Scheduling chunk generation... ⏳', 'info', statusId);
            const response = await fetch(ENDPOINTS.TRANSLATE_GENERATE_CHUNKS, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const readError = async () => {
                const contentType = response.headers.get('Content-Type') || '';
                if (contentType.includes('application/json')) {
                    try {
                        const errorData = await response.json();
                        return errorData.message || errorData.error;
                    } catch (jsonError) {
                        console.warn('Failed to parse chunk batch error response:', jsonError);
                    }
                }
                try {
                    return await response.text();
                } catch (textError) {
                    console.warn('Failed to read chunk batch error response:', textError);
                }
                return null;
            };

            if (!response.ok) {
                const message = (await readError()) || `Chunk generation failed (${response.status})`;
                showStatus(message, 'error', statusId);
                throw new Error(message);
            }

            if (!response.body) {
                const message = 'Chunk generation failed: streaming not supported in this browser.';
                showStatus(message, 'error', statusId);
                throw new Error(message);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            const newline = '\n';
            let buffer = '';
            let completed = false;
            let successfulChunks = 0;
            let failedChunks = 0;

            try {
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
                            console.warn('Failed to parse chunk batch event:', parseError, line);
                            continue;
                        }
                        if (!eventData) {
                            continue;
                        }
                        const eventType = eventData.event || 'status';
                        if (eventType === 'status') {
                            const message = eventData.message || 'Processing chunks...';
                            showStatus(message, 'info', statusId);
                        } else if (eventType === 'heartbeat') {
                            // Heartbeat event to keep connection alive - just log it
                            console.log('Chunk generation heartbeat:', eventData.message);
                        } else if (eventType === 'chunk_waiting') {
                            const waitSeconds = Math.ceil(eventData.delay_seconds || 0);
                            showStatus(
                                `Chunk ${eventData.chunk_index ?? ''} scheduled (starts in ~${waitSeconds}s).`,
                                'info',
                                statusId
                            );
                        } else if (eventType === 'chunk_start') {
                            showStatus(
                                `Chunk ${eventData.chunk_index ?? ''} generating...`,
                                'success',
                                statusId
                            );
                        } else if (eventType === 'chunk_complete') {
                            successfulChunks++;
                            if (eventData.metadata) {
                                applyChunkGenerationMetadata(eventData.metadata, eventData.audio_url, { autoSelect: false });
                            }
                            if (eventData.chunk_session_id) {
                                translateChunkSelections.delete(eventData.chunk_session_id);
                            }
                            const successMessage =
                                eventData.message || `Chunk ${eventData.chunk_index ?? ''} generated successfully.`;
                            showStatus(successMessage, 'success', statusId);
                            updateChunkBatchControlsVisibility();
                        } else if (eventType === 'chunk_error') {
                            failedChunks++;
                            const message = eventData.message || 'Chunk generation failed.';
                            showStatus(message, 'error', statusId);
                        } else if (eventType === 'complete') {
                            completed = true;
                            const detailParts = [];
                            if (typeof eventData.completed_chunks === 'number') {
                                detailParts.push(`${eventData.completed_chunks} succeeded`);
                            }
                            if (typeof eventData.failed_chunks === 'number') {
                                detailParts.push(`${eventData.failed_chunks} failed`);
                            }
                            const summaryMessage = eventData.message || 'Chunk batch generation finished.';
                            const detailMessage = detailParts.length ? ` (${detailParts.join(', ')})` : '';
                            showStatus(`${summaryMessage}${detailMessage}`, 'success', statusId);
                        }
                    }
                }
            } catch (streamError) {
                console.error('Stream read error during chunk generation:', streamError);
                // If we already processed some chunks successfully, show a partial success message
                if (successfulChunks > 0) {
                    const partialMessage = `Connection interrupted. ${successfulChunks} chunk(s) completed before error. You can retry the remaining chunks.`;
                    showStatus(partialMessage, 'warning', statusId);
                    updateChunkBatchControlsVisibility();
                    return; // Don't throw, partial success
                }
                throw streamError;
            }

            if (!completed) {
                // If we have successful chunks but didn't get complete event, show partial success
                if (successfulChunks > 0) {
                    const partialMessage = `Stream ended early. ${successfulChunks} chunk(s) completed. You can retry the remaining chunks.`;
                    showStatus(partialMessage, 'warning', statusId);
                    updateChunkBatchControlsVisibility();
                    return;
                }
                const message = 'Chunk batch generation stream ended unexpectedly.';
                showStatus(message, 'error', statusId);
                throw new Error(message);
            }
        }

        function renderMergedAudioResult(data, format) {
            const resultDiv = document.getElementById('translateResult');
            if (!resultDiv) {
                return;
            }
            const audioUrl = data.audio_url;
            if (!audioUrl) {
                return;
            }
            if (data.base_output_name && translateBaseFilenameInput) {
                translateBaseFilenameInput.value = data.base_output_name;
                translateBaseFilenameInput.dataset.userEdited = 'false';
            }
            if (data.language_code) {
                translateLanguageCodeHint = data.language_code;
            }
            updateFfmpegCommands({
                baseName: data.base_output_name,
                languageCode: data.language_code,
            });
            const downloadName = data.file_name || `merged_chunks.${format}`;
            const subtitleUrl = data.subtitle_url || (data.subtitle && data.subtitle.url) || null;
            const subtitleFileName =
                data.subtitle_file_name ||
                (data.subtitle && data.subtitle.filename) ||
                downloadName.replace(/\.[^.]+$/, '.srt');
            const originalSubtitleUrl =
                data.original_subtitle_url ||
                (data.subtitle_original && data.subtitle_original.url) ||
                null;
            const originalSubtitleFileName =
                data.original_subtitle_file_name ||
                (data.subtitle_original && data.subtitle_original.filename) ||
                downloadName.replace(/\.[^.]+$/, '_original.srt');
            renderTranslatedAudioPlayer(resultDiv, {
                audioUrl,
                downloadName,
                subtitleUrl,
                subtitleFileName,
                originalSubtitleUrl,
                originalSubtitleFileName,
                metadata: data.metadata || data,
                segments: [],
            });
        }

        function getMediaTypeForFormat(format) {
            switch (format) {
                case 'mp3':
                    return 'audio/mpeg';
                case 'wav':
                    return 'audio/wav';
                case 'flac':
                    return 'audio/flac';
                case 'aac':
                    return 'audio/aac';
                case 'opus':
                    return 'audio/opus';
                case 'ogg':
                    return 'audio/ogg';
                case 'webm':
                    return 'audio/webm';
                default:
                    return 'audio/mpeg';
            }
        }

        const FILENAME_FORBIDDEN_CHARS = /[<>:"/\\|?*]/g;
        const LANGUAGE_CODE_OVERRIDES = {
            chinese: 'chn',
            'zh-cn': 'chn',
            zh: 'chn',
            english: 'en',
            en: 'en',
            spanish: 'es',
            es: 'es',
            japanese: 'jp',
            ja: 'jp',
            korean: 'kr',
            ko: 'kr',
            german: 'de',
            de: 'de',
            french: 'fr',
            fr: 'fr',
            indonesian: 'id',
            id: 'id',
            italian: 'it',
            it: 'it',
            thai: 'th',
            th: 'th',
            portuguese: 'pt',
            pt: 'pt',
            russian: 'ru',
            ru: 'ru',
            malay: 'ms',
            ms: 'ms',
            vietnamese: 'vi',
            viet: 'vi',
            vi: 'vi',
        };
        const SUBTITLE_LANGUAGE_CODE_OVERRIDES = {
            chinese: 'zho',
            chn: 'zho',
            cnt: 'zho',
            zh: 'zho',
            english: 'eng',
            en: 'eng',
            spanish: 'spa',
            es: 'spa',
            japanese: 'jpn',
            jp: 'jpn',
            ja: 'jpn',
            korean: 'kor',
            kr: 'kor',
            ko: 'kor',
            german: 'deu',
            de: 'deu',
            french: 'fra',
            fr: 'fra',
            indonesian: 'ind',
            id: 'ind',
            italian: 'ita',
            it: 'ita',
            thai: 'tha',
            th: 'tha',
            portuguese: 'por',
            pt: 'por',
            russian: 'rus',
            ru: 'rus',
            malay: 'msa',
            ms: 'msa',
            vietnamese: 'vie',
            viet: 'vie',
            vi: 'vie',
        };

        function normalizeBaseFilenameInput(value) {
            if (!value) {
                return '';
            }
            let stem = `${value}`.replace(/\s+/g, ' ').trim();
            if (!stem) {
                return '';
            }
            stem = stem.replace(FILENAME_FORBIDDEN_CHARS, '_').replace(/__+/g, '_');
            stem = stem.replace(/^[.]+|[.]+$/g, '');
            return stem;
        }

        function deriveBaseFromFilename(filename) {
            if (!filename) {
                return '';
            }
            const lastSlash = Math.max(filename.lastIndexOf('/'), filename.lastIndexOf('\\'));
            const lastDot = filename.lastIndexOf('.');
            const withoutExt = lastDot > lastSlash ? filename.slice(0, lastDot) : filename;
            return normalizeBaseFilenameInput(withoutExt);
        }

        function getLanguageCodeForFilename(label) {
            if (!label) {
                return 'translated';
            }
            const lower = label.trim().toLowerCase();
            if (!lower) {
                return 'translated';
            }
            if (LANGUAGE_CODE_OVERRIDES[lower]) {
                return LANGUAGE_CODE_OVERRIDES[lower];
            }
            const compact = lower.replace(/[^a-z0-9]+/g, '');
            if (!compact) {
                return 'translated';
            }
            return compact.length <= 3 ? compact : compact.slice(0, 3);
        }

        function getSubtitleLanguageCode(labelOrCode) {
            if (!labelOrCode) {
                return 'und';
            }
            const lower = labelOrCode.trim().toLowerCase();
            if (!lower || lower === 'original' || lower === 'source') {
                return 'und';
            }
            if (SUBTITLE_LANGUAGE_CODE_OVERRIDES[lower]) {
                return SUBTITLE_LANGUAGE_CODE_OVERRIDES[lower];
            }
            const compact = lower.replace(/[^a-z0-9]+/g, '');
            return SUBTITLE_LANGUAGE_CODE_OVERRIDES[compact] || 'und';
        }

        function updateFfmpegCommands(options = {}) {
            if (!ffmpegPanel || !ffmpegExtractCmd || !ffmpegReplaceCmd || !ffmpegSubtitleCmd) {
                return;
            }
            const baseCandidate =
                options.baseName !== undefined
                    ? options.baseName
                    : translateBaseFilenameInput
                        ? translateBaseFilenameInput.value
                        : '';
            const normalizedBase = normalizeBaseFilenameInput(baseCandidate);
            if (translateBaseFilenameInput && options.baseName !== undefined && normalizedBase) {
                translateBaseFilenameInput.value = normalizedBase;
            }
            if (!normalizedBase) {
                ffmpegPanel.style.display = 'none';
                ffmpegExtractCmd.textContent = '';
                ffmpegReplaceCmd.textContent = '';
                if (ffmpegDualAudioCmd) {
                    ffmpegDualAudioCmd.textContent = '';
                }
                ffmpegSubtitleCmd.textContent = '';
                if (ffmpegSubtitleOriginalCmd) {
                    ffmpegSubtitleOriginalCmd.textContent = '';
                }
                if (ffmpegEmbedSubtitleCmd) {
                    ffmpegEmbedSubtitleCmd.textContent = '';
                }
                return;
            }
            const languageLabel =
                options.languageLabel ||
                (translateDestLanguageSelect && translateDestLanguageSelect.value.trim()) ||
                '';
            const preferredCode =
                options.languageCode ||
                translateLanguageCodeHint ||
                getLanguageCodeForFilename(languageLabel);
            const languageCode = preferredCode || 'translated';
            const subtitleLanguageCode = getSubtitleLanguageCode(languageCode);
            if (options.languageCode) {
                translateLanguageCodeHint = options.languageCode;
            }
            const baseMp4 = `"${normalizedBase}.mp4"`;
            const baseMp3 = `"${normalizedBase}.mp3"`;
            const translatedMp3 = `"${normalizedBase}_${languageCode}.mp3"`;
            const translatedMp4 = `"${normalizedBase}_${languageCode}.mp4"`;
            const dualAudioMp4 = `"${normalizedBase}_${languageCode}_dual_audio.mp4"`;
            const translatedSrtFilename = `${normalizedBase}_${languageCode}.srt`;
            const srtBurnedMp4 = `"${normalizedBase}_${languageCode}_srt.mp4"`;
            const originalSrtFilename = `${normalizedBase}_original.srt`;
            const originalSrtBurned = `"${normalizedBase}_original_srt.mp4"`;
            const subtitleForceStyle = "force_style='FontName=Noto Sans CJK SC,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1.5,Shadow=0.7'";
            const subtitleFontsDir = "fontsdir='fonts/noto-cjk'";
            const translatedSubtitleFilter = `"subtitles='${translatedSrtFilename.replace(/'/g, "\\'")}':${subtitleFontsDir}:${subtitleForceStyle}"`;
            const originalSubtitleFilter = `"subtitles='${originalSrtFilename.replace(/'/g, "\\'")}':${subtitleFontsDir}:${subtitleForceStyle}"`;
            ffmpegPanel.style.display = 'block';
            ffmpegExtractCmd.textContent = `ffmpeg -i ${baseMp4} -vn -codec:a libmp3lame -q:a 2 -threads 0 ${baseMp3}`;
            ffmpegReplaceCmd.textContent = `ffmpeg -i ${baseMp4} -i ${translatedMp3} -c:v copy -c:a aac -threads 0 -map 0:v:0 -map 1:a:0 ${translatedMp4}`;
            if (ffmpegDualAudioCmd) {
                ffmpegDualAudioCmd.textContent = `ffmpeg -i ${baseMp4} -i ${translatedMp3} -map 0:v:0 -map 0:a:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -metadata:s:a:0 title=Original -metadata:s:a:1 title=Translated -disposition:a:0 0 -disposition:a:1 default -threads 0 ${dualAudioMp4}`;
            }
            ffmpegSubtitleCmd.textContent = `ffmpeg -filter_threads 0 -filter_complex_threads 0 -i ${translatedMp4} -vf ${translatedSubtitleFilter} -c:v libx264 -preset veryfast -crf 18 -threads:v 0 -c:a copy ${srtBurnedMp4}`;
            if (ffmpegSubtitleOriginalCmd) {
                ffmpegSubtitleOriginalCmd.textContent = `ffmpeg -filter_threads 0 -filter_complex_threads 0 -i ${baseMp4} -vf ${originalSubtitleFilter} -c:v libx264 -preset veryfast -crf 18 -threads:v 0 -c:a copy ${originalSrtBurned}`;
            }
            if (ffmpegEmbedSubtitleCmd) {
                ffmpegEmbedSubtitleCmd.textContent = `ffmpeg -i ${translatedMp4} -i "${translatedSrtFilename}" -i "${originalSrtFilename}" -map 0:v:0 -map 0:a? -map 1:0 -map 2:0 -c:v copy -c:a copy -c:s mov_text -metadata:s:s:0 title=Translated -metadata:s:s:0 handler_name=Translated -metadata:s:s:0 language=${subtitleLanguageCode} -disposition:s:0 default -metadata:s:s:1 title=Original -metadata:s:s:1 handler_name=Original -metadata:s:s:1 language=und -disposition:s:1 0 -movflags +faststart "${normalizedBase}_${languageCode}_subtracks.mp4"`;
            }
        }

        async function handleSplitAudioRequest() {
            if (!translateSplitAudioBtn) {
                return;
            }
            const statusId = 'translateSplitStatus';
            hideStatus(statusId);
            resetChunkResults();
            const audioInput = document.getElementById('translateAudioFile');
            const downloadedVideoId = getSelectedDownloadedVideoId();
            const hasAudioFile = Boolean(audioInput && audioInput.files && audioInput.files.length);
            if (!hasAudioFile && !downloadedVideoId) {
                showStatus('Select a source audio file or downloaded video before splitting.', 'error', statusId);
                return;
            }
            const formData = new FormData();
            if (downloadedVideoId) {
                formData.append('downloaded_video_id', downloadedVideoId);
            } else {
                formData.append('audio_file', audioInput.files[0]);
            }
            if (translateDestLanguageSelect && translateDestLanguageSelect.value.trim()) {
                formData.append('dest_language', translateDestLanguageSelect.value.trim());
            }
            if (translateTtsBackendEl && translateTtsBackendEl.value) {
                formData.append('tts_backend', translateTtsBackendEl.value);
            }
            if (translateChunkMinInput && translateChunkMinInput.value) {
                formData.append('chunk_min_minutes', translateChunkMinInput.value);
            }
            if (translateChunkMaxInput && translateChunkMaxInput.value) {
                formData.append('chunk_max_minutes', translateChunkMaxInput.value);
            }
            if (translateChunkMinSilenceInput && translateChunkMinSilenceInput.value) {
                formData.append('min_silence_ms', translateChunkMinSilenceInput.value);
            }
            formData.append('super_resolution_voice', translateSuperEl && translateSuperEl.checked ? 'true' : 'false');
            formData.append('enhance_voice', translateEnhanceEl && translateEnhanceEl.checked ? 'true' : 'false');
            formData.append('enhancement_model', translateEnhancementModelEl ? translateEnhancementModelEl.value : 'MossFormerGAN_SE_16K');
            formData.append('audio_separator_enabled', translateAudioSeparatorEl && translateAudioSeparatorEl.checked ? 'true' : 'false');
            formData.append('audio_separator_model', translateAudioSeparatorModelEl ? translateAudioSeparatorModelEl.value : 'balance');
            formData.append('audio_separator_use_soundfile', translateAudioSeparatorUseSoundfileEl && translateAudioSeparatorUseSoundfileEl.checked ? 'true' : 'false');
            appendClearVoiceParallelSettings(formData);
            if (translateBaseFilenameInput && translateBaseFilenameInput.value.trim()) {
                formData.append('base_filename', translateBaseFilenameInput.value.trim());
            }

            try {
                translateSplitAudioBtn.disabled = true;
                translateChunkSummary && (translateChunkSummary.textContent = 'Splitting audio...');
                showStatus('Splitting audio into manageable chunks...', 'info', statusId);
                await streamSplitAudioRequest(formData, statusId);
            } catch (error) {
                console.error('Chunk split error:', error);
                showStatus(`Chunk split error: ${error.message}`, 'error', statusId);
            } finally {
                translateSplitAudioBtn.disabled = false;
            }
        }

        async function streamSplitAudioRequest(formData, statusId) {
            const response = await fetch(ENDPOINTS.TRANSLATE_SPLIT_AUDIO, {
                method: 'POST',
                body: formData,
            });

            const parseErrorPayload = async () => {
                const contentType = response.headers.get('Content-Type') || '';
                if (contentType.includes('application/json')) {
                    try {
                        const data = await response.json();
                        return data.message || data.error || null;
                    } catch (err) {
                        console.warn('Failed to parse split error response:', err);
                    }
                }
                try {
                    return await response.text();
                } catch (err) {
                    console.warn('Failed to read split error response:', err);
                }
                return null;
            };

            if (!response.ok) {
                const errorMessage =
                    (await parseErrorPayload()) || `Chunk split failed (HTTP ${response.status}).`;
                showStatus(errorMessage, 'error', statusId);
                throw new Error(errorMessage);
            }

            if (!response.body) {
                const message = 'Chunk split failed: streaming not supported in this browser.';
                showStatus(message, 'error', statusId);
                throw new Error(message);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            const newlineDelimiter = String.fromCharCode(10);
            let buffer = '';
            let splitCompleted = false;
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
                        console.warn('Failed to parse split event:', parseError, line);
                        continue;
                    }
                    if (!eventData) {
                        continue;
                    }
                    const eventType = eventData.event || 'status';
                    if (eventType === 'status') {
                        const message = eventData.message || 'Processing...';
                        lastStatusMessage = message;
                        showStatus(message, 'info', statusId);
                    } else if (eventType === 'heartbeat') {
                        const heartbeatMessage = lastStatusMessage
                            ? `Still splitting... ⏳ (Last step: ${lastStatusMessage})`
                            : 'Still splitting... ⏳';
                        showStatus(heartbeatMessage, 'info', statusId);
                    } else if (eventType === 'error') {
                        const message = eventData.message || 'Chunk split failed.';
                        showStatus(message, 'error', statusId);
                        throw new Error(message);
                    } else if (eventType === 'complete') {
                        splitCompleted = true;
                        const payload = {
                            chunks: Array.isArray(eventData.chunks) ? eventData.chunks : [],
                            chunk_batch_id: eventData.chunk_batch_id || null,
                            duration_label: eventData.duration_label,
                            duration_ms: eventData.duration_ms,
                        };
                        renderChunkResultsFromResponse(payload);
                        if (eventData.base_output_name && translateBaseFilenameInput) {
                            translateBaseFilenameInput.value = eventData.base_output_name;
                            translateBaseFilenameInput.dataset.userEdited = 'false';
                            updateFfmpegCommands({
                                baseName: eventData.base_output_name,
                                languageLabel: translateDestLanguageSelect && translateDestLanguageSelect.value,
                                languageCode: eventData.language_code,
                            });
                        } else {
                            updateFfmpegCommands();
                        }
                        if (translateChunkResults && translateEnableChunkSplit && translateEnableChunkSplit.checked) {
                            translateChunkResults.style.display = 'block';
                        }
                        const successMessage =
                            eventData.message ||
                            `Prepared ${payload.chunks.length} chunk(s) for advanced processing.`;
                        showStatus(successMessage, 'success', statusId);
                    }
                }
            }

            if (!splitCompleted) {
                const message = 'Chunk split stream ended unexpectedly.';
                showStatus(message, 'error', statusId);
                throw new Error(message);
            }
        }

        function applyChunkSession(chunk) {
            if (!chunk) {
                return;
            }
            currentTranslateSessionId = chunk.session_id;
            currentChunkSessionId = chunk.session_id;
            translateSelectedChunkId = chunk.session_id;
            if (!translateChunkBatchId && chunk.batch_id) {
                translateChunkBatchId = chunk.batch_id;
            }
            if (translateAudioInput && translateAudioInput.value) {
                translateAudioInput.value = '';
            }
            updateAudioInputRequirement();
            translateBackingAvailableFromSession = Boolean(chunk.backing_available);
            updateCustomBackingSummary();
            syncTranslateMergeBackState();
            const metadata = {
                session_id: chunk.session_id,
                reuse_session_id: chunk.session_id,
                duration_control: chunk.duration_control,
                backing_track: {
                    available: chunk.backing_available,
                    source: chunk.backing_source || 'none',
                },
                separation: {
                    vocals_available: true,
                    vocals_url: chunk.vocals_url,
                    backing_available: chunk.backing_available,
                    backing_url: chunk.backing_url,
                    backing_source: chunk.backing_source || 'none',
                },
            };
            autoApplyTranslateMetadata(metadata, chunk.session_id);
            renderChunkResultsFromResponse();
            updateChunkSelectionUI();
            const reuseCheckbox = document.getElementById('translateReuseSeparation');
            if (reuseCheckbox) {
                reuseCheckbox.checked = true;
            }
            showStatus(
                `Chunk ${chunk.chunk_index} ready. Enable advanced mode and reuse the separation to process this portion.`,
                'success',
                'translateStatus'
            );
        }

        resetChunkResults();
        updateAudioInputRequirement();

        if (translateEnableChunkSplit) {
            translateEnableChunkSplit.addEventListener('change', () => {
                const enabled = translateEnableChunkSplit.checked;
                // ClearVoice and Audio Separation are optional for chunk splitting
                // User may upload vocal-only audio that doesn't need preprocessing
                toggleChunkControls(enabled);
            });
            toggleChunkControls(translateEnableChunkSplit.checked);
        }

        const translateReuseCheckbox = document.getElementById('translateReuseSeparation');
        if (translateReuseCheckbox) {
            translateReuseCheckbox.addEventListener('change', () => {
                updateAudioInputRequirement();
            });
        }

        if (translateSplitAudioBtn) {
            translateSplitAudioBtn.addEventListener('click', handleSplitAudioRequest);
        }

        if (translateChunkList) {
            translateChunkList.addEventListener('click', event => {
                const target = event.target.closest('.chunk-use-btn');
                if (!target) {
                    return;
                }
                const sessionId = target.dataset.sessionId;
                const chunk = translateChunkSessions.find(entry => entry.session_id === sessionId);
                if (!chunk) {
                    showStatus('Chunk metadata missing. Please split the audio again.', 'error', 'translateStatus');
                    return;
                }
                applyChunkSession(chunk);
            });
            translateChunkList.addEventListener('change', event => {
                const checkbox = event.target.closest('.chunk-select-checkbox');
                if (!checkbox) {
                    return;
                }
                const sessionId = checkbox.dataset.sessionId;
                if (!sessionId) {
                    return;
                }
                if (checkbox.checked) {
                    translateChunkSelections.add(sessionId);
                } else {
                    translateChunkSelections.delete(sessionId);
                }
                hideStatus('translateChunkBatchStatus');
                updateChunkBatchControlsVisibility();
            });
        }

        if (translateClearChunkBtn) {
            translateClearChunkBtn.addEventListener('click', () => {
                currentChunkSessionId = null;
                translateSelectedChunkId = null;
                updateChunkSelectionUI();
                updateAudioInputRequirement();
                showStatus('Chunk selection cleared. Upload a file or choose another chunk to continue.', 'success', 'translateStatus');
            });
        }

        if (translateChunkSelectPending) {
            translateChunkSelectPending.addEventListener('change', () => {
                const pendingChunks = translateChunkSessions.filter(chunk => !chunk.generated);
                if (!pendingChunks.length) {
                    translateChunkSelectPending.checked = false;
                    return;
                }
                if (translateChunkSelectPending.checked) {
                    pendingChunks.forEach(chunk => translateChunkSelections.add(chunk.session_id));
                } else {
                    pendingChunks.forEach(chunk => translateChunkSelections.delete(chunk.session_id));
                }
                hideStatus('translateChunkBatchStatus');
                updateChunkBatchControlsVisibility();
            });
        }

        if (translateMergeChunksBtn) {
            translateMergeChunksBtn.addEventListener('click', handleMergeChunks);
        }

        if (translateGenerateChunksBtn) {
            translateGenerateChunksBtn.addEventListener('click', handleGenerateSelectedChunks);
        }

        // Download all chunks as ZIP
        if (translateDownloadChunksBtn) {
            translateDownloadChunksBtn.addEventListener('click', async () => {
                if (!translateChunkBatchId) {
                    showStatus('No chunk batch available to download.', 'error', 'translateChunkBatchStatus');
                    return;
                }
                const statusId = 'translateChunkBatchStatus';
                try {
                    translateDownloadChunksBtn.disabled = true;
                    showStatus('Preparing chunks for download...', 'info', statusId);
                    const response = await fetch(`/api/translate_download_chunks/${encodeURIComponent(translateChunkBatchId)}`);
                    if (!response.ok) {
                        const error = await response.json().catch(() => ({ message: 'Download failed' }));
                        throw new Error(error.message || 'Download failed');
                    }
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `chunks_${translateChunkBatchId}.zip`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    showStatus(`Downloaded ${translateChunkSessions.length} chunk(s) as ZIP.`, 'success', statusId);
                } catch (error) {
                    console.error('Download chunks error:', error);
                    showStatus(`Failed to download chunks: ${error.message}`, 'error', statusId);
                } finally {
                    translateDownloadChunksBtn.disabled = false;
                }
            });
        }

        // Upload transcriptions ZIP
        if (translateUploadTranscriptionsBtn && translateTranscriptionsZipInput) {
            translateUploadTranscriptionsBtn.addEventListener('click', () => {
                translateTranscriptionsZipInput.click();
            });

            translateTranscriptionsZipInput.addEventListener('change', async (event) => {
                const file = event.target.files[0];
                if (!file) return;

                const statusId = 'translateTranscriptionUploadStatus';

                if (!translateChunkBatchId) {
                    showStatus('No chunk batch available. Please split audio first.', 'error', statusId);
                    translateTranscriptionsZipInput.value = '';
                    return;
                }

                // Get current Gemini settings
                const destLanguage = translateDestLanguageSelect ? translateDestLanguageSelect.value : '';
                if (!destLanguage) {
                    showStatus('Please select a destination language first.', 'error', statusId);
                    translateTranscriptionsZipInput.value = '';
                    return;
                }

                const geminiModel = translateGeminiModel ? translateGeminiModel.value : '';
                const prompt = translateCustomPrompt ? translateCustomPrompt.value.trim() : '';

                try {
                    translateUploadTranscriptionsBtn.disabled = true;
                    showStatus('Uploading and processing transcriptions...', 'info', statusId);

                    const formData = new FormData();
                    formData.append('transcriptions_zip', file);
                    formData.append('dest_language', destLanguage);
                    if (geminiModel) formData.append('gemini_model', geminiModel);
                    if (prompt) formData.append('prompt', prompt);
                    // Include translate_enabled and ignore_non_speech to match cache key generation
                    const translateEnabled = translateWhileTranscribing ? translateWhileTranscribing.checked : true;
                    formData.append('translate_enabled', translateEnabled ? 'true' : 'false');
                    const ignoreNonSpeech = translateIgnoreNonSpeechEl ? translateIgnoreNonSpeechEl.checked : false;
                    formData.append('ignore_non_speech', ignoreNonSpeech ? 'true' : 'false');

                    const response = await fetch(`/api/translate_upload_transcriptions/${encodeURIComponent(translateChunkBatchId)}`, {
                        method: 'POST',
                        body: formData,
                    });

                    const result = await response.json();

                    if (!response.ok || result.status === 'error') {
                        throw new Error(result.message || 'Upload failed');
                    }

                    let successMsg = `✅ ${result.message || 'Cache entries created.'}`;
                    if (result.errors && result.errors.length > 0) {
                        successMsg += ` ⚠️ ${result.errors.length} warning(s)`;
                        console.warn('Transcription upload warnings:', result.errors);
                    }
                    console.log('Cache import result:', result);
                    showStatus(successMsg, 'success', statusId);

                } catch (error) {
                    console.error('Upload transcriptions error:', error);
                    showStatus(`Failed to upload transcriptions: ${error.message}`, 'error', statusId);
                } finally {
                    translateUploadTranscriptionsBtn.disabled = false;
                    translateTranscriptionsZipInput.value = '';
                }
            });
        }

        if (translateAudioInput) {
            translateAudioInput.addEventListener('change', () => {
                if (translateChunkSessions.length) {
                    resetChunkResults();
                }
            });
        }
