"use strict";

/* ---------- Translate ---------- */
        const translateBtn = document.getElementById('translateBtn');
        const translateAdvancedToggle = document.getElementById('translateAdvancedMode');
        const translateAdvancedSettings = document.getElementById('translateAdvancedSettings');
        const translateAdvancedPanel = document.getElementById('translateAdvancedPanel');
        const translateSegmentsList = document.getElementById('translateSegmentsList');
        const translateSegmentsStatus = document.getElementById('translateSegmentsStatus');
        const translateSegmentsSelectAll = document.getElementById('translateSegmentsSelectAll');
        const translateHideSilence = document.getElementById('translateHideSilence');
        const translateGenerateBtn = document.getElementById('translateGenerateBtn');
        const translateWhileTranscribing = document.getElementById('translateWhileTranscribing');
        const translateCustomPrompt = document.getElementById('translateCustomPrompt');
        const translateGeminiModel = document.getElementById('translateGeminiModel');
        const translateGeminiApiKey = document.getElementById('translateGeminiApiKey');
        const translateForceGeminiRefresh = document.getElementById('translateForceGeminiRefresh');
        const translateDestLanguageSelect = document.getElementById('translateDestLanguage');
        const translateEnhanceEl = document.getElementById('translateEnhancement');
        const translateEnhancementModelEl = document.getElementById('translateEnhancementModel');
        const translateSuperEl = document.getElementById('translateSuperResolution');
        const translateAudioSeparatorEl = document.getElementById('translateAudioSeparator');
        const translateAudioSeparatorModelEl = document.getElementById('translateAudioSeparatorModel');
        const translateAudioSeparatorUseSoundfileEl = document.getElementById('translateAudioSeparatorUseSoundfile');
        const translateTtsBackendEl = document.getElementById('translateTtsBackend');
        const translateMergeBackEl = document.getElementById('translateMergeBack');
        const translateMergeBackLabelEl = document.getElementById('translateMergeBackLabel');
        const translateCustomBackingInput = document.getElementById('translateCustomBackingFile');
        const translateCustomBackingClearBtn = document.getElementById('translateCustomBackingClear');
        const translateCustomBackingSummary = document.getElementById('translateCustomBackingSummary');
        const translateStepToggles = document.querySelectorAll('.translate-step.collapsible .step-toggle');
        const translateIgnoreNonSpeechEl = document.getElementById('translateIgnoreNonSpeech');
        const translatePreserveSilenceEl = document.getElementById('translatePreserveSilence');
        const translateSilenceVolumeInput = document.getElementById('translateSilenceVolumePercent');
        const translateSilenceVolumeGroup = document.getElementById('translateSilenceVolumeGroup');
        const translateSeparationPreview = document.getElementById('translateSeparationPreview');
        const translateMinSpeechInput = document.getElementById('translateMinSpeech');
        const translateMaxMergeInput = document.getElementById('translateMaxMerge');
        const translateManualSegmentsToggle = document.getElementById('translateManualSegmentsToggle');
        const translateManualSegmentsPanel = document.getElementById('translateManualSegmentsPanel');
        const translateManualSegmentsInput = document.getElementById('translateManualSegments');
        const translatePromptTranslation = document.getElementById('translatePromptTranslation');
        const translatePromptTranscription = document.getElementById('translatePromptTranscription');
        const translatePromptTemplates = document.getElementById('translatePromptTemplates');
        // SRT subtitle upload elements
        const translateSrtSubtitleToggle = document.getElementById('translateSrtSubtitleToggle');
        const translateSrtSubtitlePanel = document.getElementById('translateSrtSubtitlePanel');
        const translateOriginalSrtFile = document.getElementById('translateOriginalSrtFile');
        const translateTranslatedSrtFile = document.getElementById('translateTranslatedSrtFile');
        const translateOriginalSrtClear = document.getElementById('translateOriginalSrtClear');
        const translateTranslatedSrtClear = document.getElementById('translateTranslatedSrtClear');
        const translateSrtSummary = document.getElementById('translateSrtSummary');
        const translateTranscriptionPipeline = document.getElementById('translateTranscriptionPipeline');
        const translateWhisperXSettingsDiv = document.getElementById('translateWhisperXSettings');
        const translateWhisperXProxyRefiner = document.getElementById('translateWhisperXProxyRefiner');
        const translateQwenOmniVadSettingsDiv = document.getElementById('translateQwenOmniVadSettings');
        const translateQwenOmniVadEnableDiarization = document.getElementById('translateQwenOmniVadEnableDiarization');
        const translateQwenOmniVadDiarizationBackend = document.getElementById('translateQwenOmniVadDiarizationBackend');
        const translateQwenOmniVadEnableForcedAligner = document.getElementById('translateQwenOmniVadEnableForcedAligner');
        const translateQwenOmniVadDiarizationMinSeconds = document.getElementById('translateQwenOmniVadDiarizationMinSeconds');
        const translateQwenOmniVadMergeGapSeconds = document.getElementById('translateQwenOmniVadMergeGapSeconds');
        const translateLitAiSettingsDiv = document.getElementById('translateLitAiSettings');
        const translateTranslationLlmModel = document.getElementById('translateTranslationLlmModel');
        const translateGeminiSettingsDiv = document.getElementById('translateGeminiSettings');
        const translatePipelineInfo = document.getElementById('translatePipelineInfo');
        const translateAiConfigSummary = document.getElementById('translateAiConfigSummary');
        const translateAdditionalSettingsSummary = document.getElementById('translateAdditionalSettingsSummary');
        const translateDurationControlEl = document.getElementById('translateDurationControl');
        const DEFAULT_EMOTION_WEIGHT = 0.6;
        const DEFAULT_TRANSLATE_TRANSCRIPTION_PIPELINE = 'moss_transcribe';
        const TRANSLATE_SETTINGS_STORAGE_KEY = 'indexTts.translateSettings.v1';
        const MULTILINGUAL_TTS_DESTINATION_LANGUAGES = Object.freeze([
            ['', 'Select...'],
            ['Chinese', 'Chinese'],
            ['English', 'English'],
            ['Japanese', 'Japanese'],
            ['Korean', 'Korean'],
            ['German', 'German'],
            ['French', 'French'],
            ['Spanish', 'Spanish'],
            ['Indonesian', 'Indonesian'],
            ['Italian', 'Italian'],
            ['Thai', 'Thai'],
            ['Portuguese', 'Portuguese'],
            ['Russian', 'Russian'],
            ['Malay', 'Malay'],
            ['Vietnamese', 'Vietnamese'],
        ]);
        const TRANSLATE_DESTINATION_LANGUAGES_BY_BACKEND = Object.freeze({
            index: Object.freeze([
                ['', 'Select...'],
                ['English', 'English'],
                ['Chinese', 'Chinese'],
            ]),
            confucius: MULTILINGUAL_TTS_DESTINATION_LANGUAGES,
            higgs: MULTILINGUAL_TTS_DESTINATION_LANGUAGES,
        });
        const DEFAULT_TRANSLATE_DESTINATION_BY_BACKEND = Object.freeze({
            index: 'Chinese',
            confucius: 'Chinese',
            higgs: 'Chinese',
        });

        function readTranslatePersistentSettings() {
            try {
                const raw = window.localStorage
                    ? window.localStorage.getItem(TRANSLATE_SETTINGS_STORAGE_KEY)
                    : null;
                if (!raw) {
                    return {};
                }
                const parsed = JSON.parse(raw);
                return parsed && typeof parsed === 'object' ? parsed : {};
            } catch (error) {
                console.warn('Unable to read saved translation settings:', error);
                return {};
            }
        }

        function writeTranslatePersistentSettings(settings) {
            try {
                if (window.localStorage) {
                    window.localStorage.setItem(
                        TRANSLATE_SETTINGS_STORAGE_KEY,
                        JSON.stringify(settings)
                    );
                }
            } catch (error) {
                console.warn('Unable to save translation settings:', error);
            }
        }

        function selectHasOption(selectEl, value) {
            return Array.from(selectEl.options || []).some(option => option.value === value);
        }

        function getTranslateTtsBackendKey() {
            const backend = translateTtsBackendEl && translateTtsBackendEl.value
                ? translateTtsBackendEl.value
                : 'index';
            return TRANSLATE_DESTINATION_LANGUAGES_BY_BACKEND[backend] ? backend : 'index';
        }

        function syncTranslateDestinationLanguageOptions(preferredValue = '') {
            if (!translateDestLanguageSelect) {
                return;
            }
            const backendKey = getTranslateTtsBackendKey();
            const options =
                TRANSLATE_DESTINATION_LANGUAGES_BY_BACKEND[backendKey] ||
                TRANSLATE_DESTINATION_LANGUAGES_BY_BACKEND.index;
            const selectedValue = (preferredValue || translateDestLanguageSelect.value || '').trim();
            const fallbackValue = DEFAULT_TRANSLATE_DESTINATION_BY_BACKEND[backendKey] || '';

            translateDestLanguageSelect.textContent = '';
            options.forEach(([value, label]) => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = label;
                translateDestLanguageSelect.appendChild(option);
            });

            if (selectedValue && selectHasOption(translateDestLanguageSelect, selectedValue)) {
                translateDestLanguageSelect.value = selectedValue;
            } else if (fallbackValue && selectHasOption(translateDestLanguageSelect, fallbackValue)) {
                translateDestLanguageSelect.value = fallbackValue;
            }
        }

        function applyTranslatePersistentSettings() {
            const settings = readTranslatePersistentSettings();
            if (
                translateTranscriptionPipeline &&
                typeof settings.transcription_pipeline === 'string' &&
                selectHasOption(translateTranscriptionPipeline, settings.transcription_pipeline)
            ) {
                translateTranscriptionPipeline.value = settings.transcription_pipeline;
            }
            if (
                translateTtsBackendEl &&
                typeof settings.tts_backend === 'string' &&
                selectHasOption(translateTtsBackendEl, settings.tts_backend)
            ) {
                translateTtsBackendEl.value = settings.tts_backend;
            }
            syncTranslateDestinationLanguageOptions(
                typeof settings.dest_language === 'string' ? settings.dest_language : ''
            );
            if (
                translateGeminiModel &&
                typeof settings.gemini_model === 'string' &&
                selectHasOption(translateGeminiModel, settings.gemini_model)
            ) {
                translateGeminiModel.value = settings.gemini_model;
            }
            if (
                translateTranslationLlmModel &&
                typeof settings.translation_llm_model === 'string' &&
                selectHasOption(translateTranslationLlmModel, settings.translation_llm_model)
            ) {
                translateTranslationLlmModel.value = settings.translation_llm_model;
            }
            if (translateGeminiApiKey && typeof settings.gemini_api_key === 'string') {
                translateGeminiApiKey.value = settings.gemini_api_key;
            }
            if (
                translateWhisperXProxyRefiner &&
                typeof settings.whisperx_proxy_refiner === 'boolean'
            ) {
                translateWhisperXProxyRefiner.checked = settings.whisperx_proxy_refiner;
            }
            if (
                translateQwenOmniVadEnableDiarization &&
                typeof settings.qwen_omnivad_enable_diarization === 'boolean'
            ) {
                translateQwenOmniVadEnableDiarization.checked = settings.qwen_omnivad_enable_diarization;
            }
            if (
                translateQwenOmniVadDiarizationBackend &&
                typeof settings.qwen_omnivad_diarization_backend === 'string' &&
                selectHasOption(translateQwenOmniVadDiarizationBackend, settings.qwen_omnivad_diarization_backend)
            ) {
                translateQwenOmniVadDiarizationBackend.value = settings.qwen_omnivad_diarization_backend;
            }
            if (
                translateQwenOmniVadEnableForcedAligner &&
                typeof settings.qwen_omnivad_enable_forced_aligner === 'boolean'
            ) {
                translateQwenOmniVadEnableForcedAligner.checked = settings.qwen_omnivad_enable_forced_aligner;
            }
            if (
                translateQwenOmniVadDiarizationMinSeconds &&
                typeof settings.qwen_omnivad_diarization_min_seconds === 'number'
            ) {
                translateQwenOmniVadDiarizationMinSeconds.value = settings.qwen_omnivad_diarization_min_seconds;
            }
            if (
                translateQwenOmniVadMergeGapSeconds &&
                typeof settings.qwen_omnivad_merge_gap_seconds === 'number'
            ) {
                translateQwenOmniVadMergeGapSeconds.value = settings.qwen_omnivad_merge_gap_seconds;
            }
            if (
                translateIgnoreNonSpeechEl &&
                typeof settings.ignore_non_speech === 'boolean'
            ) {
                translateIgnoreNonSpeechEl.checked = settings.ignore_non_speech;
            }
        }

        function saveTranslatePersistentSettings() {
            writeTranslatePersistentSettings({
                transcription_pipeline: translateTranscriptionPipeline
                    ? translateTranscriptionPipeline.value
                    : DEFAULT_TRANSLATE_TRANSCRIPTION_PIPELINE,
                whisperx_proxy_refiner: translateWhisperXProxyRefiner
                    ? translateWhisperXProxyRefiner.checked
                    : false,
                qwen_omnivad_enable_diarization: translateQwenOmniVadEnableDiarization
                    ? translateQwenOmniVadEnableDiarization.checked
                    : true,
                qwen_omnivad_diarization_backend: translateQwenOmniVadDiarizationBackend
                    ? translateQwenOmniVadDiarizationBackend.value
                    : 'auto',
                qwen_omnivad_enable_forced_aligner: translateQwenOmniVadEnableForcedAligner
                    ? translateQwenOmniVadEnableForcedAligner.checked
                    : true,
                qwen_omnivad_diarization_min_seconds: getQwenOmniVadDiarizationMinSeconds(),
                qwen_omnivad_merge_gap_seconds: getQwenOmniVadMergeGapSeconds(),
                gemini_model: translateGeminiModel ? translateGeminiModel.value : '',
                translation_llm_model: translateTranslationLlmModel
                    ? translateTranslationLlmModel.value
                    : '',
                gemini_api_key: translateGeminiApiKey ? translateGeminiApiKey.value : '',
                dest_language: translateDestLanguageSelect ? translateDestLanguageSelect.value : '',
                tts_backend: translateTtsBackendEl ? translateTtsBackendEl.value : 'index',
                ignore_non_speech: translateIgnoreNonSpeechEl
                    ? translateIgnoreNonSpeechEl.checked
                    : false,
            });
        }

        function getQwenOmniVadDiarizationMinSeconds() {
            const rawValue = translateQwenOmniVadDiarizationMinSeconds
                ? translateQwenOmniVadDiarizationMinSeconds.value
                : '';
            const parsed = parseFloat(rawValue);
            return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0.0;
        }

        function getQwenOmniVadMergeGapSeconds() {
            const rawValue = translateQwenOmniVadMergeGapSeconds
                ? translateQwenOmniVadMergeGapSeconds.value
                : '';
            const parsed = parseFloat(rawValue);
            return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0.001;
        }

        applyTranslatePersistentSettings();

        // Pipeline selector toggle logic
        function updatePipelineVisibility() {
            const pipelineValue = translateTranscriptionPipeline
                ? translateTranscriptionPipeline.value
                : DEFAULT_TRANSLATE_TRANSCRIPTION_PIPELINE;
            const isWhisperX = pipelineValue === 'whisperx';
            const isParakeet = pipelineValue === 'parakeet';
            const isMossTranscribe = pipelineValue === 'moss_transcribe';
            const isLocalTranslationPipeline = isWhisperX || pipelineValue === 'qwen_omnivad' || isParakeet || isMossTranscribe;
            const isGemini = pipelineValue === 'gemini';
            if (translateGeminiSettingsDiv) {
                translateGeminiSettingsDiv.style.opacity = isGemini ? '1' : '0.4';
                translateGeminiSettingsDiv.style.pointerEvents = isGemini ? 'auto' : 'none';
            }
            if (translateWhisperXSettingsDiv) {
                translateWhisperXSettingsDiv.style.display = isWhisperX ? 'flex' : 'none';
            }
            if (translateQwenOmniVadSettingsDiv) {
                translateQwenOmniVadSettingsDiv.style.display = pipelineValue === 'qwen_omnivad' ? 'block' : 'none';
            }
            if (translateLitAiSettingsDiv) {
                translateLitAiSettingsDiv.style.display = isLocalTranslationPipeline ? 'grid' : 'none';
            }
            if (translatePipelineInfo) {
                let pipelineInfoText = '';
                if (isWhisperX) {
                    pipelineInfoText = 'WhisperX local pipeline - Gemini settings do not apply';
                } else if (pipelineValue === 'qwen_omnivad') {
                    pipelineInfoText = 'Qwen3-ASR + OmniVAD pipeline - Gemini settings do not apply';
                } else if (isParakeet) {
                    pipelineInfoText = 'NVIDIA Parakeet local pipeline - Gemini settings do not apply';
                } else if (isMossTranscribe) {
                    pipelineInfoText = 'MOSS Transcribe+Diarize SGLang pipeline - Gemini settings do not apply';
                }
                translatePipelineInfo.textContent = pipelineInfoText;
            }
        }
        if (translateTranscriptionPipeline) {
            translateTranscriptionPipeline.addEventListener('change', () => {
                saveTranslatePersistentSettings();
                updatePipelineVisibility();
                updateAiConfigSummary();
            });
            updatePipelineVisibility();
        }
        if (translateWhisperXProxyRefiner) {
            translateWhisperXProxyRefiner.addEventListener('change', () => {
                saveTranslatePersistentSettings();
                updateAiConfigSummary();
            });
        }
        if (translateQwenOmniVadEnableDiarization) {
            translateQwenOmniVadEnableDiarization.addEventListener('change', () => {
                saveTranslatePersistentSettings();
                updateAiConfigSummary();
            });
        }
        if (translateQwenOmniVadDiarizationBackend) {
            translateQwenOmniVadDiarizationBackend.addEventListener('change', () => {
                saveTranslatePersistentSettings();
                updateAiConfigSummary();
            });
        }
        if (translateQwenOmniVadEnableForcedAligner) {
            translateQwenOmniVadEnableForcedAligner.addEventListener('change', () => {
                saveTranslatePersistentSettings();
                updateAiConfigSummary();
            });
        }
        if (translateQwenOmniVadDiarizationMinSeconds) {
            translateQwenOmniVadDiarizationMinSeconds.addEventListener('input', () => {
                saveTranslatePersistentSettings();
                updateAiConfigSummary();
            });
        }
        if (translateQwenOmniVadMergeGapSeconds) {
            translateQwenOmniVadMergeGapSeconds.addEventListener('input', () => {
                saveTranslatePersistentSettings();
                updateAiConfigSummary();
            });
        }
        if (translateTranslationLlmModel) {
            translateTranslationLlmModel.addEventListener('change', () => {
                saveTranslatePersistentSettings();
                updateAiConfigSummary();
            });
        }
        if (translateGeminiModel) {
            translateGeminiModel.addEventListener('change', () => {
                saveTranslatePersistentSettings();
                updateAiConfigSummary();
            });
        }
        if (translateGeminiApiKey) {
            translateGeminiApiKey.addEventListener('input', () => {
                saveTranslatePersistentSettings();
                updateAiConfigSummary();
            });
            translateGeminiApiKey.addEventListener('change', () => {
                saveTranslatePersistentSettings();
                updateAiConfigSummary();
            });
        }
        const DEFAULT_VOLUME_PERCENT = 100;
        const MIN_VOLUME_PERCENT = 10;
        const MAX_VOLUME_PERCENT = 300;
        const translateVolumeInput = document.getElementById('translateVolumePercent');
        const translateBackingVolumeInput = document.getElementById('translateBackingVolumePercent');
        const translateSpeakerAssignments = document.getElementById('translateSpeakerAssignments');
        const translateDefaultSpeakerSelect = document.getElementById('translateDefaultSpeaker');
        const translateDefaultEmotionWeightInput = document.getElementById('translateDefaultEmotionWeight');
        const translateDefaultEmotionWeightValue = document.getElementById('translateDefaultEmotionWeightValue');
        const translateEnableChunkSplit = document.getElementById('translateEnableChunkSplit');
        const translateChunkSettings = document.getElementById('translateChunkSettings');
        const translateSplitAudioBtn = document.getElementById('translateSplitAudioBtn');
        const translateChunkMinInput = document.getElementById('translateChunkMinMinutes');
        const translateChunkMaxInput = document.getElementById('translateChunkMaxMinutes');
        const translateChunkMinSilenceInput = document.getElementById('translateChunkMinSilenceMs');
        const translateChunkMinHint = document.getElementById('translateChunkMinHint');
        const translateChunkMaxHint = document.getElementById('translateChunkMaxHint');
        const translateParallelToggle = document.getElementById('translateClearVoiceParallel');
        const translateParallelSettings = document.getElementById('translateParallelSettings');
        const translateParallelChunkInput = document.getElementById('translateParallelChunkSeconds');
        const translateParallelWorkersInput = document.getElementById('translateParallelMaxWorkers');
        const translateChunkResults = document.getElementById('translateChunkResults');
        const translateChunkSummary = document.getElementById('translateChunkSummary');
        const translateChunkList = document.getElementById('translateChunkList');
        const translateChunkSelectionBanner = document.getElementById('translateChunkSelection');
        const translateChunkSelectionText = document.getElementById('translateChunkSelectionText');
        const translateClearChunkBtn = document.getElementById('translateClearChunkBtn');
        const translateMergeChunksBtn = document.getElementById('translateMergeChunksBtn');
        const translateChunkBatchControls = document.getElementById('translateChunkBatchControls');
        const translateChunkSelectPending = document.getElementById('translateChunkSelectPending');
        const translateGenerateChunksBtn = document.getElementById('translateGenerateChunksBtn');
        const translateDownloadChunksBtn = document.getElementById('translateDownloadChunksBtn');
        const translateUploadTranscriptionsBtn = document.getElementById('translateUploadTranscriptionsBtn');
        const translateTranscriptionsZipInput = document.getElementById('translateTranscriptionsZipInput');
        const translateTranscriptionUploadStatus = document.getElementById('translateTranscriptionUploadStatus');
        const translateMergeStatus = document.getElementById('translateMergeStatus');
        const translateChunkBatchStatus = document.getElementById('translateChunkBatchStatus');
        const translateAudioInput = document.getElementById('translateAudioFile');
        const translateDownloadedVideoSelect = document.getElementById('translateDownloadedVideo');
        const translateDownloadedVideoHint = document.getElementById('translateDownloadedVideoHint');
        const translateRefreshVideosBtn = document.getElementById('translateRefreshVideosBtn');
        const translateBaseFilenameInput = document.getElementById('translateBaseFilename');
        const ffmpegPanel = document.getElementById('translateFfmpegPanel');
        const ffmpegExtractCmd = document.getElementById('ffmpegExtractCmd');
        const ffmpegReplaceCmd = document.getElementById('ffmpegReplaceCmd');
        const ffmpegDualAudioCmd = document.getElementById('ffmpegDualAudioCmd');
        const ffmpegSubtitleCmd = document.getElementById('ffmpegSubtitleCmd');
        const ffmpegSubtitleOriginalCmd = document.getElementById('ffmpegSubtitleOriginalCmd');
        const ffmpegEmbedSubtitleCmd = document.getElementById('ffmpegEmbedSubtitleCmd');
        let currentTranslateSessionId = null;
        let currentTranslateSegments = [];
        let downloadedVideos = [];
        let translatedVideos = [];
        let selectedDownloadedVideoId = '';
        let activeSyncedAudio = null;
        let translateSpeakerProfiles = [];
        let translateSpeakerProfileMap = {};
        let translateSpeakerOverrides = {};
        let speakerOverridesDirty = false;
        let availableSpeakerPresets = [];
        let speakerPresetMeta = {};
        let translateBackingAvailableFromSession = false;
        let promptTemplates = {
            translation: '',
            transcription: '',
            ignoreNonSpeech: '',
        };
        function refreshDefaultSpeakerOptions() {
            if (!translateDefaultSpeakerSelect) {
                return;
            }
            const previousValue = translateDefaultSpeakerSelect.value || '';
            translateDefaultSpeakerSelect.innerHTML = '<option value="">Auto (clone original voice)</option>';
            availableSpeakerPresets.forEach(name => {
                if (!name) {
                    return;
                }
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                translateDefaultSpeakerSelect.appendChild(option);
            });
            if (previousValue && availableSpeakerPresets.includes(previousValue)) {
                translateDefaultSpeakerSelect.value = previousValue;
            }
            updateAdditionalSettingsSummary();
        }
        function syncDefaultEmotionWeightDisplay() {
            if (!translateDefaultEmotionWeightInput || !translateDefaultEmotionWeightValue) {
                return;
            }
            const parsed = parseFloat(translateDefaultEmotionWeightInput.value);
            const safeValue = Number.isNaN(parsed) ? DEFAULT_EMOTION_WEIGHT : parsed;
            translateDefaultEmotionWeightValue.textContent = safeValue.toFixed(2);
            updateAdditionalSettingsSummary();
        }
        if (translateDefaultEmotionWeightInput) {
            translateDefaultEmotionWeightInput.addEventListener('input', syncDefaultEmotionWeightDisplay);
            translateDefaultEmotionWeightInput.addEventListener('change', syncDefaultEmotionWeightDisplay);
            syncDefaultEmotionWeightDisplay();
        }
        function syncTranslateTtsBackendControls() {
            const usesIndexEmotionControls = !translateTtsBackendEl || translateTtsBackendEl.value === 'index';
            syncTranslateDestinationLanguageOptions();
            if (translateDefaultEmotionWeightInput) {
                translateDefaultEmotionWeightInput.disabled = !usesIndexEmotionControls;
            }
            updateAiConfigSummary();
            updateAdditionalSettingsSummary();
        }
        if (translateTtsBackendEl) {
            translateTtsBackendEl.addEventListener('change', () => {
                syncTranslateTtsBackendControls();
                saveTranslatePersistentSettings();
                refreshPromptTemplates();
                updateFfmpegCommands();
            });
            syncTranslateTtsBackendControls();
        }
        const NON_SPEECH_PLACEHOLDER = '{non_speech_instruction}';
        let autoManualSegmentsApplied = false;
        let translateChunkSessions = [];
        let translateChunkBatchId = null;
        let translateSelectedChunkId = null;
        let translateChunkSelections = new Set();
        let currentChunkSessionId = null;
        let translateLanguageCodeHint = null;

        function selectedOptionLabel(selectEl, fallback = '') {
            if (!selectEl) {
                return fallback;
            }
            const selected = selectEl.selectedOptions && selectEl.selectedOptions[0];
            return (selected ? selected.textContent : selectEl.value || fallback).trim();
        }

        function numericInputValue(inputEl, fallback = '') {
            if (!inputEl) {
                return fallback;
            }
            const value = (inputEl.value || '').trim();
            return value || fallback;
        }

        function getTranscriptionPipelineLabel(pipelineValue) {
            switch (pipelineValue) {
                case 'whisperx':
                    return 'WhisperX';
                case 'qwen_omnivad':
                    return 'Qwen3-ASR + OmniVAD';
                case 'parakeet':
                    return 'NVIDIA Parakeet';
                case 'moss_transcribe':
                    return 'MOSS Transcribe+Diarize';
                case 'gemini':
                    return 'Gemini';
                default:
                    return pipelineValue || 'MOSS Transcribe+Diarize';
            }
        }

        function parseJsonStreamEventLine(line) {
            const trimmed = (line || '').trim();
            if (!trimmed || trimmed.startsWith(':')) {
                return null;
            }
            const jsonText = trimmed.startsWith('data:')
                ? trimmed.slice(5).trimStart()
                : trimmed;
            if (!jsonText || jsonText === '[DONE]') {
                return null;
            }
            return JSON.parse(jsonText);
        }

        function buildTranslationCompleteMessage(metadata = {}) {
            let statusMessage = '✅ Translation complete!';
            if (typeof metadata.segment_count === 'number') {
                statusMessage += ` (${metadata.segment_count} segments)`;
            }

            const pipelineValue =
                metadata.transcription_pipeline ||
                (translateTranscriptionPipeline
                    ? translateTranscriptionPipeline.value
                    : DEFAULT_TRANSLATE_TRANSCRIPTION_PIPELINE);
            if (pipelineValue && pipelineValue !== 'gemini') {
                const pipelineLabel =
                    metadata.transcription_pipeline_label ||
                    getTranscriptionPipelineLabel(pipelineValue);
                statusMessage += ` • Pipeline: ${pipelineLabel}`;
                if (metadata.translation_llm_model) {
                    statusMessage += ` • Translation model: ${metadata.translation_llm_model}`;
                }
            } else if (metadata.gemini_model) {
                statusMessage += ` • Gemini model: ${metadata.gemini_model}`;
            }

            return statusMessage;
        }

        function renderStepSummary(container, items) {
            if (!container) {
                return;
            }
            container.textContent = '';
            items
                .filter(item => item && item.value !== undefined && item.value !== null && String(item.value).trim() !== '')
                .forEach(item => {
                    const chip = document.createElement('span');
                    chip.className = 'summary-chip';
                    const label = document.createElement('strong');
                    label.textContent = `${item.label}:`;
                    const value = document.createElement('span');
                    value.className = 'summary-value';
                    value.textContent = String(item.value);
                    chip.appendChild(label);
                    chip.appendChild(value);
                    container.appendChild(chip);
                });
        }

        function updateAiConfigSummary() {
            const pipelineValue = translateTranscriptionPipeline
                ? translateTranscriptionPipeline.value
                : DEFAULT_TRANSLATE_TRANSCRIPTION_PIPELINE;
            const pipelineLabel =
                pipelineValue === 'whisperx'
                    ? 'WhisperX'
                    : pipelineValue === 'qwen_omnivad'
                        ? 'Qwen3'
                        : pipelineValue === 'parakeet'
                            ? 'Parakeet'
                            : pipelineValue === 'moss_transcribe'
                                ? 'MOSS'
                                : 'Gemini';
            const items = [
                { label: 'Pipeline', value: pipelineLabel },
                {
                    label: 'Lang',
                    value:
                        (translateDestLanguageSelect && translateDestLanguageSelect.value) ||
                        'Select...',
                },
                {
                    label: 'Input',
                    value:
                        translateSrtSubtitleToggle && translateSrtSubtitleToggle.checked
                            ? 'SRT'
                            : translateManualSegmentsToggle && translateManualSegmentsToggle.checked
                                ? 'Manual JSON'
                                : 'Auto',
                },
            ];
            if (pipelineValue === 'whisperx') {
                items.push({
                    label: 'Model',
                    value: selectedOptionLabel(translateTranslationLlmModel, 'Hy-MT2'),
                });
                items.push({
                    label: 'Refiner',
                    value:
                        translateWhisperXProxyRefiner && translateWhisperXProxyRefiner.checked
                            ? 'On'
                            : 'Off',
                });
            } else if (pipelineValue === 'qwen_omnivad') {
                items.push({
                    label: 'Model',
                    value: selectedOptionLabel(translateTranslationLlmModel, 'Hy-MT2'),
                });
                items.push({
                    label: 'Diarization',
                    value:
                        translateQwenOmniVadEnableDiarization &&
                            translateQwenOmniVadEnableDiarization.checked
                            ? selectedOptionLabel(translateQwenOmniVadDiarizationBackend, 'Auto')
                            : 'Off',
                });
                items.push({
                    label: 'Timeline',
                    value:
                        translateQwenOmniVadEnableForcedAligner &&
                            translateQwenOmniVadEnableForcedAligner.checked
                            ? 'Aligner'
                            : 'OmniVAD',
                });
                items.push({
                    label: 'Merge Gap',
                    value: `${getQwenOmniVadMergeGapSeconds()}s`,
                });
            } else if (pipelineValue === 'parakeet') {
                items.push({
                    label: 'Model',
                    value: 'Parakeet 0.6B v3',
                });
                items.push({
                    label: 'Translate',
                    value: selectedOptionLabel(translateTranslationLlmModel, 'Hy-MT2'),
                });
            } else if (pipelineValue === 'moss_transcribe') {
                items.push({
                    label: 'Model',
                    value: 'MOSS 0.9B',
                });
                items.push({
                    label: 'Backend',
                    value: 'SGLang',
                });
                items.push({
                    label: 'Translate',
                    value: selectedOptionLabel(translateTranslationLlmModel, 'Hy-MT2'),
                });
            } else {
                items.push({
                    label: 'Model',
                    value: selectedOptionLabel(translateGeminiModel, 'Default'),
                });
                items.push({
                    label: 'API key',
                    value:
                        translateGeminiApiKey && translateGeminiApiKey.value.trim()
                            ? 'Set'
                            : 'Default',
                });
            }
            items.push({
                label: 'Non-speech',
                value:
                    translateIgnoreNonSpeechEl && translateIgnoreNonSpeechEl.checked
                        ? 'Ignored'
                        : 'Kept',
            });
            items.push({
                label: 'Cache',
                value:
                    translateForceGeminiRefresh && translateForceGeminiRefresh.checked
                        ? 'Refresh'
                        : 'Use',
            });
            renderStepSummary(translateAiConfigSummary, items);
        }

        function updateAdditionalSettingsSummary() {
            const customBackingSelected =
                translateCustomBackingInput &&
                translateCustomBackingInput.files &&
                translateCustomBackingInput.files.length > 0;
            const minSpeech = numericInputValue(translateMinSpeechInput, 'Default');
            const maxMerge = numericInputValue(translateMaxMergeInput, 'Default');
            const preserveSilence =
                translatePreserveSilenceEl && translatePreserveSilenceEl.checked;
            const outputFormatSelect = document.getElementById('translateOutputFormat');
            renderStepSummary(translateAdditionalSettingsSummary, [
                { label: 'Backend', value: selectedOptionLabel(translateTtsBackendEl, 'IndexTTS') },
                { label: 'Duration', value: selectedOptionLabel(translateDurationControlEl, 'Original') },
                { label: 'Format', value: selectedOptionLabel(outputFormatSelect, 'MP3') },
                {
                    label: 'Merge',
                    value:
                        translateMergeBackEl && translateMergeBackEl.checked && !translateMergeBackEl.disabled
                            ? 'On'
                            : 'Off',
                },
                {
                    label: 'Speaker',
                    value:
                        translateDefaultSpeakerSelect && translateDefaultSpeakerSelect.value
                            ? translateDefaultSpeakerSelect.value
                            : 'Auto',
                },
                {
                    label: 'Emotion',
                    value: numericInputValue(translateDefaultEmotionWeightInput, DEFAULT_EMOTION_WEIGHT.toFixed(2)),
                },
                {
                    label: 'Vol',
                    value: `${numericInputValue(translateVolumeInput, '100')}% / ${numericInputValue(translateBackingVolumeInput, '100')}%`,
                },
                {
                    label: 'Rules',
                    value: `Min ${minSpeech} ms, gap ${maxMerge} ms`,
                },
                {
                    label: 'Backing',
                    value: customBackingSelected
                        ? 'Custom'
                        : translateBackingAvailableFromSession
                            ? 'Session'
                            : 'Default',
                },
                {
                    label: 'Silence',
                    value: preserveSilence
                        ? `Preserve ${numericInputValue(translateSilenceVolumeInput, '100')}%`
                        : 'Skip',
                },
                {
                    label: 'Soundfile',
                    value:
                        translateAudioSeparatorUseSoundfileEl &&
                            !translateAudioSeparatorUseSoundfileEl.disabled &&
                            translateAudioSeparatorUseSoundfileEl.checked
                            ? 'On'
                            : 'Off',
                },
            ]);
        }

        function updateTranslateStepSummaries() {
            updateAiConfigSummary();
            updateAdditionalSettingsSummary();
        }

        function bindSummaryInputUpdates(elements) {
            elements.forEach(element => {
                if (!element) {
                    return;
                }
                element.addEventListener('input', updateAdditionalSettingsSummary);
                element.addEventListener('change', updateAdditionalSettingsSummary);
            });
        }

        function bindAiSummaryUpdates(elements) {
            elements.forEach(element => {
                if (!element) {
                    return;
                }
                element.addEventListener('input', updateAiConfigSummary);
                element.addEventListener('change', updateAiConfigSummary);
            });
        }

        bindSummaryInputUpdates([
            translateMinSpeechInput,
            translateMaxMergeInput,
            translateDefaultSpeakerSelect,
            translateDefaultEmotionWeightInput,
            translateVolumeInput,
            translateBackingVolumeInput,
            translateDurationControlEl,
            translateAudioSeparatorUseSoundfileEl,
            translateMergeBackEl,
            translatePreserveSilenceEl,
            translateSilenceVolumeInput,
        ]);
        bindAiSummaryUpdates([
            translateForceGeminiRefresh,
            translateManualSegmentsToggle,
            translateSrtSubtitleToggle,
        ]);
        updateTranslateStepSummaries();

        if (translateBaseFilenameInput) {
            translateBaseFilenameInput.addEventListener('input', () => {
                translateBaseFilenameInput.dataset.userEdited = 'true';
                updateFfmpegCommands();
            });
        }
        if (translateAudioInput) {
            translateAudioInput.addEventListener('change', () => {
                if (translateAudioInput.files && translateAudioInput.files.length) {
                    selectedDownloadedVideoId = '';
                    if (translateDownloadedVideoSelect) {
                        translateDownloadedVideoSelect.value = '';
                    }
                    const reuseCheckbox = document.getElementById('translateReuseSeparation');
                    if (reuseCheckbox) {
                        reuseCheckbox.checked = false;
                    }
                    updateDownloadedVideoHint();
                }
                if (
                    translateBaseFilenameInput &&
                    translateAudioInput.files &&
                    translateAudioInput.files[0]
                ) {
                    const autoBase = deriveBaseFromFilename(translateAudioInput.files[0].name);
                    if (autoBase) {
                        translateBaseFilenameInput.value = autoBase;
                        translateBaseFilenameInput.dataset.userEdited = 'false';
                    }
                }
                updateAudioInputRequirement();
                updateFfmpegCommands();
            });
        }
        if (translateDownloadedVideoSelect) {
            translateDownloadedVideoSelect.addEventListener('change', () => {
                selectDownloadedVideo(translateDownloadedVideoSelect.value || '');
            });
        }
        if (translateRefreshVideosBtn) {
            translateRefreshVideosBtn.addEventListener('click', () => loadDownloadedVideos());
        }
        const videoDownloadForm = document.getElementById('videoDownloadForm');
        if (videoDownloadForm) {
            videoDownloadForm.addEventListener('submit', handleVideoDownloadSubmit);
        }
        const videoInfoBtn = document.getElementById('videoInfoBtn');
        if (videoInfoBtn) {
            videoInfoBtn.addEventListener('click', fetchVideoInfo);
        }
        const refreshDownloadedVideosBtn = document.getElementById('refreshDownloadedVideosBtn');
        if (refreshDownloadedVideosBtn) {
            refreshDownloadedVideosBtn.addEventListener('click', () => {
                loadDownloadedVideos();
                loadTranslatedVideos();
            });
        }
        const translatedVideosRefreshBtn = document.getElementById('translatedVideosRefreshBtn');
        if (translatedVideosRefreshBtn) {
            translatedVideosRefreshBtn.addEventListener('click', loadTranslatedVideos);
        }
        const cookieImportBtn = document.getElementById('cookieImportBtn');
        if (cookieImportBtn) {
            cookieImportBtn.addEventListener('click', importCookiesFromCurl);
        }
        const cookieDetectDomainBtn = document.getElementById('cookieDetectDomainBtn');
        if (cookieDetectDomainBtn) {
            cookieDetectDomainBtn.addEventListener('click', detectCookieDomain);
        }
        const cookieUploadBtn = document.getElementById('cookieUploadBtn');
        if (cookieUploadBtn) {
            cookieUploadBtn.addEventListener('click', uploadCookiesFile);
        }
        const cookieFileInput = document.getElementById('cookieFileInput');
        if (cookieFileInput) {
            cookieFileInput.addEventListener('change', () => {
                const file = cookieFileInput.files[0];
                if (file) {
                    const name = file.name.toLowerCase();
                    const domainMatch = name.match(/([a-z0-9-]+\.[a-z]{2,})(?:_cookies)?\.txt$/);
                    if (domainMatch) {
                        const domainInput = document.getElementById('cookieUploadDomainInput');
                        if (domainInput && !domainInput.value.trim()) {
                            domainInput.value = domainMatch[1];
                        }
                    } else {
                        const parts = name.split(/[._-]/);
                        for (let i = 0; i < parts.length - 1; i++) {
                            if (parts[i] && parts[i + 1] && parts[i + 1].length >= 2 && ['com', 'net', 'org', 'edu', 'tv', 'co', 'io'].includes(parts[i + 1])) {
                                const domainInput = document.getElementById('cookieUploadDomainInput');
                                if (domainInput && !domainInput.value.trim()) {
                                    domainInput.value = `${parts[i]}.${parts[i + 1]}`;
                                    break;
                                }
                            }
                        }
                    }
                }
            });
        }
        function updateParallelSettingsVisibility() {
            if (!translateParallelToggle || !translateParallelSettings) {
                return;
            }
            const enhancementEnabled = translateEnhanceEl && translateEnhanceEl.checked;
            translateParallelToggle.disabled = !enhancementEnabled;
            if (!enhancementEnabled) {
                translateParallelToggle.checked = false;
            }
            translateParallelSettings.style.display =
                enhancementEnabled && translateParallelToggle.checked ? 'flex' : 'none';
        }
        if (translateParallelToggle) {
            translateParallelToggle.addEventListener('change', updateParallelSettingsVisibility);
        }
        if (translateEnhanceEl) {
            translateEnhanceEl.addEventListener('change', updateParallelSettingsVisibility);
        }
        updateParallelSettingsVisibility();

        function appendClearVoiceParallelSettings(formData) {
            if (!formData || !translateParallelToggle) {
                return;
            }
            const enhancementEnabled = translateEnhanceEl && translateEnhanceEl.checked;
            const enabled = enhancementEnabled && translateParallelToggle.checked && !translateParallelToggle.disabled;
            formData.append('clearvoice_parallel_enabled', enabled ? 'true' : 'false');
            if (enabled && translateParallelChunkInput && translateParallelChunkInput.value) {
                formData.append('clearvoice_parallel_chunk_seconds', translateParallelChunkInput.value);
            }
            if (enabled && translateParallelWorkersInput && translateParallelWorkersInput.value) {
                formData.append('clearvoice_parallel_max_workers', translateParallelWorkersInput.value);
            }
        }

        translateStepToggles.forEach((toggle) => {
            const step = toggle.closest('.translate-step');
            if (!step) {
                return;
            }
            const syncToggleState = (isOpen) => {
                toggle.textContent = isOpen ? 'Collapse' : 'Expand';
                toggle.setAttribute('aria-expanded', String(isOpen));
            };
            syncToggleState(!step.classList.contains('collapsed'));
            toggle.addEventListener('click', () => {
                step.classList.toggle('collapsed');
                syncToggleState(!step.classList.contains('collapsed'));
            });
        });

        function updateAudioInputRequirement() {
            if (!translateAudioInput) {
                return;
            }
            const reuseCheckbox = document.getElementById('translateReuseSeparation');
            const reuseCandidateActive =
                reuseCheckbox && reuseCheckbox.checked && (currentChunkSessionId || currentTranslateSessionId);
            const downloadedVideoActive = Boolean(getSelectedDownloadedVideoId());
            const needsFile = !downloadedVideoActive && !currentChunkSessionId && !reuseCandidateActive;
            translateAudioInput.required = needsFile;
        }

        function resetChunkResults() {
            translateChunkSessions = [];
            translateChunkBatchId = null;
            translateSelectedChunkId = null;
            currentChunkSessionId = null;
            translateChunkSelections = new Set();
            if (translateChunkSummary) {
                translateChunkSummary.textContent = 'Chunks will appear here after splitting.';
            }
            if (translateChunkList) {
                translateChunkList.innerHTML = '';
            }
            if (translateChunkResults) {
                translateChunkResults.style.display = 'none';
            }
            if (translateMergeChunksBtn) {
                translateMergeChunksBtn.style.display = 'none';
            }
            if (translateMergeStatus) {
                hideStatus('translateMergeStatus');
            }
            if (translateChunkSelectPending) {
                translateChunkSelectPending.checked = false;
                translateChunkSelectPending.disabled = true;
            }
            hideStatus('translateChunkBatchStatus');
            hideStatus('translateSplitStatus');
            updateChunkSelectionUI();
            updateChunkBatchControlsVisibility();
            updateAudioInputRequirement();
        }

        function toggleChunkControls(enabled) {
            if (translateChunkSettings) {
                translateChunkSettings.style.display = enabled ? 'block' : 'none';
            }
            if (!enabled && translateChunkResults) {
                translateChunkResults.style.display = 'none';
            } else if (enabled && translateChunkResults && translateChunkSessions.length) {
                translateChunkResults.style.display = 'block';
            }
            updateChunkBatchControlsVisibility();
        }

        function updateChunkLengthHints() {
            if (!translateChunkMinHint && !translateChunkMaxHint) {
                return;
            }
            const seconds = CHUNK_SPLIT_MIN_SILENCE_MS / 1000;
            const secondsLabel = Number.isInteger(seconds) ? seconds.toString() : seconds.toFixed(1);
            const message = `Hard minimum silence gap: ${secondsLabel}s (${CHUNK_SPLIT_MIN_SILENCE_MS} ms)`;
            if (translateChunkMinHint) {
                translateChunkMinHint.textContent = message;
            }
            if (translateChunkMaxHint) {
                translateChunkMaxHint.textContent = message;
            }
            if (translateChunkMinSilenceInput && !translateChunkMinSilenceInput.value) {
                translateChunkMinSilenceInput.value = CHUNK_SPLIT_MIN_SILENCE_MS;
            }
        }
        updateChunkLengthHints();

        function renderChunkResultsFromResponse(payload) {
            if (payload && Array.isArray(payload.chunks)) {
                translateChunkSessions = payload.chunks.slice();
                translateChunkBatchId = payload.chunk_batch_id || translateChunkBatchId;
            }
            const validChunkIds = new Set(translateChunkSessions.map(chunk => chunk.session_id));
            translateChunkSelections = new Set(
                Array.from(translateChunkSelections).filter(id => validChunkIds.has(id))
            );
            if (!translateChunkResults || !translateChunkSummary || !translateChunkList) {
                return;
            }
            if (!translateChunkSessions.length) {
                translateChunkSummary.textContent = 'Chunk split did not produce usable segments.';
                translateChunkResults.style.display = 'none';
                translateChunkList.innerHTML = '';
                if (translateMergeChunksBtn) {
                    translateMergeChunksBtn.style.display = 'none';
                }
                if (translateMergeStatus) {
                    hideStatus('translateMergeStatus');
                }
                updateChunkSelectionUI();
                return;
            }
            translateChunkResults.style.display =
                translateEnableChunkSplit && !translateEnableChunkSplit.checked ? 'none' : 'block';
            const totalLabel =
                (payload && typeof payload.duration_label === 'string' && payload.duration_label) || '';
            const summaryParts = [`Prepared ${translateChunkSessions.length} chunk(s)`];
            if (totalLabel) {
                summaryParts.push(`Total ${totalLabel}`);
            }
            translateChunkSummary.textContent = summaryParts.join(' • ');
            if (translateMergeChunksBtn) {
                translateMergeChunksBtn.style.display = translateChunkSessions.length > 1 ? 'inline-flex' : 'none';
            }
            if (translateMergeStatus && translateChunkSessions.length <= 1) {
                hideStatus('translateMergeStatus');
            }
            const cacheBuster = Date.now();
            translateChunkList.innerHTML = translateChunkSessions
                .map(chunk => {
                    const isSelected = translateSelectedChunkId === chunk.session_id;
                    const isBatchSelected = translateChunkSelections.has(chunk.session_id);
                    const cardClasses = ['segment-card', 'chunk-card'];
                    if (isSelected) {
                        cardClasses.push('selected');
                    }
                    const borderStyle = isSelected ? 'border: 2px solid #6370ff; box-shadow: 0 0 0 2px rgba(99,112,255,0.2);' : '';
                    const vocalsUrl = chunk.vocals_url
                        ? `${chunk.vocals_url}?session=${chunk.session_id}&t=${cacheBuster}`
                        : '';
                    const backingUrl =
                        chunk.backing_available && chunk.backing_url
                            ? `${chunk.backing_url}?session=${chunk.session_id}&t=${cacheBuster}`
                            : '';
                    const translatedUrl = chunk.audio_url
                        ? `${chunk.audio_url}?session=${chunk.session_id}&t=${cacheBuster}`
                        : '';
                    // Build compact audio row with Vocal and Translated side by side
                    let audioRowHtml = '';
                    if (vocalsUrl || translatedUrl) {
                        audioRowHtml = `<div style="display:flex;gap:12px;margin-top:8px;">`;
                        if (vocalsUrl) {
                            audioRowHtml += `<div class="audio-cell" style="flex:1;"><span class="audio-label">Vocal:</span><audio controls preload="none" style="flex:1;height:32px;min-width:0;"><source src="${vocalsUrl}" type="audio/mpeg"></audio></div>`;
                        }
                        if (translatedUrl) {
                            audioRowHtml += `<div class="audio-cell" style="flex:1;"><span class="audio-label">Trans:</span><audio controls preload="none" style="flex:1;height:32px;min-width:0;"><source src="${translatedUrl}" type="audio/mpeg"></audio></div>`;
                        }
                        audioRowHtml += `</div>`;
                    }
                    const statusBadge = chunk.generated
                        ? '<span style="background: rgba(16, 185, 129, 0.18); color: var(--brand-emerald); padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">Generated</span>'
                        : '<span style="background: rgba(251, 191, 36, 0.25); color: #9b6f00; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">Pending</span>';
                    return `
                            <div class="${cardClasses.join(' ')}" data-session-id="${chunk.session_id}" style="${borderStyle}padding:10px;">
                                <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
                                    <div style="display:flex;align-items:center;gap:8px;">
                                        <input type="checkbox" class="chunk-select-checkbox" data-session-id="${chunk.session_id}" ${isBatchSelected ? 'checked' : ''}>
                                        <span style="font-weight:600;">Chunk ${chunk.chunk_index ?? ''}</span>
                                        <span style="font-size:0.8rem;color:var(--text-muted);">${chunk.start_label || ''} → ${chunk.end_label || ''} (${chunk.duration_label || ''})</span>
                                    </div>
                                    <div style="display:flex;align-items:center;gap:8px;">
                                        ${statusBadge}
                                        <button type="button" class="btn btn-secondary chunk-use-btn" data-session-id="${chunk.session_id}" style="padding:4px 10px;font-size:0.8rem;">Use</button>
                                    </div>
                                </div>
                                ${audioRowHtml}
                            </div>
                        `;
                })
                .join('');
            updateChunkSelectionUI();
            updateChunkBatchControlsVisibility();
        }
