"use strict";

const CHUNK_SPLIT_MIN_SILENCE_MS = Number(document.querySelector('meta[name="chunk-split-min-silence-ms"]')?.content || 1000);

        /** Mirrors server routes — single inventory for review and fetch calls */
        const ENDPOINTS = Object.freeze({
            AUDIO_ROLES: '/audio_roles',
            ADD_SPEAKER: '/add_speaker',
            DELETE_SPEAKER: '/delete_speaker',
            DELETE_ALL_SPEAKERS: '/delete_all_speakers',
            SPEAKER_EFFECTS: '/speaker_effects',
            SPEAK: '/speak',
            CLONE_VOICE: '/clone_voice',
            TRANSLATE_AUDIO: '/api/translate_audio',
            TRANSLATE_SEGMENTS: '/api/translate_segments',
            TRANSLATE_GENERATE_SEGMENTS: '/api/translate_generate_segments',
            TRANSLATE_MERGE_CHUNKS: '/api/translate_merge_chunks',
            TRANSLATE_GENERATE_CHUNKS: '/api/translate_generate_chunks',
            TRANSLATE_SPLIT_AUDIO: '/api/translate_split_audio',
            TRANSLATE_SEGMENT_PREVIEW: '/api/translate_segment_preview',
            DOWNLOADED_VIDEOS: '/api/downloaded_videos',
            TRANSLATED_VIDEOS: '/api/translated_videos',
            VIDEO_INFO: '/api/video_info',
            VIDEO_DOWNLOAD: '/api/video_download',
            VIDEO_REPLACE_AUDIO: '/api/video_replace_audio',
            COOKIES_LIST: '/api/cookies',
            COOKIES_IMPORT_CURL: '/api/cookies/import_curl',
            COOKIES_UPLOAD: '/api/cookies/upload',
            DESIGN_VOICE: '/api/design-voice',
            DESIGN_VOICE_STATUS: '/api/design-voice/status',
            DESIGN_VOICE_SAVE_PRESET: '/api/design-voice/save-preset',
            ESTIMATE_DURATION: '/api/estimate_duration',
            CLEAR_OUTPUTS: '/api/clear_outputs',
            PROMPT_TEMPLATES: '/api/prompt_templates',
            STABLE_AUDIO_STATUS: '/api/stable-audio/status',
            STABLE_AUDIO_MODELS: '/api/stable-audio/models',
            STABLE_AUDIO_GENERATE: '/api/stable-audio/generate',
            MODELS_STATUS: '/api/models/status',
            MODELS_UNLOAD: '/api/models/unload',
            MODELS_WAKE: '/api/models/wake',
        });

        const DURATION_CONTROL_STORAGE_KEY = 'indexTts.durationControl.v1';
        const DURATION_CONTROL_VALUES = new Set(['original', 'ffmpeg']);

        function normalizeDurationControl(value) {
            const normalized = String(value || '').trim().toLowerCase();
            return DURATION_CONTROL_VALUES.has(normalized) ? normalized : 'original';
        }

        function getDurationControlSelects() {
            return Array.from(document.querySelectorAll('[data-duration-control-select]'));
        }

        function readDurationControlSetting() {
            try {
                if (window.localStorage) {
                    return normalizeDurationControl(window.localStorage.getItem(DURATION_CONTROL_STORAGE_KEY));
                }
            } catch (error) {
                console.warn('Unable to read duration control setting:', error);
            }
            return 'original';
        }

        function writeDurationControlSetting(value) {
            const normalized = normalizeDurationControl(value);
            try {
                if (window.localStorage) {
                    window.localStorage.setItem(DURATION_CONTROL_STORAGE_KEY, normalized);
                }
            } catch (error) {
                console.warn('Unable to save duration control setting:', error);
            }
            return normalized;
        }

        function syncDurationControlSelects(value) {
            const normalized = normalizeDurationControl(value);
            getDurationControlSelects().forEach(select => {
                select.value = normalized;
            });
            return normalized;
        }

        function getDurationControlMode() {
            const firstSelect = getDurationControlSelects()[0];
            return normalizeDurationControl(firstSelect ? firstSelect.value : readDurationControlSetting());
        }

        function initDurationControlControls() {
            syncDurationControlSelects(readDurationControlSetting());
            getDurationControlSelects().forEach(select => {
                select.addEventListener('change', () => {
                    const normalized = writeDurationControlSetting(select.value);
                    syncDurationControlSelects(normalized);
                    if (typeof updateAdditionalSettingsSummary === 'function') {
                        updateAdditionalSettingsSummary();
                    }
                });
            });
            if (typeof updateAdditionalSettingsSummary === 'function') {
                updateAdditionalSettingsSummary();
            }
        }

        /* ---------- Helpers (HTTP, UI binding) ---------- */

        function renderModelManager(data) {
            const gpuEl = document.getElementById('modelManagerGpu');
            const listEl = document.getElementById('modelManagerList');
            if (!gpuEl || !listEl) return;
            const gpu = data.gpu || {};
            if (gpu.available) {
                const pct = gpu.total_mb ? Math.round(gpu.used_mb / gpu.total_mb * 100) : 0;
                gpuEl.innerHTML = `<strong>GPU Memory: ${pct}% used</strong><br><span style="color: var(--text-secondary)">${Number(gpu.used_mb).toLocaleString()} MiB used / ${Number(gpu.total_mb).toLocaleString()} MiB total · ${Number(gpu.free_mb).toLocaleString()} MiB free</span>`;
            } else {
                gpuEl.textContent = 'CUDA GPU memory information is unavailable.';
            }
            const models = Array.isArray(data.models) ? data.models : [];
            const rows = models.map(model => `
                    <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px; border:1px solid var(--border); border-radius:10px; background:var(--surface-muted);">
                        <div><strong>${escapeHtml(model.name)}</strong><br><small style="color:var(--text-secondary)">${escapeHtml(model.kind || 'Optional model')} · ${escapeHtml(model.state || 'loaded')}</small></div>
                        ${model.state === 'sleeping'
                    ? `<button type="button" class="btn btn-secondary" data-action="models-wake" data-model-key="${escapeHtml(model.key)}">Wake</button>`
                    : `<button type="button" class="btn btn-danger" data-action="models-unload" data-model-key="${escapeHtml(model.key)}">${model.key.endsWith('_vllm') ? 'Sleep' : 'Unload'}</button>`}
                    </div>`);
            listEl.innerHTML = rows.join('');
        }

        async function loadModelManager(showStatus = false) {
            const statusEl = document.getElementById('modelManagerStatus');
            try {
                const response = await fetch(ENDPOINTS.MODELS_STATUS);
                if (!response.ok) throw new Error(await parseHttpError(response, 'Failed to load model status'));
                renderModelManager(await response.json());
                if (showStatus && statusEl) statusEl.textContent = 'Model status refreshed.';
            } catch (error) {
                if (statusEl) statusEl.textContent = `Error: ${error.message}`;
            }
        }

        async function unloadManagedModel(modelKey) {
            const statusEl = document.getElementById('modelManagerStatus');
            if (statusEl) statusEl.textContent = 'Unloading model and releasing VRAM…';
            try {
                const response = await fetch(ENDPOINTS.MODELS_UNLOAD, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_key: modelKey }),
                });
                if (!response.ok) throw new Error(await parseHttpError(response, 'Failed to unload model'));
                const data = await response.json();
                renderModelManager(data);
                if (statusEl) statusEl.textContent = data.unloaded.length ? `Unloaded: ${data.unloaded.join(', ')}` : 'No loaded models matched.';
            } catch (error) {
                if (statusEl) statusEl.textContent = `Error: ${error.message}`;
            }
        }

        async function wakeManagedModel(modelKey) {
            const statusEl = document.getElementById('modelManagerStatus');
            if (statusEl) statusEl.textContent = 'Waking model…';
            try {
                const response = await fetch(ENDPOINTS.MODELS_WAKE, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_key: modelKey }),
                });
                if (!response.ok) throw new Error(await parseHttpError(response, 'Failed to wake model'));
                const data = await response.json();
                renderModelManager(data);
                if (statusEl) statusEl.textContent = `Woke: ${data.woken.join(', ')}`;
            } catch (error) {
                if (statusEl) statusEl.textContent = `Error: ${error.message}`;
            }
        }

        async function parseHttpError(response, fallbackMessage) {
            let detail = fallbackMessage || 'Request failed';
            try {
                const clone = response.clone();
                const ct = clone.headers.get('content-type') || '';
                if (ct.includes('application/json')) {
                    const body = await clone.json();
                    detail = body.detail || body.message || body.error || detail;
                } else {
                    const t = await clone.text();
                    if (t) {
                        detail = t.slice(0, 500);
                    }
                }
            } catch (_e) {
                /* ignore */
            }
            return detail;
        }

        function bindRangeOutputs() {
            const pairs = [
                ['firstChunkSize', 'firstChunkSizeValue'],
                ['emotionWeight', 'emotionWeightValue'],
                ['diffusionSteps', 'diffusionStepsValue'],
                ['maxTextTokens', 'maxTextTokensValue'],
                ['stableAudioBatchCount', 'stableAudioBatchCountValue'],
                ['stableAudioDuration', 'stableAudioDurationValue'],
                ['stableAudioSteps', 'stableAudioStepsValue'],
                ['stableAudioCfg', 'stableAudioCfgValue'],
            ];
            pairs.forEach(([inputId, labelId]) => {
                const input = document.getElementById(inputId);
                const label = document.getElementById(labelId);
                if (!input || !label) {
                    return;
                }
                const sync = () => {
                    label.textContent = input.value;
                };
                input.addEventListener('input', sync);
                sync();
            });
        }

        function bindDelegatedActions() {
            const root = document.querySelector('.app-shell');
            if (!root) {
                return;
            }

            // Bind tabs directly so their behavior does not depend on the
            // dimensions or hit-testing of the full-width app container.
            root.querySelectorAll('.hero-card[data-tab]').forEach((tabCard) => {
                tabCard.addEventListener('click', () => {
                    switchTab(tabCard.dataset.tab);
                });
                tabCard.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        switchTab(tabCard.dataset.tab);
                    }
                });
            });

            root.addEventListener('click', (e) => {
                const demoCn = e.target.closest('[data-demo-chinese]');
                if (demoCn && root.contains(demoCn)) {
                    setChineseDemo(demoCn.dataset.demoChinese);
                    return;
                }

                const demoEm = e.target.closest('[data-demo-emotion]');
                if (demoEm && root.contains(demoEm)) {
                    setEmotionDemo(demoEm.dataset.demoEmotion);
                    return;
                }

                const actionEl = e.target.closest('[data-action]');
                if (!actionEl || !root.contains(actionEl)) {
                    return;
                }
                const action = actionEl.dataset.action;
                if (action === 'estimate-duration') {
                    estimateDuration();
                    return;
                }
                if (action === 'clear-outputs') {
                    clearOutputs();
                    return;
                }
                if (action === 'refresh-speakers') {
                    loadSpeakerList();
                    return;
                }
                if (action === 'delete-all-speakers') {
                    deleteAllSpeakers();
                    return;
                }
                if (action === 'voice-design-generate') {
                    generateVoiceDesign();
                    return;
                }
                if (action === 'voice-design-save') {
                    saveVoiceDesignToPreset();
                    return;
                }
                if (action === 'stable-audio-generate') {
                    generateStableAudio();
                    return;
                }
                if (action === 'stable-audio-refresh') {
                    loadStableAudioModels(true);
                    return;
                }
                if (action === 'models-refresh') {
                    loadModelManager(true);
                    return;
                }
                if (action === 'models-unload-all') {
                    unloadManagedModel('all');
                    return;
                }
                if (action === 'models-unload') {
                    unloadManagedModel(actionEl.dataset.modelKey || '');
                    return;
                }
                if (action === 'models-wake') {
                    wakeManagedModel(actionEl.dataset.modelKey || '');
                    return;
                }
                if (action === 'delete-speaker') {
                    const raw = actionEl.dataset.speakerName;
                    if (raw !== undefined && raw !== '') {
                        try {
                            deleteSpeaker(decodeURIComponent(raw));
                        } catch (_err) {
                            deleteSpeaker(raw);
                        }
                    }
                    return;
                }
            });

        }

        /* ---------- Tabs ---------- */

        function switchTab(evtOrName, maybeTabName) {
            const tabName = typeof evtOrName === 'string' ? evtOrName : maybeTabName;
            const trigger = (evtOrName && evtOrName.currentTarget) ? evtOrName.currentTarget : null;

            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });

            document.querySelectorAll('.hero-card[data-tab]').forEach(card => {
                card.classList.remove('active');
            });

            const content = document.getElementById(tabName);
            if (content) {
                content.classList.add('active');
            }

            if (trigger) {
                trigger.classList.add('active');
            } else {
                const heroForTab = document.querySelector(`.hero-card[data-tab="${tabName}"]`);
                if (heroForTab) {
                    heroForTab.classList.add('active');
                }
            }

            if (tabName === 'speakers') {
                loadSpeakerList();
            }
            if (tabName === 'video-download') {
                loadDownloadedVideos();
                loadTranslatedVideos();
                loadCookieSites();
            }
            if (tabName === 'stable-audio') {
                loadStableAudioModels();
            }
            if (tabName === 'models') {
                loadModelManager();
            }

            if (content && window.matchMedia('(max-width: 1024px)').matches) {
                content.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }

        function setChineseDemo(type) {
            const textArea = document.getElementById('text');
            const demos = {
                '现代文本': '你好！欢迎使用IndexTTS中文语音合成系统。这是一个功能强大的AI语音生成工具，能够准确处理中文语音合成任务。系统支持多种语音风格，让您的文本转换为自然流畅的语音。',
                '古诗词': '床前明月光，疑是地上霜。举头望明月，低头思故乡。这首《静夜思》是李白的名作，表达了诗人对故乡的深深思念之情。',
                '数字日期': '今天是2025年1月11日，时间是下午3点30分。这款产品的价格是12,999元，性价比很高。我的电话号码是138-8888-8888，欢迎联系。',
                '中英混合': '我正在使用IndexTTS和vLLM技术进行AI语音合成。This system supports both Chinese and English perfectly. 这个系统的RTF约为0.1，比原版快3倍！GPU memory utilization设置为85%。'
            };

            textArea.value = demos[type];
            textArea.focus();

            // Show a brief tooltip
            showStatus(`已设置${type}演示文本`, 'success');
            setTimeout(() => hideStatus(), 2000);
        }

        function setEmotionDemo(emotionType) {
            const textArea = document.getElementById('text');
            const emotionText = document.getElementById('emotionText');
            const emotionWeight = document.getElementById('emotionWeight');

            const emotionDemos = {
                '开心': {
                    text: '今天真是太开心了！我收到了好消息，心情特别愉快。阳光明媚，鸟儿在歌唱，一切都是那么美好！',
                    emotion: 'happy and joyful',
                    weight: 0.8
                },
                '悲伤': {
                    text: '雨滴轻敲着窗台，就像我内心的忧伤。离别的时刻总是让人难过，回忆如潮水般涌来。',
                    emotion: 'sad and melancholic',
                    weight: 0.7
                },
                '愤怒': {
                    text: '这实在太过分了！我再也无法忍受这种不公正的待遇。愤怒在我心中燃烧，必须要说出来！',
                    emotion: 'angry and frustrated',
                    weight: 0.6
                },
                '平静': {
                    text: '静坐在湖边，微风轻拂过脸颊。内心如湖水般平静，思绪缓缓流淌，享受这宁静的时光。',
                    emotion: 'calm and peaceful',
                    weight: 0.3
                }
            };

            const demo = emotionDemos[emotionType];
            if (demo) {
                textArea.value = demo.text;
                emotionText.value = demo.emotion;
                emotionWeight.value = demo.weight;
                document.getElementById('emotionWeightValue').textContent = demo.weight;
                textArea.focus();

                // Show a brief tooltip
                showStatus(`已设置${emotionType}情感演示 (${demo.emotion})`, 'success');
                setTimeout(() => hideStatus(), 3000);
            }
        }

        function showStatus(message, type, elementId = 'status') {
            const status = document.getElementById(elementId);
            status.textContent = message;
            status.className = `status ${type}`;
            status.style.display = 'block';
        }

        function hideStatus(elementId = 'status') {
            document.getElementById(elementId).style.display = 'none';
        }
