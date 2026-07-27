"use strict";

let stableAudioModels = [];
        let stableAudioModelsLoaded = false;
        let stableAudioDefaultsApplied = false;

        function getStableAudioModel(key) {
            return stableAudioModels.find((model) => model.key === key) || stableAudioModels[0] || null;
        }

        function updateStableAudioModelStatus(model) {
            if (!model) {
                return;
            }
            const status = model.checkpoint && model.checkpoint.ready ? 'ready' : 'missing checkpoint';
            const loaded = model.loaded ? ', loaded' : '';
            const statusEl = document.getElementById('stableAudioModelStatus');
            if (statusEl) {
                statusEl.textContent = `${model.label}: ${status}${loaded}. Checkpoint: ${model.checkpoint ? model.checkpoint.path : ''}`;
            }
        }

        function updateStableAudioModelDefaults(resetValues = true) {
            const select = document.getElementById('stableAudioModel');
            const duration = document.getElementById('stableAudioDuration');
            const prompt = document.getElementById('stableAudioPrompt');
            const maskEnd = document.getElementById('stableAudioMaskEnd');
            if (!select || !duration || !prompt) {
                return;
            }
            const model = getStableAudioModel(select.value);
            if (!model) {
                return;
            }
            const maxSeconds = model.max_seconds || 360;
            const defaultDuration = Math.min(model.default_duration || 60, maxSeconds);
            duration.max = String(maxSeconds);
            if (resetValues) {
                duration.value = String(defaultDuration);
            } else if (Number(duration.value || 0) > maxSeconds) {
                duration.value = String(maxSeconds);
            }
            document.getElementById('stableAudioDurationValue').textContent = duration.value;
            prompt.placeholder = model.placeholder || prompt.placeholder;
            if (resetValues && !prompt.value.trim()) {
                prompt.value = model.placeholder || '';
            }
            if (maskEnd) {
                maskEnd.max = String(maxSeconds);
                if (resetValues) {
                    maskEnd.value = String(defaultDuration);
                } else if (Number(maskEnd.value || 0) > maxSeconds) {
                    maskEnd.value = String(maxSeconds);
                }
            }
            updateStableAudioModelStatus(model);
        }

        async function loadStableAudioModels(force = false, options = {}) {
            if (stableAudioModelsLoaded && !force) {
                return;
            }
            const preserveControls = options.preserveControls !== false;
            try {
                const response = await fetch(ENDPOINTS.STABLE_AUDIO_MODELS, { cache: 'no-cache' });
                if (!response.ok) {
                    throw new Error(await parseHttpError(response, 'Failed to load Stable Audio models'));
                }
                const data = await response.json();
                stableAudioModels = Array.isArray(data.models) ? data.models : [];
                const select = document.getElementById('stableAudioModel');
                if (select && stableAudioModels.length) {
                    const previousValue = preserveControls ? select.value : '';
                    const hadOptions = select.options.length > 0;
                    select.innerHTML = stableAudioModels.map((model) => {
                        const label = `${model.label}${model.checkpoint && model.checkpoint.ready ? '' : ' (missing)'}`;
                        return `<option value="${model.key}">${label}</option>`;
                    }).join('');
                    const hasPrevious = previousValue && stableAudioModels.some((model) => model.key === previousValue);
                    select.value = hasPrevious ? previousValue : (data.default_variant || 'medium');
                    select.onchange = () => updateStableAudioModelDefaults(true);
                    const shouldApplyDefaults = !stableAudioDefaultsApplied || !preserveControls || !hadOptions || !hasPrevious;
                    updateStableAudioModelDefaults(shouldApplyDefaults);
                    stableAudioDefaultsApplied = true;
                }
                stableAudioModelsLoaded = true;
                if (force) {
                    showStatus('Stable Audio status refreshed.', 'success', 'stableAudioStatus');
                }
            } catch (error) {
                showStatus(`Stable Audio status error: ${error.message}`, 'error', 'stableAudioStatus');
            }
        }

        async function generateStableAudio() {
            await loadStableAudioModels();
            const promptEl = document.getElementById('stableAudioPrompt');
            const prompt = promptEl ? promptEl.value.trim() : '';
            if (!prompt) {
                showStatus('Enter a music or sound effect prompt first.', 'error', 'stableAudioStatus');
                return;
            }

            const startTime = performance.now();
            const outputFormat = document.getElementById('stableAudioOutputFormat').value || 'mp3';
            const batchCount = Math.min(4, Math.max(1, Number(document.getElementById('stableAudioBatchCount').value || 1)));
            const formData = new FormData();
            formData.append('variant_key', document.getElementById('stableAudioModel').value || 'medium');
            formData.append('prompt', prompt);
            formData.append('negative_prompt', document.getElementById('stableAudioNegativePrompt').value || '');
            formData.append('batch_count', String(batchCount));
            formData.append('duration', document.getElementById('stableAudioDuration').value || '60');
            formData.append('steps', document.getElementById('stableAudioSteps').value || '8');
            formData.append('cfg_scale', document.getElementById('stableAudioCfg').value || '1.0');
            formData.append('sampler_type', document.getElementById('stableAudioSampler').value || 'pingpong');
            formData.append('seed', document.getElementById('stableAudioSeed').value || '0');
            formData.append('sigma_max', document.getElementById('stableAudioSigmaMax').value || '1.0');
            formData.append('apg_scale', document.getElementById('stableAudioApgScale').value || '1.0');
            formData.append('duration_padding_sec', document.getElementById('stableAudioDurationPadding').value || '6.0');
            formData.append('cut_to_seconds_total', document.getElementById('stableAudioCutToDuration').checked ? 'true' : 'false');
            formData.append('init_noise_level', document.getElementById('stableAudioInitNoise').value || '0.9');
            formData.append('mask_start_sec', document.getElementById('stableAudioMaskStart').value || '0');
            formData.append('mask_end_sec', document.getElementById('stableAudioMaskEnd').value || '0');
            formData.append('response_format', outputFormat);

            const initFile = document.getElementById('stableAudioInitAudio').files[0];
            const inpaintFile = document.getElementById('stableAudioInpaintAudio').files[0];
            if (initFile) {
                formData.append('init_audio_file', initFile);
            }
            if (inpaintFile) {
                formData.append('inpaint_audio_file', inpaintFile);
            }

            const generateButton = document.querySelector('[data-action="stable-audio-generate"]');
            if (generateButton) {
                generateButton.disabled = true;
                generateButton.textContent = batchCount > 1 ? `Generating ${batchCount} audios...` : 'Generating audio...';
            }
            showStatus(batchCount > 1 ? `Generating ${batchCount} Stable Audio 3 outputs in parallel...` : 'Generating Stable Audio 3 output...', 'info', 'stableAudioStatus');
            const resultEl = document.getElementById('stableAudioResult');
            resultEl.innerHTML = '';
            try {
                const response = await fetch(ENDPOINTS.STABLE_AUDIO_GENERATE, {
                    method: 'POST',
                    body: formData,
                });
                if (!response.ok) {
                    throw new Error(await parseHttpError(response, 'Stable Audio generation failed'));
                }
                const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
                const extension = outputFormat === 'mp3' ? 'mp3' : outputFormat;
                const contentType = response.headers.get('content-type') || '';
                if (contentType.includes('application/json')) {
                    const data = await response.json();
                    const items = Array.isArray(data.items) ? data.items : [];
                    if (!items.length) {
                        throw new Error('Stable Audio generation returned no audio.');
                    }
                    resultEl.innerHTML = `
                            <h3>Generated Audio (${items.length} clips, ${elapsed}s)</h3>
                            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:10px;">
                                ${items.map((item, index) => {
                        const metadata = item.metadata || {};
                        const seedLabel = metadata.seed && metadata.seed > 0 ? `Seed ${metadata.seed}` : 'Random seed';
                        const audioUrl = `data:${item.media_type || `audio/${extension}`};base64,${item.audio_base64 || ''}`;
                        const filename = item.filename || `stable-audio-3-${index + 1}.${extension}`;
                        return `
                                        <div class="segment-card" style="padding:14px;">
                                            <div class="segment-header" style="margin-bottom:8px;">Clip ${index + 1}</div>
                                            <small style="color:var(--text-muted);">${escapeHtml(seedLabel)}${metadata.elapsed_seconds ? ` - ${Number(metadata.elapsed_seconds).toFixed(2)}s backend` : ''}</small>
                                            <audio controls preload="none" style="width:100%;margin:10px 0;" src="${audioUrl}"></audio>
                                            <a href="${audioUrl}" download="${escapeHtml(filename)}" class="btn btn-secondary" style="padding:8px 12px;font-size:0.82rem;">Download</a>
                                        </div>
                                    `;
                    }).join('')}
                            </div>
                        `;
                    showStatus(`Generated ${items.length} Stable Audio clips in ${elapsed}s.`, 'success', 'stableAudioStatus');
                } else {
                    const blob = await response.blob();
                    const audioUrl = URL.createObjectURL(blob);
                    resultEl.innerHTML = `
                            <h3>Generated Audio (${elapsed}s)</h3>
                            <audio controls autoplay preload="none" style="width: 100%; margin: 10px 0;" src="${audioUrl}"></audio>
                            <br>
                            <a href="${audioUrl}" download="stable-audio-3.${extension}" class="btn">Download</a>
                        `;
                    showStatus(`Stable Audio generated in ${elapsed}s.`, 'success', 'stableAudioStatus');
                }
                stableAudioModelsLoaded = false;
                loadStableAudioModels(true, { preserveControls: true });
            } catch (error) {
                showStatus(`Stable Audio error: ${error.message}`, 'error', 'stableAudioStatus');
            } finally {
                if (generateButton) {
                    generateButton.disabled = false;
                    generateButton.textContent = 'Generate Audio';
                }
            }
        }
