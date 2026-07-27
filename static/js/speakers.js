"use strict";

async function loadSpeakers() {
            try {
                const response = await fetch(ENDPOINTS.AUDIO_ROLES);
                const data = await response.json();
                const select = document.getElementById('speaker');
                const previousSpeaker = select ? select.value : '';

                // Clear existing options except first
                select.innerHTML = '<option value="">Select a speaker...</option>';

                if (data.success) {
                    const speakerMeta = data.speakers || {};
                    speakerPresetMeta = speakerMeta;
                    const roles = data.roles && data.roles.length ? data.roles : Object.keys(speakerMeta || {});
                    availableSpeakerPresets = roles || [];
                    refreshDefaultSpeakerOptions();
                    (roles || []).forEach(speaker => {
                        const option = document.createElement('option');
                        option.value = speaker;
                        option.textContent = speaker;
                        select.appendChild(option);
                    });
                    if (previousSpeaker && roles.includes(previousSpeaker)) {
                        select.value = previousSpeaker;
                    }
                    renderSpeakerAssignments();
                }
            } catch (error) {
                console.error('Failed to load speakers:', error);
            }
        }

        function renderSpeakerEffects(effects) {
            const container = document.getElementById('speakerEffectsList');
            if (!container) {
                return;
            }
            if (!Array.isArray(effects) || !effects.length) {
                container.innerHTML = '<div class="speaker-effects-empty">No speaker effects available.</div>';
                return;
            }
            container.innerHTML = effects.map(effect => {
                const id = String(effect.id || '').trim();
                if (!id) {
                    return '';
                }
                const name = effect.name || id.replace(/_/g, ' ');
                const description = effect.description || effect.use_case || '';
                return `
                        <label class="speaker-effect-option" title="${escapeHtml(description)}">
                            <input type="checkbox" name="speakerEffects" value="${escapeHtml(id)}">
                            <span class="speaker-effect-name">${escapeHtml(name)}</span>
                        </label>
                    `;
            }).join('');
        }

        async function loadSpeakerEffects() {
            const container = document.getElementById('speakerEffectsList');
            if (!container) {
                return;
            }
            try {
                const response = await fetch(ENDPOINTS.SPEAKER_EFFECTS);
                const data = await response.json();
                if (data.available && Array.isArray(data.effects)) {
                    renderSpeakerEffects(data.effects);
                } else {
                    const message = data.error ? `Speaker effects unavailable: ${data.error}` : 'Speaker effects unavailable.';
                    container.innerHTML = `<div class="speaker-effects-empty">${escapeHtml(message)}</div>`;
                }
            } catch (error) {
                console.error('Failed to load speaker effects:', error);
                container.innerHTML = '<div class="speaker-effects-empty">Speaker effects unavailable.</div>';
            }
        }

        function getSelectedSpeakerEffects() {
            return Array.from(document.querySelectorAll('input[name="speakerEffects"]:checked'))
                .map(input => input.value)
                .filter(Boolean);
        }

        function formatSpeakerEffectsStatus(effectIds) {
            if (!Array.isArray(effectIds) || !effectIds.length) {
                return '';
            }
            return ` Effects: ${effectIds.map(id => id.replace(/_/g, ' ')).join(' + ')}`;
        }

        async function loadSpeakerList() {
            try {
                const response = await fetch(ENDPOINTS.AUDIO_ROLES);
                const data = await response.json();
                const listDiv = document.getElementById('speakerList');

                if (data.success) {
                    const speakerMeta = data.speakers || {};
                    const speakers = data.roles && data.roles.length ? data.roles : Object.keys(speakerMeta);

                    if (!speakers.length) {
                        listDiv.innerHTML = '<p>No speakers found.</p>';
                        return;
                    }

                    let html = `<h4>📊 ${speakers.length} Speakers Available</h4>`;

                    for (const name of speakers) {
                        const info = speakerMeta[name] || {};
                        const description = info.description && info.description.trim() !== '' ? info.description : 'Speaker preset';
                        const previewUrl = info.preview_url;
                        const previewSection = previewUrl
                            ? `<audio controls preload="none" src="${previewUrl.replace(/"/g, '&quot;')}"></audio>`
                            : `<small style="color: #888;">No preview available</small>`;

                        html += `
                                <div class="speaker-item">
                                    <div class="speaker-info">
                                        <h4>🎭 ${escapeHtml(name)}</h4>
                                        <small>${escapeHtml(description)}</small>
                                        <div class="speaker-preview">
                                            <label>Preview</label>
                                            ${previewSection}
                                        </div>
                                    </div>
                                    <button type="button" class="btn btn-danger" data-action="delete-speaker" data-speaker-name="${encodeURIComponent(name)}">🗑️ Delete</button>
                                </div>
                            `;
                    }

                    listDiv.innerHTML = html;
                } else {
                    listDiv.innerHTML = '<p>No speakers found.</p>';
                }
            } catch (error) {
                console.error('Failed to load speaker list:', error);
                document.getElementById('speakerList').innerHTML = '<p>Error loading speakers.</p>';
            }
        }

        async function deleteSpeaker(speakerName) {
            if (!confirm(`Are you sure you want to delete speaker "${speakerName}"?`)) {
                return;
            }

            try {
                const formData = new FormData();
                formData.append('name', speakerName);

                const response = await fetch(ENDPOINTS.DELETE_SPEAKER, {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                showStatus(result.success ? 'Speaker deleted successfully' : result.error, result.success ? 'success' : 'error', 'speakerStatus');

                if (result.success) {
                    loadSpeakerList();
                    loadSpeakers(); // Refresh dropdown
                }
            } catch (error) {
                showStatus(`Error deleting speaker: ${error.message}`, 'error', 'speakerStatus');
            }
        }

        async function deleteAllSpeakers() {
            const currentCount = availableSpeakerPresets.length || Object.keys(speakerPresetMeta || {}).length;
            const countLabel = currentCount ? ` ${currentCount}` : '';
            if (!confirm(`Remove all${countLabel} speakers from the speaker library? This cannot be undone.`)) {
                return;
            }

            try {
                const response = await fetch(ENDPOINTS.DELETE_ALL_SPEAKERS, {
                    method: 'POST',
                });
                const result = await response.json();
                if (!response.ok || result.success === false) {
                    throw new Error(result.error || result.message || 'Failed to remove speakers');
                }
                const deletedCount = typeof result.deleted_count === 'number' ? result.deleted_count : 0;
                const message = result.message || `Removed ${deletedCount} speakers.`;
                const statusType = result.partial ? 'error' : 'success';
                showStatus(message, statusType, 'speakerStatus');
                await loadSpeakerList();
                await loadSpeakers();
            } catch (error) {
                showStatus(`Error removing speakers: ${error.message}`, 'error', 'speakerStatus');
            }
        }

        // ============================================================================
        // Voice Design (Qwen3-TTS) Functions
        // ============================================================================

        const VOICE_DESIGN_PRESETS = {
            "young_female": "A bright, energetic young female voice with a clear tone, speaking with enthusiasm and warmth.",
            "young_male": "A confident young male voice with clear articulation, professional and authoritative tone.",
            "elderly_wise": "A seasoned, wise voice with a low, mellow timbre, speaking slowly and thoughtfully.",
            "narrator": "A warm, engaging audiobook narrator voice with perfect pacing, clear pronunciation, and emotional depth.",
            "news_anchor": "A clear, authoritative news anchor voice with neutral accent, steady pace, and professional delivery.",
            "excited": "An excited, upbeat voice with high energy, fast pace, and enthusiastic delivery.",
            "calm": "A calm, soothing voice like a meditation guide, speaking slowly with gentle intonation.",
            "child": "A playful, high-pitched child's voice with innocent enthusiasm and natural variation."
        };

        function applyVoiceDesignPreset() {
            const select = document.getElementById('voiceDesignPreset');
            const descriptionTextarea = document.getElementById('voiceDesignDescription');
            const presetValue = select.value;

            if (presetValue && VOICE_DESIGN_PRESETS[presetValue]) {
                descriptionTextarea.value = VOICE_DESIGN_PRESETS[presetValue];
            }
        }

        function showVoiceDesignStatus(message, type = 'info') {
            const statusDiv = document.getElementById('voiceDesignStatus');
            statusDiv.textContent = message;
            statusDiv.style.color = type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : 'var(--text-secondary)';
        }

        async function generateVoiceDesign() {
            const text = document.getElementById('voiceDesignText').value.trim();
            const voiceDescription = document.getElementById('voiceDesignDescription').value.trim();
            const language = document.getElementById('voiceDesignLanguage').value;
            const generateBtn = document.getElementById('voiceDesignGenerateBtn');

            // Validation
            if (!text) {
                showVoiceDesignStatus('Please enter text to synthesize.', 'error');
                return;
            }
            if (!voiceDescription) {
                showVoiceDesignStatus('Please enter a voice description.', 'error');
                return;
            }

            // Show loading state
            generateBtn.disabled = true;
            generateBtn.innerHTML = '⏳ Generating...';
            showVoiceDesignStatus('Generating voice with Qwen3-TTS... This may take a moment.');

            try {
                const response = await fetch(ENDPOINTS.DESIGN_VOICE, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: text,
                        voice_description: voiceDescription,
                        language: language,
                        output_format: 'mp3'
                    })
                });

                if (!response.ok) {
                    const detail = await parseHttpError(response, `Generation failed (${response.status})`);
                    throw new Error(detail);
                }

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);

                // Show audio player
                const audioElement = document.getElementById('voiceDesignAudio');
                audioElement.src = url;
                document.getElementById('voiceDesignResultSection').style.display = 'block';

                // Clear preset name input
                document.getElementById('voiceDesignPresetName').value = '';

                // Check if preset saving is available
                try {
                    const statusResp = await fetch(ENDPOINTS.DESIGN_VOICE_STATUS);
                    const statusData = await statusResp.json();
                    const saveStatusDiv = document.getElementById('voiceDesignSaveStatus');
                    const saveBtn = document.getElementById('voiceDesignSaveBtn');

                    if (statusData.preset_save_available) {
                        saveStatusDiv.textContent = '';
                        saveBtn.disabled = false;
                        showVoiceDesignStatus('✅ Voice generated successfully! You can save it to Speaker Presets.', 'success');
                    } else {
                        saveStatusDiv.innerHTML = '<span style="color: #f59e0b;">⚠️ IndexTTS is still initializing. Preset saving will be available shortly.</span>';
                        saveBtn.disabled = true;
                        showVoiceDesignStatus('✅ Voice generated successfully! Preset saving will be available once IndexTTS is ready.', 'success');
                    }
                } catch (e) {
                    document.getElementById('voiceDesignSaveStatus').textContent = '';
                    showVoiceDesignStatus('✅ Voice generated successfully! You can save it to Speaker Presets.', 'success');
                }

            } catch (error) {
                console.error('Voice Design error:', error);
                showVoiceDesignStatus(`❌ Error: ${error.message}`, 'error');
            } finally {
                generateBtn.disabled = false;
                generateBtn.innerHTML = '🎵 Generate Voice';
            }
        }

        async function saveVoiceDesignToPreset() {
            const presetName = document.getElementById('voiceDesignPresetName').value.trim();
            const saveBtn = document.getElementById('voiceDesignSaveBtn');
            const saveStatusDiv = document.getElementById('voiceDesignSaveStatus');

            if (!presetName) {
                saveStatusDiv.innerHTML = '<span style="color: #ef4444;">Please enter a preset name.</span>';
                return;
            }

            // Show loading state
            saveBtn.disabled = true;
            saveBtn.innerHTML = '⏳ Saving...';
            saveStatusDiv.innerHTML = '<span style="color: var(--text-secondary);">Saving to Speaker Presets...</span>';

            try {
                const response = await fetch(ENDPOINTS.DESIGN_VOICE_SAVE_PRESET, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        preset_name: presetName
                    })
                });

                const result = await response.json();

                if (response.ok && result.success) {
                    saveStatusDiv.innerHTML = `<span style="color: #10b981;">✅ Saved as "${presetName}" in Speaker Presets!</span>`;
                    // Refresh speaker list
                    loadSpeakerList();
                    loadSpeakers();
                } else {
                    const errorMsg = result.detail || result.error || 'Failed to save preset';
                    saveStatusDiv.innerHTML = `<span style="color: #ef4444;">❌ ${errorMsg}</span>`;
                }
            } catch (error) {
                console.error('Save preset error:', error);
                saveStatusDiv.innerHTML = `<span style="color: #ef4444;">❌ Error: ${error.message}</span>`;
            } finally {
                saveBtn.disabled = false;
                saveBtn.innerHTML = '💾 Save Preset';
            }
        }

        async function estimateDuration() {
            const text = document.getElementById('text').value;
            if (!text.trim()) {
                showStatus('Please enter text first', 'error');
                return;
            }

            try {
                showStatus('Estimating duration...', 'success');

                const response = await fetch(ENDPOINTS.ESTIMATE_DURATION, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text, language: 'auto' })
                });

                const result = await response.json();
                if (result.status === 'success') {
                    const estimateDiv = document.getElementById('durationEstimate');
                    estimateDiv.innerHTML = `📊 Estimated: <strong>${result.duration_s}s</strong> (${result.duration_ms}ms)<br>🌐 Language: ${result.detected_language} | 📝 Characters: ${result.char_count}`;
                    estimateDiv.style.display = 'block';
                    document.getElementById('speechLength').value = result.duration_ms;
                    showStatus(`Duration estimated: ${result.duration_s}s`, 'success');
                } else {
                    showStatus(`Error: ${result.message}`, 'error');
                }
            } catch (error) {
                showStatus(`Error estimating duration: ${error.message}`, 'error');
            }
        }

        async function clearOutputs() {
            if (!confirm('Are you sure you want to clear all generated output files? This action cannot be undone.')) {
                return;
            }

            try {
                showStatus('Clearing outputs...', 'success');

                const response = await fetch(ENDPOINTS.CLEAR_OUTPUTS, {
                    method: 'POST'
                });

                const result = await response.json();

                if (result.status === 'success') {
                    const message = `✅ ${result.message}\n📁 Files deleted: ${result.files_deleted}\n💾 Space freed: ${result.space_freed_mb} MB`;
                    showStatus(message, 'success');

                    // Clear the audio result display
                    document.getElementById('audioResult').innerHTML = '';
                } else {
                    showStatus(`Error: ${result.message}`, 'error');
                }
            } catch (error) {
                showStatus(`Error clearing outputs: ${error.message}`, 'error');
            }
        }

        /* ---------- Speakers ---------- */
        // Add Speaker Form
        document.getElementById('addSpeakerForm').addEventListener('submit', async function (e) {
            e.preventDefault();

            const speakerName = document.getElementById('speakerName').value;
            const audioFiles = document.getElementById('speakerAudioFiles').files;

            if (!audioFiles || audioFiles.length === 0) {
                showStatus('Please select at least one audio file', 'error', 'speakerStatus');
                return;
            }

            try {
                showStatus('Adding speaker...', 'success', 'speakerStatus');

                const formData = new FormData();
                formData.append('name', speakerName);
                formData.append('audio_file', audioFiles[0]); // /add_speaker uses single file
                formData.append('enhance_voice', document.getElementById('applyEnhancement').checked ? 'true' : 'false');
                formData.append('enhancement_model', document.getElementById('enhancementModel').value);
                formData.append('super_resolution_voice', document.getElementById('applySuperResolution').checked ? 'true' : 'false');

                const response = await fetch(ENDPOINTS.ADD_SPEAKER, {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    showStatus(`Speaker "${speakerName}" added successfully!`, 'success', 'speakerStatus');
                    loadSpeakerList();
                    loadSpeakers(); // Refresh dropdown
                } else {
                    showStatus(`Error: ${result.error}`, 'error', 'speakerStatus');
                }
            } catch (error) {
                showStatus(`Error adding speaker: ${error.message}`, 'error', 'speakerStatus');
            }
        });
