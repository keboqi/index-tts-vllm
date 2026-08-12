"use strict";

/* ---------- Synthesis ---------- */
        // TTS Form
        const ttsForm = document.getElementById('ttsForm');
        const generateBtn = document.getElementById('generateBtn');
        const generateBtnDefaultHtml = generateBtn ? generateBtn.innerHTML : '';
        if (ttsForm) {
            ttsForm.addEventListener('submit', async function (e) {
                e.preventDefault();

                const formData = new FormData(this);
                const text = formData.get('text');
                const speaker = formData.get('speaker');
                const ttsBackend = formData.get('tts_backend') || 'index';
                const ttsLanguage = formData.get('language') || 'auto';
                const emotionText = document.getElementById('emotionText').value;
                const emotionWeight = parseFloat(document.getElementById('emotionWeight').value);
                const diffusionSteps = parseInt(document.getElementById('diffusionSteps').value);
                const maxTextTokens = parseInt(document.getElementById('maxTextTokens').value);
                const streamingMode = document.getElementById('streamingMode').checked;
                const speakerEffects = getSelectedSpeakerEffects();

                if (!text.trim()) {
                    showStatus('Please enter some text to synthesize.', 'error');
                    return;
                }

                if (generateBtn) {
                    generateBtn.disabled = true;
                    generateBtn.innerHTML = '⏳ Generating...';
                }

                try {
                    const startTime = performance.now();

                    const useStreamingTransport = streamingMode;
                    if (useStreamingTransport) {
                        // Streaming transport: uses chunked response with keepalive frames.
                        await handleStreamingRequest(text, speaker, emotionText, emotionWeight, diffusionSteps, maxTextTokens, speakerEffects, formData, startTime, ttsBackend, ttsLanguage);
                    } else {
                        // Regular mode
                        await handleRegularRequest(text, speaker, emotionText, emotionWeight, diffusionSteps, maxTextTokens, speakerEffects, formData, startTime, ttsBackend, ttsLanguage);
                    }
                } catch (error) {
                    showStatus(`Network error: ${error.message}`, 'error');
                } finally {
                    if (generateBtn) {
                        generateBtn.disabled = false;
                        generateBtn.innerHTML = generateBtnDefaultHtml || '🎵 Generate Speech';
                    }
                }
            });
        }
