"use strict";

async function handleRegularRequest(text, speaker, emotionText, emotionWeight, diffusionSteps, maxTextTokens, speakerEffects, formData, startTime, ttsBackend, ttsLanguage) {
            let response;
            const voiceFiles = document.getElementById('voice_files').files;

            if (speaker) {
                // Use /speak endpoint with speaker preset
                const requestData = {
                    text: text,
                    name: speaker,  // API uses 'name' not 'speaker'
                    tts_backend: ttsBackend,
                    language: ttsLanguage,
                    emotion_text: emotionText || "",
                    emotion_weight: emotionWeight,
                    speech_length: parseInt(document.getElementById('speechLength').value) || 0,
                    duration_control: getDurationControlMode(),
                    diffusion_steps: diffusionSteps,
                    max_text_tokens_per_sentence: maxTextTokens,
                    speaker_effects: speakerEffects,
                    response_format: "mp3"
                };

                response = await fetch(ENDPOINTS.SPEAK, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestData)
                });
            } else if (voiceFiles && voiceFiles.length > 0) {
                // Use /clone_voice endpoint with uploaded voice file
                const cloneFormData = new FormData();
                cloneFormData.append('text', text);
                cloneFormData.append('tts_backend', ttsBackend);
                cloneFormData.append('language', ttsLanguage);
                cloneFormData.append('reference_audio_file', voiceFiles[0]);
                cloneFormData.append('emotion_text', emotionText || "");
                cloneFormData.append('emotion_weight', emotionWeight.toString());
                cloneFormData.append('speech_length', (parseInt(document.getElementById('speechLength').value) || 0).toString());
                cloneFormData.append('duration_control', getDurationControlMode());
                cloneFormData.append('diffusion_steps', diffusionSteps.toString());
                cloneFormData.append('max_text_tokens_per_sentence', maxTextTokens.toString());
                cloneFormData.append('speaker_effects', JSON.stringify(speakerEffects || []));
                cloneFormData.append('response_format', 'mp3');

                response = await fetch(ENDPOINTS.CLONE_VOICE, {
                    method: 'POST',
                    body: cloneFormData
                });
            } else {
                showStatus('Please select a speaker preset or upload a voice file', 'error');
                return;
            }

            if (response.ok) {
                const endTime = performance.now();
                const duration = ((endTime - startTime) / 1000).toFixed(2);

                const blob = await response.blob();
                const audioUrl = URL.createObjectURL(blob);

                document.getElementById('audioResult').innerHTML = `
                            <h3>🎵 Generated Speech (${duration}s)</h3>
                        <audio controls autoplay preload="none" style="width: 100%; margin: 10px 0;">
                                <source src="${audioUrl}" type="audio/mpeg">
                            </audio>
                            <br>
                            <a href="${audioUrl}" download="speech.mp3" class="btn">💾 Download</a>
                        `;
                // Show enhanced status message with emotion info
                let statusMessage = `Speech generated in ${duration}s! 🚀`;
                if (ttsBackend === 'index' && emotionText && emotionText.trim()) {
                    statusMessage += ` 😊 Emotion: "${emotionText}" (${emotionWeight})`;
                }
                statusMessage += formatSpeakerEffectsStatus(speakerEffects);
                showStatus(statusMessage, 'success');
            } else {
                const error = await response.text();
                showStatus(`Error: ${error}`, 'error');
            }
        }

        async function handleStreamingRequest(text, speaker, emotionText, emotionWeight, diffusionSteps, maxTextTokens, speakerEffects, formData, startTime, ttsBackend, ttsLanguage) {
            showStatus('⚡ Streaming: Waiting for first chunk...', 'success');

            // Get first chunk size setting
            const firstChunkSize = parseInt(document.getElementById('firstChunkSize').value) || 40;

            let endpoint, requestOptions;

            const voiceFiles = document.getElementById('voice_files').files;

            if (speaker) {
                // Use /speak_stream endpoint with speaker preset
                endpoint = '/speak_stream';
                requestOptions = {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: text,
                        name: speaker,  // API uses 'name' not 'speaker'
                        tts_backend: ttsBackend,
                        language: ttsLanguage,
                        emotion_text: emotionText || "",
                        emotion_weight: emotionWeight,
                        speech_length: parseInt(document.getElementById('speechLength').value) || 0,
                        duration_control: getDurationControlMode(),
                        diffusion_steps: diffusionSteps,
                        max_text_tokens_per_sentence: maxTextTokens,
                        speaker_effects: speakerEffects,
                        response_format: "mp3"
                    })
                };
            } else if (voiceFiles && voiceFiles.length > 0) {
                // Use /clone_voice_stream endpoint with uploaded voice file
                endpoint = '/clone_voice_stream';
                const cloneFormData = new FormData();
                cloneFormData.append('text', text);
                cloneFormData.append('tts_backend', ttsBackend);
                cloneFormData.append('language', ttsLanguage);
                cloneFormData.append('reference_audio_file', voiceFiles[0]);
                cloneFormData.append('emotion_text', emotionText || "");
                cloneFormData.append('emotion_weight', emotionWeight.toString());
                cloneFormData.append('speech_length', (parseInt(document.getElementById('speechLength').value) || 0).toString());
                cloneFormData.append('duration_control', getDurationControlMode());
                cloneFormData.append('diffusion_steps', diffusionSteps.toString());
                cloneFormData.append('max_text_tokens_per_sentence', maxTextTokens.toString());
                cloneFormData.append('speaker_effects', JSON.stringify(speakerEffects || []));
                cloneFormData.append('response_format', 'mp3');
                requestOptions = {
                    method: 'POST',
                    body: cloneFormData
                };
            } else {
                showStatus('Please select a speaker preset or upload a voice file for streaming', 'error');
                return;
            }

            const response = await fetch(endpoint, requestOptions);

            if (!response.ok) {
                const error = await response.text();
                showStatus(`Error: ${error}`, 'error');
                return;
            }

            const reader = response.body.getReader();
            const audioChunks = [];
            let buffer = new Uint8Array();
            let chunkCount = 0;
            let firstChunkTime = null;
            let audioContext = null;
            let audioSource = null;
            let nextStartTime = 0;

            // Create audio context for streaming playback
            audioContext = new (window.AudioContext || window.webkitAudioContext)();

            try {
                while (true) {
                    const { done, value } = await reader.read();

                    if (done) {
                        break;
                    }

                    // Append new data to buffer
                    const newBuffer = new Uint8Array(buffer.length + value.length);
                    newBuffer.set(buffer);
                    newBuffer.set(value, buffer.length);
                    buffer = newBuffer;

                    // Try to parse chunks from buffer
                    while (true) {
                        // Look for header: CHUNK:idx:size:status\\n
                        // Find newline character (10 = '\\n' in ASCII)
                        let headerEnd = -1;
                        for (let i = 0; i < buffer.length; i++) {
                            if (buffer[i] === 10) {
                                headerEnd = i;
                                break;
                            }
                        }

                        if (headerEnd === -1) break;

                        const headerText = new TextDecoder().decode(buffer.slice(0, headerEnd));

                        if (headerText.startsWith('ERROR:')) {
                            showStatus(`Streaming error: ${headerText.substring(6)}`, 'error');
                            return;
                        }

                        if (headerText.startsWith('KEEPALIVE:')) {
                            const parts = headerText.split(':');
                            const parsedPayloadSize = parts.length >= 2 ? parseInt(parts[1]) : 0;
                            const payloadSize = Number.isFinite(parsedPayloadSize) && parsedPayloadSize > 0
                                ? parsedPayloadSize
                                : 0;
                            const payloadStart = headerEnd + 1;
                            const payloadEnd = payloadStart + payloadSize;

                            if (buffer.length < payloadEnd) break;

                            const payloadData = buffer.slice(payloadStart, payloadEnd);
                            buffer = buffer.slice(payloadEnd);

                            let payload = {};
                            try {
                                const payloadText = new TextDecoder().decode(payloadData);
                                payload = payloadText ? JSON.parse(payloadText) : {};
                            } catch (parseError) {
                                console.warn('Failed to parse streaming keepalive:', parseError);
                            }
                            const elapsed = typeof payload.elapsed_seconds === 'number'
                                ? ` (${payload.elapsed_seconds}s)`
                                : '';
                            const message = payload.message || `External TTS backend is still working${elapsed}.`;
                            showStatus(message, 'info');
                            if (firstChunkTime === null) {
                                document.getElementById('audioResult').innerHTML = `
                                    <div style="background: rgba(30, 41, 59, 0.55); padding: 16px; border-radius: 10px; margin: 10px 0; border: 1px solid var(--border);">
                                        <h3 style="margin: 0 0 8px 0;">External backend warming up...</h3>
                                        <div style="color: var(--text-muted);">${message}</div>
                                    </div>
                                `;
                            }
                            continue;
                        }

                        if (!headerText.startsWith('CHUNK:')) break;

                        const parts = headerText.split(':');
                        if (parts.length !== 4) break;

                        const chunkIdx = parseInt(parts[1]);
                        const chunkSize = parseInt(parts[2]);
                        const isLast = parts[3] === 'LAST';

                        // Check if we have the complete chunk
                        const chunkStart = headerEnd + 1;
                        const chunkEnd = chunkStart + chunkSize;

                        if (buffer.length < chunkEnd) break;

                        // Extract chunk data
                        const chunkData = buffer.slice(chunkStart, chunkEnd);
                        buffer = buffer.slice(chunkEnd);

                        chunkCount++;

                        if (firstChunkTime === null) {
                            firstChunkTime = performance.now();
                            const ttfb = ((firstChunkTime - startTime) / 1000).toFixed(2);

                            // Show first chunk performance prominently
                            const firstChunkSize = document.getElementById('firstChunkSize').value;
                            showStatus(`⚡ First chunk ready in ${ttfb}s! (${firstChunkSize} tokens) Playing now...`, 'success');

                            // Show real-time performance indicator
                            document.getElementById('audioResult').innerHTML = `
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; margin: 10px 0;">
                                    <h3 style="margin: 0; display: flex; align-items: center; gap: 10px;">
                                        <span class="loading"></span>
                                        Streaming in progress...
                                    </h3>
                                    <div style="margin-top: 15px; background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                                        <div style="font-size: 1.2em; margin-bottom: 5px;">
                                            ⚡ First Chunk Generated
                                        </div>
                                        <div style="font-size: 2em; font-weight: bold;">
                                            ${ttfb}s
                                        </div>
                                        <div style="font-size: 0.9em; opacity: 0.9; margin-top: 5px;">
                                            🎵 Audio playing • Receiving chunk ${chunkCount}/${chunkCount}...
                                        </div>
                                    </div>
                                </div>
                            `;
                        } else {
                            // Update chunk counter during streaming
                            const currentDisplay = document.getElementById('audioResult').innerHTML;
                            if (currentDisplay.includes('Streaming in progress')) {
                                const ttfb = ((firstChunkTime - startTime) / 1000).toFixed(2);
                                document.getElementById('audioResult').innerHTML = `
                                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; margin: 10px 0;">
                                        <h3 style="margin: 0; display: flex; align-items: center; gap: 10px;">
                                            <span class="loading"></span>
                                            Streaming in progress...
                                        </h3>
                                        <div style="margin-top: 15px; background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                                            <div style="font-size: 1.2em; margin-bottom: 5px;">
                                                ⚡ First Chunk Generated
                                            </div>
                                            <div style="font-size: 2em; font-weight: bold;">
                                                ${ttfb}s
                                            </div>
                                            <div style="font-size: 0.9em; opacity: 0.9; margin-top: 5px;">
                                                🎵 Audio playing • Received ${chunkCount} chunks...
                                            </div>
                                        </div>
                                    </div>
                                `;
                            }
                        }

                        // Decode and play audio chunk
                        try {
                            const audioBlob = new Blob([chunkData], { type: 'audio/mpeg' });
                            const arrayBuffer = await audioBlob.arrayBuffer();
                            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

                            // Schedule playback
                            const source = audioContext.createBufferSource();
                            source.buffer = audioBuffer;
                            source.connect(audioContext.destination);

                            const currentTime = audioContext.currentTime;
                            if (nextStartTime < currentTime) {
                                nextStartTime = currentTime;
                            }

                            source.start(nextStartTime);
                            nextStartTime += audioBuffer.duration;

                            // Store for later download
                            audioChunks.push(chunkData);

                            showStatus(`⚡ Streaming: Playing chunk ${chunkIdx + 1}...`, 'success');
                        } catch (decodeError) {
                            console.error('Error decoding audio chunk:', decodeError);
                            showStatus(`⚠️ Error decoding chunk ${chunkIdx}: ${decodeError.message}`, 'error');
                        }

                        if (isLast) {
                            const endTime = performance.now();
                            const duration = ((endTime - startTime) / 1000).toFixed(2);
                            const firstChunkDuration = ((firstChunkTime - startTime) / 1000).toFixed(2);

                            // Combine all chunks for download
                            const combinedBlob = new Blob(audioChunks, { type: 'audio/mpeg' });
                            const audioUrl = URL.createObjectURL(combinedBlob);

                            // Calculate performance metrics
                            const totalGenTime = duration;
                            const firstChunkPercent = ((firstChunkDuration / totalGenTime) * 100).toFixed(0);

                            document.getElementById('audioResult').innerHTML = `
                                <h3>🎵 Streamed Speech (${chunkCount} chunks)</h3>
                                <audio controls src="${audioUrl}" style="width: 100%; margin: 10px 0;"></audio>
                                <br>
                                <div style="background: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 10px; margin: 10px 0; border: 1px solid var(--border);">
                                    <h4 style="margin-top: 0;">⚡ Performance Metrics</h4>
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">
                                        <div style="background: white; padding: 10px; border-radius: 5px; border-left: 4px solid #667eea;">
                                            <strong style="color: #667eea;">⏱️ First Chunk:</strong><br>
                                            <span style="font-size: 1.5em; font-weight: bold;">${firstChunkDuration}s</span>
                                        </div>
                                        <div style="background: white; padding: 10px; border-radius: 5px; border-left: 4px solid #764ba2;">
                                            <strong style="color: #764ba2;">🕐 Total Time:</strong><br>
                                            <span style="font-size: 1.5em; font-weight: bold;">${totalGenTime}s</span>
                                        </div>
                                    </div>
                                    <div style="background: white; padding: 10px; border-radius: 5px;">
                                        <strong>📊 First Chunk Speed:</strong> ${firstChunkPercent}% of total time<br>
                                        <div style="background: rgba(51, 65, 85, 0.5); height: 10px; border-radius: 5px; margin-top: 5px; overflow: hidden;">
                                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 100%; width: ${firstChunkPercent}%;"></div>
                                        </div>
                                    </div>
                                </div>
                                <a href="${audioUrl}" download="speech.mp3" class="btn">💾 Download</a>
                            `;

                            let statusMessage = `✅ Streaming complete! First chunk: ${firstChunkDuration}s, Total: ${totalGenTime}s (${chunkCount} chunks)`;
                            if (ttsBackend === 'index' && emotionText && emotionText.trim()) {
                                statusMessage += ` 😊 Emotion: "${emotionText}" (${emotionWeight})`;
                            }
                            statusMessage += formatSpeakerEffectsStatus(speakerEffects);
                            showStatus(statusMessage, 'success');
                            return;
                        }
                    }
                }
            } catch (streamError) {
                console.error('Streaming error:', streamError);
                showStatus(`Network error: ${streamError.message}`, 'error');
            }
        }
