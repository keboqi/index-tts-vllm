"use strict";

function formatTimestamp(ms) {
            const totalMs = Math.max(0, Math.round(ms || 0));
            const minutes = Math.floor(totalMs / 60000);
            const seconds = (totalMs % 60000) / 1000;
            const secondsStr =
                seconds < 10
                    ? `0${seconds.toFixed(3)}`.replace(/([.][0-9]*?[1-9])0+$/, '$1').replace(/[.]0+$/, '')
                    : `${seconds.toFixed(3)}`.replace(/([.][0-9]*?[1-9])0+$/, '$1').replace(/[.]0+$/, '');
            return `${String(minutes).padStart(2, '0')}:${secondsStr}`;
        }

        // Format timestamp as HH:MM:SS for compact view
        function formatTimestampHHMMSS(ms) {
            const totalMs = Math.max(0, Math.round(ms || 0));
            const hours = Math.floor(totalMs / 3600000);
            const minutes = Math.floor((totalMs % 3600000) / 60000);
            const seconds = Math.floor((totalMs % 60000) / 1000);
            return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        }

        // Truncate text with ellipsis
        function truncateText(text, maxLength) {
            if (!text) return '';
            if (text.length <= maxLength) return text;
            return text.substring(0, maxLength - 1) + '…';
        }

        // Escape HTML special characters
        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function normalizeSubtitleUrl(url) {
            return typeof url === 'string' && url.trim() ? url.trim() : null;
        }

        function parseSubtitleTimestamp(value) {
            const raw = String(value || '').trim().replace(',', '.');
            const parts = raw.split(':').map(part => part.trim()).filter(Boolean);
            if (parts.length < 2) {
                return 0;
            }
            const seconds = parseFloat(parts.pop() || '0');
            const minutes = parseInt(parts.pop() || '0', 10);
            const hours = parts.length ? parseInt(parts.pop() || '0', 10) : 0;
            if (Number.isNaN(seconds) || Number.isNaN(minutes) || Number.isNaN(hours)) {
                return 0;
            }
            return Math.max(0, hours * 3600 + minutes * 60 + seconds);
        }

        function formatPlayerTime(seconds) {
            const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
            const wholeSeconds = Math.floor(safeSeconds);
            const hours = Math.floor(wholeSeconds / 3600);
            const minutes = Math.floor((wholeSeconds % 3600) / 60);
            const secs = wholeSeconds % 60;
            if (hours > 0) {
                return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            }
            return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
        }

        function parseSrtCues(srtText) {
            const normalized = String(srtText || '').replace(/\r/g, '').trim();
            if (!normalized) {
                return [];
            }
            return normalized
                .split(/\n{2,}/)
                .map((block, blockIndex) => {
                    const lines = block.split('\n');
                    const timeIndex = lines.findIndex(line => line.includes('-->'));
                    if (timeIndex < 0) {
                        return null;
                    }
                    const [startRaw, endRawWithSettings] = lines[timeIndex].split(/\s*-->\s*/);
                    if (!startRaw || !endRawWithSettings) {
                        return null;
                    }
                    const endRaw = endRawWithSettings.trim().split(/\s+/)[0];
                    const start = parseSubtitleTimestamp(startRaw);
                    const end = parseSubtitleTimestamp(endRaw);
                    const text = lines
                        .slice(timeIndex + 1)
                        .map(line => line.trim())
                        .filter(Boolean)
                        .join('\n')
                        .trim();
                    if (!text || end <= start) {
                        return null;
                    }
                    return {
                        id: `srt-${blockIndex}`,
                        start,
                        end,
                        text,
                    };
                })
                .filter(Boolean)
                .sort((a, b) => a.start - b.start);
        }

        function buildCuesFromSegments(segments = [], textKind = 'translated') {
            if (!Array.isArray(segments)) {
                return [];
            }
            return segments
                .map((segment, index) => {
                    if (!segment) {
                        return null;
                    }
                    const startMs = Number(segment.start_ms);
                    const endMs = Number(segment.end_ms);
                    if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) {
                        return null;
                    }
                    const primaryText =
                        textKind === 'source'
                            ? segment.source_text
                            : segment.translated_text || segment.source_text;
                    const text = String(primaryText || '').trim();
                    if (!text) {
                        return null;
                    }
                    return {
                        id: `segment-${segment.index ?? index}`,
                        start: startMs / 1000,
                        end: endMs / 1000,
                        text,
                    };
                })
                .filter(Boolean)
                .sort((a, b) => a.start - b.start);
        }

        async function loadSubtitleCues(url) {
            const subtitleUrl = normalizeSubtitleUrl(url);
            if (!subtitleUrl) {
                return [];
            }
            const response = await fetch(subtitleUrl, { cache: 'no-cache' });
            if (!response.ok) {
                throw new Error(`Subtitle fetch failed (${response.status})`);
            }
            return parseSrtCues(await response.text());
        }

        function createDownloadLink(url, fileName, label, secondary = false) {
            const link = document.createElement('a');
            link.href = url;
            link.download = fileName || '';
            link.className = secondary ? 'btn btn-secondary' : 'btn';
            link.textContent = label;
            return link;
        }

        function cacheBustUrl(url) {
            if (!url) {
                return '';
            }
            return `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}`;
        }

        function fileNameFromUrl(url, fallback = '') {
            try {
                const parsed = new URL(url, window.location.href);
                const rawName = parsed.pathname.split('/').filter(Boolean).pop() || fallback;
                return decodeURIComponent(rawName || fallback || '');
            } catch (error) {
                const clean = String(url || fallback || '').split(/[?#]/)[0];
                return clean.split('/').pop() || fallback || '';
            }
        }

        function resolveSubtitleFileName(explicitName, url, fallback) {
            const fromUrl = fileNameFromUrl(url, '');
            const fromExplicit = fileNameFromUrl(explicitName, '');
            if (fromExplicit && fromExplicit !== fallback) {
                return fromExplicit;
            }
            return fromUrl || fromExplicit || fallback || '';
        }

        function lazyVideoMarkup(url, posterUrl = '', maxHeight = '220px') {
            const safeUrl = escapeHtml(url || '');
            const safePoster = posterUrl ? ` poster="${escapeHtml(posterUrl)}"` : '';
            return `
                    <div data-lazy-video-wrap style="position:relative;margin-top:10px;border-radius:8px;overflow:hidden;background:#000;">
                        <video class="lazy-video" controls preload="none" data-lazy-src="${safeUrl}"${safePoster} style="width:100%;max-height:${escapeHtml(maxHeight)};display:block;background:#000;"></video>
                        <button type="button" class="btn lazy-video-play" style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);padding:8px 16px;">Play</button>
                    </div>
                `;
        }

        function loadLazyVideo(video, autoplay = false) {
            if (!video) {
                return;
            }
            const src = video.dataset.lazySrc || '';
            if (src && !video.getAttribute('src')) {
                video.src = src;
                video.load();
            }
            const wrapper = video.closest('[data-lazy-video-wrap]');
            const button = wrapper ? wrapper.querySelector('.lazy-video-play') : null;
            if (button) {
                button.style.display = 'none';
            }
            if (autoplay) {
                video.play().catch(error => console.warn('Failed to play video after lazy load', error));
            }
        }

        function bindLazyVideos(root = document) {
            root.querySelectorAll('video.lazy-video:not([data-lazy-bound])').forEach(video => {
                video.dataset.lazyBound = 'true';
                const wrapper = video.closest('[data-lazy-video-wrap]');
                const button = wrapper ? wrapper.querySelector('.lazy-video-play') : null;
                if (button) {
                    button.addEventListener('click', () => loadLazyVideo(video, true));
                }
                video.addEventListener('click', () => {
                    if (!video.getAttribute('src')) {
                        loadLazyVideo(video, true);
                    }
                });
                video.addEventListener('keydown', event => {
                    if ((event.key === 'Enter' || event.key === ' ') && !video.getAttribute('src')) {
                        event.preventDefault();
                        loadLazyVideo(video, true);
                    }
                });
            });
        }

        function resolveSourceVideo(metadata = {}) {
            const candidates = [
                metadata.source_video,
                metadata.downloaded_video,
                metadata.sources && metadata.sources.source_video,
            ];
            const sourceVideo = candidates.find(candidate => candidate && typeof candidate === 'object');
            if (!sourceVideo) {
                return null;
            }
            const id = sourceVideo.id || sourceVideo.filename || sourceVideo.name || '';
            if (!id) {
                return null;
            }
            return {
                ...sourceVideo,
                id,
            };
        }

        function renderTranslatedVideoPreview(container, data = {}) {
            if (!container || !data.video_url) {
                return;
            }
            let panel = container.querySelector('[data-translated-video-panel]');
            if (!panel) {
                panel = document.createElement('section');
                panel.className = 'segment-card';
                panel.dataset.translatedVideoPanel = 'true';
                panel.style.marginTop = '16px';
                container.appendChild(panel);
            }
            panel.innerHTML = '';

            const title = document.createElement('div');
            title.className = 'segment-header';
            title.textContent = data.audio_mode === 'original'
                ? 'Subtitled Video'
                : data.audio_mode === 'both'
                    ? 'Video with Audio Tracks'
                    : 'Translated Video';
            panel.appendChild(title);

            const posterUrl = data.poster_url ? cacheBustUrl(data.poster_url) : `${data.video_url}/snapshot`;
            const videoWrap = document.createElement('div');
            videoWrap.innerHTML = lazyVideoMarkup(cacheBustUrl(data.video_url), posterUrl, '420px').trim();
            panel.appendChild(videoWrap.firstElementChild);
            bindLazyVideos(panel);

            const actions = document.createElement('div');
            actions.style.display = 'flex';
            actions.style.gap = '8px';
            actions.style.flexWrap = 'wrap';
            actions.style.marginTop = '10px';

            actions.appendChild(createDownloadLink(data.video_url, data.file_name || 'translated_video.mp4', 'Download video', true));
            panel.appendChild(actions);
        }

        async function renderTranslatedVideoFromAudio({ sourceVideo, audioFileName, subtitleFileName, embeddedSubtitleFileNames = [], audioMode = 'translated', sessionId, outputBaseName, container, button }) {
            const normalizedAudioMode = ['translated', 'original', 'both'].includes(audioMode) ? audioMode : 'translated';
            const needsTranslatedAudio = normalizedAudioMode === 'translated' || normalizedAudioMode === 'both';
            if (!sourceVideo || (needsTranslatedAudio && !audioFileName)) {
                showStatus('Translated video cannot be rendered because source video metadata is missing.', 'error', 'translateStatus');
                return;
            }
            const originalLabel = button ? button.textContent : '';
            if (button) {
                button.disabled = true;
                button.textContent = 'Rendering video...';
            }
            showStatus(
                normalizedAudioMode === 'original'
                    ? 'Rendering video with the original audio track...'
                    : normalizedAudioMode === 'both'
                        ? 'Rendering video with original and translated audio tracks...'
                        : subtitleFileName
                            ? 'Replacing audio and rendering the selected subtitle track...'
                            : 'Replacing the original video audio track...',
                'info',
                'translateStatus'
            );
            try {
                const safeBase = deriveBaseFromFilename(outputBaseName || audioFileName || sourceVideo.filename || 'translated_video');
                const audioSuffix = normalizedAudioMode === 'original'
                    ? '_original_audio'
                    : normalizedAudioMode === 'both'
                        ? '_dual_audio'
                        : '_translated_video';
                const outputSuffix = subtitleFileName ? `${audioSuffix}_subtitled` : audioSuffix;
                const response = await fetch(ENDPOINTS.VIDEO_REPLACE_AUDIO, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        downloaded_video_id: sourceVideo.id || sourceVideo.filename,
                        session_id: sessionId || null,
                        audio_file_name: needsTranslatedAudio ? audioFileName : null,
                        subtitle_file_name: subtitleFileName || null,
                        embedded_subtitle_file_names: Array.isArray(embeddedSubtitleFileNames) ? embeddedSubtitleFileNames : [],
                        audio_mode: normalizedAudioMode,
                        output_filename: safeBase ? `${safeBase}${outputSuffix}` : '',
                    }),
                });
                const data = await response.json();
                if (!response.ok || data.status === 'error') {
                    throw new Error(data.message || data.error || `Video render failed (${response.status})`);
                }
                renderTranslatedVideoPreview(container, data);
                loadTranslatedVideos();
                showStatus(data.message || 'Video created.', 'success', 'translateStatus');
            } catch (error) {
                showStatus(`Video render failed: ${error.message}`, 'error', 'translateStatus');
            } finally {
                if (button) {
                    button.disabled = false;
                    button.textContent = originalLabel || 'Render video';
                }
            }
        }

        function resolveOriginalAudioUrl(metadata = {}) {
            const separation = metadata && metadata.separation ? metadata.separation : {};
            const chunk = metadata && metadata.chunk ? metadata.chunk : {};
            const sources = metadata && metadata.sources ? metadata.sources : {};
            const candidates = [
                metadata.vocals_url,
                separation.vocals_url,
                chunk.vocals_url,
                sources.vocals_url,
                metadata.original_audio_url,
                metadata.source_audio_url,
            ];
            return normalizeSubtitleUrl(candidates.find(candidate => typeof candidate === 'string' && candidate.trim()));
        }

        function resolveBackingAudioUrl(metadata = {}) {
            const separation = metadata && metadata.separation ? metadata.separation : {};
            const chunk = metadata && metadata.chunk ? metadata.chunk : {};
            const backing = metadata && metadata.backing_track ? metadata.backing_track : {};
            const sources = metadata && metadata.sources ? metadata.sources : {};
            const candidates = [
                metadata.backing_url,
                separation.backing_url,
                chunk.backing_url,
                backing.preview_url,
                sources.backing_url,
            ];
            return normalizeSubtitleUrl(candidates.find(candidate => typeof candidate === 'string' && candidate.trim()));
        }

        function resolveTranslatedVocalUrl(metadata = {}, fallbackUrl = '') {
            const candidates = [
                metadata.translated_vocals_url,
                metadata.translated_unmixed_url,
                metadata.translated_dry_url,
                fallbackUrl,
            ];
            return normalizeSubtitleUrl(candidates.find(candidate => typeof candidate === 'string' && candidate.trim()));
        }

        function renderTranslatedAudioPlayer(resultDiv, options = {}) {
            if (!resultDiv || !options.audioUrl) {
                return;
            }

            const audioUrl = options.audioUrl;
            const downloadName = options.downloadName || 'translated_speech.mp3';
            const subtitleUrl = normalizeSubtitleUrl(options.subtitleUrl);
            const originalSubtitleUrl = normalizeSubtitleUrl(options.originalSubtitleUrl);
            const translatedSubtitleFileName = subtitleUrl
                ? resolveSubtitleFileName(options.subtitleFileName, subtitleUrl, 'translated_speech.srt')
                : '';
            const originalSubtitleFileName = originalSubtitleUrl
                ? resolveSubtitleFileName(options.originalSubtitleFileName, originalSubtitleUrl, 'translated_speech_original.srt')
                : '';
            const metadata = options.metadata || {};
            const fallbackSegments = Array.isArray(options.segments) ? options.segments : [];
            const sourceVideo = resolveSourceVideo(metadata);
            const translatedAudioFileName = metadata.audio_file_name || options.audioFileName || options.downloadName || downloadName;
            const originalAudioUrl = resolveOriginalAudioUrl(metadata);
            const backingAudioUrl = resolveBackingAudioUrl(metadata);
            const translatedVocalUrl = resolveTranslatedVocalUrl(metadata, audioUrl);
            const audioSources = {
                translated: {
                    label: 'Translated Audio',
                    url: translatedVocalUrl,
                    downloadName,
                },
            };
            if (originalAudioUrl && originalAudioUrl !== audioUrl) {
                audioSources.original = {
                    label: 'Original Audio',
                    url: originalAudioUrl,
                    downloadName: 'original_audio.mp3',
                };
            }

            resultDiv.innerHTML = '';

            const player = document.createElement('section');
            player.className = 'synced-audio-player';

            const audio = document.createElement('audio');
            audio.className = 'synced-audio-native';
            audio.preload = 'metadata';
            audio.src = audioSources.translated.url;
            player.appendChild(audio);

            const backingAudio = document.createElement('audio');
            backingAudio.className = 'synced-audio-native';
            backingAudio.preload = 'metadata';
            if (backingAudioUrl) {
                backingAudio.src = backingAudioUrl;
            }
            player.appendChild(backingAudio);

            const main = document.createElement('div');
            main.className = 'synced-audio-main';
            player.appendChild(main);

            const stage = document.createElement('div');
            stage.className = 'synced-audio-stage';
            main.appendChild(stage);

            const topbar = document.createElement('div');
            topbar.className = 'synced-audio-topbar';
            stage.appendChild(topbar);

            const title = document.createElement('div');
            title.className = 'synced-audio-title';
            const strong = document.createElement('strong');
            strong.textContent = 'Translated Audio';
            const sub = document.createElement('span');
            const segmentCount = typeof metadata.segment_count === 'number' ? metadata.segment_count : null;
            sub.textContent = segmentCount ? `${segmentCount} segments` : 'Ready';
            title.appendChild(strong);
            title.appendChild(sub);
            topbar.appendChild(title);

            const topbarRight = document.createElement('div');
            topbarRight.className = 'synced-audio-actions';
            const sourceTabs = document.createElement('div');
            sourceTabs.className = 'synced-audio-source-tabs';
            const subtitleCount = document.createElement('span');
            subtitleCount.className = 'synced-subtitle-count';
            subtitleCount.textContent = subtitleUrl || fallbackSegments.length ? 'Loading subtitles' : 'No subtitles';
            topbarRight.appendChild(sourceTabs);
            topbarRight.appendChild(subtitleCount);
            topbar.appendChild(topbarRight);

            const now = document.createElement('div');
            now.className = 'synced-now';
            const nowTime = document.createElement('div');
            nowTime.className = 'synced-now-time';
            nowTime.textContent = '00:00';
            const nowText = document.createElement('div');
            nowText.className = 'synced-now-text';
            nowText.textContent = 'Loading subtitles...';
            now.appendChild(nowTime);
            now.appendChild(nowText);
            stage.appendChild(now);

            const controls = document.createElement('div');
            controls.className = 'synced-controls';
            stage.appendChild(controls);

            const playButton = document.createElement('button');
            playButton.type = 'button';
            playButton.className = 'synced-play-btn';
            playButton.textContent = 'Play';
            playButton.title = 'Play or pause';
            controls.appendChild(playButton);

            const seekWrap = document.createElement('div');
            const seek = document.createElement('input');
            seek.type = 'range';
            seek.className = 'synced-seek';
            seek.min = '0';
            seek.max = '1000';
            seek.step = '1';
            seek.value = '0';
            const timeRow = document.createElement('div');
            timeRow.className = 'synced-time';
            const elapsed = document.createElement('span');
            elapsed.textContent = '00:00';
            const durationLabel = document.createElement('span');
            durationLabel.textContent = '00:00';
            timeRow.appendChild(elapsed);
            timeRow.appendChild(durationLabel);
            seekWrap.appendChild(seek);
            seekWrap.appendChild(timeRow);
            controls.appendChild(seekWrap);

            const sideControls = document.createElement('div');
            sideControls.className = 'synced-side-controls';
            const backButton = document.createElement('button');
            backButton.type = 'button';
            backButton.className = 'synced-mini-btn';
            backButton.textContent = '-10';
            backButton.title = 'Back 10 seconds';
            const forwardButton = document.createElement('button');
            forwardButton.type = 'button';
            forwardButton.className = 'synced-mini-btn';
            forwardButton.textContent = '+10';
            forwardButton.title = 'Forward 10 seconds';
            const rateSelect = document.createElement('select');
            rateSelect.className = 'synced-rate-select select-inline-dark';
            ['0.75', '1', '1.25', '1.5', '2'].forEach(rate => {
                const option = document.createElement('option');
                option.value = rate;
                option.textContent = `${rate}x`;
                if (rate === '1') {
                    option.selected = true;
                }
                rateSelect.appendChild(option);
            });
            sideControls.appendChild(backButton);
            sideControls.appendChild(forwardButton);
            sideControls.appendChild(rateSelect);
            controls.appendChild(sideControls);

            const downloads = document.createElement('div');
            downloads.className = 'synced-downloads';
            downloads.appendChild(createDownloadLink(audioUrl, downloadName, 'Download audio'));
            if (backingAudioUrl) {
                downloads.appendChild(createDownloadLink(backingAudioUrl, 'backing_track.mp3', 'Backing track', true));
            }
            if (subtitleUrl) {
                downloads.appendChild(
                    createDownloadLink(
                        subtitleUrl,
                        translatedSubtitleFileName || 'translated_speech.srt',
                        'Translated SRT',
                        true
                    )
                );
            }
            if (originalSubtitleUrl) {
                downloads.appendChild(
                    createDownloadLink(
                        originalSubtitleUrl,
                        originalSubtitleFileName || 'translated_speech_original.srt',
                        'Original SRT',
                        true
                    )
                );
            }
            if (sourceVideo && translatedAudioFileName) {
                let renderSubtitleSelect = null;
                let embedSubtitleSelect = null;
                if (translatedSubtitleFileName || originalSubtitleFileName) {
                    renderSubtitleSelect = document.createElement('select');
                    renderSubtitleSelect.className = 'select-inline-dark';
                    renderSubtitleSelect.title = 'Subtitle track to burn into video frames';
                    renderSubtitleSelect.style.minWidth = '150px';

                    const noneOption = document.createElement('option');
                    noneOption.value = '';
                    noneOption.textContent = 'Burn: none';
                    renderSubtitleSelect.appendChild(noneOption);

                    if (translatedSubtitleFileName) {
                        const translatedOption = document.createElement('option');
                        translatedOption.value = translatedSubtitleFileName;
                        translatedOption.textContent = 'Burn: translated';
                        renderSubtitleSelect.appendChild(translatedOption);
                    }
                    if (originalSubtitleFileName) {
                        const originalOption = document.createElement('option');
                        originalOption.value = originalSubtitleFileName;
                        originalOption.textContent = 'Burn: original';
                        renderSubtitleSelect.appendChild(originalOption);
                    }
                    downloads.appendChild(renderSubtitleSelect);

                    embedSubtitleSelect = document.createElement('select');
                    embedSubtitleSelect.className = 'select-inline-dark';
                    embedSubtitleSelect.title = 'Subtitle tracks to embed in the MP4';
                    embedSubtitleSelect.style.minWidth = '170px';

                    const embedOptions = [['', 'Embed: none']];
                    if (translatedSubtitleFileName) {
                        embedOptions.push(['translated', 'Embed: translated']);
                    }
                    if (originalSubtitleFileName) {
                        embedOptions.push(['original', 'Embed: original']);
                    }
                    if (translatedSubtitleFileName && originalSubtitleFileName) {
                        embedOptions.push(['both', 'Embed: both']);
                    }
                    embedOptions.forEach(([value, label]) => {
                        const option = document.createElement('option');
                        option.value = value;
                        option.textContent = label;
                        embedSubtitleSelect.appendChild(option);
                    });
                    downloads.appendChild(embedSubtitleSelect);
                }

                const renderAudioModeSelect = document.createElement('select');
                renderAudioModeSelect.className = 'select-inline-dark';
                renderAudioModeSelect.title = 'Audio track for rendered video';
                renderAudioModeSelect.style.minWidth = '170px';
                [
                    ['translated', 'Translated audio'],
                    ['original', 'Original audio'],
                    ['both', 'Both audio tracks'],
                ].forEach(([value, label]) => {
                    const option = document.createElement('option');
                    option.value = value;
                    option.textContent = label;
                    renderAudioModeSelect.appendChild(option);
                });
                downloads.appendChild(renderAudioModeSelect);

                const renderVideoBtn = document.createElement('button');
                renderVideoBtn.type = 'button';
                renderVideoBtn.className = 'btn btn-secondary';
                renderVideoBtn.textContent = 'Render video';
                renderVideoBtn.addEventListener('click', () => {
                    const selectedSubtitleFileName = renderSubtitleSelect ? renderSubtitleSelect.value : '';
                    const selectedEmbedMode = embedSubtitleSelect ? embedSubtitleSelect.value : '';
                    const embeddedSubtitleFileNames = [];
                    if ((selectedEmbedMode === 'translated' || selectedEmbedMode === 'both') && translatedSubtitleFileName) {
                        embeddedSubtitleFileNames.push(translatedSubtitleFileName);
                    }
                    if ((selectedEmbedMode === 'original' || selectedEmbedMode === 'both') && originalSubtitleFileName) {
                        embeddedSubtitleFileNames.push(originalSubtitleFileName);
                    }
                    const selectedAudioMode = renderAudioModeSelect ? renderAudioModeSelect.value : 'translated';
                    renderTranslatedVideoFromAudio({
                        sourceVideo,
                        audioFileName: translatedAudioFileName,
                        subtitleFileName: selectedSubtitleFileName,
                        embeddedSubtitleFileNames,
                        audioMode: selectedAudioMode,
                        sessionId: metadata.session_id || metadata.reuse_session_id || options.sessionId || '',
                        outputBaseName: metadata.output_base_name || metadata.base_output_name || translatedAudioFileName,
                        container: resultDiv,
                        button: renderVideoBtn,
                    });
                });
                downloads.appendChild(renderVideoBtn);
            }
            stage.appendChild(downloads);

            const subtitlePanel = document.createElement('aside');
            subtitlePanel.className = 'synced-subtitle-panel';
            main.appendChild(subtitlePanel);

            const subtitleHeader = document.createElement('div');
            subtitleHeader.className = 'synced-subtitle-header';
            const subtitleHeading = document.createElement('strong');
            subtitleHeading.textContent = 'Subtitles';
            const subtitleTabs = document.createElement('div');
            subtitleTabs.className = 'synced-subtitle-tabs';
            subtitleHeader.appendChild(subtitleHeading);
            subtitleHeader.appendChild(subtitleTabs);
            subtitlePanel.appendChild(subtitleHeader);

            const subtitleList = document.createElement('div');
            subtitleList.className = 'synced-subtitle-list';
            subtitlePanel.appendChild(subtitleList);

            resultDiv.appendChild(player);

            const subtitleSets = {
                translated: {
                    label: 'Translated',
                    url: subtitleUrl,
                    cues: buildCuesFromSegments(fallbackSegments, 'translated'),
                },
                original: {
                    label: 'Original',
                    url: originalSubtitleUrl,
                    cues: buildCuesFromSegments(fallbackSegments, 'source'),
                },
            };
            let activeSubtitleKind = subtitleSets.translated.url || subtitleSets.translated.cues.length
                ? 'translated'
                : 'original';
            let activeAudioKind = 'translated';
            let backingEnabled = Boolean(backingAudioUrl);
            let cueButtons = [];
            let activeCueIndex = -1;
            let manualScrollUntil = 0;
            let programmaticSubtitleScroll = false;
            let pendingSubtitleLoads = [subtitleUrl, originalSubtitleUrl].filter(Boolean).length;

            const getActiveCues = () => (subtitleSets[activeSubtitleKind] || subtitleSets.translated).cues || [];
            const getActiveAudioSource = () => audioSources[activeAudioKind] || audioSources.translated;
            const getBackingVolume = () => {
                const backing = metadata && metadata.backing_track ? metadata.backing_track : {};
                const rawVolume =
                    typeof backing.volume_percent === 'number'
                        ? backing.volume_percent
                        : typeof metadata.backing_volume_percent === 'number'
                            ? metadata.backing_volume_percent
                            : 100;
                return Math.max(0, Math.min(1, rawVolume / 100));
            };
            const syncBackingPosition = () => {
                if (!backingAudioUrl || !backingAudio.src) {
                    return;
                }
                const currentTime = audio.currentTime || 0;
                try {
                    backingAudio.currentTime = currentTime;
                } catch (_error) {
                    /* backing metadata may not be ready yet */
                }
            };
            const pauseBacking = () => {
                if (backingAudioUrl) {
                    backingAudio.pause();
                }
            };
            const playBacking = () => {
                if (!backingEnabled || !backingAudioUrl) {
                    pauseBacking();
                    return;
                }
                backingAudio.volume = getBackingVolume();
                backingAudio.playbackRate = audio.playbackRate || 1;
                syncBackingPosition();
                backingAudio.play().catch(error => console.warn('Failed to play backing track', error));
            };
            const updateAudioSourceTitle = () => {
                const source = getActiveAudioSource();
                strong.textContent = source.label;
                if (activeAudioKind === 'original') {
                    sub.textContent = segmentCount ? `Original vocal - ${segmentCount} segments` : 'Original vocal';
                } else {
                    sub.textContent = segmentCount ? `${segmentCount} segments` : 'Ready';
                }
            };
            const getEffectiveDuration = () => {
                if (Number.isFinite(audio.duration) && audio.duration > 0) {
                    return audio.duration;
                }
                const cues = getActiveCues();
                const lastCue = cues.length ? cues[cues.length - 1] : null;
                return lastCue ? lastCue.end : 0;
            };
            const seekTo = seconds => {
                const duration = getEffectiveDuration();
                const nextTime = Math.max(0, duration ? Math.min(seconds, duration) : seconds);
                try {
                    audio.currentTime = nextTime;
                } catch (_error) {
                    /* metadata may not be ready yet */
                }
                syncBackingPosition();
                updatePlaybackUi();
            };
            const seekBy = delta => seekTo((audio.currentTime || 0) + delta);
            const updateAudioSourceTabs = () => {
                sourceTabs.innerHTML = '';
                const entries = Object.entries(audioSources);
                if (entries.length >= 2) {
                    entries.forEach(([kind, source]) => {
                        const tab = document.createElement('button');
                        tab.type = 'button';
                        tab.className = `synced-audio-source-tab${kind === activeAudioKind ? ' active' : ''}`;
                        tab.textContent = kind === 'original' ? 'Original' : 'Translated';
                        tab.title = `Switch to ${source.label}`;
                        tab.addEventListener('click', () => switchAudioSource(kind));
                        sourceTabs.appendChild(tab);
                    });
                }
                if (backingAudioUrl) {
                    const backingTab = document.createElement('button');
                    backingTab.type = 'button';
                    backingTab.className = `synced-audio-source-tab${backingEnabled ? ' active' : ''}`;
                    backingTab.textContent = backingEnabled ? 'Backing On' : 'Backing Off';
                    backingTab.title = backingEnabled ? 'Turn backing off' : 'Turn backing on';
                    backingTab.addEventListener('click', () => {
                        backingEnabled = !backingEnabled;
                        updateAudioSourceTabs();
                        if (backingEnabled && !audio.paused) {
                            playBacking();
                        } else {
                            pauseBacking();
                        }
                    });
                    sourceTabs.appendChild(backingTab);
                }
            };
            const switchAudioSource = kind => {
                const nextSource = audioSources[kind];
                if (!nextSource || kind === activeAudioKind) {
                    return;
                }
                const wasPlaying = !audio.paused;
                const currentTime = audio.currentTime || 0;
                activeAudioKind = kind;
                if (subtitleSets[kind] && (subtitleSets[kind].url || subtitleSets[kind].cues.length)) {
                    activeSubtitleKind = kind;
                }
                activeCueIndex = -1;
                updateAudioSourceTitle();
                updateAudioSourceTabs();
                updateSubtitleTabs();
                renderSubtitleList();
                audio.src = nextSource.url;
                audio.load();
                pauseBacking();
                const restorePosition = () => {
                    seekTo(currentTime);
                    updatePlaybackUi();
                    if (wasPlaying) {
                        audio.play().catch(error => console.warn('Failed to resume audio after source switch', error));
                    }
                };
                audio.addEventListener('loadedmetadata', restorePosition, { once: true });
                window.setTimeout(() => {
                    if (audio.readyState > 0) {
                        restorePosition();
                    }
                }, 250);
                updatePlaybackUi();
            };
            const updateSubtitleCount = () => {
                const cues = getActiveCues();
                if (cues.length) {
                    subtitleCount.textContent = `${cues.length} subtitles`;
                } else if (pendingSubtitleLoads > 0) {
                    subtitleCount.textContent = 'Loading subtitles';
                } else {
                    subtitleCount.textContent = 'No subtitles';
                }
            };
            const updateSubtitleTabs = () => {
                subtitleTabs.innerHTML = '';
                ['translated', 'original'].forEach(kind => {
                    const set = subtitleSets[kind];
                    if (!set.url && !set.cues.length) {
                        return;
                    }
                    const tab = document.createElement('button');
                    tab.type = 'button';
                    tab.className = `synced-subtitle-tab${kind === activeSubtitleKind ? ' active' : ''}`;
                    tab.textContent = set.label;
                    tab.addEventListener('click', () => {
                        activeSubtitleKind = kind;
                        activeCueIndex = -1;
                        updateSubtitleTabs();
                        renderSubtitleList();
                        updatePlaybackUi();
                    });
                    subtitleTabs.appendChild(tab);
                });
            };
            const renderSubtitleList = () => {
                const cues = getActiveCues();
                subtitleList.innerHTML = '';
                cueButtons = [];
                if (!cues.length) {
                    const empty = document.createElement('div');
                    empty.className = 'synced-empty';
                    empty.textContent = pendingSubtitleLoads > 0 ? 'Loading subtitles...' : 'No synced subtitles available.';
                    subtitleList.appendChild(empty);
                    nowText.textContent = pendingSubtitleLoads > 0 ? 'Loading subtitles...' : 'No synced subtitles available.';
                    nowTime.textContent = formatPlayerTime(audio.currentTime || 0);
                    updateSubtitleCount();
                    return;
                }
                cues.forEach((cue, index) => {
                    const item = document.createElement('button');
                    item.type = 'button';
                    item.className = 'synced-subtitle-item';
                    item.dataset.index = String(index);
                    const cueTime = document.createElement('span');
                    cueTime.className = 'synced-subtitle-time';
                    cueTime.textContent = `${formatPlayerTime(cue.start)} - ${formatPlayerTime(cue.end)}`;
                    const cueText = document.createElement('span');
                    cueText.className = 'synced-subtitle-text';
                    cueText.textContent = cue.text;
                    item.appendChild(cueTime);
                    item.appendChild(cueText);
                    item.addEventListener('click', () => {
                        const wasPlaying = !audio.paused;
                        manualScrollUntil = Date.now() + 1200;
                        seekTo(cue.start);
                        if (wasPlaying) {
                            audio.play().catch(error => console.warn('Failed to resume audio after subtitle seek', error));
                        }
                    });
                    subtitleList.appendChild(item);
                    cueButtons.push(item);
                });
                updateSubtitleCount();
            };
            const updateActiveCue = currentTime => {
                const cues = getActiveCues();
                let nextIndex = -1;
                for (let i = 0; i < cues.length; i += 1) {
                    if (currentTime >= cues[i].start) {
                        nextIndex = i;
                    } else {
                        break;
                    }
                }
                if (nextIndex !== activeCueIndex) {
                    if (cueButtons[activeCueIndex]) {
                        cueButtons[activeCueIndex].classList.remove('active');
                    }
                    activeCueIndex = nextIndex;
                    if (cueButtons[activeCueIndex]) {
                        cueButtons[activeCueIndex].classList.add('active');
                        if (!audio.paused && Date.now() > manualScrollUntil) {
                            programmaticSubtitleScroll = true;
                            cueButtons[activeCueIndex].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
                            window.setTimeout(() => {
                                programmaticSubtitleScroll = false;
                            }, 450);
                        }
                    }
                }
                const activeCue = cues[activeCueIndex];
                if (activeCue) {
                    nowTime.textContent = `${formatPlayerTime(activeCue.start)} - ${formatPlayerTime(activeCue.end)}`;
                    nowText.textContent = activeCue.text;
                } else {
                    nowTime.textContent = formatPlayerTime(currentTime);
                    nowText.textContent = cues.length ? '' : (pendingSubtitleLoads > 0 ? 'Loading subtitles...' : 'No synced subtitles available.');
                }
            };
            function updatePlaybackUi() {
                const duration = getEffectiveDuration();
                const currentTime = audio.currentTime || 0;
                const progress = duration > 0 ? Math.min(1000, Math.max(0, (currentTime / duration) * 1000)) : 0;
                seek.value = String(progress);
                elapsed.textContent = formatPlayerTime(currentTime);
                durationLabel.textContent = formatPlayerTime(duration);
                updateActiveCue(currentTime);
            }
            const loadSubtitleSet = async kind => {
                const set = subtitleSets[kind];
                if (!set || !set.url) {
                    return;
                }
                try {
                    const cues = await loadSubtitleCues(set.url);
                    if (cues.length) {
                        set.cues = cues;
                    }
                } catch (error) {
                    console.warn(`Failed to load ${kind} subtitles`, error);
                } finally {
                    pendingSubtitleLoads = Math.max(0, pendingSubtitleLoads - 1);
                    if (!getActiveCues().length && set.cues.length) {
                        activeSubtitleKind = kind;
                    }
                    activeCueIndex = -1;
                    updateSubtitleTabs();
                    renderSubtitleList();
                    updatePlaybackUi();
                }
            };

            subtitleList.addEventListener('wheel', () => {
                manualScrollUntil = Date.now() + 2500;
            }, { passive: true });
            subtitleList.addEventListener('scroll', () => {
                if (!programmaticSubtitleScroll) {
                    manualScrollUntil = Date.now() + 1800;
                }
            }, { passive: true });
            subtitleList.addEventListener('touchstart', () => {
                manualScrollUntil = Date.now() + 2500;
            }, { passive: true });
            playButton.addEventListener('click', () => {
                if (audio.paused) {
                    audio.play().catch(error => console.warn('Failed to play translated audio', error));
                } else {
                    audio.pause();
                }
            });
            seek.addEventListener('input', () => {
                const duration = getEffectiveDuration();
                if (!duration) {
                    return;
                }
                seekTo((parseFloat(seek.value) / 1000) * duration);
            });
            backButton.addEventListener('click', () => seekBy(-10));
            forwardButton.addEventListener('click', () => seekBy(10));
            rateSelect.addEventListener('change', () => {
                const nextRate = parseFloat(rateSelect.value);
                audio.playbackRate = Number.isFinite(nextRate) ? nextRate : 1;
                backingAudio.playbackRate = audio.playbackRate;
            });
            audio.addEventListener('play', () => {
                if (activeSyncedAudio && activeSyncedAudio !== audio) {
                    activeSyncedAudio.pause();
                }
                activeSyncedAudio = audio;
                playButton.textContent = 'Pause';
                playBacking();
            });
            audio.addEventListener('pause', () => {
                playButton.textContent = 'Play';
                pauseBacking();
            });
            audio.addEventListener('ended', () => {
                playButton.textContent = 'Play';
                pauseBacking();
            });
            audio.addEventListener('loadedmetadata', updatePlaybackUi);
            audio.addEventListener('durationchange', updatePlaybackUi);
            audio.addEventListener('timeupdate', () => {
                if (
                    backingEnabled &&
                    backingAudioUrl &&
                    !audio.paused &&
                    Math.abs((backingAudio.currentTime || 0) - (audio.currentTime || 0)) > 0.45
                ) {
                    syncBackingPosition();
                }
                updatePlaybackUi();
            });

            updateAudioSourceTitle();
            updateAudioSourceTabs();
            updateSubtitleTabs();
            renderSubtitleList();
            updatePlaybackUi();
            loadSubtitleSet('translated');
            loadSubtitleSet('original');
        }

        // Toggle segment card expand/collapse
