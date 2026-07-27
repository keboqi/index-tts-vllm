"use strict";

function getSelectedDownloadedVideoId() {
            return (selectedDownloadedVideoId || (translateDownloadedVideoSelect ? translateDownloadedVideoSelect.value : '') || '').trim();
        }

        function getSelectedDownloadedVideo() {
            const id = getSelectedDownloadedVideoId();
            if (!id) {
                return null;
            }
            return downloadedVideos.find(video => video.id === id) || null;
        }

        function renderDownloadedVideoList() {
            const list = document.getElementById('downloadedVideoList');
            if (!list) {
                return;
            }
            if (!downloadedVideos.length) {
                list.innerHTML = '<div class="segment-empty">No downloaded videos yet.</div>';
                return;
            }
            list.innerHTML = downloadedVideos.map(video => {
                const title = escapeHtml(video.title || video.filename || 'Downloaded video');
                const size = typeof video.size_mb === 'number' ? `${video.size_mb} MB` : '';
                const duration = video.duration_label ? ` • ${escapeHtml(video.duration_label)}` : '';
                const url = video.url || `${ENDPOINTS.DOWNLOADED_VIDEOS}/${encodeURIComponent(video.id)}`;
                const posterUrl = video.poster_url || `${url}/snapshot`;
                return `
                        <div class="segment-card" data-video-id="${escapeHtml(video.id)}" style="padding: 14px;">
                            <div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start;">
                                <div style="min-width:0;">
                                    <div class="segment-header" title="${title}">${title}</div>
                                    <small style="color:var(--text-muted);">${escapeHtml(video.extension || 'video').toUpperCase()}${size ? ` • ${size}` : ''}${duration}</small>
                                </div>
                            </div>
                            ${lazyVideoMarkup(url, posterUrl, '220px')}
                            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;">
                                <button type="button" class="btn btn-secondary downloaded-video-use" data-video-id="${escapeHtml(video.id)}" style="padding:6px 12px;font-size:0.82rem;">Use in Translate</button>
                                <a class="btn btn-secondary" href="${url}" download="${escapeHtml(video.filename || video.id)}" style="padding:6px 12px;font-size:0.82rem;">Download</a>
                                <button type="button" class="btn btn-danger downloaded-video-delete" data-video-id="${escapeHtml(video.id)}" data-video-title="${title}" style="padding:6px 12px;font-size:0.82rem;">🗑️ Delete</button>
                            </div>
                        </div>
                    `;
            }).join('');
            bindLazyVideos(list);
            list.querySelectorAll('.downloaded-video-use').forEach(button => {
                button.addEventListener('click', () => {
                    selectDownloadedVideo(button.dataset.videoId || '');
                    switchTab('translate');
                });
            });
            list.querySelectorAll('.downloaded-video-delete').forEach(button => {
                button.addEventListener('click', () => {
                    deleteDownloadedVideo(button.dataset.videoId || '', button.dataset.videoTitle || '');
                });
            });
        }

        function populateDownloadedVideoSelect() {
            if (!translateDownloadedVideoSelect) {
                return;
            }
            const previous = selectedDownloadedVideoId || translateDownloadedVideoSelect.value || '';
            translateDownloadedVideoSelect.innerHTML = '<option value="">Use uploaded audio</option>';
            downloadedVideos.forEach(video => {
                const option = document.createElement('option');
                option.value = video.id;
                const bits = [video.title || video.filename || video.id];
                if (video.duration_label) {
                    bits.push(video.duration_label);
                }
                if (typeof video.size_mb === 'number') {
                    bits.push(`${video.size_mb} MB`);
                }
                option.textContent = bits.join(' - ');
                translateDownloadedVideoSelect.appendChild(option);
            });
            if (previous && downloadedVideos.some(video => video.id === previous)) {
                translateDownloadedVideoSelect.value = previous;
                selectedDownloadedVideoId = previous;
            } else {
                translateDownloadedVideoSelect.value = '';
                selectedDownloadedVideoId = '';
            }
            updateDownloadedVideoHint();
            updateAudioInputRequirement();
        }

        async function loadDownloadedVideos(options = {}) {
            try {
                const response = await fetch(ENDPOINTS.DOWNLOADED_VIDEOS, { cache: 'no-cache' });
                const data = await response.json();
                downloadedVideos = Array.isArray(data.videos) ? data.videos : [];
                if (options.selectId) {
                    selectedDownloadedVideoId = options.selectId;
                }
                populateDownloadedVideoSelect();
                renderDownloadedVideoList();
            } catch (error) {
                console.warn('Failed to load downloaded videos', error);
                const list = document.getElementById('downloadedVideoList');
                if (list) {
                    list.innerHTML = '<div class="segment-empty">Failed to load downloaded videos.</div>';
                }
            }
        }

        function renderTranslatedVideoList() {
            const list = document.getElementById('translatedVideoList');
            if (!list) {
                return;
            }
            if (!translatedVideos.length) {
                list.innerHTML = '<div class="segment-empty">No translated videos yet.</div>';
                return;
            }
            list.innerHTML = translatedVideos.map(video => {
                const id = video.id || video.filename || '';
                const title = escapeHtml(video.title || video.filename || 'Translated video');
                const size = typeof video.size_mb === 'number' ? `${video.size_mb} MB` : '';
                const duration = video.duration_label ? ` • ${escapeHtml(video.duration_label)}` : '';
                const url = video.url || `/api/translate_outputs/${encodeURIComponent(video.filename || id)}`;
                const posterUrl = video.poster_url || `${url}/snapshot`;
                const extension = escapeHtml(video.extension || 'video').toUpperCase();
                const createdLabel = video.mtime_label ? ` • ${escapeHtml(video.mtime_label)}` : '';
                const encodedId = encodeURIComponent(id);
                return `
                        <div class="segment-card" data-translated-video-id="${encodedId}" style="padding: 14px;">
                            <div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start;">
                                <div style="min-width:0;">
                                    <div class="segment-header" title="${title}">${title}</div>
                                    <small style="color:var(--text-muted);">${extension}${size ? ` • ${size}` : ''}${duration}${createdLabel}</small>
                                </div>
                            </div>
                            ${lazyVideoMarkup(url, posterUrl, '220px')}
                            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;">
                                <a class="btn btn-secondary" href="${escapeHtml(url)}" download="${escapeHtml(video.filename || id)}" style="padding:6px 12px;font-size:0.82rem;">Download</a>
                                <button type="button" class="btn btn-danger translated-video-delete" data-video-id="${encodedId}" style="padding:6px 12px;font-size:0.82rem;">🗑️ Delete</button>
                            </div>
                        </div>
                    `;
            }).join('');
            bindLazyVideos(list);
            list.querySelectorAll('.translated-video-delete').forEach(button => {
                button.addEventListener('click', () => {
                    let videoId = button.dataset.videoId || '';
                    try {
                        videoId = decodeURIComponent(videoId);
                    } catch (error) {
                        console.warn('Failed to decode translated video id', error);
                    }
                    const video = translatedVideos.find(item => (item.id || item.filename || '') === videoId);
                    deleteTranslatedVideo(videoId, video ? (video.title || video.filename || '') : '');
                });
            });
        }

        async function loadTranslatedVideos() {
            const list = document.getElementById('translatedVideoList');
            try {
                const response = await fetch(ENDPOINTS.TRANSLATED_VIDEOS, { cache: 'no-cache' });
                const data = await response.json();
                if (!response.ok || data.status === 'error') {
                    throw new Error(data.message || 'Failed to load translated videos');
                }
                translatedVideos = Array.isArray(data.videos) ? data.videos : [];
                renderTranslatedVideoList();
            } catch (error) {
                console.warn('Failed to load translated videos', error);
                if (list) {
                    list.innerHTML = '<div class="segment-empty">Failed to load translated videos.</div>';
                }
            }
        }

        function updateDownloadedVideoHint() {
            if (!translateDownloadedVideoHint) {
                return;
            }
            const video = getSelectedDownloadedVideo();
            if (!video) {
                translateDownloadedVideoHint.textContent = 'Select a downloaded video to auto-extract MP3 with FFmpeg.';
                return;
            }
            const details = [];
            if (video.duration_label) {
                details.push(video.duration_label);
            }
            if (typeof video.size_mb === 'number') {
                details.push(`${video.size_mb} MB`);
            }
            translateDownloadedVideoHint.textContent = `Using ${video.filename || video.id}${details.length ? ` (${details.join(', ')})` : ''}.`;
        }

        function selectDownloadedVideo(videoId) {
            selectedDownloadedVideoId = videoId || '';
            if (translateDownloadedVideoSelect) {
                translateDownloadedVideoSelect.value = selectedDownloadedVideoId;
            }
            const video = getSelectedDownloadedVideo();
            if (video && translateBaseFilenameInput && !translateBaseFilenameInput.dataset.userEdited) {
                const autoBase = deriveBaseFromFilename(video.filename || video.title || video.id);
                if (autoBase) {
                    translateBaseFilenameInput.value = autoBase;
                    translateBaseFilenameInput.dataset.userEdited = 'false';
                }
            }
            if (video && translateAudioInput) {
                translateAudioInput.value = '';
            }
            if (video && (currentChunkSessionId || translateChunkSessions.length)) {
                resetChunkResults();
            }
            if (video) {
                const reuseCheckbox = document.getElementById('translateReuseSeparation');
                if (reuseCheckbox) {
                    reuseCheckbox.checked = false;
                }
            }
            updateDownloadedVideoHint();
            updateAudioInputRequirement();
            updateFfmpegCommands();
        }

        async function fetchVideoInfo() {
            const urlInput = document.getElementById('videoDownloadUrl');
            const infoDiv = document.getElementById('videoDownloadInfo');
            const url = urlInput ? urlInput.value.trim() : '';
            if (!url) {
                showStatus('Enter a video URL first.', 'error', 'videoDownloadStatus');
                return;
            }
            showStatus('Fetching video info...', 'info', 'videoDownloadStatus');
            try {
                const response = await fetch(ENDPOINTS.VIDEO_INFO, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url }),
                });
                const data = await response.json();
                if (!response.ok || data.status === 'error') {
                    throw new Error(data.message || data.error || `Info failed (${response.status})`);
                }
                if (infoDiv) {
                    const formats = Array.isArray(data.formats) ? data.formats.slice(0, 8) : [];
                    infoDiv.innerHTML = `
                            <div class="segment-card" style="padding:14px;">
                                <div class="segment-header">${escapeHtml(data.title || 'Video')}</div>
                                <small style="color:var(--text-muted);">${escapeHtml(data.duration_label || '')}</small>
                                ${data.thumbnail ? `<img src="${data.thumbnail}" alt="" style="width:100%;max-height:260px;object-fit:cover;border-radius:8px;margin-top:10px;">` : ''}
                                ${formats.length ? `<div style="margin-top:10px;color:var(--text-muted);font-size:0.82rem;">${formats.map(fmt => escapeHtml([fmt.resolution, fmt.ext, fmt.note].filter(Boolean).join(' / '))).join('<br>')}</div>` : ''}
                                ${data.format_warning ? `<div style="margin-top:10px;color:var(--error);font-size:0.82rem;">${escapeHtml(data.format_warning)}</div>` : ''}
                            </div>
                        `;
                }

                const qualitySelect = document.getElementById('videoDownloadQuality');
                if (qualitySelect) {
                    qualitySelect.innerHTML = '';
                    const availableFormats = Array.isArray(data.formats)
                        ? data.formats.filter(fmt => fmt && fmt.format_id)
                        : [];
                    if (availableFormats.length > 0) {
                        availableFormats.forEach(fmt => {
                            const option = document.createElement('option');
                            const isVideo = fmt.vcodec && fmt.vcodec !== 'none';
                            const isAudio = fmt.acodec && fmt.acodec !== 'none';

                            let selector = fmt.format_id;
                            let typeLabel = '';
                            if (isVideo && !isAudio) {
                                selector = `${fmt.format_id}+bestaudio/${fmt.format_id}/best`;
                                typeLabel = '📹 Video';
                            } else if (isAudio && !isVideo) {
                                typeLabel = '🎵 Audio';
                            } else {
                                typeLabel = '🎥 Combined';
                            }

                            option.value = selector;

                            const res = fmt.resolution || '';
                            const ext = fmt.ext || '';
                            const note = fmt.note || '';
                            const size = fmt.filesize ? ` (${(fmt.filesize / (1024 * 1024)).toFixed(1)} MB)` : '';
                            option.textContent = `${typeLabel} - ${res} (${ext}) ${note}${size}`;
                            qualitySelect.appendChild(option);
                        });
                        qualitySelect.selectedIndex = 0;
                    } else {
                        const option = document.createElement('option');
                        option.value = 'best';
                        option.textContent = 'Best available (yt-dlp fallback)';
                        qualitySelect.appendChild(option);
                    }
                }

                showStatus(data.format_warning || 'Video info loaded.', data.format_warning ? 'error' : 'success', 'videoDownloadStatus');
            } catch (error) {
                showStatus(`Failed to fetch info: ${error.message}`, 'error', 'videoDownloadStatus');
            }
        }

        async function handleVideoDownloadSubmit(event) {
            event.preventDefault();
            const urlInput = document.getElementById('videoDownloadUrl');
            const qualitySelect = document.getElementById('videoDownloadQuality');
            const downloadBtn = document.getElementById('videoDownloadBtn');
            const url = urlInput ? urlInput.value.trim() : '';
            const quality = qualitySelect ? qualitySelect.value : 'best';
            if (!url) {
                showStatus('Enter a video URL first.', 'error', 'videoDownloadStatus');
                return;
            }
            if (downloadBtn) {
                downloadBtn.disabled = true;
            }
            showStatus('Starting video download...', 'info', 'videoDownloadStatus');
            try {
                const response = await fetch(ENDPOINTS.VIDEO_DOWNLOAD, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url, quality }),
                });
                if (!response.ok) {
                    const message = await parseHttpError(response, `Download failed (${response.status})`);
                    throw new Error(message);
                }
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let completedVideo = null;
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) {
                        break;
                    }
                    buffer += decoder.decode(value, { stream: true });
                    let newlineIndex = buffer.indexOf('\n');
                    while (newlineIndex !== -1) {
                        const line = buffer.slice(0, newlineIndex).trim();
                        buffer = buffer.slice(newlineIndex + 1);
                        newlineIndex = buffer.indexOf('\n');
                        if (!line) {
                            continue;
                        }
                        const eventData = parseJsonStreamEventLine(line);
                        if (!eventData) {
                            continue;
                        }
                        const eventType = eventData.event || 'status';
                        if (eventType === 'progress') {
                            const percent = typeof eventData.percent === 'number' ? ` ${eventData.percent.toFixed(1)}%` : '';
                            showStatus(`${eventData.message || 'Downloading video...'}${percent}`, 'info', 'videoDownloadStatus');
                        } else if (eventType === 'status') {
                            showStatus(eventData.message || 'Processing video...', 'info', 'videoDownloadStatus');
                        } else if (eventType === 'error') {
                            throw new Error(eventData.message || 'Video download failed.');
                        } else if (eventType === 'complete') {
                            completedVideo = eventData.video || null;
                            showStatus(eventData.message || 'Video downloaded.', 'success', 'videoDownloadStatus');
                        }
                    }
                }
                await loadDownloadedVideos({ selectId: completedVideo ? completedVideo.id : selectedDownloadedVideoId });
            } catch (error) {
                showStatus(`Download failed: ${error.message}`, 'error', 'videoDownloadStatus');
            } finally {
                if (downloadBtn) {
                    downloadBtn.disabled = false;
                }
            }
        }

        // ============================================================================
        // Cookie Management Functions
        // ============================================================================

        async function loadCookieSites() {
            const container = document.getElementById('cookieSitesList');
            if (!container) return;
            try {
                const response = await fetch(ENDPOINTS.COOKIES_LIST, { cache: 'no-cache' });
                const data = await response.json();
                if (data.status !== 'ok' || !data.sites || Object.keys(data.sites).length === 0) {
                    container.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">No cookies saved yet.</span>';
                    return;
                }
                container.innerHTML = Object.entries(data.sites).map(([domain, info]) => {
                    const downloadUrl = info.download_url || `${ENDPOINTS.COOKIES_LIST}/${encodeURIComponent(domain)}/download`;
                    return `
                            <div style="display: flex; align-items: center; gap: 8px; padding: 4px 0;">
                                <span style="color: #51cf66; font-size: 0.88rem;">✓ ${escapeHtml(domain)}</span>
                                <span style="color: var(--text-muted); font-size: 0.8rem;">(${info.count} cookies)</span>
                                <a class="btn btn-secondary" href="${escapeHtml(downloadUrl)}" download style="padding: 2px 8px; font-size: 0.75rem; margin-left: auto;">Backup</a>
                                <button type="button" class="btn btn-danger cookie-delete-btn" data-domain="${escapeHtml(domain)}" style="padding: 2px 8px; font-size: 0.75rem;">✕</button>
                            </div>
                        `;
                }).join('');
                container.querySelectorAll('.cookie-delete-btn').forEach(btn => {
                    btn.addEventListener('click', () => deleteCookieSite(btn.dataset.domain));
                });
            } catch (error) {
                console.warn('Failed to load cookie sites', error);
            }
        }

        async function importCookiesFromCurl() {
            const curlInput = document.getElementById('cookieCurlInput');
            const domainInput = document.getElementById('cookieDomainInput');
            const curlText = curlInput ? curlInput.value.trim() : '';
            const domain = domainInput ? domainInput.value.trim() : '';
            if (!curlText) {
                showStatus('Paste a cURL command first.', 'error', 'cookieImportStatus');
                return;
            }
            showStatus('Importing cookies...', 'info', 'cookieImportStatus');
            try {
                const body = { curl_text: curlText };
                if (domain) body.domain = domain;
                const response = await fetch(ENDPOINTS.COOKIES_IMPORT_CURL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                const data = await response.json();
                if (!response.ok || data.status === 'error') {
                    throw new Error(data.message || data.error || 'Import failed');
                }
                showStatus(data.message || 'Cookies imported!', 'success', 'cookieImportStatus');
                if (curlInput) curlInput.value = '';
                if (domainInput) domainInput.value = '';
                loadCookieSites();
            } catch (error) {
                showStatus(`Import failed: ${error.message}`, 'error', 'cookieImportStatus');
            }
        }

        async function uploadCookiesFile() {
            const fileInput = document.getElementById('cookieFileInput');
            const domainInput = document.getElementById('cookieUploadDomainInput');
            const file = fileInput ? fileInput.files[0] : null;
            const domain = domainInput ? domainInput.value.trim() : '';

            if (!file) {
                showStatus('Select a cookies.txt file first.', 'error', 'cookieUploadStatus');
                return;
            }
            if (!domain) {
                showStatus('Enter the domain for these cookies.', 'error', 'cookieUploadStatus');
                return;
            }

            showStatus('Uploading cookies...', 'info', 'cookieUploadStatus');
            try {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('domain', domain);

                const response = await fetch(ENDPOINTS.COOKIES_UPLOAD, {
                    method: 'POST',
                    body: formData,
                });
                const data = await response.json();
                if (!response.ok || data.status === 'error') {
                    throw new Error(data.message || data.error || 'Upload failed');
                }
                showStatus(data.message || 'Cookies uploaded successfully!', 'success', 'cookieUploadStatus');
                if (fileInput) fileInput.value = '';
                if (domainInput) domainInput.value = '';
                loadCookieSites();
            } catch (error) {
                showStatus(`Upload failed: ${error.message}`, 'error', 'cookieUploadStatus');
            }
        }

        function detectCookieDomain() {
            const curlInput = document.getElementById('cookieCurlInput');
            const domainInput = document.getElementById('cookieDomainInput');
            const curlText = curlInput ? curlInput.value.trim() : '';
            if (!curlText) return;
            // Simple client-side domain detection from URL in cURL
            const urlMatch = curlText.match(/https?:\/\/(?:www\.)?([^\s\/'";\ ]+)/);
            if (urlMatch) {
                let domain = urlMatch[1];
                const parts = domain.split('.');
                if (parts.length > 2) domain = parts.slice(-2).join('.');
                if (domainInput) domainInput.value = domain;
            }
        }

        async function deleteCookieSite(domain) {
            if (!confirm(`Delete cookies for ${domain}?`)) return;
            try {
                const response = await fetch(`${ENDPOINTS.COOKIES_LIST}/${encodeURIComponent(domain)}`, {
                    method: 'DELETE',
                });
                const data = await response.json();
                if (!response.ok || data.status === 'error') {
                    throw new Error(data.message || 'Delete failed');
                }
                loadCookieSites();
            } catch (error) {
                console.error('Failed to delete cookies:', error);
            }
        }

        // ============================================================================
        // Video Delete Function
        // ============================================================================

        async function deleteDownloadedVideo(videoId, videoTitle) {
            if (!videoId) return;
            const displayName = videoTitle || videoId;
            if (!confirm(`Delete video "${displayName}"? This cannot be undone.`)) return;
            try {
                const response = await fetch(`${ENDPOINTS.DOWNLOADED_VIDEOS}/${encodeURIComponent(videoId)}`, {
                    method: 'DELETE',
                });
                const data = await response.json();
                if (!response.ok || data.status === 'error') {
                    throw new Error(data.message || 'Delete failed');
                }
                showStatus(data.message || 'Video deleted.', 'success', 'videoDownloadStatus');
                await loadDownloadedVideos();
            } catch (error) {
                showStatus(`Delete failed: ${error.message}`, 'error', 'videoDownloadStatus');
            }
        }

        async function deleteTranslatedVideo(videoId, videoTitle) {
            if (!videoId) return;
            const displayName = videoTitle || videoId;
            if (!confirm(`Delete translated video "${displayName}"? This cannot be undone.`)) return;
            try {
                const response = await fetch(`${ENDPOINTS.TRANSLATED_VIDEOS}/${encodeURIComponent(videoId)}`, {
                    method: 'DELETE',
                });
                const data = await response.json();
                if (!response.ok || data.status === 'error') {
                    throw new Error(data.message || 'Delete failed');
                }
                showStatus(data.message || 'Translated video deleted.', 'success', 'videoDownloadStatus');
                await loadTranslatedVideos();
            } catch (error) {
                showStatus(`Delete failed: ${error.message}`, 'error', 'videoDownloadStatus');
            }
        }
