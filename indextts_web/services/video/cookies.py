"""Parse cookie exports and locate saved Netscape cookie files for video URLs."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any


def extract_domain_from_url(url: str) -> str:
    """Extract the main domain from a video URL."""
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    if match:
        domain = match.group(1)
        parts = domain.split(".")
        if len(parts) > 2:
            domain = ".".join(parts[-2:])
        if domain == "youtu.be":
            domain = "youtube.com"
        return domain
    return "unknown"


def extract_domain_from_curl(curl_text: str) -> str:
    """Extract the main domain from a cURL command."""
    url_match = re.search(r"curl\s+['\"]?(https?://[^'\"\s]+)", curl_text)
    if not url_match:
        url_match = re.search(r"https?://([^\s/'\"]+)", curl_text)
    if url_match:
        url = url_match.group(1) if "://" in url_match.group(1) else url_match.group(0)
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        if domain_match:
            domain = domain_match.group(1)
            parts = domain.split(".")
            if len(parts) > 2:
                domain = ".".join(parts[-2:])
            if domain == "youtu.be":
                domain = "youtube.com"
            return domain
    return "unknown"


def parse_curl_cookies(curl_text: str) -> dict[str, str]:
    """Parse cookies from a cURL command string."""
    cookies: dict[str, str] = {}
    patterns = [
        r"-H\s+['\"]Cookie:\s*([^'\"]+)['\"]",
        r"--header\s+['\"]Cookie:\s*([^'\"]+)['\"]",
        r"-b\s+['\"]([^'\"]+)['\"]",
        r"--cookie\s+['\"]([^'\"]+)['\"]",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, curl_text, re.IGNORECASE)
        for match in matches:
            pairs = match.split(";")
            for pair in pairs:
                pair = pair.strip()
                if "=" in pair:
                    name, value = pair.split("=", 1)
                    cookies[name.strip()] = value.strip()
    return cookies


def normalize_cookie_domain(domain: str) -> str:
    normalized = (domain or "").strip().lower()
    if normalized.startswith("www."):
        normalized = normalized[4:]
    normalized = normalized.strip(".")
    if not normalized:
        raise ValueError("Domain is required.")
    if not re.fullmatch(r"[a-z0-9.-]+", normalized):
        raise ValueError("Invalid domain.")
    return normalized


def cookie_filename_for_domain(domain: str) -> str:
    normalized = normalize_cookie_domain(domain)
    return normalized.replace(".", "_") + "_cookies.txt"


class CookieStore:
    """Saved ``<domain>_cookies.txt`` files: one writable directory plus read-only fallbacks."""

    def __init__(self, cookies_dir: str, fallback_dirs: Iterable[str] = ()) -> None:
        self.cookies_dir = cookies_dir
        self._dirs: list[str] = []
        for candidate in (cookies_dir, *fallback_dirs):
            normalized = os.path.abspath(candidate)
            if normalized not in self._dirs:
                self._dirs.append(normalized)

    def candidate_dirs(self) -> list[str]:
        """Cookie directories from every possible output root, deduped."""
        return list(self._dirs)

    def saved_sites(self) -> dict[str, int]:
        """Saved cookie sites with their cookie counts."""
        sites: dict[str, int] = {}
        for cookies_dir in self._dirs:
            if not os.path.exists(cookies_dir):
                continue
            for f in os.listdir(cookies_dir):
                if f.endswith("_cookies.txt"):
                    domain = f.replace("_cookies.txt", "").replace("_", ".")
                    filepath = os.path.join(cookies_dir, f)
                    try:
                        with open(filepath, encoding="utf-8") as fh:
                            lines = [line for line in fh.readlines() if line.strip() and not line.startswith("#")]
                            sites[domain] = max(sites.get(domain, 0), len(lines))
                    except Exception:
                        sites.setdefault(domain, 0)
        return sites

    def path_for_domain(self, domain: str, cookies_dir: str | None = None) -> str:
        filename = cookie_filename_for_domain(domain)
        root = os.path.abspath(cookies_dir or self.cookies_dir)
        path = os.path.abspath(os.path.join(root, filename))
        if os.path.commonpath([root, path]) != root:
            raise ValueError("Invalid cookie path.")
        return path

    def find_saved_file_for_domain(self, domain: str) -> tuple[str, str]:
        filename = cookie_filename_for_domain(domain)
        for root in self._dirs:
            path = os.path.abspath(os.path.join(root, filename))
            if os.path.commonpath([root, path]) == root and os.path.exists(path):
                return path, normalize_cookie_domain(domain)
        raise FileNotFoundError("No cookies found for this domain.")

    def file_for_url(self, url: str) -> str | None:
        """The cookie file for a video URL: exact domain first, then subdomain matches."""
        domain = extract_domain_from_url(url)
        filename = domain.replace(".", "_") + "_cookies.txt"
        for cookies_dir in self._dirs:
            filepath = os.path.join(cookies_dir, filename)
            if os.path.exists(filepath):
                return filepath
        for cookies_dir in self._dirs:
            if not os.path.exists(cookies_dir):
                continue
            for f in os.listdir(cookies_dir):
                if f.endswith("_cookies.txt"):
                    saved_domain = f.replace("_cookies.txt", "").replace("_", ".")
                    if saved_domain in domain or domain in saved_domain:
                        return os.path.join(cookies_dir, f)
        return None

    def ytdlp_opts_for_url(self, url: str) -> dict[str, Any]:
        """yt-dlp cookie options for a URL, or an empty dict."""
        cookie_file = self.file_for_url(url)
        return {"cookiefile": cookie_file} if cookie_file else {}
