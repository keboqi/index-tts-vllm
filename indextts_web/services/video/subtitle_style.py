"""CJK font discovery and FFmpeg ``subtitles`` filter arguments for burned-in subtitles."""

from __future__ import annotations

import os

from ...infrastructure.ffmpeg import (
    escape_filter_path,
    escape_filter_quoted_value,
    ffprobe_video_dimensions,
)


def existing_path_or_none(path: str | None) -> str | None:
    if not path:
        return None
    normalized = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
    return normalized if os.path.exists(normalized) else None


def infer_font_name_from_path(path: str | None) -> str | None:
    if not path:
        return None
    lower_name = os.path.basename(path).lower()
    if "msyh" in lower_name or "yahei" in lower_name:
        return "Microsoft YaHei"
    if "simhei" in lower_name:
        return "SimHei"
    if "simsun" in lower_name:
        return "SimSun"
    if "notosanscjk" in lower_name or "noto sans cjk" in lower_name:
        return "Noto Sans CJK SC"
    if "notosanssc" in lower_name or "noto sans sc" in lower_name:
        return "Noto Sans SC"
    if "wqy" in lower_name or "wenquanyi" in lower_name:
        return "WenQuanYi Micro Hei"
    if "sourcehansans" in lower_name or "source han sans" in lower_name:
        return "Source Han Sans SC"
    return None


def font_candidates(local_fonts: str) -> list[tuple[str, list[str]]]:
    """Known CJK fonts in preference order: bundled fonts first, then system locations."""
    bundled_noto_fonts = os.path.join(local_fonts, "noto-cjk")
    return [
        (
            "Noto Sans CJK SC",
            [
                os.path.join(bundled_noto_fonts, "NotoSansCJKsc-Regular.otf"),
                os.path.join(local_fonts, "NotoSansCJK-Regular.ttc"),
                os.path.join(local_fonts, "NotoSansCJKsc-Regular.otf"),
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
                "/usr/local/share/fonts/NotoSansCJK-Regular.ttc",
            ],
        ),
        (
            "Noto Sans SC",
            [
                os.path.join(local_fonts, "NotoSansSC-Regular.otf"),
                "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
                "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf",
                "/usr/local/share/fonts/NotoSansSC-Regular.otf",
            ],
        ),
        (
            "Source Han Sans SC",
            [
                os.path.join(local_fonts, "SourceHanSansSC-Regular.otf"),
                "/usr/share/fonts/opentype/adobe-source-han-sans/SourceHanSansSC-Regular.otf",
                "/usr/local/share/fonts/SourceHanSansSC-Regular.otf",
            ],
        ),
        (
            "WenQuanYi Micro Hei",
            [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc",
            ],
        ),
        (
            "Microsoft YaHei",
            [
                r"C:\Windows\Fonts\msyh.ttc",
                r"C:\Windows\Fonts\msyh.ttf",
                r"C:\Windows\Fonts\msyhbd.ttc",
            ],
        ),
        ("SimHei", [r"C:\Windows\Fonts\simhei.ttf"]),
        ("SimSun", [r"C:\Windows\Fonts\simsun.ttc"]),
        (
            "PingFang SC",
            [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
            ],
        ),
    ]


class SubtitleStyle:
    """Configured subtitle font plus size scaling for the target video resolution."""

    def __init__(
        self,
        *,
        font_name: str,
        font_file: str,
        fonts_dir: str,
        font_size: int,
        local_fonts_dir: str,
    ) -> None:
        self.font_name = font_name
        self.font_file = font_file
        self.fonts_dir = fonts_dir
        self.font_size = font_size
        self.local_fonts_dir = local_fonts_dir
        self._resolved_font: tuple[str | None, str | None] | None = None

    def resolve_font(self) -> tuple[str | None, str | None]:
        """(font name, fonts dir) for libass; resolved once per process."""
        if self._resolved_font is None:
            self._resolved_font = self._resolve_font()
        return self._resolved_font

    def _resolve_font(self) -> tuple[str | None, str | None]:
        configured_font_name = self.font_name or None
        configured_fonts_dir = existing_path_or_none(self.fonts_dir)
        configured_font_file = existing_path_or_none(self.font_file)

        if configured_font_file:
            return (
                configured_font_name or infer_font_name_from_path(configured_font_file),
                os.path.dirname(configured_font_file),
            )
        if self.font_file:
            print(f"⚠️ VIDEO_SUBTITLE_FONT_FILE does not exist: {self.font_file}")

        if configured_font_name:
            return configured_font_name, configured_fonts_dir

        for font_name, candidate_paths in font_candidates(self.local_fonts_dir):
            for candidate_path in candidate_paths:
                font_path = existing_path_or_none(candidate_path)
                if font_path:
                    return font_name, os.path.dirname(font_path)

        print(
            "⚠️ No CJK subtitle font found for FFmpeg rendering. "
            "Install a CJK font such as Noto Sans CJK, or set VIDEO_SUBTITLE_FONT_FILE "
            "and optionally VIDEO_SUBTITLE_FONT."
        )
        return None, configured_fonts_dir

    def metrics_for_video(self, video_path: str | None) -> tuple[int, float, float, int]:
        """(font size, outline, shadow, vertical margin) scaled to the video's short side."""
        dimensions = ffprobe_video_dimensions(video_path) if video_path else None
        if not dimensions:
            return self.font_size, 1.5, 0.7, 36
        width, height = dimensions
        short_side = max(1, min(width, height))
        scale = short_side / 1080.0
        font_size = int(round(self.font_size * scale))
        font_size = max(12, min(self.font_size, font_size))
        outline = round(max(0.8, min(1.5, 1.5 * scale)), 2)
        shadow = round(max(0.35, min(0.7, 0.7 * scale)), 2)
        margin_v = int(round(36 * scale))
        margin_v = max(14, min(36, margin_v))
        return font_size, outline, shadow, margin_v

    def force_style(self, font_name: str | None, video_path: str | None = None) -> str:
        font_size, outline, shadow, margin_v = self.metrics_for_video(video_path)
        style_parts = [
            f"FontSize={font_size}",
            "PrimaryColour=&H00FFFFFF",
            "OutlineColour=&H00000000",
            "BorderStyle=1",
            f"Outline={outline}",
            f"Shadow={shadow}",
            f"MarginV={margin_v}",
        ]
        if font_name:
            style_parts.insert(0, f"FontName={font_name}")
        return ",".join(style_parts)

    def filter_arg(self, subtitle_path: str, video_path: str | None = None) -> str:
        font_name, fonts_dir = self.resolve_font()
        options = [f"filename='{escape_filter_path(subtitle_path)}'"]
        ext = os.path.splitext(subtitle_path)[1].lstrip(".").lower()
        if ext in {"srt", "vtt"}:
            options.append("charenc=UTF-8")
        if fonts_dir:
            options.append(f"fontsdir='{escape_filter_path(fonts_dir)}'")
        force_style = self.force_style(font_name, video_path)
        if force_style:
            options.append(f"force_style='{escape_filter_quoted_value(force_style)}'")
        return "subtitles=" + ":".join(options)
