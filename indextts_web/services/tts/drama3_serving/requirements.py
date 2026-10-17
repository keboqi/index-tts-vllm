"""Pinned isolated runtime from minimax_h3 (not installed in the web app venv)."""
from pathlib import Path, PurePath

DRAMA3_SGLANG_TORCH_BACKEND = "cu130"
DRAMA3_SGLANG_PYTHON = "3.12"
DRAMA3_SGLANG_OMNI_REPO = "https://github.com/sgl-project/sglang-omni.git"
DRAMA3_SGLANG_OMNI_REF = "caae7eb18d8c8d22643d4cf8e377226a3b1259e9"
DRAMA3_HF_REPO = "fishaudio/drama-3"
DRAMA3_HF_REVISION = "e0a9218aa48888270f06ef1ed9ea67622c0e17d2"

DRAMA3_SGLANG_READY_MARKER = "h3-drama3-ready.txt"

DRAMA3_SGLANG_SMOKE = (
    "import importlib.metadata as m, torch; "
    "assert torch.version.cuda, 'SGLang-Omni installed CPU-only Torch'; "
    "print('sglang', m.version('sglang'), 'sglang-omni', m.version('sglang-omni'), 'cuda', torch.version.cuda)"
)


# Mirrors sglang-omni's [tool.uv] override-dependencies for Linux.
DRAMA3_SGLANG_OVERRIDES = "protobuf>=6.31.1,<7.0.0\n"


def drama3_sglang_fingerprint() -> str:
    """Identifies an installed SGLang-Omni environment built from these exact pins."""
    import hashlib

    pins = "|".join((DRAMA3_SGLANG_PYTHON, DRAMA3_SGLANG_OMNI_REPO, DRAMA3_SGLANG_OMNI_REF,
                     DRAMA3_SGLANG_TORCH_BACKEND, DRAMA3_SGLANG_OVERRIDES))
    return hashlib.sha256(pins.encode("utf-8")).hexdigest()


def drama3_sglang_install_commands(venv: str | Path, uv: str | list[str] = "uv") -> list[list[str]]:
    """Install steps for the SGLang-Omni Drama 3 environment (image build and first use)."""
    base = venv if isinstance(venv, PurePath) else Path(venv)
    python = str(base / "bin" / "python")
    overrides = str(base / "h3-overrides.txt")
    uv = [uv] if isinstance(uv, str) else list(uv)
    return [
        [*uv, "venv", "--python", DRAMA3_SGLANG_PYTHON, "--seed", str(base)],
        # The [fishaudio] extra's descript-audiotools pins protobuf<3.20, while SGLang
        # needs >=6.30. The project overrides it in [tool.uv], which uv ignores for a
        # git-URL install, so apply the same override explicitly.
        [python, "-c", f"open({overrides!r}, 'w').write({DRAMA3_SGLANG_OVERRIDES!r})"],
        # SGLang pins Torch 2.13; name the CUDA wheel because builders have no GPU.
        [*uv, "pip", "install", "--python", python, "--prerelease=allow",
         f"--torch-backend={DRAMA3_SGLANG_TORCH_BACKEND}", "--overrides", overrides,
         f"sglang-omni[fishaudio] @ git+{DRAMA3_SGLANG_OMNI_REPO}@{DRAMA3_SGLANG_OMNI_REF}"],
        [python, "-c", DRAMA3_SGLANG_SMOKE],
    ]

