from types import ModuleType

from .confucius import ConfuciusBackend
from .higgs import HiggsBackend
from .index import IndexBackend
from .registry import BackendRegistry


def build_backend_registry(legacy: ModuleType) -> BackendRegistry:
    return BackendRegistry(
        [IndexBackend(legacy), ConfuciusBackend(legacy), HiggsBackend(legacy)],
        default=legacy.SETTINGS.tts_backend,
    )

