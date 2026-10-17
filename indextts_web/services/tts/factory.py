from types import ModuleType

from .confucius import ConfuciusBackend
from .drama3 import Drama3Backend
from .index import IndexBackend
from .index25 import IndexTTS25Backend
from .registry import BackendRegistry


def build_backend_registry(legacy: ModuleType) -> BackendRegistry:
    return BackendRegistry(
        [Drama3Backend(legacy), IndexBackend(legacy), IndexTTS25Backend(legacy), ConfuciusBackend(legacy)],
        default=legacy.SETTINGS.tts_backend,
    )
