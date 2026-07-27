"""Application-layer package for the IndexTTS vLLM web service.

Importing this package is intentionally side-effect free. Use
``indextts_web.app.create_app`` to assemble the FastAPI application or
``indextts_web.main.main`` to start the server.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"

