# Legacy entry-point audit

The supported WebUI/API command is:

```bash
python fastapi_webui_v2.py
```

The following files remain for compatibility and are not part of the modular
application assembly:

| File | Status | Replacement |
| --- | --- | --- |
| `api_server.py` | Legacy IndexTTS API; retained for old clients and Docker `APP_SERVER=legacy-api` | `fastapi_webui_v2.py` |
| `webui.py` | Original Gradio UI | Modern HTML WebUI |
| `webui_v2.py` | IndexTTS2 Gradio prototype | Modern HTML WebUI |
| `webui_with_presets.py` | Preset-enabled Gradio prototype | Speaker routes and modern HTML WebUI |
| `templates/index.html` | Earlier HTML template; not served | `index_new.html` plus `static/` |
| `simple_test.py` | Manual HTTP load generator, not an automated test | `tests/` for contracts; retain this file for benchmarking |

No new feature work should target these entry points. Remove one only after
checking deployment commands and downstream clients for direct references.

