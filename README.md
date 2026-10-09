中文 | [English](README_EN.md)

# IndexTTS-vLLM

基于 vLLM 的 IndexTTS 2.0 语音工作台，支持音色预设、情绪控制、流式合成、
语音翻译和分段编辑。可选集成包括 IndexTTS 2.5、Confucius4-TTS、Qwen3-TTS
声音设计、Stable Audio 3 音乐/音效、视频下载和参考音频增强。

## 快速开始

Google Colab 用户可打开 [index_tts_vllm_colab.ipynb](index_tts_vllm_colab.ipynb)，
选择 **G4 GPU** 后按顺序运行。笔记本会克隆本仓库、创建独立 Python 3.12/CUDA
环境、下载 IndexTTS 2.0 权重，并通过临时 Cloudflare 链接访问 WebUI。
默认按快速启动流程准备主环境、IndexTTS 2.0 与 HY-MT 权重及可选后端源码。
可选服务在首次使用时安装依赖、下载权重；提前准备服务和额外模型的选项均默认关闭，
需要首次进入即可使用某项功能时，可在运行前勾选对应选项。Google Drive 存储默认关闭。
预热与 S2Mel 编译共用开关，也默认关闭。服务选项、凭据要求及验证范围见
[Colab 环境与部署对照](docs/colab_setup_parity.md)。

自动安装脚本面向 Linux 和 NVIDIA CUDA GPU。它会安装音频工具和 Python
依赖，下载 IndexTTS 2.0 与 HY-MT 翻译权重，准备可选后端仓库，然后在
`http://localhost:8000` 启动 WebUI：

```bash
git clone https://github.com/keboqi/index-tts-vllm.git
cd index-tts-vllm
EXPORT_TUNNEL=0 bash quickstart.sh
```

使用 `bash quickstart.sh --setup-only` 只安装、不启动服务。常用变量：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `VENV_DIR` / `PYTHON_VERSION` | `.venv` / `3.12` | 主 Python 环境 |
| `MODEL_DIR` | `checkpoints` | IndexTTS 2.0 权重目录 |
| `INSTALL_CONFUCIUS` / `INSTALL_INDEXTTS25` | `1` / `1` | 设为 `0` 跳过相应可选后端仓库 |
| `UPDATE_EXTERNAL_REPOS` | `1` | 设为 `0` 保留已有外部仓库版本 |
| `DOWNLOAD_MODEL` / `DOWNLOAD_HY_MT_MODEL` | `1` / `1` | 控制权重下载 |
| `SERVER_PORT` | `8000` | WebUI/API 端口 |
| `EXPORT_TUNNEL` | `1` | 设为 `0` 禁用可选 Cloudflare 隧道 |

已有模型环境可直接启动：

```bash
python fastapi_webui_v2.py --model_dir checkpoints --host 0.0.0.0 --port 8000
```

显存预算、批处理和合成并发根据检测到的 VRAM 自动配置；显式 CLI/环境变量
可覆盖自动值。使用 `--use_torch_compile` / `--no-use_torch_compile` 控制编译。
完整参数见 [indextts_web/config.py](indextts_web/config.py)。

## 部署

Modal 部署在 [deploy_vllm_indextts_v2.py](deploy_vllm_indextts_v2.py) 中手动修改
`IndexTTSVllmServer` 的 `gpu=`，选择 `"L4"`、`"L40S"` 或 `"RTX-PRO-6000"`。
新建持久化卷时先准备模型，再部署：

```bash
modal deploy deploy_vllm_indextts_v2.py
```

直接打开 Modal 输出的 `prepare_model` Web 地址，点击 **Initialize** 准备缺失的模型文件。
页面也支持独立的仓库更新（Update to latest）和模型下载/续传（Download / resume）。
已有模型无需重新准备；空 Volume 准备完成后再打开 GPU Studio。
组件依赖仍在 Modal 镜像构建时安装，修改依赖后重新部署即可，无需额外的环境卷。
页面显示文件是否已下载；其他模型仍按需加载到 GPU，实际加载状态见 Studio 的模型管理器。
受限模型需要同一 Secret 中具有仓库访问权限的 `HF_TOKEN`。

初始化和更新在同一个 CPU Web 函数内执行，页面显示进度和错误。
配置完成后再次部署，以重新生成 GPU 快照并应用更新。也可以通过
`modal serve deploy_vllm_indextts_v2.py` 临时打开管理页面。已移除清缓存功能。

卷、Secret、自动参数、覆盖方式、模型管理和快照验证流程统一见
[GPU_DEPLOYMENT.md](GPU_DEPLOYMENT.md)。CPU 测试覆盖配置与启动命令；
各 GPU 的实际冷启动、推理峰值和快照恢复仍需测量验证。

Docker 使用 [Dockerfile](Dockerfile)、[docker-compose.yaml](docker-compose.yaml)
和 [entrypoint.sh](entrypoint.sh)。运行 `docker compose up --build` 前检查
`.env.example`，默认访问 `http://localhost:8000`。现代 WebUI 使用
`APP_SERVER=web` 和已转换的 IndexTTS 2.0
权重；旧版 API 使用 `APP_SERVER=legacy-api` 和 IndexTTS 1.x 权重。
可选功能需另外安装依赖和权重。
WebUI 端口应与托管后端区分（Confucius 默认 `8001`，IndexTTS 2.5 默认
`8092`）；若占用这些端口，需同时调整对应后端端口参数。

## 可选后端与功能

默认后端是 `index`（IndexTTS 2.0）。可以在 UI、单次 API 请求或
`--tts_backend` 参数中选择 `index25` 或 `confucius`。

| 后端 | 仓库与环境 | 本地 API | 说明 |
| --- | --- | --- | --- |
| IndexTTS 2.0 | 本仓库主环境 | 主 WebUI | 情绪文本/音频、时长控制、分块流式合成 |
| IndexTTS 2.5 | `../index-tts-2.5-vllm-omni-experiment`，独立 Python 3.11 环境 | `127.0.0.1:8092` | 首次请求自动准备并启动；支持中、英、日、西班牙、阿拉伯语 |
| Confucius4-TTS | `../Confucius4-TTS`，独立后端启动器 | `127.0.0.1:8001` | 首次请求启动；多语言合成；忽略 IndexTTS 情绪文本控制 |

自定义路径使用 `--indextts25_repo_dir` / `--confucius_repo_dir`，自定义服务
启动可使用对应的 `--*_start_command` 和超时参数。外部后端冷启动期间输出
keepalive 帧；IndexTTS 2.5 模型本身非流式，完成后输出整段音频。
切换托管后端会休眠或停止其他 TTS 引擎，但不会卸载所有辅助模型。

翻译支持 MOSS Transcribe+Diarize（默认）、Gemini、WhisperX、
Qwen3-ASR + OmniVAD 和 NVIDIA Parakeet。本地 MOSS Docker 服务可提前准备：

```bash
bash sglang_omni_moss_transcribe.sh deploy
```

首次转录请求会按需启动。Modal 使用独立的
[moss_transcribe_server.py](moss_transcribe_server.py) 服务。

手动安装时，根据使用的功能安装 `requirements-optional.txt`。
Qwen3-ASR 与 Qwen3-TTS 的 Transformers 依赖版本冲突，需要独立环境，并通过
`QWEN_OMNIVAD_PYTHON` 指定其 Python。不要把 `qwen-asr[vllm]` 装入固定使用
`vllm==0.10.2` 的主环境。Modal 已配置独立 ASR 环境。
Stable Audio 3 的安装和权重路径见
[英文安装说明](README_EN.md#stable-audio-3)，运行时从本地权重目录加载。

## API 与开发

启动后的 `/docs`、`/openapi.json` 和 WebUI 的 API 页提供当前接口说明。
主要合成入口是 `/speak`、`/clone_voice` 和对应的 `_stream` 接口。
旧版 IndexTTS 1.x 的 [api_server.py](api_server.py) 单独提供 OpenAI 兼容的
`/audio/speech`、`/audio/voices`。先在 UI 创建 `my_speaker_preset`，再调用：

```bash
curl --fail http://127.0.0.1:8000/speak \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，欢迎使用 IndexTTS。","name":"my_speaker_preset","tts_backend":"index"}' \
  --output output.mp3
```

合成流使用二进制帧：`CHUNK:{idx}:{size}:{MORE|LAST}\n{audio_bytes}`，
可包含 `KEEPALIVE:{size}\n{json}`。客户端应缓存跨网络读取的部分头部/负载，
按声明字节数读取，不能当作 SSE 解析。翻译进度接口使用 SSE。

目录结构、兼容规则和旧入口说明见 [ARCHITECTURE.md](ARCHITECTURE.md)。

开发和部署请从仓库目录运行。下面的可编辑安装用于提供开发工具；模型代码和
应用资源依赖仓库目录结构，目前不支持通过独立 wheel 部署。

```bash
pip install -e '.[dev]'
python -m unittest discover -s tests -v
ruff check indextts_web tests fastapi_webui_v2.py
python -m compileall -q indextts_web tests fastapi_webui_v2.py fastapi_webui_v2_impl.py
```

CPU 测试不需要 CUDA 或模型权重。前端脚本通过 `node --check` 检查语法；
发布前还需在 GPU 环境验证真实推理、流式合成、后端切换和快照恢复。
