<a href="README.md">中文</a> ｜ <a href="README_EN.md">English</a>

<div align="center">

# IndexTTS-vLLM
</div>

## Modal GPU 选择

在 [`deploy_vllm_indextts_v2.py`](deploy_vllm_indextts_v2.py) 的
`IndexTTSVllmServer` 装饰器中，手动将 `gpu=` 设置为 `"L4"`（24 GB）、
`"L40S"`（48 GB）或 `"RTX-PRO-6000"`（96 GB），然后运行
`modal deploy deploy_vllm_indextts_v2.py`。GPU 型号由部署者手动选择。
容器启动后根据实际显存自动调整 vLLM 显存比例、批处理、编译和推理并发参数，
无需随 GPU 手动修改这些参数。

具体参数、覆盖方式和测试流程见 [GPU 部署说明](GPU_DEPLOYMENT.md)。
小显存配置已实现并通过 CPU 和启动命令测试；实际 Modal GPU 冷启动、推理峰值
及快照恢复仍待验证。现代 `fastapi_webui_v2.py`、quickstart 和 Docker WebUI
同样使用自动配置；下文旧版 IndexTTS 1.x API 的启动方式保持不变。

Modal 会为 Qwen3-ASR 配置单独的 Python 环境，并通过
`QWEN_OMNIVAD_PYTHON` 调用工作进程，避免其 Transformers 依赖影响 TTS。
此更新需要重新部署以构建镜像；无需为安装依赖重新运行 `prepare_model`。

Working on IndexTTS2 support, coming soon... 0.0

## 项目简介
该项目在 [index-tts](https://github.com/index-tts/index-tts) 的基础上使用 vllm 库重新实现了 gpt 模型的推理，加速了 index-tts 的推理过程。

推理速度在单卡 RTX 4090 上的提升为：
- 单个请求的 RTF (Real-Time Factor)：≈0.3 -> ≈0.1
- 单个请求的 gpt 模型 decode 速度：≈90 token / s -> ≈280 token / s
- 并发量：gpu_memory_utilization设置为0.15时，可按实际显存容量测试并发量（测速脚本参考 `simple_test.py`）

## 新特性
- 支持多角色音频混合：可以传入多个参考音频，TTS 输出的角色声线为多个参考音频的混合版本（输入多个参考音频会导致输出的角色声线不稳定，可以抽卡抽到满意的声线再作为参考音频）

## 性能
Word Error Rate (WER) Results for IndexTTS and Baseline Models on the [**seed-test**](https://github.com/BytedanceSpeech/seed-tts-eval)

| model                   | zh    | en    |
| ----------------------- | ----- | ----- |
| Human                   | 1.254 | 2.143 |
| index-tts (num_beams=3) | 1.005 | 1.943 |
| index-tts (num_beams=1) | 1.107 | 2.032 |
| index-tts-vllm      | 1.12  | 1.987 |

基本保持了原项目的性能

## 更新日志

- **[2025-08-07]** 支持 Docker 全自动化一键部署 API 服务：`docker compose up`

- **[2025-08-06]** 支持 openai 接口格式调用：
    1. 添加 /audio/speech api 路径，兼容 OpenAI 接口
    2. 添加 /audio/voices api 路径， 获得 voice/character 列表
    - 对应：[createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

- **[2025-09-22]** 支持了 vllm v1 版本，IndexTTS2 正在兼容中

## 使用步骤

### 1. git 本项目
```bash
git clone https://github.com/Ksuriuri/index-tts-vllm.git
cd index-tts-vllm
```


### 2. 创建并激活 conda 环境
```bash
conda create -n index-tts-vllm python=3.12
conda activate index-tts-vllm
```


### 3. 安装 pytorch

需要 pytorch 版本 2.8.0（对应 vllm 0.10.2），具体安装指令请参考：[pytorch 官网](https://pytorch.org/get-started/locally/)


### 4. 安装依赖
```bash
pip install -r requirements.txt
```


### 5. 下载模型权重

此为官方权重文件，下载到本地任意路径即可，支持 IndexTTS-1.5 的权重

| **HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |
| [😁IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |

### 6. 模型权重转换

```bash
bash convert_hf_format.sh /path/to/your/model_dir
```

此操作会将官方的模型权重转换为 transformers 库兼容的版本，保存在模型权重路径下的 `vllm` 文件夹中，方便后续 vllm 库加载模型权重

### 7. webui 启动！
将 [`webui.py`](webui.py) 中的 `model_dir` 修改为模型权重下载路径，然后运行：

```bash
python webui.py
```
第一次启动可能会久一些，因为要对 bigvgan 进行 cuda 核编译


## API

使用 fastapi 封装了 api 接口，启动示例如下，请将 `--model_dir` 改为你的模型的实际路径：

```bash
python api_server.py --model_dir /your/path/to/Index-TTS
```

### 启动参数
- `--model_dir`: 必填，模型权重路径
- `--host`: 服务ip地址，默认为 `6006`
- `--port`: 服务端口，默认为 `0.0.0.0`
- `--gpu_memory_utilization`: IndexTTS2 vLLM 显存占用率，默认设置为 `0.15`
- `--qwenemo_gpu_memory_utilization`: QwenEmotion vLLM 显存占用率，默认设置为 `0.05`

### 请求示例
参考 `api_example.py`

### OpenAI API
- 添加 /audio/speech api 路径，兼容 OpenAI 接口
- 添加 /audio/voices api 路径， 获得 voice/character 列表

详见：[createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

## 并发测试
参考 [`simple_test.py`](simple_test.py)，需先启动 API 服务
