#!/bin/bash
VLLM_USE_V1=1 python3 api_server.py --model_dir assets/checkpoints/ --port 8001