import time
from typing import List, Optional, Tuple, Union

from packaging import version
import importlib
vllm_version = version.parse(importlib.import_module("vllm").__version__)

# 在 vllm 中注册自定义的 GPT2TTSModel
from vllm import ModelRegistry
# For vLLM v1.x, use the new model
if vllm_version >= version.parse("1.0.0"):
    from indextts.gpt.index_tts_gpt2_new import GPT2TTSModel
elif vllm_version > version.parse("0.7.3"):
    from indextts.gpt.index_tts_gpt2_new import GPT2TTSModel
else:
    from indextts.gpt.index_tts_gpt2 import GPT2TTSModel
ModelRegistry.register_model("GPT2InferenceModel", GPT2TTSModel)
print("✅ Registry GPT2TTSModel to vllm")



# 解除 vllm 对 repetition_penalty 的限制 (v1 compatible)
from vllm.sampling_params import SamplingParams

# Check if the patch is still needed in v1
if vllm_version >= version.parse("1.0.0"):
    # In v1, repetition_penalty limits may have been relaxed
    try:
        original_verify_args = SamplingParams._verify_args

        def patched_verify_args(self) -> None:
            repetition_penalty_temp = -1
            if self.repetition_penalty > 2.0:
                repetition_penalty_temp = self.repetition_penalty
                self.repetition_penalty = 2.0
            original_verify_args(self)
            if repetition_penalty_temp != -1:
                self.repetition_penalty = repetition_penalty_temp

        SamplingParams._verify_args = patched_verify_args
        print("⚠️  SamplingParams._verify_args Patched (v1)")
    except Exception as e:
        print(f"⚠️  SamplingParams patch not needed in v1: {e}")
else:
    # Original v0.9 patch
    original_verify_args = SamplingParams._verify_args

    def patched_verify_args(self) -> None:
        repetition_penalty_temp = -1
        if self.repetition_penalty > 2.0:
            repetition_penalty_temp = self.repetition_penalty
            self.repetition_penalty = 2.0
        original_verify_args(self)
        if repetition_penalty_temp != -1:
            self.repetition_penalty = repetition_penalty_temp

    SamplingParams._verify_args = patched_verify_args
    print("⚠️  SamplingParams._verify_args Patched (v0.9)")



# Multi-modal data scheduling patch (disabled for v1 - may not be needed)
# This patch was used in v0.9 to ensure multi_modal_data is passed through scheduling
# In v1, this may be handled differently or not needed at all


# 将 position_ids 减去 prefill 的长度再加 2，以便计算每一步 decode 的 position embed (v1 compatible)
if vllm_version >= version.parse("1.0.0"):
    # In v1, the model runner architecture may have changed
    try:
        from vllm.worker.model_runner import ModelInputForGPUBuilder
        from vllm.sequence import SequenceGroupMetadata
        from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

        def patched_compute_lens_v1(self, inter_data: ModelInputForGPUBuilder.InterDataForSeqGroup, seq_idx: int,
                            seq_group_metadata: SequenceGroupMetadata):
            """Compute context length, sequence length and tokens
            for the given sequence data. (v1 compatible)
            """
            seq_data = seq_group_metadata.seq_data[inter_data.seq_ids[seq_idx]]
            token_chunk_size = seq_group_metadata.token_chunk_size

            # Compute context length (the number of tokens that are
            # already computed) and sequence length (total number of tokens).

            seq_len = seq_data.get_len()
            if inter_data.is_prompt:
                context_len = seq_data.get_num_computed_tokens()
                seq_len = min(seq_len, context_len + token_chunk_size)
            elif hasattr(self, 'runner') and hasattr(self.runner, 'scheduler_config') and \
                (self.runner.scheduler_config.is_multi_step or \
                 self.runner.model_config.is_encoder_decoder):
                context_len = seq_len - 1
            else:
                context_len = seq_data.get_num_computed_tokens()

            # Compute tokens.
            tokens = seq_data.get_token_ids()[context_len:seq_len]
            token_types = getattr(seq_group_metadata, 'token_type_ids', None)

            inter_data.seq_lens[seq_idx] = seq_len
            inter_data.orig_seq_lens[seq_idx] = seq_len
            inter_data.context_lens[seq_idx] = context_len
            inter_data.input_tokens[seq_idx].extend(tokens)
            # Custom position calculation for TTS
            pos_bias = seq_data.get_prompt_len() - 2
            inter_data.input_positions[seq_idx].extend(range(context_len-pos_bias, seq_len-pos_bias))
            if token_types:
                inter_data.token_types[seq_idx].extend(token_types)
            inter_data.query_lens[seq_idx] = seq_len - context_len

            # Handle MRoPE if available
            if hasattr(seq_data, 'mrope_position_delta') and seq_data.mrope_position_delta is not None:
                if inter_data.mrope_input_positions is None:
                    inter_data.mrope_input_positions = [None] * inter_data.n_seqs

                inter_data.mrope_input_positions[
                    seq_idx] = MRotaryEmbedding.get_next_input_positions(
                        seq_data.mrope_position_delta,
                        context_len,
                        seq_len,
                    )

        ModelInputForGPUBuilder._compute_lens = patched_compute_lens_v1
        print("⚠️  ModelInputForGPUBuilder._compute_lens Patched (v1)")
    except Exception as e:
        print(f"⚠️  ModelInputForGPUBuilder patch failed in v1: {e}")
else:
    # Original v0.9 patch
    from vllm.worker.model_runner import ModelInputForGPUBuilder
    from vllm.sequence import SequenceGroupMetadata
    from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

    def patched_compute_lens(self, inter_data: ModelInputForGPUBuilder.InterDataForSeqGroup, seq_idx: int,
                        seq_group_metadata: SequenceGroupMetadata):
        """Compute context length, sequence length and tokens
        for the given sequence data.
        """
        seq_data = seq_group_metadata.seq_data[inter_data.seq_ids[seq_idx]]
        token_chunk_size = seq_group_metadata.token_chunk_size

        # Compute context length (the number of tokens that are
        # already computed) and sequence length (total number of tokens).

        seq_len = seq_data.get_len()
        if inter_data.is_prompt:
            context_len = seq_data.get_num_computed_tokens()
            seq_len = min(seq_len, context_len + token_chunk_size)
        elif self.runner.scheduler_config.is_multi_step or \
            self.runner.model_config.is_encoder_decoder:
            context_len = seq_len - 1
        else:
            context_len = seq_data.get_num_computed_tokens()

        # Compute tokens.
        tokens = seq_data.get_token_ids()[context_len:seq_len]
        token_types = seq_group_metadata.token_type_ids

        inter_data.seq_lens[seq_idx] = seq_len
        inter_data.orig_seq_lens[seq_idx] = seq_len
        inter_data.context_lens[seq_idx] = context_len
        inter_data.input_tokens[seq_idx].extend(tokens)
        # inter_data.input_positions[seq_idx].extend(range(context_len, seq_len))
        pos_bias = seq_data.get_prompt_len() - 2
        inter_data.input_positions[seq_idx].extend(range(context_len-pos_bias, seq_len-pos_bias))
        inter_data.token_types[seq_idx].extend(
            token_types if token_types else [])
        inter_data.query_lens[seq_idx] = seq_len - context_len

        if seq_data.mrope_position_delta is not None:
            if inter_data.mrope_input_positions is None:
                inter_data.mrope_input_positions = [None] * inter_data.n_seqs

            inter_data.mrope_input_positions[
                seq_idx] = MRotaryEmbedding.get_next_input_positions(
                    seq_data.mrope_position_delta,
                    context_len,
                    seq_len,
                )

    ModelInputForGPUBuilder._compute_lens = patched_compute_lens
    print("⚠️  ModelInputForGPUBuilder._compute_lens Patched (v0.9)")



# Hidden states patches (disabled for v1)
# The extensive hidden states patches used in v0.9 are commented out
# as they may not be compatible with v1 architecture changes.
# These patches modified the engine's step_async and process_model_outputs
# methods to return hidden states, which may be handled differently in v1.

print("✅ vLLM patches loaded for v1 compatibility")