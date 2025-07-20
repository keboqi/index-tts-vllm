#!/usr/bin/env python3
"""
Test script to verify vLLM v1 upgrade for IndexTTS
"""

import os
import sys
import traceback

# Enable vLLM v1
os.environ["VLLM_USE_V1"] = "1"

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    
    try:
        import vllm
        print(f"✅ vLLM version: {vllm.__version__}")
        
        # Test if v1 is enabled
        from vllm.config import VllmConfig
        print("✅ VllmConfig import successful")
        
        # Test model registry
        from vllm import ModelRegistry
        print("✅ ModelRegistry import successful")
        
        # Test async engine
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        print("✅ AsyncLLMEngine imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_patch_loading():
    """Test that patches load without errors"""
    print("\nTesting patch loading...")
    
    try:
        # This should load our patches
        import patch_vllm
        print("✅ Patches loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Patch loading failed: {e}")
        traceback.print_exc()
        return False

def test_model_registration():
    """Test that our custom model is registered"""
    print("\nTesting model registration...")
    
    try:
        from vllm import ModelRegistry
        
        # Check if our model is registered
        registered_models = ModelRegistry._get_supported_archs()
        if "GPT2InferenceModel" in registered_models:
            print("✅ GPT2TTSModel registered successfully")
            return True
        else:
            print("❌ GPT2TTSModel not found in registered models")
            print(f"Available models: {list(registered_models.keys())}")
            return False
    except Exception as e:
        print(f"❌ Model registration test failed: {e}")
        traceback.print_exc()
        return False

def test_engine_args():
    """Test that engine args work with v1"""
    print("\nTesting engine arguments...")
    
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        
        # Test creating engine args with v1 parameters
        engine_args = AsyncEngineArgs(
            model="dummy_model",
            tensor_parallel_size=1,
            dtype="auto",
            gpu_memory_utilization=0.25,
            max_num_seqs=32,
            max_num_batched_tokens=8192,
            use_v2_block_manager=True  # v1 specific
        )
        print("✅ AsyncEngineArgs creation successful")
        return True
    except Exception as e:
        print(f"❌ Engine args test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== vLLM v1 Upgrade Test ===\n")
    
    tests = [
        test_imports,
        test_patch_loading,
        test_model_registration,
        test_engine_args
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n=== Test Results ===")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! vLLM v1 upgrade appears successful.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())