# Project Issues Summary: FineTuning LoRA & qLoRA

## Overview
This document summarizes the main technical problems encountered while setting up and running the LoRA and qLoRA workflow, with emphasis on environment setup, model configuration, and inference stability.

## Main Issues

### 1. Environment and GPU Setup Compatibility
**Problem**: The project initially could not use the NVIDIA GPU reliably, and PyTorch setup failed multiple times.

**Root Cause**:
- PyTorch was initially installed without CUDA support
- the system CUDA and driver setup was outdated for the required PyTorch version
- Python 3.14 was incompatible with the Windows PyTorch wheels needed for this project
- multiple virtual environments made package setup harder to track

**Resolution**:
- upgraded CUDA to a compatible version
- switched to a Python 3.11 virtual environment
- installed a GPU-enabled PyTorch build in the correct environment

**Impact**: Until this was fixed, GPU acceleration was unavailable and model loading was either failing or extremely slow.

### 2. Model and Script Configuration Mismatches
**Problem**: Several scripts failed because their configuration did not match the intended model architecture or runtime requirements.

**Root Cause**:
- full LoRA fine-tuning with `microsoft/phi-1.5` was too heavy for the available hardware, so a smaller model (`gpt2`) had to be used for the standard LoRA path
- the LoRA fine-tuning setup depended on target module names that must match the selected base model architecture
- the qLoRA training script was incorrectly configured to run with `device_map="cpu"`, even though 4-bit quantization via bitsandbytes requires GPU execution

**Resolution**:
- treated model choice as hardware-dependent: standard LoRA used `gpt2`, while baseline inference and qLoRA could still use `microsoft/phi-1.5`
- aligned LoRA target modules with the actual base model architecture
- changed qLoRA training to use `device_map="auto"`

**Impact**: These configuration issues caused failures during model load, adapter setup, and training, and they also forced different model choices for LoRA versus qLoRA on the same machine.

**Key Observation**: One of the most useful takeaways was seeing the practical effect of qLoRA directly: the same `microsoft/phi-1.5` model was too heavy for standard LoRA fine-tuning on this hardware, but it became trainable once the qLoRA path used 4-bit quantization.

### 3. qLoRA Adapter Inference Instability
**Problem**: Post-training inference with `script3_chat_with_adapter.py` initially failed with PEFT and Accelerate offload errors and later caused laptop instability.

**Root Cause**:
- the qLoRA adapter contains only adapter weights, so inference still requires loading the base model
- loading the base Phi model through the regular or mixed offload path created memory pressure on the available hardware
- the inference path was not initially aligned with the successful qLoRA training path

**Resolution**:
- updated `script3_chat_with_adapter.py` to load the base model with `BitsAndBytesConfig(load_in_4bit=True, ...)`
- kept `device_map="auto"`
- loaded the saved adapter from `models/adapters/qlora_acme`
- used the same instruction and response prompt format as training

**Impact**: After aligning inference with the qLoRA training setup, the adapter script ran successfully and produced stable results.

### 4. Project Organization and Clarity
**Problem**: Adapter folders and script intent were initially harder to track than necessary.

**Root Cause**: Generic naming made it less clear which adapter belonged to which model or training method.

**Resolution**: Kept adapter paths and script roles model-specific and method-specific.

**Impact**: This reduced confusion during training and inference and made debugging easier.

## Current Status
- CUDA 12.1 installed and verified
- Python 3.11 virtual environment created
- PyTorch GPU support working
- LoRA and qLoRA scripts functional for the intended models (`gpt2` for standard LoRA, `microsoft/phi-1.5` for baseline and qLoRA)
- qLoRA adapter inference script functional with 4-bit base-model loading
- Adapter folders organized by model type
- Baseline evaluation and fine-tuning scripts ready

## Lessons Learned
1. Always verify model names from official Hugging Face repositories.
2. Verify Python, CUDA, and PyTorch compatibility before debugging script-level issues.
3. Model choice is also hardware-dependent: `microsoft/phi-1.5` worked for baseline inference and qLoRA, but standard LoRA needed a lighter model such as `gpt2`.
4. qLoRA requires GPU execution and a compatible quantized loading path.
5. LoRA target modules must match the selected model architecture.
6. A qLoRA adapter still requires the original base model at inference time.
7. For low-memory GPUs, inference should use the same 4-bit loading strategy used during qLoRA training.
8. Clear naming for adapters and scripts reduces debugging time.
9. qLoRA’s main practical advantage was clearly visible in this project: quantization made it possible to train `microsoft/phi-1.5` on hardware where standard LoRA fine-tuning of the same model was not feasible.

## Next Steps
1. Run GPT-2 LoRA fine-tuning: `python .\script2_lora_finetune.py`
2. Test GPT-2 adapter inference: `python .\script3_gpt2_chat_with_adapter.py`
3. Run Phi qLoRA fine-tuning: `python .\script2_qlora_finetune.py`
4. Run Phi qLoRA adapter inference: `python .\script3_chat_with_adapter.py`
5. Compare performance between LoRA and qLoRA approaches.
6. Document final results and model comparisons.