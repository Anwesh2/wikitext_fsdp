# Finetuning GPT2 using FSDP
Finetuning GPT2 on wikitext using distributed training (FSDP)

**Instance used** - EC2 g4dn.12xlarge instance

**Dataset used** - [wikitext-2-v1] (https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1)

**Model used** - [GPT-2 (144M)] (https://huggingface.co/openai-community/gpt2)

**Processing** - a per-gpu batch size of 24, which totals to a batch size of 96 (across all 4 GPUs) with FSDP leveraged

**Token size** - 512 

## Experiments


| Method | Batch Size Max ($BS) | Approx Train Time (minutes) | Perplexity | Notes
| --- | --- | --- | --- | --- |
| FSDP + Full_Shard | 64 |  |  |  |
| FSDP + Full_Shard + Transformer_based_wrap | 128 |  |  |  |
| FSDP + Full_Shard + min_num_params = 2K + No-Prefetch + No-use_original_params + MPT + fp16 | 256 |  |  |  |
| FSDP + Full_Shard + min_num_params = 2K + No-Prefetch + No-use_original_params + MPT + fp16 + Gradient checkpointing  | 512 |  |  |  |

**Table 1: Finetuning GPT-2(144M) model 0- different strategies and batch sizes**



| Batch Size Max ($BS) | Full-Sharding | Wrapping Strategy | Prefetch | Forward-fetch | use_original_params | CPU-RAM Offloading+Efficient Loading | Mixed Precision Training
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | ✅ | Transformer_based_wrap |✅  | ✅ | ✅ | ✅ |❌  |
| 128 | ✅ | Transformer_based_wrap | ✅ | ✅ | ✅ |✅  | ❌ |
| 256 | ✅ | Size based wrap - 2k params | ❌ | ❌ | ❌ | ✅ | ✅ |
| 512  | ✅ | Size based wrap - 2k params |  ❌| ❌ | ❌ | ✅ | ✅ |

**Table 2: FSDP Sharding Strategies for different batch sizes**

## What worked
1. Mixed precision - fp 16
2. Gradient checkpointing (excluding activation checpointing)
3. CPU Offloading
4. Full Sharding

## Challenges 
1. Making the model fit onto the available GPU RAMs
2. Experimenting with FSDP parameters
3. Making fp8 work
4. Access to budget/billing usage

## Limitations
1. Make the code more modular to make the trainign work for different datasets and models 
2. Build custom training class to have more control on the FSDP processes
3. Incorporate hyperparametertuning for better performance

## Things to Do
1. Generate training and validation plots
2. Genrate GPU utilisation graphs

## Things to Try
1. PEFT LoRA QloRA
2. Automated experiments of FSDP parameters
3. Improve GPU utilization
4. Optimise using Dynamo





