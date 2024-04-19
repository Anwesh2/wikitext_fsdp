# Finetuning GPT2 using FSDP
Finetuning GPT2 on wikitext using distributed training (FSDP)

**Instance used** - EC2 g4dn.12xlarge instance

**Dataset used** - [wikitext-2-v1] (https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1)

**Model used** - [GPT-2 (144M)] (https://huggingface.co/openai-community/gpt2)

**Processing** - a per-gpu batch size of 24, which totals to a batch size of 96 (across all 4 GPUs) with FSDP leveraged

**Token size** - 512 

## My process and thinking
1. Created a base pipeline on Colab with dataset loading, model loading, tokenization, model finetuning.
   
2. Attempted parallelization on Kaggle since it has multiple GPUs.
   
3. Setup the AWS instances and made the base pipelines work on a single GPU, batch_size = 8
   
4. Had a choice to either create a custom training class or leverage Huggingface's trainer and accelerate modules. Chose the later because of time constraints, sacrificing control for quicker experimentation.
   
5. Experimented with different parameters of FSDP to facilitate as high a size of batch_size and token_length as possible.
   
   a.Full sharding + size based wrapping strategy + not pre-fetching + mixed precision training (fp 16) made batch_size = 256 work

   b. batch_size = 512 was still failing. There was a slight gap between memory reserved and memory allocation. Thought mixed precision training (fp 8) might do the trick. Didn't work as it's not yet properly supported.
   
   c. Attempted to figureout the throughput and GPU utilisation by plotting the GPU utilisation graphs - there is some scope here as there was not 100% utlisation of the RAM at all times.

   d. Tried to figure out the most efficient resizing strategies for the embeddings as I could see it being  slightly inefficient.
   
   e. Attempted gradient checkpointing. batch_size = 512 and higher was working now.



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
2. Gradient checkpointing
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
5. Integrate experiment tracking 





