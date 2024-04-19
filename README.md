# Finetuning GPT2 using FSDP
Finetuning GPT2 on wikitext using distributed training (FSDP)

   **Instance used** - EC2 g4dn.12xlarge instance

   **Dataset used** - [wikitext-2-v1](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-v1)

   **Model used** - [GPT-2 (144M)](https://huggingface.co/openai-community/gpt2)

   **Processing** - a per-gpu batch size of 24, which totals to a batch size of 96 (across all 4 GPUs) with FSDP leveraged

   **Token size** - 512 

## My process and thinking
1. Created a base pipeline on Colab with dataset loading, model loading, tokenization, model finetuning.
   
2. Attempted parallelization on Kaggle since it has multiple GPUs.
   
3. Setup the AWS instances and made the base pipelines work on a single GPU, `batch_size` = 8
   
4. Had a choice to either create a custom training class or leverage Huggingface's trainer and accelerate modules. Chose the later because of time constraints, sacrificing control for quicker experimentation.
   
5. Experimented with different parameters of FSDP to facilitate as high a size of batch_size and token_length as possible.
   
   a.Full sharding + size based wrapping strategy + not pre-fetching + mixed precision training (fp 16) made `batch_size = 256` work

   b. `batch_size = 512` was still failing. There was a slight gap between memory reserved and memory allocation. Thought mixed precision training (fp 8) might do the trick. Spent tons of time here but it didn't work as it's not yet properly supported.
   
   c. Attempted to figure out the throughput and GPU utilization by plotting the GPU utilization graphs - there is some scope here as there was not 100% utilization of the RAM at all times.

   d. Tried to figure out the most efficient resizing strategies for the embeddings as I could see it being  slightly inefficient.
   
   e. Attempted gradient checkpointing. `batch_size = 512` and higher was working now. The tradeoff is initial accuracy halved/perplexity doubled.
   
6. Saved, pushed, and created a custom fine-tuned model on the Huggingface hub. Easier to access for testing/inferencing.



## Experiments


| Method | Batch Size Max ($BS) | Approx Train Time per epoch(minutes) (*) | Perplexity | Notes
| --- | --- | --- | --- | --- |
| FSDP + full_shard | 64 | 1:52 |85  |  |
| FSDP + full_shard + transformer_based_wrap | 128 | 1:47 | 62 |  |
| FSDP + full_shard + min_num_params = 2K + no-prefetch + no-use_original_params + MPT + fp16 | 256 | 1:22 | 45.36 |  |
| FSDP + full_shard + min_num_params = 2K + no-prefetch + no-use_original_params + MPT + fp16 + Gradient checkpointing  | 512 |1:39  | 35.09 - ~24 |  |

**Table 1: Finetuning GPT-2(144M) model 0- different strategies and batch sizes**
* - need to re-confirm/recalculate

| Batch Size Max ($BS) | Full-Sharding | Wrapping Strategy | Prefetch | Forward-fetch | use_original_params | CPU-RAM Offloading+Efficient Loading | Mixed Precision Training
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | ✅ | transformer_based_wrap |✅  | ✅ | ✅ | ✅ |❌  |
| 128 | ✅ | transformer_based_wrap | ✅ | ✅ | ✅ |✅  | ❌ |
| 256 | ✅ | size based wrap - 2k params | ❌ | ❌ | ❌ | ✅ | ✅ |
| 512  | ✅ | size based wrap - 2k params |  ❌| ❌ | ❌ | ✅ | ✅ |

**Table 2: FSDP Sharding Strategies for Different Batch Sizes**

## What worked
1.  Mixed precision - fp 16
2.  Gradient checkpointing
3.  CPU Offloading
4.  Full Sharding

## Challenges 
1.  Making the model fit onto the available GPU RAMs
2.  Experimenting with FSDP parameters
3.  Making fp8 work
4.  Access to budget/billing usage

## Limitations
1.  Make the code more modular to make the training work for different datasets and models. 
2.  Build a custom training class to have more control over the FSDP processes.
3.  Incorporate hyperparameter tuning for better performance.

## Things to Do
1.  Generate training and validation plots.
2.  Generate proper GPU utilization graphs.
3.  Implement more metrics (`ROUGE`, `mauve` holistic accuracy metrics - `creativity`, `coherence`, `diversity`.)
4.  PEFT LoRA QloRA
5.  Automated experiments of FSDP parameters
6.  Improve GPU utilization
7.  Optimise using Dynamo
8.  Integrate experiment tracking 
9.  Better inferencing pipeline




