# ASTER: Agentic Scaling with Tool-integrated Extended Reasoning


## 📢 News

📊 **Experimental Results**: Our experimental results are publicly available on [![WandB](https://img.shields.io/badge/Weights_&_Biases-FFCC00?style=flat-square)](https://wandb.ai/aster_rl/ASTER)

📁 **AIME25 Samples**: AIME25 sampling results with 16 samples per problem: [`./asserts/aime25.jsonl`](./asserts/aime25.jsonl)

🎉 **Community Reproduction Available!** We are excited to announce that [Rainyrou](https://github.com/Rainyrou/ASTER) has successfully reproduced our work and made their implementation publicly available.

🔗 **Repository**: [**👉 Rainyrou/ASTER**](https://github.com/Rainyrou/ASTER) - Check out their repository for an alternative implementation of ASTER!

## 🚀 Overview

ASTER (**A**gentic **S**caling with **T**ool-integrated **E**xtended **R**easoning) is a two-stage framework that combines targeted cold-start supervised fine-tuning with reinforcement learning to scale Tool-Integrated Reasoning (TIR) capabilities in Large Language Models.

### Core Problem

Traditional approaches to Tool-Integrated Reasoning often suffer from **interaction collapse** during RL training: a pathological failure mode where models fail to sustain multi-turn tool usage, instead degenerating into extensive internal reasoning followed by trivial, post-hoc code verification, rather than engaging in genuine iterative planning.

### Solution

ASTER addresses this challenge through an **interaction-dense cold-start strategy**:
- **Interaction Density**: Prioritizing long trajectories with multiple tool invocations
- **Behavioral Prior**: Establishing multi-turn tool-use behavioral patterns through cold-start SFT
- **Multi-stage RL**: Employing a two-stage curriculum learning strategy that progressively increases context length from 18K to 32K tokens, enabling the model to first learn efficient tool usage and then tackle more challenging problems requiring longer reasoning trajectories

## 🔄 Algorithm Pipeline

ASTER follows a two-stage training pipeline:

### Stage 1: Cold-start Supervised Fine-Tuning (SFT)

- **Data Construction**: We synthesize tool-augmented trajectories using GPT-OSS-20B and curate a small expert dataset of **4K trajectories**, each containing **more than nine tool-interaction turns**
- **Training Objective**: While this interaction-dense design may yield lower post-SFT accuracy (even falling below the base model's baseline), it establishes a robust exploration prior that prevents premature convergence to short-horizon, suboptimal policies
- **Key Features**:
  - **Interaction Density**: Long trajectories with frequent tool invocations
  - **Causal Tool Use**: Tool calls arise from iterative decision-making (plan-execute-interpret-refine loops)
  - **State Tracking**: Stateful Python environment with persistent variables across turns

### Stage 2: Multi-stage Reinforcement Learning (RL)

- **Algorithm**: Group Relative Policy Optimization (GRPO) for outcome-only RL training
  - **Token-mean Objective**: Averages the policy-gradient term over all generated tokens in a group to improve stability under long trajectories
  - **Asymmetric Clipping (Clip-Higher)**: Uses asymmetric clipping with a larger upper bound (ε_h = 0.28) to avoid overly conservative updates, without KL divergence penalty
  - **Maximum Tool Invocations**: Sets the maximum allowed tool invocations to 50 per trajectory to support long-horizon tool use and iterative self-correction
- **Training Strategy**:
  - **Stage 1**: Maximum context length of 18K tokens, training the model to use tools efficiently
  - **Stage 2**: Maximum context length of 32K tokens, filtering out prompts that consistently produce correct outcomes to focus on more challenging instances
- **Reward Mechanism**: Binary reward based solely on final answer correctness (exact match = 1, otherwise = 0)
- **Tool Environment**: Stateful Python sandbox that persists execution state (variables and functions) across turns

## 📊 Experimental Results

ASTER achieves state-of-the-art performance across competitive mathematical reasoning benchmarks:

### Main Results (ASTER-4B, 90K Inference Budget)

| Benchmark | Accuracy |
|-----------|----------|
| **AIME 2024** | **85.8%** |
| **AIME 2025** | **90.0%** |
| **HMMT 2025** | **77.1%** |
| **BeyondAIME** | **61.7%** |

### Performance Comparison (30K Inference Budget)

- **AIME 2025**: 85.0%
- **HMMT 2025**: 73.3%

### Comprehensive Benchmark Comparison

We compare ASTER against state-of-the-art text-only reasoning models and agentic reasoning systems:

| Model | AIME2024 | AIME2025 | HMMT2025 | BeyondAIME |
|-------|----------|----------|----------|------------|
| *Text-Only Reasoning Models* |
| Qwen3-1.7B-Thinking(30K) | 47.5 | 38.3 | 25.8 | 20.8 |
| Qwen3-4B-Thinking-2507(30K) | 76.7 | 72.7 | 48.1 | 43.6 |
| Qwen3-4B-Thinking-2507(80K) | -- | 81.3 | 55.5 | -- |
| OpenAI o3-mini (medium) | 79.6 | 77.0 | 53.0 | -- |
| POLARIS-4B-Preview(90K) | 81.2 | 79.4 | 58.7 | -- |
| OpenReasoning-Nemotron-7B(64K) | 84.7 | 78.2 | 63.5 | -- |
| Qwen3-235B-A22B-Thinking | <u>85.7</u> | 81.5 | 62.5 | -- |
| *Agentic Reasoning Models* |
| ReTool-32B | 72.5 | 54.3 | -- | -- |
| rStar2-Agent-14B | 80.6 | 69.8 | 52.7 | -- |
| DemyAgent-4B | 72.6 | 70.0 | 52.9† | 35.3† |
| *ASTER* |
| ASTER-1.7B-SFT | 19.4 | 19.0 | 11.3 | 6.4 |
| ASTER-1.7B | 64.6 | 59.6 | 47.5 | 26.3 |
| ASTER-4B-SFT | 62.5 | 54.6 | 43.3 | 27.4 |
| ASTER-4B (30K) | 82.3 | 85.0 | 73.3 | 53.9 |
| **ASTER-4B (90K)** | **85.8** | **90.0** | **77.1** | **61.7** |

*Note: All results are reported as average accuracy over 16 samples (avg@16) following the DeepSeek-R1 assessment framework (temperature=0.6, top_p=0.95). Results marked with † denote independent evaluation using officially recommended configurations restricted to a 30K inference budget.*

### Key Advantages

- **Model Efficiency**: Despite being trained from a 4B base model, ASTER-4B achieves 90.0% on AIME 2025, outperforming [MiniMax M2.5](https://www.minimax.io/news/minimax-m25)(86.3%)
- **Inference Scalability**: Performance improves from 85.0% to 90.0% on AIME 2025 as inference budget increases from 30K to 90K, demonstrating that interaction-dense training unlocks scalable agentic intelligence

## Acknowledgments

- Built on top of [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) and [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) base models
- Use [LlamaFactory](https://github.com/hiyouga/LlamaFactory) for multi-turn SFT
- Use [SandboxFusion](https://github.com/bytedance/SandboxFusion) for RL training
- Uses [verl](https://github.com/verl-project/verl) framework for RL training

## 📖 Citation

If you find our work useful, please consider citing:

```bibtex
@misc{zhang2026aster,
  title        = {ASTER: Agentic Scaling with Tool-integrated Extended Reasoning},
  author       = {Zhang, Xuqin and He, Quan and Zheng, Zhenrui and Zhang, Zongzhang and He, Xu and Li, Dong},
  year         = {2026},
  eprint       = {2602.01204},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL}
}
```
