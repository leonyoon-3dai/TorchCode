# 🇰🇷 TorchCode — 한국어 해설집

> PyTorch 로 GPT/Transformer 부품들을 처음부터 구현하는 [TorchCode](../README.md) 의 **모든 40개 예제**를 한국어로 line-by-line 해설하고, 시각 자료를 덧붙인 학습 자료입니다.

각 페이지 구성:
- 📐 개념과 수식
- 📊 시각 자료 (matplotlib 으로 생성한 PNG)
- 💻 라인 단위 코드 해설
- ✅ 검증 코드
- 🚀 관련 예제와 한 걸음 더

## 📚 전체 예제 목록

### 🔰 기본 레이어 & 활성화

| # | 주제 | 난이도 | 요약 |
|---|------|--------|------|
| [01](01_relu.md) | ReLU | 🟢 | 가장 단순한 비선형 활성화 |
| [02](02_softmax.md) | Softmax | 🟢 | 수치 안정 확률분포 변환 |
| [03](03_linear.md) | Linear Layer | 🟡 | $y = xW^\top + b$ 기초 |
| [04](04_layernorm.md) | LayerNorm | 🟡 | Transformer 표준 정규화 |
| [07](07_batchnorm.md) | BatchNorm | 🟡 | 배치 축 정규화 (train/eval 구분) |
| [08](08_rmsnorm.md) | RMSNorm | 🟡 | LayerNorm 경량판 (LLaMA) |
| [17](17_dropout.md) | Dropout | 🟢 | 랜덤 masking 으로 과적합 방지 |
| [18](18_embedding.md) | Embedding | 🟢 | 토큰 ID → 벡터 룩업 |
| [19](19_gelu.md) | GELU | 🟢 | 부드러운 ReLU |
| [20](20_weight_init.md) | Kaiming Init | 🟢 | ReLU 용 초기화 |

### 🧠 Attention 계열

| # | 주제 | 난이도 | 요약 |
|---|------|--------|------|
| [05](05_attention.md) | Scaled Dot-Product Attention | 🔴 | Transformer 의 심장 |
| [06](06_multihead_attention.md) | Multi-Head Attention | 🔴 | 여러 관점 병렬 처리 |
| [09](09_causal_attention.md) | Causal Attention | 🔴 | 미래 차단 마스킹 |
| [10](10_gqa.md) | Grouped Query Attention | 🔴 | KV 헤드 공유 |
| [11](11_sliding_window.md) | Sliding Window Attention | 🔴 | 인접 토큰만 보기 |
| [12](12_linear_attention.md) | Linear Attention | 🔴 | O(N) 근사 |
| [23](23_cross_attention.md) | Cross-Attention | 🟡 | Q 와 KV 의 출처 분리 |
| [24](24_rope.md) | RoPE | 🔴 | 회전 position embedding |
| [25](25_flash_attention.md) | Flash Attention | 🔴 | 블록 타일링 + online softmax |

### 🏗️ 모델 블록

| # | 주제 | 난이도 | 요약 |
|---|------|--------|------|
| [13](13_gpt2_block.md) | GPT-2 Block | 🔴 | Pre-norm + causal attn + MLP |
| [14](14_kv_cache.md) | KV Cache | 🔴 | 자동회귀 생성 가속 |
| [15](15_mlp.md) | SwiGLU MLP | 🟠 | LLaMA 의 gated FFN |
| [22](22_conv2d.md) | Conv2D | 🟡 | CNN 의 기초 연산 |
| [27](27_vit_patch.md) | ViT Patch Embedding | 🟡 | 이미지를 토큰 시퀀스로 |
| [28](28_moe.md) | Mixture of Experts | 🔴 | 조건부 연산 (Mixtral) |

### 🎯 Loss & Optimization

| # | 주제 | 난이도 | 요약 |
|---|------|--------|------|
| [16](16_cross_entropy.md) | Cross-Entropy Loss | 🟢 | 분류/LM 표준 loss |
| [21](21_gradient_clipping.md) | Gradient Clipping | 🟢 | 기울기 폭발 방지 |
| [29](29_adam.md) | Adam Optimizer | 🟡 | 1차/2차 모멘트 + bias correction |
| [30](30_cosine_lr.md) | Cosine LR Scheduler | 🟢 | Warmup + cosine decay |
| [31](31_gradient_accumulation.md) | Gradient Accumulation | 🟢 | 큰 배치 효과 with 작은 메모리 |
| [40](40_linear_regression.md) | Linear Regression | 🟡 | 세 가지 접근으로 본 첫 ML |

### 🚀 추론 & 생성

| # | 주제 | 난이도 | 요약 |
|---|------|--------|------|
| [32](32_topk_sampling.md) | Top-k / Top-p Sampling | 🟡 | 확률 기반 디코딩 |
| [33](33_beam_search.md) | Beam Search | 🟡 | 탐색 기반 디코딩 |
| [34](34_speculative_decoding.md) | Speculative Decoding | 🔴 | draft + target 2 단계 |
| [35](35_bpe.md) | Byte-Pair Encoding | 🟡 | 토크나이저 원리 |

### ⚡ 효율화

| # | 주제 | 난이도 | 요약 |
|---|------|--------|------|
| [26](26_lora.md) | LoRA | 🟡 | 저차원 fine-tune |
| [36](36_int8_quantization.md) | INT8 Quantization | 🟡 | 메모리 1/4, 속도↑ |

### 🤖 RLHF 계열

| # | 주제 | 난이도 | 요약 |
|---|------|--------|------|
| [37](37_dpo_loss.md) | DPO Loss | 🔴 | preference 직접 최적화 |
| [38](38_grpo_loss.md) | GRPO Loss | 🔴 | 그룹 상대 정책 (DeepSeek-R1) |
| [39](39_ppo_loss.md) | PPO Loss | 🔴 | 클립 surrogate (ChatGPT 계열) |

## 🗺️ 학습 경로 추천

### 딥러닝 입문

1. [Linear Regression (40)](40_linear_regression.md) — 경사 하강법의 모든 것
2. [Linear Layer (03)](03_linear.md) + [ReLU (01)](01_relu.md) — 기본 부품
3. [Cross-Entropy (16)](16_cross_entropy.md) + [Softmax (02)](02_softmax.md) — 분류 기초
4. [Adam (29)](29_adam.md) + [Kaiming Init (20)](20_weight_init.md) + [Gradient Clipping (21)](21_gradient_clipping.md) — 학습 안정화

### Transformer 구축

1. [Attention (05)](05_attention.md) → [MHA (06)](06_multihead_attention.md)
2. [LayerNorm (04)](04_layernorm.md) → [GELU (19)](19_gelu.md) → [Dropout (17)](17_dropout.md)
3. [Embedding (18)](18_embedding.md) + [GPT-2 Block (13)](13_gpt2_block.md) → **GPT 완성**

### LLM 최적화

1. [KV Cache (14)](14_kv_cache.md) → [GQA (10)](10_gqa.md) → [Flash Attention (25)](25_flash_attention.md)
2. [RoPE (24)](24_rope.md) → [SwiGLU (15)](15_mlp.md) → [RMSNorm (08)](08_rmsnorm.md) — **LLaMA 완성**
3. [LoRA (26)](26_lora.md) + [INT8 (36)](36_int8_quantization.md) — 효율화

### RLHF / 생성

1. [BPE (35)](35_bpe.md) → [Top-k/Top-p (32)](32_topk_sampling.md) / [Beam Search (33)](33_beam_search.md)
2. [DPO (37)](37_dpo_loss.md) → [PPO (39)](39_ppo_loss.md) → [GRPO (38)](38_grpo_loss.md)
3. [Speculative Decoding (34)](34_speculative_decoding.md) — 속도

### Vision

1. [Conv2D (22)](22_conv2d.md) → [ViT Patch (27)](27_vit_patch.md) → [Cross-Attention (23)](23_cross_attention.md)
2. [BatchNorm (07)](07_batchnorm.md) — CNN 용

### MoE

1. [MoE (28)](28_moe.md) — Mixtral 원리

## 📦 시각 자료 재생성

```bash
cd docs_ko/assets
pip install matplotlib numpy
python3 gen_plots.py
```

모든 PNG 가 이 스크립트로 생성되었습니다. (Korean font 는 `AppleGothic` / `Nanum Gothic` 자동 fallback)

## 📖 원본 영문 문서

각 페이지 상단의 `📓 원본 notebook` 링크로 이동 가능합니다. 또는 [루트 README](../README.md) 참고.

## 🙏 원작자

원본 프로젝트: [duoan/TorchCode](https://github.com/duoan/TorchCode) — 🔥 LeetCode for PyTorch.

이 한국어 해설집은 원본 솔루션을 기반으로 한 **학습용 2차 자료**입니다.
