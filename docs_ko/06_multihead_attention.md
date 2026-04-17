# 06. Multi-Head Attention — 여러 "관점" 병렬 처리

> 📓 [원본 notebook](../solutions/06_multihead_attention_solution.ipynb) · 난이도 🔴

## 개념

한 번의 attention 대신 **여러 개를 병렬로** 돌리고, 결과를 이어 붙이는(concat) 방식입니다. 각 head 는 서로 다른 관계(문법, 의미, 위치 등)를 학습할 수 있습니다.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

각 head 는 작은 $d_k = d_{\text{model}} / h$ 차원으로 projection 후 독립적으로 attention 수행.

![MHA 구조](assets/mha_diagram.png)

## 코드 line-by-line

```python
class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
```

| 라인 | 설명 |
|------|------|
| 3 | head 당 차원 = 전체 차원 / head 수. 예: d_model=32, h=4 → d_k=8 |
| 5-7 | Q/K/V 각각에 대한 projection. 출력이 `d_model` 이지만 내부적으로 **`h × d_k` 로 쪼갬** |
| 8 | Concat 결과를 다시 섞어주는 출력 projection |

```python
    def forward(self, Q, K, V):
        B, S_q, _ = Q.shape
        S_k = K.shape[1]

        q = self.W_q(Q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
```

### **핵심 트릭**: view → transpose

1. `W_q(Q)` shape: `(B, S_q, d_model)`
2. `.view(B, S_q, h, d_k)` — 마지막 축을 `h × d_k` 로 쪼갬
3. `.transpose(1, 2)` — `(B, h, S_q, d_k)` 로 head 축을 앞으로

이렇게 하면 이후의 matmul 이 **각 head 별로 독립 계산**됩니다.

```python
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
```

| 라인 | 설명 |
|------|------|
| scores | `(B, h, S_q, d_k) @ (B, h, d_k, S_k) = (B, h, S_q, S_k)`. head 마다 attention score. |
|        | `k.transpose(-2, -1)` 는 마지막 두 축만 뒤집음 (`(..., S_k, d_k) → (..., d_k, S_k)`). |
| weights | softmax 로 확률화. |
| attn  | `(B, h, S_q, S_k) @ (B, h, S_k, d_k) = (B, h, S_q, d_k)`. |

```python
        out = attn.transpose(1, 2).contiguous().view(B, S_q, -1)
        return self.W_o(out)
```

| 라인 | 설명 |
|------|------|
| `.transpose(1,2)` | `(B, S_q, h, d_k)` 로 되돌림 (원래 순서). |
| `.contiguous()` | transpose 는 stride 만 바꿈. view 가 가능하도록 연속된 메모리 배치로 복사. |
| `.view(B, S_q, -1)` | 마지막 두 축 합침 → `(B, S_q, d_model)` (h × d_k = d_model). **이것이 concat 의 연산적 구현**. |
| `self.W_o(out)` | 마지막 혼합 projection. |

## 차원 흐름 한눈에

```
Q         : (B, S, D)
W_q(Q)    : (B, S, D)
.view     : (B, S, h, d_k)
.transpose: (B, h, S, d_k)   ← 각 head 병렬 준비
attention : (B, h, S, d_k)   ← head 별 출력
.transpose: (B, S, h, d_k)
.view     : (B, S, D)         ← concat
W_o       : (B, S, D)         ← 최종
```

## 왜 그냥 head 를 for loop 으로 안 돌리는가?

`view + transpose + batched matmul` 을 쓰면 **모든 head 가 GPU 에서 병렬**로 한 번에 계산됩니다. for loop 은 head 수 만큼 커널 호출이 생겨 느립니다.

## 한 걸음 더

- **GQA** ([10번](10_gqa.md)): Q 는 많이, K/V 는 적게 — KV 메모리 절약
- **Cross-Attention** ([23번](23_cross_attention.md)): Q 와 K/V 의 소스가 다름 (encoder-decoder)
- **Causal mask** + MHA → GPT 블록 ([13번](13_gpt2_block.md))
