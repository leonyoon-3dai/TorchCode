# 05. Scaled Dot-Product Attention — Transformer 의 심장

> 📓 [원본 notebook](../solutions/05_attention_solution.ipynb) · 난이도 🔴

## 개념

Attention 은 **Query (Q)** 가 **Keys (K)** 와 얼마나 유사한지 계산해, 그 가중치로 **Values (V)** 를 섞어 새 표현을 만드는 연산입니다.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

핵심 직관: "각 query 가 어떤 key 에 얼마나 주목할지 softmax 로 결정하고, 그 비율로 value 를 섞자."

## 왜 $\sqrt{d_k}$ 로 나누는가?

$q \cdot k$ 의 분산은 차원 $d_k$ 에 비례해 커집니다. 그러면 softmax 입력이 커져 한 곳에만 몰리고, 기울기가 거의 0 이 됩니다. $\sqrt{d_k}$ 로 나눠 분산을 1 근처로 맞춰야 softmax 가 "부드러운" 분포를 유지합니다.

![attention 마스크 종류](assets/attention_masks.png)

## 코드 line-by-line

```python
def scaled_dot_product_attention(Q, K, V):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```

| 라인 | 코드 | 설명 |
|------|------|------|
| 2 | `d_k = K.size(-1)` | 마지막 축, 즉 key 벡터의 차원. |
| 3 | `Q.shape = (B, S_q, d_k)`, `K.shape = (B, S_k, d_k)` | 배치, 쿼리 길이, 키 길이. |
|   | `K.transpose(1, 2)` | `(B, d_k, S_k)` 로 바꿔 내적이 가능하게. |
|   | `torch.bmm(Q, K.T)` | **Batched matmul** — `(B, S_q, d_k) @ (B, d_k, S_k) = (B, S_q, S_k)`. 각 query-key 쌍의 유사도. |
|   | `/ math.sqrt(d_k)` | 위에서 설명한 스케일. |
| 4 | `torch.softmax(scores, dim=-1)` | 각 query 행에 대해 key 축으로 softmax. 결과 행 합 = 1 (확률분포). |
| 5 | `torch.bmm(weights, V)` | `(B, S_q, S_k) @ (B, S_k, d_v) = (B, S_q, d_v)`. query 별로 value 들의 가중 평균. |

## 차원 흐름 정리

```
Q      : (B, S_q, d_k)
K      : (B, S_k, d_k)   ── K.T → (B, d_k, S_k)
V      : (B, S_k, d_v)

scores : Q @ K.T  → (B, S_q, S_k)  ── attention 유사도
weights: softmax  → (B, S_q, S_k)  ── 확률
output : weights @ V → (B, S_q, d_v)
```

**cross-attention** (예: encoder→decoder) 은 `S_q ≠ S_k`, `d_v ≠ d_k` 여도 동일하게 동작합니다.

## 검증 (cross-attention 포함)

```python
Q2 = torch.randn(1, 3, 16)  # 3 queries
K2 = torch.randn(1, 5, 16)  # 5 keys
V2 = torch.randn(1, 5, 32)  # 5 values, d_v=32
out = scaled_dot_product_attention(Q2, K2, V2)
print(out.shape)  # torch.Size([1, 3, 32])
```

Query 가 3 개, Value 차원이 32 → 출력은 `(1, 3, 32)`.

## 한 걸음 더

- **Multi-Head Attention**: 여러 개의 attention 을 병렬로 ([06번](06_multihead_attention.md))
- **Causal mask**: 미래 토큰을 못 보게 ([09번](09_causal_attention.md))
- **Sliding window**: 가까운 토큰만 보게 ([11번](11_sliding_window.md))
- **KV cache**: 생성 시 key/value 재사용 ([14번](14_kv_cache.md))
- **Flash Attention**: 블록 타일링으로 메모리 절약 ([25번](25_flash_attention.md))
- **Linear Attention**: softmax 를 kernel 로 근사 → O(N) ([12번](12_linear_attention.md))
