# 09. Causal Attention — 미래를 보지 않는 self-attention

> 📓 [원본 notebook](../solutions/09_causal_attention_solution.ipynb) · 난이도 🔴

## 개념

GPT 같은 **자동회귀 모델** 은 학습 시에도 "다음 토큰 예측" 이 목표입니다. 위치 $i$ 에서 위치 $j > i$ 의 토큰을 보면 답을 그대로 보는 꼴이라 학습이 망가집니다.

해결: 상삼각(upper-triangular) 영역의 attention score 를 $-\infty$ 로 마스킹 → softmax 후 0 → 미래 토큰 무시.

$$\text{mask}_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

![attention mask](assets/attention_masks.png)

## 코드 line-by-line

```python
def causal_attention(Q, K, V):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    S = scores.size(-1)
    mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool),
                      diagonal=1)
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```

| 라인 | 코드 | 설명 |
|------|------|------|
| 2-3 | 기본 scaled dot-product ([05번](05_attention.md) 참고) |
| 4 | `S = scores.size(-1)` | 시퀀스 길이 |
| 5 | `torch.ones(S, S, dtype=torch.bool)` | `S×S` 전부 `True` 인 boolean 행렬 |
|   | `torch.triu(..., diagonal=1)` | 상삼각 1 로 채움. **`diagonal=1`** 이 핵심 — 대각선은 포함하지 않음 (즉, 자기 자신은 볼 수 있음). |
| 6 | `mask.unsqueeze(0)` | `(S, S) → (1, S, S)` 로 배치 축 추가해 broadcast. |
|   | `.masked_fill(mask, -inf)` | mask 가 True 인 위치를 `-inf` 로. softmax 후 0 이 됨. |
| 7 | `torch.softmax(scores, dim=-1)` | 각 row (query) 에서 정규화. 미래 토큰 가중치는 0. |

## 왜 `diagonal=1` 인가?

`torch.triu` 는 기본 `diagonal=0` 이면 **대각선 포함 상삼각**. 우리는 자기 위치는 허용해야 하므로 한 칸 위로 올려 `diagonal=1`:

```
diagonal=0 (틀림):       diagonal=1 (맞음):
1 1 1 1                  0 1 1 1
0 1 1 1                  0 0 1 1
0 0 1 1                  0 0 0 1
0 0 0 1                  0 0 0 0
```

## 왜 0 이 아니라 `-inf` ?

attention 을 "보지 않는" 건 **softmax 전** 에 해야 합니다. `scores = 0` 은 softmax 후에 **일정한 작은 확률** 이 남습니다. `-inf` 로 두면 $e^{-\infty} = 0$ 이 되어 확률 = 0. 그리고 정규화도 제대로 됩니다.

## 검증

```python
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = causal_attention(Q, K, V)
# 위치 0 은 자기 자신만 볼 수 있으므로 softmax([score_0]) = [1], out[0] ≈ V[0]
print(torch.allclose(out[:, 0], V[:, 0], atol=1e-5))  # True
```

위치 0 의 출력이 `V[0]` 과 일치 → mask 가 제대로 적용됨.

## 한 걸음 더

- Multi-head 와 결합 → [GPT-2 Block (13번)](13_gpt2_block.md)
- 생성 시 KV 재사용 → [KV Cache (14번)](14_kv_cache.md)
- Window mask 조합 → [Sliding Window (11번)](11_sliding_window.md)
