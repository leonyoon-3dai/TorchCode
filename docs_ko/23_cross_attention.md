# 23. Multi-Head Cross-Attention — Q 와 KV 의 출처가 다름

> 📓 [원본 notebook](../solutions/23_cross_attention_solution.ipynb) · 난이도 🟡

## 개념

Self-attention 은 같은 시퀀스에서 Q, K, V 를 만듭니다. **Cross-attention** 은 Q 를 한 시퀀스(예: decoder 의 현재 상태), K/V 를 **다른 시퀀스**(예: encoder 의 출력) 에서 뽑습니다.

쓰임새:
- **Encoder-Decoder** (원본 Transformer, T5, Whisper 등)
- **Conditional generation**: 이미지/오디오 → 텍스트 매핑
- **Diffusion 모델**: text embedding 을 이미지 feature 에 주입

## 코드 line-by-line

```python
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv):
        B, S_q, _ = x_q.shape
        S_kv = x_kv.shape[1]
        q = self.W_q(x_q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, S_q, -1))
```

[MHA (06번)](06_multihead_attention.md) 과 거의 동일. **유일한 차이**:

| 항목 | Self-attention | Cross-attention |
|------|----------------|-----------------|
| Q 입력 | `x` | `x_q` |
| K, V 입력 | `x` | `x_kv` |
| S_q vs S_kv | 같음 | **다를 수 있음** |

그 결과 scores shape 은 `(B, h, S_q, S_kv)` — 직사각형일 수 있습니다.

## 언제 쓰는가 — 전형적 구조

**Transformer decoder block**:

```
x (decoder input) ──▶ LN ──▶ Self-attention ──▶ + ──┐
                                                      │
encoder_out ──────────▶─┐                             │
x ──▶ LN ──▶ Cross-attention(Q=x, KV=encoder_out) ──▶ +
                                                      │
                    ──▶ LN ──▶ MLP ──▶ + ──▶ output
```

Decoder 는 먼저 자기 시퀀스로 self-attention, 다음 encoder 출력에 cross-attention, 마지막 MLP.

## 검증

```python
attn = MultiHeadCrossAttention(64, 4)
x_q = torch.randn(2, 6, 64)    # decoder 측, 6 tokens
x_kv = torch.randn(2, 10, 64)  # encoder 측, 10 tokens
attn(x_q, x_kv).shape          # (2, 6, 64) — Q 길이에 맞춤
```

출력 길이 = Q 길이. K/V 길이는 결과에 영향 없음 (가중 평균될 뿐).

## Causal mask 와의 관계

Cross-attention 은 보통 mask 불필요 — decoder 의 query 가 encoder 의 **전체** 출력을 볼 수 있어야 함 (미래 개념 없음). 단, padding mask 는 종종 사용 (패딩된 K 위치 무시).

## 한 걸음 더

- Image captioning: encoder = CNN/ViT, decoder = Transformer text generator
- Whisper: audio encoder + text decoder (cross-attention 으로 audio 참조)
- Stable Diffusion: U-Net 이 cross-attention 으로 text embedding 참조
