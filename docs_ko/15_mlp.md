# 15. SwiGLU MLP — LLaMA/Mistral 의 feed-forward

> 📓 [원본 notebook](../solutions/15_mlp_solution.ipynb) · 난이도 🟠

## 개념

Transformer 블록의 MLP 는 원래 `Linear → GELU → Linear` (GPT-2) 였습니다. **SwiGLU** 는 **gated** 구조를 도입해 표현력을 크게 높였습니다:

$$\text{SwiGLU}(x) = (\text{SiLU}(xW_{gate}) \odot (x W_{up})) W_{down}$$

- $\odot$: 원소별 곱
- SiLU = $x \cdot \sigma(x)$, 부드러운 활성화
- **Gate** 가 up 경로의 값을 선택적으로 통과시킴 → learned attention-like mixing

LLaMA, Mistral, PaLM 등이 채택. GLU 계열의 variant (ReGLU, GeGLU, SwiGLU) 중 경험적으로 SwiGLU 가 가장 강함.

## 코드 line-by-line

```python
class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

| 라인 | 코드 | 설명 |
|------|------|------|
| 4 | `gate_proj` | gate 계산용 projection. d_model → d_ff. |
| 5 | `up_proj` | value projection. d_model → d_ff. **gate 와 같은 입력을 다르게 투영** |
| 6 | `down_proj` | 확장된 차원을 다시 d_model 로 축소 |
| 9 | `F.silu(gate(x))` | $x \cdot \sigma(x)$. 음수 부분에서 부드럽게 감쇠. |
|   | `* up(x)` | 원소별 곱 — **gating**. gate 값에 따라 up 출력을 조절 |
|   | `down_proj(...)` | 최종 투영 |

## 왜 gate 가 효과적인가

비교: 일반 MLP `W_2 · \phi(W_1 x)` 는 **activation $\phi$** 만으로 비선형을 만듭니다.

SwiGLU 는:
1. Gate 와 up 두 독립 경로
2. 곱으로 **multiplicative** interaction
3. gate 가 "이 값은 통과시켜, 이 값은 억제해" 역할

결과: 같은 파라미터 수에서 표현력 우수. 다만 파라미터가 늘어나 (2개 projection) 실제 비교 시 d_ff 를 줄임.

### LLaMA 의 실용 팁
LLaMA 원본은 `d_ff = (2/3) × 4 × d_model` 로 쓰고 8의 배수로 반올림해, GELU MLP 와 파라미터 수를 비슷하게 맞춥니다.

## 차원 흐름

```
x            : (B, S, d_model)
gate_proj(x) : (B, S, d_ff)
silu(...)    : (B, S, d_ff)
up_proj(x)   : (B, S, d_ff)
* (element)  : (B, S, d_ff)
down_proj    : (B, S, d_model)
```

## 파라미터 수

d_model=64, d_ff=128 기준:

- gate_proj: 64×128 + 128 = 8320
- up_proj: 64×128 + 128 = 8320
- down_proj: 128×64 + 64 = 8256
- **합계**: 24,896

동일 d_model/d_ff 의 GPT-2 MLP 는 2 개 Linear 라 약 16K. SwiGLU 가 1.5 배 정도 큼.

## 한 걸음 더

- Activation 비교: GELU vs SiLU (사실상 Swish). SiLU 는 $x \cdot \sigma(x)$ 로 GELU 와 매우 유사
- 다른 GLU 계열:
  - **ReGLU**: ReLU gate
  - **GeGLU**: GELU gate
- 본 예제의 gate/up 을 하나의 `Linear(d_model, 2*d_ff)` 로 합쳐 두 projection 을 한 번에 하는 구현도 흔함 (속도 이득)
