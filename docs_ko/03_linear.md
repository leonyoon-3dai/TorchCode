# 03. Linear Layer — 가장 기본이 되는 학습층

> 📓 [원본 notebook](../solutions/03_linear_solution.ipynb) · 난이도 🟡

## 개념

**Linear (Fully-Connected) 층**은 입력 벡터를 선형 변환하는 가장 기초적인 layer 입니다.

$$y = xW^\top + b$$

- $x \in \mathbb{R}^{B \times d_{in}}$: 배치 입력
- $W \in \mathbb{R}^{d_{out} \times d_{in}}$: 가중치 행렬
- $b \in \mathbb{R}^{d_{out}}$: 편향
- $y \in \mathbb{R}^{B \times d_{out}}$: 출력

## 코드 line-by-line

```python
class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
        self.weight.requires_grad_(True)
        self.bias = torch.zeros(out_features, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias
```

| 라인 | 코드 | 설명 |
|------|------|------|
| 3 | `torch.randn(out_features, in_features)` | 표준정규분포 $\mathcal{N}(0,1)$ 에서 샘플링. shape 은 `(d_out, d_in)` — PyTorch 관례와 동일. |
|   | `* (1 / math.sqrt(in_features))` | **Xavier-like 초기화**. 분산을 $1/d_{in}$ 로 줄여 layer 를 거쳐도 activations 의 분산이 유지되게 함. 초기화를 안 하면 깊은 네트워크에서 **폭발/소실** 이 발생. → [20번 Kaiming 초기화](20_weight_init.md) |
| 4 | `.requires_grad_(True)` | autograd 추적 대상으로 등록. |
| 5 | `torch.zeros(..., requires_grad=True)` | bias 는 **0 으로 초기화** 가 표준. 랜덤 bias 는 대칭성에 영향을 주지 않고 수렴을 지연시킬 뿐. |
| 8 | `x @ self.weight.T + self.bias` | `@` 는 matmul. `W.T` 로 transpose 해서 `(B, d_in) @ (d_in, d_out) = (B, d_out)`. bias broadcast 로 마지막 축에 더해짐. |

## 왜 `W` 를 `(d_out, d_in)` 로 저장하는가?

수식상 $y = xW^\top$ 이므로 그냥 `(d_in, d_out)` 로 저장해도 됩니다. 하지만 PyTorch 의 `nn.Linear` 관례는 `(d_out, d_in)` 을 따릅니다. 이는:

- 행이 **출력 뉴런**, 열이 **입력 특징** 으로 직관적
- `nn.Linear.weight.T @ x.T` 보다 `x @ W.T` 가 메모리 액세스 패턴이 좋음

## 차원 흐름

```
x       : (B, d_in)
W       : (d_out, d_in)
W.T     : (d_in, d_out)
x @ W.T : (B, d_out)
+ b     : (B, d_out)  — bias 는 (d_out,) 이 broadcast 됨
```

## 검증

```python
layer = SimpleLinear(8, 4)
print(layer.weight.shape)  # torch.Size([4, 8])
print(layer.bias.shape)    # torch.Size([4])
x = torch.randn(2, 8)
print(layer.forward(x).shape)  # torch.Size([2, 4])
```

## 한 걸음 더

- 초기화 스케일 $1/\sqrt{d_{in}}$ 은 **forward pass** 의 분산을 보존. 만약 ReLU 뒤에서 분산이 절반이 되는 걸 고려하면 $\sqrt{2/d_{in}}$ (**Kaiming He**) 가 맞음 → [20번](20_weight_init.md)
- LoRA 에서 이 층을 **동결 + 저차원 업데이트** 로 효율화 → [26번](26_lora.md)
- 정수 양자화 → [36번 INT8 Quantized Linear](36_int8_quantization.md)
