# 01. ReLU — 가장 기본이 되는 활성화 함수

> 📓 [원본 notebook](../solutions/01_relu_solution.ipynb) · 난이도 🟢

## 개념

ReLU (Rectified Linear Unit) 는 딥러닝의 **가장 대표적인 비선형 활성화 함수**입니다. 입력이 양수이면 그대로 내보내고, 음수이면 0으로 끊어버리는 단순한 규칙입니다.

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \le 0 \end{cases}$$

**왜 쓰는가?**

- 비선형성을 주어 신경망이 복잡한 함수를 학습할 수 있게 해줌
- 계산이 극히 단순 (비교 한 번)
- 기울기(derivative) 도 단순: 양수면 1, 음수면 0 → **기울기 소실(vanishing gradient) 문제 완화**

![활성화 함수 비교](assets/activations.png)

## 코드 line-by-line

```python
def relu(x: torch.Tensor) -> torch.Tensor:
    return x * (x > 0).float()
```

| 라인 | 코드 | 설명 |
|------|------|------|
| 1 | `def relu(x: torch.Tensor) -> torch.Tensor:` | `torch.Tensor` 를 받아서 같은 타입을 반환하는 함수 선언. |
| 2 | `(x > 0)` | 원소별 비교. 각 위치가 0보다 크면 `True`, 아니면 `False` 인 **boolean tensor** 생성. |
|   | `.float()` | `True → 1.0`, `False → 0.0` 으로 변환. 즉 **indicator 함수** $\mathbb{1}[x>0]$. |
|   | `x * (x > 0).float()` | 원소별 곱. 양수는 그대로(×1), 음수·0 은 0 으로(×0) 출력. |

## 왜 `max(0, x)` 대신 곱셈을 쓰는가?

같은 결과지만 위 구현은 **branchless (분기 없음)** 이라 GPU 에서 벡터화가 잘 됩니다. `torch.max(x, torch.zeros_like(x))` 도 가능하지만 텐서를 하나 더 만드는 비용이 있습니다.

## 수식과 대응

| 수식 | 코드 |
|------|------|
| $\mathbb{1}[x>0]$ | `(x > 0).float()` |
| $x \cdot \mathbb{1}[x>0]$ | `x * (x > 0).float()` |

## 검증 코드

```python
x = torch.tensor([-2., -1., 0., 1., 2.])
print("Output:", relu(x))   # tensor([0., 0., 0., 1., 2.])
```

음수 입력은 모두 0, 양수는 그대로 통과하는 것을 확인할 수 있습니다. 정확히 경계값 `x=0` 은 `(0 > 0) = False` 이므로 0 이 됩니다.

## 한 걸음 더

- **Dead ReLU 문제**: 입력이 계속 음수인 뉴런은 기울기가 항상 0 이라 학습이 멈춤 → [Leaky ReLU, GELU (19번)](19_gelu.md) 등 개선형이 등장한 이유.
- Transformer 블록에서는 ReLU 대신 **GELU** 나 **SwiGLU** 가 주로 쓰임 → [15번](15_mlp.md) 참고.
