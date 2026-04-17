# 02. Softmax — 수치 안정 버전

> 📓 [원본 notebook](../solutions/02_softmax_solution.ipynb) · 난이도 🟢

## 개념

Softmax 는 임의의 실수 벡터를 **합이 1인 확률분포**로 바꿔줍니다. 분류기의 마지막 층, Attention 가중치 등 모든 확률 출력의 표준입니다.

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

하지만 $x_i$ 가 크면 $e^{x_i}$ 가 금방 `inf` 가 되어 NaN 이 나옵니다. 그래서 **수치 안정 버전**을 쓰는데, 모든 원소에서 max 를 빼도 결과가 동일하다는 성질을 이용합니다:

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

입력을 최대 0 으로 잘라주니 `exp` 가 폭발하지 않습니다.

![softmax 예시](assets/activations.png)

## 코드 line-by-line

```python
def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(dim=dim, keepdim=True)
```

| 라인 | 코드 | 설명 |
|------|------|------|
| 1 | `dim: int = -1` | 기본값은 마지막 축. 예: `(B, C)` 에서는 C 축. |
| 2 | `x.max(dim=dim, keepdim=True)` | 지정 축에 대해 최대값을 구함. **`.values`** 만 쓰고 인덱스는 무시. `keepdim=True` 는 원래 차원을 유지해야 **broadcast** 가 정확히 들어맞기 때문. |
| 3 | `torch.exp(x - x_max)` | 원소별로 max 를 뺀 뒤 지수. 값이 최대 0 이하 → 안전. |
| 4 | `e_x.sum(dim=dim, keepdim=True)` | 같은 축으로 합계. `keepdim=True` 로 shape 유지. |
|   | `e_x / e_x.sum(...)` | broadcast 나눗셈. 각 원소를 자기 그룹의 합으로 정규화. |

## 왜 `keepdim=True` 가 중요한가?

`x` shape = `(2, 3)`, `dim=-1` 일 때:

- `x.max(dim=-1)` → shape `(2,)` (차원 축소됨)
- `x.max(dim=-1, keepdim=True)` → shape `(2, 1)` (축 유지)

`x - x_max` 는 broadcast 가 필요한데, `(2,3) - (2,)` 는 오류를 내거나 원하지 않는 broadcasting 이 일어날 수 있습니다. `(2,3) - (2,1)` 은 명확하게 각 행에서 해당 행의 max 를 빼는 동작이 됩니다.

## 검증

```python
x = torch.tensor([1.0, 2.0, 3.0])
print(my_softmax(x))           # tensor([0.0900, 0.2447, 0.6652])
print(my_softmax(x).sum())     # 1.0 (확률 정규화 확인)
```

큰 값일수록 더 큰 확률을 갖고, 합이 정확히 1 입니다.

## 수치 안정성 시험

```python
x = torch.tensor([1000., 1001., 1002.])
torch.exp(x)              # inf, inf, inf  (이대로는 망함)
my_softmax(x)             # 수치 안정 버전은 정상 동작
```

## 한 걸음 더

- Attention 의 핵심 연산 → [05번](05_attention.md)
- `log_softmax` 는 $\log$ 를 취해 더 안정적 → Cross-Entropy loss 에서 쓰임 ([16번](16_cross_entropy.md))
