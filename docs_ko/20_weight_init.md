# 20. Kaiming Initialization — ReLU 네트워크를 위한 초기화

> 📓 [원본 notebook](../solutions/20_weight_init_solution.ipynb) · 난이도 🟢

## 개념

신경망 초기화의 핵심 목표: **forward 와 backward 에서 activations / gradients 의 분산이 유지**되어야 함. 분산이 층마다 기하급수적으로 증가/감소하면 **폭발/소실** 발생.

**He (Kaiming) 초기화**: ReLU 가 뒤에 있을 때를 가정 — ReLU 는 평균적으로 입력의 절반을 0 으로 만들어 분산을 절반으로 줄임. 이를 **보정하는 인수 2** 를 곱합니다:

$$W \sim \mathcal{N}\left(0, \frac{2}{\text{fan\_in}}\right)$$

**Xavier/Glorot** 초기화는 $2$ 대신 $1$ — tanh/sigmoid 같은 대칭 활성화용.

## 코드 line-by-line

```python
def kaiming_init(weight):
    fan_in = weight.shape[1] if weight.dim() >= 2 else weight.shape[0]
    std = math.sqrt(2.0 / fan_in)
    with torch.no_grad():
        weight.normal_(0, std)
    return weight
```

| 라인 | 코드 | 설명 |
|------|------|------|
| 2 | `weight.shape[1]` | Linear 가중치 `(out, in)` 에서 **입력 차원** = fan_in. dim>=2 가정. |
|   | 1D fallback | 1D 파라미터(예: bias) 는 거의 안 쓰지만 방어적 처리. |
| 3 | `sqrt(2.0 / fan_in)` | Kaiming normal std. 분산 = 2/fan_in. |
| 4 | `torch.no_grad()` | in-place 쓰기는 autograd 에 보이면 안 됨 (원래 값은 learning 대상이지만 초기화는 grad 그래프 밖). |
| 5 | `weight.normal_(0, std)` | **in-place** 정규분포 샘플링. trailing `_` 는 in-place 관례. |

## 왜 `2 / fan_in` 인가

Forward pass 분산 보존 유도:

$$\text{Var}[y_i] = \sum_{j=1}^{\text{fan\_in}} \text{Var}[W_{ij} x_j] = \text{fan\_in} \cdot \text{Var}[W] \cdot \text{Var}[x]$$

$\text{Var}[y] = \text{Var}[x]$ 가 되려면 $\text{Var}[W] = 1/\text{fan\_in}$. 이게 **Xavier**.

ReLU 를 통과하면 분산이 절반으로 줄어드니 입력 분산을 2 배 키우려면 **$\text{Var}[W] = 2/\text{fan\_in}$**. 이게 **Kaiming**.

## fan_in vs fan_out

- `fan_in` 모드: forward 분산 보존
- `fan_out` 모드: backward 분산 보존

둘 다 완벽히는 안 되고, 실용적으로 `fan_in` 이 기본값.

## 검증

```python
w = torch.empty(256, 512)
kaiming_init(w)
print(f'Mean: {w.mean():.4f}')   # ~0
print(f'Std:  {w.std():.4f}')    # ~sqrt(2/512) ≈ 0.0625
```

## PyTorch 기본값

`nn.Linear` 는 기본적으로 **Kaiming uniform** (`a=sqrt(5)` 로 튜닝된 legacy 값) 을 씀. 이는 `sqrt(2/fan_in)` 에 해당하는 uniform 버전.

```python
# PyTorch 내부
nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
```

## 한 걸음 더

- 최근 대형 모델은 **1/sqrt(d) · 일부 스케일 다운** (e.g., `std = 0.02` 고정) 을 사용 — GPT-2 방식
- Transformer 끝 residual branch 에는 더 작은 std (`0.02 / sqrt(2N)`, N=layer 수) 를 써서 residual 경로 분산을 억제 (GPT-2 trick)
- 잘못된 초기화는 NaN, loss stall, slow convergence 의 주범
