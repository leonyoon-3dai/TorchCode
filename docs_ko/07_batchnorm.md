# 07. BatchNorm — 배치 축으로 정규화

> 📓 [원본 notebook](../solutions/07_batchnorm_solution.ipynb) · 난이도 🟡

## 개념

Batch Normalization 은 **배치 안의 여러 샘플에 걸쳐** feature 별 평균/분산을 추정해 정규화합니다. CNN 에 주로 쓰이며, 학습 시와 추론 시의 동작이 다릅니다.

- **Training**: 현재 배치의 통계로 정규화 + running stats 업데이트
- **Inference**: 학습 중 축적한 running stats 사용 (배치 크기 1 에서도 동작)

![정규화 축 비교](assets/norm_axes.png)

## 코드 line-by-line

```python
def my_batch_norm(x, gamma, beta, running_mean, running_var,
                  eps=1e-5, momentum=0.1, training=True):
    if training:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        running_mean.mul_(1 - momentum).add_(momentum * batch_mean.detach())
        running_var.mul_(1 - momentum).add_(momentum * batch_var.detach())

        mean = batch_mean
        var = batch_var
    else:
        mean = running_mean
        var = running_var

    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta
```

### Training 분기

| 코드 | 설명 |
|------|------|
| `x.mean(dim=0)` | **배치 축(0축)** 평균. feature 별로 하나. `x.shape=(B,C)` → `(C,)`. |
| `x.var(dim=0, unbiased=False)` | 분산 (biased, 즉 $1/N$). PyTorch `F.batch_norm` 과 동일. |
| `.mul_(1-momentum)` | **in-place** 곱. `running_mean *= (1 - momentum)` 과 같음. 기존 값을 감쇠. |
| `.add_(momentum * ...)` | 새 관측값을 momentum 비율로 더함. 수식: $\mu_{run} \leftarrow (1-m)\mu_{run} + m \mu_{batch}$. |
| `.detach()` | **중요**: running stats 업데이트 시 autograd 그래프 끊어야 함. 아니면 backward 때 이상한 그래디언트 경로 생김. |

### Inference 분기

running stats 가 이미 "안정화된" 모집단 추정치라 그대로 사용. 배치 1 로도 정상 동작.

### 마지막

```python
x_norm = (x - mean) / torch.sqrt(var + eps)
return gamma * x_norm + beta
```

LayerNorm 과 동일한 affine (γ, β).

## Momentum 의 의미

PyTorch 의 `momentum` 은 **새로운 값의 반영 비율**. `0.1` 이면 매 배치마다 10% 씩 새 통계로 업데이트. 이는 지수이동평균 (EMA) 입니다:

$$\mu_{run}^{(t)} = (1-m) \mu_{run}^{(t-1)} + m \mu_{batch}^{(t)}$$

## BN 의 약점

1. **배치 크기 의존성**: 배치가 작으면 통계가 부정확 → LayerNorm 이 더 안정
2. **시퀀스 가변성**: RNN/Transformer 에서 시퀀스 길이가 달라지면 문제
3. **train ≠ eval**: 배포 시 모드 스위치 실수가 흔한 버그

그래서 **Transformer** 는 [LayerNorm](04_layernorm.md) / [RMSNorm](08_rmsnorm.md) 을 선호합니다.

## 검증

```python
x = torch.randn(8, 4)
gamma, beta = torch.ones(4), torch.zeros(4)
running_mean, running_var = torch.zeros(4), torch.ones(4)
out = my_batch_norm(x, gamma, beta, running_mean, running_var, training=True)
# 출력 각 열의 평균 ≈ 0, 표준편차 ≈ 1
```

## 한 걸음 더

- **GroupNorm / InstanceNorm**: 배치와 독립적인 변형들.
- CNN 에서는 `(B, C, H, W)` 에서 C 를 제외한 `(B, H, W)` 축으로 평균 → 본 예제는 2D 버전.
