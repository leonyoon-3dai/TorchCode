# 08. RMSNorm — LayerNorm 의 경량화 버전

> 📓 [원본 notebook](../solutions/08_rmsnorm_solution.ipynb) · 난이도 🟡

## 개념

RMSNorm 은 **LayerNorm 에서 평균 제거(centering) 단계를 없앤** 변형입니다. 분산 대신 **RMS (Root Mean Square)** 로 나눕니다.

$$\text{RMS}(x) = \sqrt{\frac{1}{D}\sum_i x_i^2 + \epsilon}, \quad \text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

**왜?** 평균 제거 없이도 학습이 잘 되고, 한 번의 연산(평균) 이 줄어 빠릅니다. LLaMA, Mistral, T5 등 최신 LLM 이 채택.

## 코드 line-by-line

```python
def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight
```

| 라인 | 코드 | 설명 |
|------|------|------|
| 2 | `x.pow(2)` | 원소별 제곱 = $x_i^2$. |
|   | `.mean(dim=-1, keepdim=True)` | feature 축 평균 = $\frac{1}{D}\sum x_i^2$. keepdim 유지로 broadcast 준비. |
|   | `+ eps` | 0-division 방지. LN 보다 더 작은 `1e-6` 이 일반적. |
|   | `torch.sqrt(...)` | RMS 완성. |
| 3 | `x / rms` | 원소별 나눗셈. 결과 벡터의 RMS = 1 에 가까워짐. |
|   | `* weight` | 학습 가능한 scale (γ). **bias 는 없음** (LN 과의 또 다른 차이). |

## LayerNorm vs RMSNorm

| 항목 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 평균 제거 | ✅ | ❌ |
| 분산 vs RMS | $\sigma = \sqrt{\text{Var}}$ | $\sqrt{\text{mean}(x^2)}$ |
| 학습 파라미터 | γ, β | γ 만 |
| 연산량 | 평균 1회 + 분산 1회 | 제곱평균 1회 |
| 속도 | 기준 | **더 빠름** |

수학적으로, **평균이 0 근처라면** 분산 = mean(x²) 라 두 정규화가 거의 같아집니다. 실제로 많은 활성화 출력은 이미 평균이 0 근처이므로 RMSNorm 이 충분히 잘 동작합니다.

## 검증

```python
x = torch.randn(2, 8)
out = rms_norm(x, torch.ones(8))
print(out.pow(2).mean(dim=-1).sqrt())
# tensor([1.0000, 1.0000])  — 출력의 RMS 가 정확히 1
```

## 한 걸음 더

- [LayerNorm (04번)](04_layernorm.md) — 전체 수식과 BN 과의 비교
- LLaMA 같은 모델은 모든 Transformer 블록에서 LN 대신 RMSNorm 을 씀
- **Pre-norm vs Post-norm** 구조도 연관 — Pre-norm + RMSNorm 이 사실상 표준
