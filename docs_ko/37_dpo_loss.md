# 37. DPO Loss — RLHF 없이 preference 학습

> 📓 [원본 notebook](../solutions/37_dpo_loss_solution.ipynb) · 난이도 🔴

## 개념

**DPO (Direct Preference Optimization)**: RLHF 의 reward model + PPO 를 **한 번의 classification-like loss** 로 대체. human preference 데이터 `(chosen, rejected)` 만으로 policy 를 직접 최적화.

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$

- $y_w$: 선호 응답 (chosen), $y_l$: 거부 응답 (rejected)
- $\pi_\text{ref}$: 참조 (보통 SFT 모델)
- $\beta$: KL 정규화 세기. 클수록 ref 에서 멀어짐에 패널티

직관: "chosen 의 log-ratio 가 rejected 보다 커지도록 학습". sigmoid 는 Bradley-Terry preference 모델.

## 코드 line-by-line

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

| 라인 | 설명 |
|------|------|
| `policy_chosen_logps` | 현재 policy 가 chosen response 에 주는 log-prob 합 (토큰 축 sum) |
| `ref_chosen_logps` | 참조 모델의 동일 계산 |
| `chosen_rewards = β·(policy - ref)` | 이게 바로 **암묵적 reward**. DPO 의 수학적 유도에 따르면 이 값이 reward 역할. |
| `rejected_rewards` | 마찬가지 |
| `chosen_rewards - rejected_rewards` | 선호 차이 (logit) |
| `F.logsigmoid(·)` | $\log \sigma(·)$. 수치 안정 구현 |
| `-(...).mean()` | 음수 로그우도 평균 |

## 왜 `logsigmoid`?

Bradley-Terry 모델: $P(y_w \succ y_l) = \sigma(r_w - r_l)$. 이를 log-likelihood 로 바꾸면 $\log \sigma(\cdot)$. `F.logsigmoid` 는 `log(1 + exp(-x))` 를 수치 안정하게 계산.

## DPO vs PPO

| 항목 | DPO | PPO ([39번](39_ppo_loss.md)) |
|------|-----|------|
| Reward model | 필요 없음 | 필요 |
| Online rollout | 필요 없음 (off-policy) | 필요 |
| 구현 복잡도 | 단순 (loss 하나) | 복잡 (policy+value+reward) |
| 데이터 | `(chosen, rejected)` 쌍 | scalar reward |

DPO 는 구현과 튜닝이 훨씬 쉬워 LLM 미세조정에서 급속 확산.

## β 의 역할

- 작은 β (예: 0.1): ref 와 가까이 유지 (보수적)
- 큰 β (예: 1.0): 큰 변화 허용 (공격적)
- 너무 크면 ref 에서 멀어져 **reward hacking** 위험

## 검증

```python
chosen = torch.tensor([0.0, 0.0])       # 높은 log-prob
rejected = torch.tensor([-5.0, -5.0])   # 낮은 log-prob
ref_c = torch.tensor([-1.0, -1.0])
ref_r = torch.tensor([-1.0, -1.0])
dpo_loss(chosen, rejected, ref_c, ref_r, beta=0.1).item()
# 작은 값 — policy 가 이미 preference 잘 반영
```

## 실제 구현 주의

- **log-probs** 는 각 토큰의 `log p(y_t | y_<t)` 를 **응답 길이 만큼 합** 함
- **padding mask** 로 padding 토큰 제외
- **batched pair**: chosen/rejected 를 concat 해 한 번의 forward 로 효율화

## 한 걸음 더

- **IPO**: DPO 의 수치 불안정 개선
- **KTO** (Kahneman-Tversky Optimization): 단일 응답 + binary label
- **SLiC**: sequence-level contrastive
- DPO 가 너무 보수적인 경우 PPO/GRPO 와 결합하기도 함
