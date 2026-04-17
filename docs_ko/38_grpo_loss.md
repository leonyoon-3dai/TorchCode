# 38. GRPO Loss — DeepSeek 스타일 그룹 상대 정책 최적화

> 📓 [원본 notebook](../solutions/38_grpo_loss_solution.ipynb) · 난이도 🔴

## 개념

**GRPO (Group Relative Policy Optimization)**: PPO 의 value model 을 제거하고, **같은 prompt 에서 샘플링된 여러 응답의 reward 를 그룹 내 정규화** 하여 advantage 를 계산.

- PPO 는 critic (value network) 필요 → 메모리 부담
- GRPO 는 한 prompt 에 G 개 응답 → 그룹 평균 빼고 표준편차로 나눠 advantage

$$A_i = \frac{r_i - \text{mean}(r_{\text{group}})}{\text{std}(r_{\text{group}}) + \epsilon}$$

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}[A_i \cdot \log \pi_\theta(y_i|x)]$$

DeepSeek-R1, Qwen-Math 등 강력한 추론 모델이 채택.

## 코드 line-by-line

```python
def grpo_loss(logps, rewards, group_ids, eps=1e-5):
    unique_ids = group_ids.unique()
    advantages = torch.empty_like(rewards)
    for gid in unique_ids:
        mask = group_ids == gid
        r_g = rewards[mask]
        mean_g = r_g.mean()
        std_g = r_g.std(unbiased=False)
        advantages[mask] = (r_g - mean_g) / (std_g + eps)

    advantages_detached = advantages.detach()
    return -(advantages_detached * logps).mean()
```

| 라인 | 설명 |
|------|------|
| `logps`, `rewards`, `group_ids` | 모두 `(B,)` shape. group_ids 로 같은 prompt 에서 온 샘플 식별 |
| `unique_ids` | 고유 그룹 수 만큼 순회 |
| `mask = group_ids == gid` | 해당 그룹에 속한 샘플 boolean mask |
| `r_g.mean(), r_g.std()` | **그룹 내** 평균/표준편차 |
| `(r_g - mean_g) / (std_g + eps)` | z-score 정규화. 그룹 내에서 얼마나 잘했는지의 상대값 |

### Detach 이유

```python
advantages_detached = advantages.detach()
```

Advantage 는 **reward 로부터 유도**된 값이지만, gradient 는 `logps` 에만 흐르게 해야 함. reward 로 역전파하면 의미가 깨짐.

### 최종 loss

```python
return -(advantages_detached * logps).mean()
```

Policy gradient 의 기본 형태: "좋은 action (A > 0) 의 log-prob 을 증가, 나쁜 것은 감소". 평균으로 배치 reduction.

## PPO 와의 차이

| 항목 | PPO | GRPO |
|------|-----|------|
| Advantage | critic value 기반 GAE | 그룹 내 z-score |
| Critic 학습 | 필요 | 없음 |
| 메모리 | policy + value | policy 만 |
| Clip | 있음 | 원 논문은 있음 (본 구현은 단순화) |

본 예제는 GRPO 의 core 만 구현. 전체 구현은 PPO-style clip 과 KL penalty 도 포함.

## 왜 그룹 정규화가 효과적인가

수학 문제 RL 을 예로 들면:
- 쉬운 문제: 모든 샘플 reward 1 → advantage 0 (차이 없음, update 작음)
- 어려운 문제: 일부만 성공 → 성공 샘플이 큰 양 advantage

즉 **난이도에 자동 적응**하는 효과. Critic 없이도.

## 검증

```python
logps = torch.tensor([0.0, -0.5, -1.0, -1.5])
rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])
group_ids = torch.tensor([0, 0, 1, 1])
grpo_loss(logps, rewards, group_ids)
# 양수 loss. 그룹 0 에서 logps=0.0 이 reward 1.0 이라 advantage 양수
```

그룹 0: `mean = 0.9, std = 0.1`. 샘플 0 의 advantage = `(1.0 - 0.9) / 0.1 = 1.0`.

## 한 걸음 더

- **DeepSeek-R1**: GRPO + rule-based reward (수학 정답 체크) → R1 reasoning
- **Clip + KL penalty** 를 추가한 full GRPO 는 더 안정
- Group size G: 보통 8~64. 너무 작으면 분산 큼, 너무 크면 generation 비용
