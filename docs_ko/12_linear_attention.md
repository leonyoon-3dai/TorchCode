# 12. Linear Attention — softmax 를 kernel 로 근사

> 📓 [원본 notebook](../solutions/12_linear_attention_solution.ipynb) · 난이도 🔴

## 개념

Softmax attention 은 $\text{softmax}(QK^\top) V$ 가 $O(S^2 d)$ 메모리/연산을 요구합니다.

**Linear attention** 의 핵심 아이디어: softmax 를 두 feature map 의 내적 $\phi(Q) \phi(K)^\top$ 으로 근사하면, **결합법칙**으로 순서를 바꿔 $O(Sd^2)$ 로 줄일 수 있습니다.

$$\text{softmax}(QK^\top)V \approx \frac{\phi(Q)(\phi(K)^\top V)}{\phi(Q) \phi(K)^\top \mathbf{1}}$$

- $\phi(x) = \text{elu}(x) + 1$ 이 일반적 (양수 보장)
- 분모는 정규화 (softmax 의 행 합 = 1 역할)

## 코드 line-by-line

```python
def linear_attention(Q, K, V):
    Q_prime = F.elu(Q) + 1
    K_prime = F.elu(K) + 1
    KV = torch.bmm(K_prime.transpose(1, 2), V)       # (B, D_k, D_v)
    Z = K_prime.sum(dim=1, keepdim=True)              # (B, 1, D_k)
    num = torch.bmm(Q_prime, KV)                      # (B, S, D_v)
    den = torch.bmm(Q_prime, Z.transpose(1, 2))       # (B, S, 1)
    return num / (den + 1e-6)
```

| 라인 | 코드 | 설명 |
|------|------|------|
| 2-3 | `F.elu(x) + 1` | ELU 는 음수는 부드럽게 수렴, 양수는 identity. `+1` 로 **항상 양수**로 만듦 (확률 해석 유지). |
| 4 | `K_prime.T @ V` | shape `(B, d_k, S) @ (B, S, d_v) = (B, d_k, d_v)`. **S 가 먼저 사라짐 → 핵심** |
| 5 | `K_prime.sum(dim=1)` | `(B, 1, d_k)`. 분모 정규화용. softmax 의 $\sum e^{k}$ 와 유사. |
| 6 | `Q_prime @ KV` | `(B, S, d_k) @ (B, d_k, d_v) = (B, S, d_v)`. 실제 출력. |
| 7 | `Q_prime @ Z.T` | `(B, S, d_k) @ (B, d_k, 1) = (B, S, 1)`. 각 query 의 정규화 상수. |
| 8 | `num / (den + 1e-6)` | 각 query 별 가중합 / 정규화. `1e-6` 은 0-division 방지. |

## 왜 빨라지는가

일반 attention:
```
scores = Q @ K.T     # (S, S) — S² 메모리
out    = scores @ V  # (S, d) — S²d 연산
```

Linear attention:
```
KV = K.T @ V   # (d_k, d_v) — S 가 사라짐!
out = Q @ KV   # (S, d_v) — Sd 연산
```

복잡도: $O(Sd^2)$ vs $O(S^2 d)$. $S \gg d$ 일 때 압도적 이득.

## trade-off

- **장점**: 긴 시퀀스 (수천~수만 토큰) 에 유리. 메모리 상수.
- **단점**: softmax 의 sharp attention 이 사라져 **집중도**가 떨어짐. 복잡한 의존성 학습이 어려울 수 있음.
- 해결: Performer, RetNet, Mamba 등 다양한 개선 연구.

## 한 걸음 더

- Performer: 가우시안 커널 + 무작위 특징 사상
- RetNet, Mamba: state-space / retention 관점의 재구성
- 본 구현은 causal 아님 — causal 버전은 누적합으로 $O(Sd^2)$ 유지
