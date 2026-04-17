# 18. Embedding — 토큰 ID → 벡터

> 📓 [원본 notebook](../solutions/18_embedding_solution.ipynb) · 난이도 🟢

## 개념

**Embedding layer** 는 **룩업 테이블** 입니다. 정수 ID 를 밀집 벡터로 변환:

$$\text{Embedding}: \{0, 1, ..., V-1\} \to \mathbb{R}^D$$

행렬 $W \in \mathbb{R}^{V \times D}$ 를 두고, ID `i` 를 받으면 $W[i]$ 를 돌려줍니다. 이 행은 학습 대상. LLM 의 가장 앞단에 위치해 토큰을 벡터로 바꿔주는 역할.

## 코드 line-by-line

```python
class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, indices):
        return self.weight[indices]
```

| 라인 | 코드 | 설명 |
|------|------|------|
| 4 | `nn.Parameter(torch.randn(V, D))` | `(V, D)` 행렬을 **학습 파라미터** 로 등록. `Parameter` 로 감싸야 `model.parameters()` 에 포함됨. |
| 7 | `self.weight[indices]` | advanced indexing. `indices.shape = (*)` 이면 `(*, D)` 반환. batch, seq 차원 자유. |

## 왜 `Parameter` 인가

```python
nn.Parameter(x)  # autograd 추적 + optimizer 에 자동 포함
x                # 그냥 텐서 — 업데이트 안 됨
```

`nn.Module.parameters()` 는 `Parameter` 속성만 수집합니다. 이 라벨이 없으면 옵티마이저가 학습을 못 함.

## Indexing 의 힘

```python
indices = torch.tensor([0, 3, 7])    # (3,)
emb(indices).shape                    # (3, D)

indices = torch.tensor([[0, 1], [2, 3]])  # (2, 2)
emb(indices).shape                          # (2, 2, D)
```

임의 shape 의 정수 텐서 → 마지막 축에 D 가 추가된 출력. 배치 + 시퀀스 자연스럽게 처리.

## 학습의 의미

Cross-entropy 로 역전파하면 **해당 ID 의 행만** 업데이트됩니다. 다른 행은 기울기 0. 그래서 자주 등장한 토큰만 풍부하게 학습됨.

PyTorch 의 `nn.Embedding` 에는 `sparse=True` 옵션으로 이를 활용한 sparse gradient 지원.

## 차원 흐름 (전형적 LLM)

```
token_ids   : (B, S)   — 정수
embedding   : (B, S, D)
+ pos emb   : (B, S, D)  [학습가능 position]  ← GPT-2 스타일
                          (RoPE 는 attention 안에서 적용)
```

## 크기 감각

- GPT-2: V=50257, D=768 → 38.6M 파라미터 (**모델 총 파라미터의 30%+**)
- LLaMA: 같은 embedding 이 **LM head 와 가중치 공유** (weight tying)

## 한 걸음 더

- [Linear Layer (03번)](03_linear.md) 과의 관계: embedding 은 원-핫 입력에 대한 Linear 와 수학적 동치
- **Weight tying**: 입력 embedding 과 출력 projection (LM head) 의 가중치를 공유 → 파라미터 절감
- **Positional embedding** vs **RoPE** ([24번](24_rope.md)): 위치 정보 주입 방법들
