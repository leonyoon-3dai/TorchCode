# 32. Top-k / Top-p Sampling — LM 생성 디코딩

> 📓 [원본 notebook](../solutions/32_topk_sampling_solution.ipynb) · 난이도 🟡

## 개념

LM 의 출력 logits 를 바로 greedy argmax 하면 **단조롭고 반복적**인 텍스트가 나옵니다. 샘플링하면 다양성이 생기지만 너무 무작위하면 난잡. 타협책:

- **Top-k**: 확률 상위 k 개 토큰 중에서만 샘플링
- **Top-p (nucleus)**: 누적 확률 p 이하의 최소 집합에서 샘플링 (개수 가변)
- **Temperature**: logits 를 T 로 나눠 분포 날카로움 조절 (T<1 샤프, T>1 플랫)

## 코드 line-by-line

```python
def sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0):
    logits = logits / max(temperature, 1e-8)

    if top_k > 0:
        top_k_val = logits.topk(top_k).values[-1]
        logits[logits < top_k_val] = float('-inf')

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = (cumsum - probs) > top_p
        sorted_logits[mask] = float('-inf')
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()
```

### Temperature

```python
logits = logits / max(temperature, 1e-8)
```

- `T < 1`: logits 커짐 → softmax 날카로움 → 더 deterministic
- `T > 1`: logits 작아짐 → softmax 평평 → 더 랜덤
- `T = 1`: 원본 그대로

### Top-k

```python
top_k_val = logits.topk(top_k).values[-1]    # k 번째로 큰 값
logits[logits < top_k_val] = float('-inf')   # 나머지는 -inf (sampling 에서 제외)
```

상위 k 개만 남기고 나머지는 softmax 후 0 이 됨.

### Top-p (nucleus)

```python
sorted_logits, sorted_idx = torch.sort(logits, descending=True)
probs = torch.softmax(sorted_logits, dim=-1)
cumsum = torch.cumsum(probs, dim=-1)
mask = (cumsum - probs) > top_p
```

- `sorted_logits` : 큰 순서로 정렬
- `cumsum` : 누적 확률 `[p1, p1+p2, p1+p2+p3, ...]`
- `cumsum - probs` : **해당 토큰 직전까지의 누적** (토큰 자신은 빼고)
- 이 값이 `top_p` 를 넘는 토큰부터 잘라냄 → "확률 상위부터 누적 p 만큼만 남김"

```python
sorted_logits[mask] = float('-inf')
logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
```

`scatter_` 로 **원래 인덱스 위치**에 되돌려 놓음. 안 하면 정렬된 상태로 남아 토큰 ID 가 뒤틀림.

### Multinomial

```python
probs = torch.softmax(logits, dim=-1)
return torch.multinomial(probs, 1).item()
```

확률로 한 개 샘플링 — 이게 실제 토큰 선택.

## 왜 `cumsum - probs` 로 edge case 처리?

`cumsum > top_p` 만 쓰면 **첫 번째로 넘긴** 토큰도 제외됨. 그러면 `top_p=0.01` 같은 극단에서 남는 토큰이 없을 수 있음. `cumsum - probs > top_p` 는 **최소 하나는** 포함 보장.

## 검증

```python
logits = torch.tensor([1.0, 5.0, 2.0, 0.5])

# top_k=1 → 항상 index 1 (logit 5.0)
sample_top_k_top_p(logits.clone(), top_k=1)   # 1

# top_p=0.5 → 거의 index 1
sample_top_k_top_p(logits.clone(), top_p=0.5) # 1 (대부분)
```

## 실전 디코딩 레시피

- **Deterministic**: greedy or `top_p=0.0` or `T=0`
- **다양성 필요**: `T=1.0, top_p=0.9, top_k=50`
- **창의적 글쓰기**: `T=1.2, top_p=0.95`

대부분 공개 모델은 top-p + temperature 기본값을 권장.

## 한 걸음 더

- **Repetition penalty**: 같은 토큰 반복시 logits 에 패널티
- **Min-p sampling**: 최상 확률 대비 비율로 필터링 (최근 기법)
- **Typical sampling**: 정보 이론적 정의
- **Beam search** ([33번](33_beam_search.md)) 는 샘플링 대신 탐색 기반
