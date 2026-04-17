"""한국어 문서용 시각 자료 생성 스크립트."""
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "font.family": ["AppleGothic", "Nanum Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ─── 1. Activation functions (ReLU / Softmax input / GELU / SiLU) ───
def save_activations():
    x = np.linspace(-4, 4, 400)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    axes[0, 0].plot(x, np.maximum(0, x), color="#2a9d8f", lw=2)
    axes[0, 0].set_title("ReLU: max(0, x)")
    axes[0, 0].axhline(0, color="k", lw=0.5); axes[0, 0].axvline(0, color="k", lw=0.5)

    def gelu(x): return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    axes[0, 1].plot(x, gelu(x), color="#e76f51", lw=2)
    axes[0, 1].plot(x, np.maximum(0, x), "--", color="gray", lw=1, label="ReLU")
    axes[0, 1].set_title("GELU: 0.5·x·(1+erf(x/√2))")
    axes[0, 1].legend()
    axes[0, 1].axhline(0, color="k", lw=0.5); axes[0, 1].axvline(0, color="k", lw=0.5)

    silu = x / (1 + np.exp(-x))
    axes[1, 0].plot(x, silu, color="#264653", lw=2)
    axes[1, 0].set_title("SiLU/Swish: x·σ(x)")
    axes[1, 0].axhline(0, color="k", lw=0.5); axes[1, 0].axvline(0, color="k", lw=0.5)

    # softmax of a simple vector
    v = np.array([1.0, 2.0, 3.0])
    sm = np.exp(v - v.max()); sm /= sm.sum()
    axes[1, 1].bar(["x=1", "x=2", "x=3"], sm, color=["#adc178", "#6a994e", "#386641"])
    axes[1, 1].set_title("Softmax(1,2,3) = 확률분포")
    axes[1, 1].set_ylabel("확률")
    for i, v in enumerate(sm):
        axes[1, 1].text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(OUT / "activations.png", bbox_inches="tight")
    plt.close()


# ─── 2. LayerNorm vs BatchNorm vs RMSNorm (정규화 축 비교) ───
def save_norm_axes():
    np.random.seed(0)
    data = np.random.randn(4, 6) * 2 + 1

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    titles = ["BatchNorm\n(feature 축, 배치 평균)",
              "LayerNorm\n(sample 축, 각 행 평균)",
              "RMSNorm\n(sample 축, mean 제거 X)"]
    for ax, t in zip(axes, titles):
        im = ax.imshow(data, cmap="RdBu_r", vmin=-4, vmax=4)
        ax.set_title(t)
        ax.set_xlabel("feature dim")
        ax.set_ylabel("batch dim")
    # 화살표로 정규화 축 표시
    axes[0].annotate("", xy=(0, 3.8), xytext=(0, -0.2), arrowprops=dict(arrowstyle="<->", color="orange", lw=2))
    axes[1].annotate("", xy=(5.8, 0), xytext=(-0.2, 0), arrowprops=dict(arrowstyle="<->", color="orange", lw=2))
    axes[2].annotate("", xy=(5.8, 0), xytext=(-0.2, 0), arrowprops=dict(arrowstyle="<->", color="orange", lw=2))
    plt.tight_layout()
    plt.savefig(OUT / "norm_axes.png", bbox_inches="tight")
    plt.close()


# ─── 3. Attention heatmap (causal vs sliding vs full) ───
def save_attention_masks():
    S = 8
    full = np.ones((S, S))
    causal = np.tril(np.ones((S, S)))
    idx = np.arange(S)
    sw = (np.abs(idx[:, None] - idx[None, :]) <= 2).astype(float)
    sw = sw * causal  # sliding + causal

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, m, t in zip(axes, [full, causal, sw], ["Full attention", "Causal (미래 차단)", "Sliding window (w=2)"]):
        ax.imshow(m, cmap="Greens")
        ax.set_title(t)
        ax.set_xlabel("key pos"); ax.set_ylabel("query pos")
    plt.tight_layout()
    plt.savefig(OUT / "attention_masks.png", bbox_inches="tight")
    plt.close()


# ─── 4. Cosine LR schedule ───
def save_cosine_lr():
    total, warmup, max_lr = 100, 10, 1e-3
    def sched(s):
        if s < warmup: return max_lr * s / warmup
        if s >= total: return 0
        p = (s - warmup) / (total - warmup)
        return 0.5 * max_lr * (1 + math.cos(math.pi * p))
    steps = np.arange(total + 1)
    lrs = [sched(s) for s in steps]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, lrs, color="#bc4749", lw=2)
    ax.axvline(warmup, color="gray", ls="--", label=f"warmup 끝 ({warmup})")
    ax.set_xlabel("step"); ax.set_ylabel("learning rate")
    ax.set_title("Cosine LR with Warmup")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "cosine_lr.png", bbox_inches="tight")
    plt.close()


# ─── 5. Multi-head attention 구조 ───
def save_mha_diagram():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    # input
    ax.add_patch(plt.Rectangle((0.2, 2.5), 1.5, 1, facecolor="#a8dadc", edgecolor="k"))
    ax.text(0.95, 3, "x\n(B,S,D)", ha="center", va="center")
    # Linears
    for i, (label, color) in enumerate([("W_q", "#457b9d"), ("W_k", "#1d3557"), ("W_v", "#e63946")]):
        y = 4.5 - i * 1.5
        ax.add_patch(plt.Rectangle((2.5, y - 0.4), 1.2, 0.8, facecolor=color, edgecolor="k"))
        ax.text(3.1, y, label, ha="center", va="center", color="white", fontweight="bold")
        ax.annotate("", xy=(2.5, y), xytext=(1.7, 3), arrowprops=dict(arrowstyle="->"))
    # split heads
    ax.add_patch(plt.Rectangle((4.3, 1.5), 1.8, 3, facecolor="#f1faee", edgecolor="k"))
    ax.text(5.2, 3, "split\nheads\n(B,h,S,d_k)", ha="center", va="center", fontsize=9)
    # attention
    ax.add_patch(plt.Rectangle((6.6, 1.5), 1.8, 3, facecolor="#fcbf49", edgecolor="k"))
    ax.text(7.5, 3, "Attention\nsoftmax(QK/√d)V", ha="center", va="center", fontsize=9)
    # concat + W_o
    ax.add_patch(plt.Rectangle((8.7, 2.5), 1.1, 1, facecolor="#2a9d8f", edgecolor="k"))
    ax.text(9.25, 3, "concat\n+ W_o", ha="center", va="center", color="white", fontsize=9)
    ax.annotate("", xy=(4.3, 3), xytext=(3.7, 3), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(6.6, 3), xytext=(6.1, 3), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(8.7, 3), xytext=(8.4, 3), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Multi-Head Attention 구조", fontsize=13)
    plt.savefig(OUT / "mha_diagram.png", bbox_inches="tight")
    plt.close()


# ─── 6. Dropout 시각화 ───
def save_dropout():
    np.random.seed(0)
    x = np.ones((5, 10))
    mask = np.random.rand(5, 10) > 0.5
    out = x * mask / 0.5
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.5))
    axes[0].imshow(x, cmap="Blues", vmin=0, vmax=2); axes[0].set_title("입력 (모두 1)")
    axes[1].imshow(mask.astype(float), cmap="Blues"); axes[1].set_title("mask (p=0.5)")
    axes[2].imshow(out, cmap="Blues", vmin=0, vmax=2); axes[2].set_title("출력 = x·mask / (1-p)")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUT / "dropout.png", bbox_inches="tight")
    plt.close()


# ─── 7. GQA (Grouped Query Attention) ───
def save_gqa():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    # MHA
    for i in range(8):
        ax.add_patch(plt.Rectangle((0.5 + i * 0.35, 2.8), 0.3, 0.5, facecolor="#457b9d"))
        ax.add_patch(plt.Rectangle((0.5 + i * 0.35, 2.2), 0.3, 0.5, facecolor="#e63946"))
        ax.add_patch(plt.Rectangle((0.5 + i * 0.35, 1.6), 0.3, 0.5, facecolor="#f4a261"))
    ax.text(1.9, 3.5, "MHA: 8 Q + 8 KV", ha="center", fontweight="bold")

    # GQA: 8 Q, 2 KV
    for i in range(8):
        ax.add_patch(plt.Rectangle((4.5 + i * 0.35, 2.8), 0.3, 0.5, facecolor="#457b9d"))
    for i in range(2):
        ax.add_patch(plt.Rectangle((4.5 + i * 1.4, 2.2), 1.3, 0.5, facecolor="#e63946"))
        ax.add_patch(plt.Rectangle((4.5 + i * 1.4, 1.6), 1.3, 0.5, facecolor="#f4a261"))
    ax.text(5.9, 3.5, "GQA: 8 Q + 2 KV (공유)", ha="center", fontweight="bold")

    # labels on the left
    ax.text(0.2, 3.05, "Q", ha="right", va="center")
    ax.text(0.2, 2.45, "K", ha="right", va="center")
    ax.text(0.2, 1.85, "V", ha="right", va="center")
    ax.set_title("MHA vs GQA — KV 헤드 공유로 메모리/대역폭 절감")
    plt.savefig(OUT / "gqa.png", bbox_inches="tight")
    plt.close()


# ─── 8. ViT patch embedding ───
def save_vit_patch():
    np.random.seed(1)
    img = np.random.rand(16, 16, 3)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    axes[0].imshow(img); axes[0].set_title("입력 이미지 (H×W×C)")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(img); axes[1].set_title("4×4 패치로 분할")
    for i in range(1, 4):
        axes[1].axhline(i * 4 - 0.5, color="white", lw=2)
        axes[1].axvline(i * 4 - 0.5, color="white", lw=2)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    tokens = np.random.rand(16, 8)
    axes[2].imshow(tokens, cmap="viridis", aspect="auto")
    axes[2].set_title("Linear 투영 → (N_patches, D)")
    axes[2].set_xlabel("embed_dim"); axes[2].set_ylabel("patch index")
    plt.tight_layout()
    plt.savefig(OUT / "vit_patch.png", bbox_inches="tight")
    plt.close()


# ─── 9. KV cache 시각화 ───
def save_kv_cache():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    # without cache: O(S^2)
    s = np.arange(1, 11)
    axes[0].plot(s, s * s, "r-o", label="Cache X: O(S²)")
    axes[0].plot(s, s, "g-o", label="Cache O: O(S)")
    axes[0].set_xlabel("생성 토큰 수"); axes[0].set_ylabel("연산량 (step당)")
    axes[0].set_title("KV Cache 효과")
    axes[0].legend()

    # visual of cache filling up
    cache = np.zeros((4, 8))
    for i in range(4):
        cache[i, : i + 1] = 1
    axes[1].imshow(cache, cmap="Greens", aspect="auto")
    axes[1].set_title("KV 캐시 확장 (step마다 1개씩 추가)")
    axes[1].set_xlabel("cache slot"); axes[1].set_ylabel("step")
    plt.tight_layout()
    plt.savefig(OUT / "kv_cache.png", bbox_inches="tight")
    plt.close()


# ─── 10. Adam optimizer trajectory ───
def save_adam_traj():
    # Simple bowl loss
    def loss(x, y): return x ** 2 + 10 * y ** 2
    def grad(x, y): return np.array([2 * x, 20 * y])

    def train(optimizer):
        path = [(3.0, 1.5)]
        state = {"m": np.zeros(2), "v": np.zeros(2), "t": 0}
        p = np.array([3.0, 1.5])
        for _ in range(40):
            g = grad(*p)
            if optimizer == "sgd":
                p = p - 0.05 * g
            else:
                state["t"] += 1
                state["m"] = 0.9 * state["m"] + 0.1 * g
                state["v"] = 0.999 * state["v"] + 0.001 * g ** 2
                m_hat = state["m"] / (1 - 0.9 ** state["t"])
                v_hat = state["v"] / (1 - 0.999 ** state["t"])
                p = p - 0.3 * m_hat / (np.sqrt(v_hat) + 1e-8)
            path.append(tuple(p))
        return np.array(path)

    sgd = train("sgd"); adam = train("adam")
    xs, ys = np.meshgrid(np.linspace(-4, 4, 60), np.linspace(-2, 2, 60))
    zs = xs ** 2 + 10 * ys ** 2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.contour(xs, ys, zs, levels=20, cmap="gray", alpha=0.5)
    ax.plot(sgd[:, 0], sgd[:, 1], "r-o", ms=3, label="SGD")
    ax.plot(adam[:, 0], adam[:, 1], "b-o", ms=3, label="Adam")
    ax.set_title("SGD vs Adam — 좁은 골짜기에서의 경로")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "adam_traj.png", bbox_inches="tight")
    plt.close()


# ─── 11. RoPE 회전 시각화 ───
def save_rope():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pos_list = [0, 1, 3, 7]
    colors = plt.cm.viridis(np.linspace(0, 1, len(pos_list)))
    ax = axes[0]
    theta_list = []
    for p, c in zip(pos_list, colors):
        theta = p * 1.0
        ax.plot([0, np.cos(theta)], [0, np.sin(theta)], "-o", color=c, label=f"pos={p}")
        theta_list.append(theta)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_title("RoPE: 위치에 따라 벡터 회전")
    ax.legend()
    circle = plt.Circle((0, 0), 1, color="gray", fill=False, ls="--")
    ax.add_patch(circle)

    # dot product decay with distance
    dists = np.arange(16)
    dot = np.cos(dists * 0.5)
    axes[1].plot(dists, dot, "-o", color="#e76f51")
    axes[1].set_xlabel("위치 거리 |i-j|"); axes[1].set_ylabel("상대 내적 성분")
    axes[1].set_title("거리에 따라 감쇠하는 상대적 패턴")
    plt.tight_layout()
    plt.savefig(OUT / "rope.png", bbox_inches="tight")
    plt.close()


# ─── 12. Flash attention tiling ───
def save_flash():
    fig, ax = plt.subplots(figsize=(7, 6))
    S = 12; bs = 4
    for i in range(0, S, bs):
        for j in range(0, S, bs):
            color = "#ffe5ec" if (i + j) % 2 == 0 else "#ffc2d1"
            ax.add_patch(plt.Rectangle((j, S - i - bs), bs, bs, facecolor=color, edgecolor="black"))
    ax.set_xlim(0, S); ax.set_ylim(0, S); ax.set_aspect("equal")
    ax.set_xlabel("Key positions (블록 단위 순회)")
    ax.set_ylabel("Query positions")
    ax.set_title("Flash Attention: 블록 타일링 + online softmax")
    plt.tight_layout()
    plt.savefig(OUT / "flash.png", bbox_inches="tight")
    plt.close()


# ─── 13. LoRA ───
def save_lora():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    # Frozen W
    ax.add_patch(plt.Rectangle((1, 1), 2, 2, facecolor="#d4d4d4", edgecolor="k"))
    ax.text(2, 2, "W\n(frozen)\nd×d", ha="center", va="center", fontweight="bold")
    # LoRA A
    ax.add_patch(plt.Rectangle((4.5, 1.7), 0.4, 0.6, facecolor="#90e0ef", edgecolor="k"))
    ax.text(4.7, 2, "A\nr×d", ha="center", va="center", fontsize=8)
    # LoRA B
    ax.add_patch(plt.Rectangle((5.3, 1), 0.6, 2, facecolor="#f72585", edgecolor="k"))
    ax.text(5.6, 2, "B\nd×r", ha="center", va="center", color="white", fontsize=8)
    ax.text(7, 2, "+ α/r", ha="center", va="center", fontsize=11)
    ax.text(8.5, 2, "= ΔW", ha="center", va="center", fontweight="bold")
    ax.text(5, 3.3, "학습 파라미터: 2 × d × r  ≪  d²", ha="center", fontsize=10, color="#6a040f")
    ax.set_title("LoRA — 저차원 행렬 A, B만 학습", fontsize=13)
    plt.savefig(OUT / "lora.png", bbox_inches="tight")
    plt.close()


# ─── 14. PPO / DPO / GRPO RL 비교 ───
def save_rl_losses():
    ratios = np.linspace(0.5, 1.5, 200)
    adv_pos = 1.0; adv_neg = -1.0
    clip = 0.2
    for fname, adv, title in [("ppo_pos.png", adv_pos, "PPO clipped (A>0)"),
                              ("ppo_neg.png", adv_neg, "PPO clipped (A<0)")]:
        unclip = ratios * adv
        clipped = np.clip(ratios, 1 - clip, 1 + clip) * adv
        obj = np.minimum(unclip, clipped)
        fig, ax = plt.subplots(figsize=(6, 3.3))
        ax.plot(ratios, unclip, "--", color="gray", label="r·A (unclip)")
        ax.plot(ratios, clipped, ":", color="orange", label="clipped")
        ax.plot(ratios, obj, "-", color="#8338ec", lw=2, label="min(·)")
        ax.axvline(1 - clip, color="red", ls=":", alpha=0.5)
        ax.axvline(1 + clip, color="red", ls=":", alpha=0.5)
        ax.set_xlabel("r = π_new/π_old"); ax.set_ylabel("Objective")
        ax.set_title(title); ax.legend()
        plt.tight_layout()
        plt.savefig(OUT / fname, bbox_inches="tight")
        plt.close()


# ─── 15. Beam search tree ───
def save_beam_tree():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 5); ax.set_ylim(0, 4); ax.axis("off")
    # Depth 1
    ax.plot([0.3, 1.5], [2, 3], "k-"); ax.plot([0.3, 1.5], [2, 1], "k-")
    ax.annotate("<s>", (0.3, 2), ha="center", fontsize=10, bbox=dict(boxstyle="circle", facecolor="#e0aaff"))
    ax.annotate("A\n-0.3", (1.5, 3), ha="center", fontsize=9, bbox=dict(boxstyle="round", facecolor="#c77dff"))
    ax.annotate("B\n-0.8", (1.5, 1), ha="center", fontsize=9, bbox=dict(boxstyle="round", facecolor="#c77dff"))
    # Depth 2 — each branch has 2 more
    ax.plot([1.5, 3], [3, 3.5], "k-"); ax.plot([1.5, 3], [3, 2.5], "k-")
    ax.plot([1.5, 3], [1, 1.5], "k-"); ax.plot([1.5, 3], [1, 0.5], "k-")
    for x, y, text, sc in [(3, 3.5, "AA", -0.5), (3, 2.5, "AB", -1.2), (3, 1.5, "BA", -1.0), (3, 0.5, "BB", -2.0)]:
        color = "#9d4edd" if sc > -1.3 else "#d4d4d4"
        ax.annotate(f"{text}\n{sc}", (x, y), ha="center", fontsize=9, bbox=dict(boxstyle="round", facecolor=color))
    ax.text(4.3, 3, "beam=2:\nAA, BA 유지", fontsize=10, color="#6a040f")
    ax.set_title("Beam Search (beam_width=2)")
    plt.savefig(OUT / "beam.png", bbox_inches="tight")
    plt.close()


# ─── 16. Quantization ───
def save_quant():
    x = np.linspace(-1, 1, 400)
    scale = 1.0 / 127
    x_q = np.round(x / scale).clip(-128, 127) * scale
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(x, x, label="FP32", color="gray")
    axes[0].plot(x, x_q, label="INT8 반양자화", color="#d00000", lw=1)
    axes[0].legend(); axes[0].set_title("INT8 양자화 계단 함수")

    bits = [32, 16, 8, 4]
    mem = [4, 2, 1, 0.5]
    axes[1].bar([f"FP{b}" if b >= 16 else f"INT{b}" for b in bits], mem, color=["#023e8a", "#0077b6", "#0096c7", "#00b4d8"])
    axes[1].set_ylabel("weight당 바이트"); axes[1].set_title("정밀도별 메모리")
    for i, v in enumerate(mem):
        axes[1].text(i, v + 0.05, f"{v}B", ha="center")
    plt.tight_layout()
    plt.savefig(OUT / "quant.png", bbox_inches="tight")
    plt.close()


# ─── 17. Linear regression fit ───
def save_linreg():
    np.random.seed(0)
    X = np.random.randn(50)
    y = 2 * X + 1 + np.random.randn(50) * 0.5
    # closed form
    X_aug = np.c_[X, np.ones_like(X)]
    theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    xs = np.linspace(X.min(), X.max(), 100)
    ys = theta[0] * xs + theta[1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(X, y, alpha=0.6, label="데이터")
    ax.plot(xs, ys, "r-", lw=2, label=f"회귀직선 y={theta[0]:.2f}x + {theta[1]:.2f}")
    ax.legend()
    ax.set_title("Linear Regression — 최소제곱 피팅")
    plt.tight_layout()
    plt.savefig(OUT / "linreg.png", bbox_inches="tight")
    plt.close()


# ─── 18. Conv2d 시각화 ───
def save_conv2d():
    np.random.seed(2)
    img = np.zeros((8, 8))
    img[2:6, 2:6] = 1
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4.0
    out = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            out[i, j] = (img[i:i+3, j:j+3] * kernel).sum()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("입력 (정사각형)")
    axes[1].imshow(kernel, cmap="RdBu_r"); axes[1].set_title("커널 (Sobel-X)")
    axes[2].imshow(out, cmap="RdBu_r"); axes[2].set_title("출력 = 엣지 검출")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUT / "conv2d.png", bbox_inches="tight")
    plt.close()


# ─── 19. Cross entropy ───
def save_cross_entropy():
    p = np.linspace(0.01, 0.99, 200)
    ce = -np.log(p)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p, ce, "b-", lw=2)
    ax.axvline(1.0, ls="--", color="green")
    ax.text(0.85, 4, "정답일 때 loss=0\n(확률 1에 수렴)", color="green", fontsize=9)
    ax.set_xlabel("정답 토큰의 확률 p"); ax.set_ylabel("-log p")
    ax.set_title("Cross-Entropy Loss — 확률이 낮을수록 큰 패널티")
    plt.tight_layout()
    plt.savefig(OUT / "cross_entropy.png", bbox_inches="tight")
    plt.close()


# ─── 20. Gradient clipping ───
def save_grad_clip():
    np.random.seed(0)
    steps = np.arange(50)
    norms = np.abs(np.random.randn(50) * 2 + 0.5)
    norms[10] = 8; norms[25] = 12  # exploding spikes
    clipped = np.minimum(norms, 1.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, norms, "r-o", label="원본 grad norm")
    ax.plot(steps, clipped, "g-o", label="clipped (max=1.0)")
    ax.axhline(1.0, ls="--", color="gray")
    ax.set_xlabel("step"); ax.set_ylabel("grad norm")
    ax.set_title("Gradient Clipping — 폭발 방지")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "grad_clip.png", bbox_inches="tight")
    plt.close()


# ─── 21. MoE routing ───
def save_moe():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    # tokens
    for i in range(4):
        ax.add_patch(plt.Rectangle((0.5, 0.5 + i), 1, 0.8, facecolor="#a8dadc", edgecolor="k"))
        ax.text(1, 0.9 + i, f"token {i}", ha="center", va="center", fontsize=8)
    # router
    ax.add_patch(plt.Rectangle((2.5, 2), 1.2, 1, facecolor="#f1c40f", edgecolor="k"))
    ax.text(3.1, 2.5, "Router", ha="center", va="center", fontweight="bold")
    # experts
    colors = ["#e63946", "#2a9d8f", "#e9c46a", "#f4a261"]
    for i, c in enumerate(colors):
        ax.add_patch(plt.Rectangle((5.5, 0.3 + i * 1.1), 1.5, 0.9, facecolor=c, edgecolor="k"))
        ax.text(6.25, 0.75 + i * 1.1, f"Expert {i}", ha="center", va="center", color="white", fontweight="bold")
    # arrows
    ax.annotate("", xy=(2.5, 2.5), xytext=(1.5, 2.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    for y in [0.75, 1.85]:
        ax.annotate("", xy=(5.5, y), xytext=(3.7, 2.5), arrowprops=dict(arrowstyle="->", color="green", lw=1.3))
    ax.text(7.3, 2.5, "top-k=2 활성화\n→ 가중 합", fontsize=10, color="#6a040f")
    ax.set_title("Mixture of Experts — 토큰별 전문가 선택")
    plt.savefig(OUT / "moe.png", bbox_inches="tight")
    plt.close()


# ─── 22. BPE merges ───
def save_bpe():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    rows = [
        ("iter 0", "l o w </w>", "#e0aaff"),
        ("iter 1 (l+o)", "lo w </w>", "#c77dff"),
        ("iter 2 (lo+w)", "low </w>", "#9d4edd"),
        ("iter 3 (low+</w>)", "low</w>", "#7b2cbf"),
    ]
    for i, (lbl, seq, c) in enumerate(rows):
        ax.text(0.5, 4 - i, lbl, fontsize=10, va="center")
        ax.add_patch(plt.Rectangle((3, 3.7 - i), 4, 0.6, facecolor=c, edgecolor="k"))
        ax.text(5, 4 - i, seq, ha="center", va="center", color="white" if i > 1 else "black", fontweight="bold")
    ax.set_title("BPE — 가장 빈번한 쌍을 반복 병합")
    plt.savefig(OUT / "bpe.png", bbox_inches="tight")
    plt.close()


# ─── 23. Gradient accumulation ───
def save_gradaccum():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    # 4 micro batches accumulating
    for i in range(4):
        ax.add_patch(plt.Rectangle((0.3 + i * 1.5, 2.5), 1.2, 0.8, facecolor="#90e0ef", edgecolor="k"))
        ax.text(0.9 + i * 1.5, 2.9, f"micro\nbatch {i}", ha="center", va="center", fontsize=8)
        ax.add_patch(plt.Rectangle((0.3 + i * 1.5, 1.3), 1.2, 0.8, facecolor="#caf0f8", edgecolor="k"))
        ax.text(0.9 + i * 1.5, 1.7, "backward\n(/N)", ha="center", va="center", fontsize=8)
        if i < 3:
            ax.annotate("", xy=(1.8 + i * 1.5, 1.7), xytext=(1.5 + i * 1.5, 1.7), arrowprops=dict(arrowstyle="->"))
    ax.add_patch(plt.Rectangle((6.7, 1.3), 1.8, 2, facecolor="#ffb4a2", edgecolor="k"))
    ax.text(7.6, 2.3, "optimizer\n.step()\n(1회)", ha="center", va="center", fontweight="bold")
    ax.annotate("", xy=(6.7, 2.3), xytext=(6.3, 2.3), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Gradient Accumulation — N개 micro-batch → 1 step")
    plt.savefig(OUT / "grad_accum.png", bbox_inches="tight")
    plt.close()


# ─── 24. speculative decoding timing ───
def save_spec_decode():
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    # target timeline
    ax.text(0, 2.3, "Target (큰 모델):", fontsize=10, fontweight="bold")
    for i in range(8):
        ax.add_patch(plt.Rectangle((2 + i * 0.8, 2.0), 0.7, 0.5, facecolor="#f72585", edgecolor="k"))
        ax.text(2.35 + i * 0.8, 2.25, f"{i+1}", ha="center", color="white", fontsize=8)
    ax.text(0, 1.3, "Draft (작은 모델):", fontsize=10, fontweight="bold")
    for i in range(4):
        ax.add_patch(plt.Rectangle((2 + i * 0.2, 1.0), 0.18, 0.5, facecolor="#90e0ef", edgecolor="k"))
    ax.text(0, 0.4, "Speculative:", fontsize=10, fontweight="bold")
    ax.add_patch(plt.Rectangle((2, 0.2), 0.8, 0.5, facecolor="#90e0ef", edgecolor="k"))
    ax.text(2.4, 0.45, "draft", ha="center", fontsize=8)
    ax.add_patch(plt.Rectangle((2.8, 0.2), 1.5, 0.5, facecolor="#f72585", edgecolor="k"))
    ax.text(3.55, 0.45, "target 1회로 K개 검증", ha="center", color="white", fontsize=8)
    ax.set_title("Speculative Decoding — 작은 모델로 후보 생성, 큰 모델로 일괄 검증")
    plt.savefig(OUT / "spec_decode.png", bbox_inches="tight")
    plt.close()


def main():
    save_activations()
    save_norm_axes()
    save_attention_masks()
    save_cosine_lr()
    save_mha_diagram()
    save_dropout()
    save_gqa()
    save_vit_patch()
    save_kv_cache()
    save_adam_traj()
    save_rope()
    save_flash()
    save_lora()
    save_rl_losses()
    save_beam_tree()
    save_quant()
    save_linreg()
    save_conv2d()
    save_cross_entropy()
    save_grad_clip()
    save_moe()
    save_bpe()
    save_gradaccum()
    save_spec_decode()
    print("모든 PNG 생성 완료:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
