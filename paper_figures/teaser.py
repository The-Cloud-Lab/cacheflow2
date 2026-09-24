"""Page-1 teaser figure for the CacheFlow ICLR submission.

(a) Analytic (paper Eq. 11): memory to cache the workload's 10 x 20K-token
    prompts (Llama-3.2-3B, BF16 KV = 22.9 GB) when N GPUs serve the same
    agents: vLLM keeps a copy per GPU, CacheFlow one shared copy on the
    SmartNIC. GPU counts are illustrative; the experiments are single-node.
(b) Measured: mean TTFT, Qwen3-4B, QPS=1 (paper Table 1 / Table 3).
(c) Measured: total token throughput from Table 1, averaged over QPS 1-10
    per system; labels are the gain over vLLM's average. The averaging is ours; the
    script prints every derived number next to its source.

Run:  python3 paper_figures/teaser.py   -> teaser.pdf / teaser.png
"""

from pathlib import Path
from statistics import mean

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent

# ---------------------------------------------------------------- style
# Validated categorical slots (blue / orange / aqua), mapped to keep the
# paper's existing convention: vLLM warm, CacheFlow blue, LMCache green-ish.
C_CF, C_VLLM, C_LMC = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------- data
# (a) KV bytes/token = 2 (K,V) * layers * kv_heads * head_dim * 2 B (BF16).
#     Shapes verified against each model's config.json on the HF Hub.
GB = 1e9
PREFIX_TOKENS = 20_000                              # workload, App. A.1
MODELS_KV = {  # name: (num_hidden_layers, num_key_value_heads, head_dim)
    "Qwen3-4B": (36, 8, 128),
    "Llama-3.2-3B": (28, 8, 128),
    "Ministral-3B": (26, 8, 128),
}
per_prefix = {m: PREFIX_TOKENS * 2 * L * h * d * 2 / GB
              for m, (L, h, d) in MODELS_KV.items()}
GPU_SIZES = [24, 48, 80]                            # common GPU memory sizes (GB)
N_WORKLOAD = 10                                     # prefixes in App. A.1
N_GPUS = [1, 2, 4, 8]                               # illustrative

# (b) Table 1, Qwen3-4B, QPS=1, mean TTFT (ms)
ttft = {"vLLM": 27_256, "LMCache": 8_817, "CacheFlow": 5_038}

# (c) Table 1 throughput (tok/s), QPS 1..10
tput = {
    "Qwen3-4B": {
        "vLLM": [12931, 13371, 13378, 13385, 13390, 13395, 13398, 13364, 13397, 13398],
        "LMCache": [16652, 16869, 16925, 16889, 16902, 16893, 16918, 16919, 16905, 16912],
        "CacheFlow": [17945, 18770, 17723, 20004, 19007, 18729, 18143, 20004, 20324, 19374],
    },
    "Llama-3.2-3B": {
        "vLLM": [19398, 27697, 28566, 28762, 28773, 28710, 28683, 28692, 28689, 28691],
        "LMCache": [19575, 37311, 38928, 39369, 39596, 40208, 39776, 40108, 39088, 38739],
        "CacheFlow": [19651, 36959, 44961, 48083, 44676, 42792, 41660, 49462, 42187, 47927],
    },
    "Ministral-3B": {
        "vLLM": [19813, 35990, 37321, 37191, 37229, 37366, 37245, 37671, 37479, 37541],
        "LMCache": [19922, 37850, 46120, 49540, 48320, 51400, 50850, 51500, 50200, 51500],
        "CacheFlow": [19922, 39313, 56947, 61632, 56745, 61428, 56786, 56538, 63663, 56843],
    },
}

COLORS = {"vLLM": C_VLLM, "LMCache": C_LMC, "CacheFlow": C_CF}

fig, axes = plt.subplots(
    1, 3, figsize=(5.5, 2.05),
    gridspec_kw={"width_ratios": [1.15, 1.0, 1.2], "wspace": 0.6},
)

# ---------------------------------------------------------------- (a)
# Memory to cache the same 10 x 20K-token agent prompts when N GPUs serve
# them: vLLM keeps a copy per GPU, CacheFlow one shared copy on the
# SmartNIC.  Llama-3.2-3B KV size (Eq. 11).
ax = axes[0]
pool = per_prefix["Llama-3.2-3B"] * N_WORKLOAD     # GB for 10 prompts
gpus = np.array(N_GPUS)
xa = np.arange(len(gpus))
wa = 0.38
ax.bar(xa - wa / 2, pool * gpus, width=wa, color=C_VLLM, edgecolor="white", linewidth=0.8)
ax.bar(xa + wa / 2, np.full(len(gpus), pool), width=wa, color=C_CF,
       edgecolor="white", linewidth=0.8)
# label only the largest group: the one number worth remembering
ax.text(xa[-1] - wa / 2, pool * gpus[-1] + 5, f"{pool * gpus[-1]:.0f} GB",
        ha="center", va="bottom", fontsize=6.8, color=INK)
ax.text(xa[-1] + 0.02, pool + 5, f"{pool:.0f} GB", ha="left", va="bottom",
        fontsize=6.8, color=INK, fontweight="bold")
ax.set_xticks(xa)
ax.set_xticklabels([str(g) for g in gpus])
ax.tick_params(axis="x", length=0)
ax.set_xlabel("Number of GPUs")
ax.set_ylabel("memory occupied by\nprefix cache (GB)", fontsize=7.5, linespacing=1.05)
ax.set_ylim(0, 270)
ax.set_xlim(-0.55, len(gpus) - 0.3)
ax.set_yticks([0, 50, 100, 150, 200])
ax.text(0.03, 0.98, "avoided by CacheFlow", transform=ax.transAxes,
        fontsize=7, color=INK, va="top")
ax.text(0.03, 0.89, "10 prompts × 20K tokens", transform=ax.transAxes,
        fontsize=6.3, color=MUTED, va="top")
ax.set_title("(a) Redundant caching", loc="left", color=INK)
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)

# ---------------------------------------------------------------- (b)
ax = axes[1]
names = ["vLLM", "LMCache", "CacheFlow"]
xb = np.arange(len(names))
vals = [ttft[k] for k in names]                     # ms, exactly as in Table 1
ax.bar(xb, vals, width=0.62, color=[COLORS[k] for k in names],
       edgecolor="white", linewidth=1.0)
for xi, v in zip(xb, vals):
    ax.text(xi, v + 500, f"{v:,}", ha="center", va="bottom", fontsize=7, color=INK)
ax.set_xticks([])
ax.set_xlabel("mean, Qwen3-4B, QPS = 1")
ax.set_ylim(0, 33000)
ax.set_yticks([0, 10000, 20000, 30000])
ax.set_yticklabels(["0", "10K", "20K", "30K"])
ax.set_ylabel("TTFT (ms)")
ax.set_title("(b) 5.4$\\times$ lower TTFT", loc="left", color=INK)
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)

# ---------------------------------------------------------------- (c)
ax = axes[2]
models = list(tput)
w = 0.26
x = np.arange(len(models))
for i, sysname in enumerate(names):
    avg = [mean(tput[m][sysname]) for m in models]
    bars = ax.bar(x + (i - 1) * w, avg, width=w, color=COLORS[sysname],
                  edgecolor="white", linewidth=0.8, label=sysname)
    if sysname == "CacheFlow":
        for b, m, a in zip(bars, models, avg):
            gain = a / mean(tput[m]["vLLM"]) - 1
            ax.text(b.get_x() + b.get_width() / 2, a + 1200,
                    f"+{gain * 100:.0f}%", ha="center", va="bottom",
                    fontsize=7, color=INK, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["Qwen3", "Llama3.2", "Ministral"], fontsize=6.5)
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 62000)
ax.set_yticks([0, 20000, 40000, 60000])
ax.set_yticklabels(["0", "20K", "40K", "60K"])
ax.set_ylabel("throughput (tokens/s)")
ax.set_title("(c) 1.4–1.5$\\times$ throughput", loc="left", color=INK)
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)

handles = [mpl.patches.Patch(color=COLORS[k], label=k) for k in names]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.08), fontsize=7.5, handlelength=1.2,
           columnspacing=1.4)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"teaser.{ext}", bbox_inches="tight", pad_inches=0.02,
                dpi=300)
print("== (a) analytic, Eq. 11 ==")
for m, v in per_prefix.items():
    print(f"  {m:13s} KV/20K-token prefix = {v:.2f} GB;  x{N_WORKLOAD} = {v * N_WORKLOAD:.1f} GB")
print("== (b) Table 1, Qwen3-4B, QPS=1, mean TTFT ==")
for k, v in ttft.items():
    print(f"  {k:9s} {v:>6,} ms  -> plotted {v:,} ms")
print(f"  vLLM/CacheFlow = {ttft['vLLM'] / ttft['CacheFlow']:.2f}x  (paper Sec. 3.4 states 5.41x)")
print(f"  LMCache/CacheFlow = {ttft['LMCache'] / ttft['CacheFlow']:.2f}x")
print("== (c) Table 1 throughput, mean over QPS 1-10, / vLLM mean ==")
for m in tput:
    mv = mean(tput[m]["vLLM"])
    print(f"  {m:13s} " + "  ".join(
        f"{k}={mean(tput[m][k]):,.0f} ({mean(tput[m][k]) / mv:.3f}x)" for k in names))
