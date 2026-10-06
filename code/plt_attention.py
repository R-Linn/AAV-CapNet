# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os

# 绘图配置
plt.rcParams["font.family"] = ["Arial", "SimHei"]
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['savefig.dpi'] = 600
os.makedirs("./figures", exist_ok=True)

# AAV 已知功能区域（按 AAV2 VP1 编号）
REGIONS = {
    "Receptor Binding\nDomain (RBD)": (452, 478),
    "Hypervariable\nRegion 1 (HVR1)": (146, 152),
    "Hypervariable\nRegion 2 (HVR2)": (182, 186),
    "Hypervariable\nRegion 3 (HVR3)": (262, 264),
    "Hypervariable\nRegion 5 (HVR5)": (450, 474),
    "Hypervariable\nRegion 9 (HVR9)": (534, 555),
}
HIGHLIGHT_COLOR = "red"
RBD_COLOR = "#00FF00"

# 目标区间
TARGET_START = 150
TARGET_END = 300

def format_cell_label(name):
    if "_" not in name:
        return name
    first, rest = name.split("_", 1)
    return f"{first} ({rest.replace('_', ' ')})"

# 主程序
if __name__ == "__main__":
    # 加载注意力权重
    attn_path = "./visualization/attention_weights.npy"
    attn_data = np.load(attn_path, allow_pickle=True).item()
    attn_matrix = attn_data.get("attn_matrix", np.array([]))
    cell_labels = attn_data.get("cell_labels", ["Unknown"])
    print("已获取注意力权重")

    # 格式化细胞名
    cell_labels = [format_cell_label(name) for name in cell_labels]

    # 截取目标区间（假设原始矩阵列索引 0 对应残基 1）
    attn_matrix = attn_matrix[:, TARGET_START - 1:TARGET_END]
    seq_positions = np.arange(TARGET_START, TARGET_END + 1)
    n_rows = len(cell_labels)

    # 绘制热图
    fig, ax = plt.subplots(figsize=(24, 6))

    im = ax.imshow(
        attn_matrix,
        cmap="Blues",
        aspect="auto",
        vmin=attn_matrix.min(),
        vmax=attn_matrix.max()
    )

    # 标注区域
    for region_name, (start, end) in REGIONS.items():
        if end < TARGET_START or start > TARGET_END:
            continue

        draw_start = max(start, TARGET_START)
        draw_end = min(end, TARGET_END)
        rel_start = draw_start - TARGET_START
        rel_end = draw_end - TARGET_START
        mid = (rel_start + rel_end) / 2

        is_rbd = "RBD" in region_name
        is_HVR1 = "HVR1" in region_name

        if is_rbd:
            color = RBD_COLOR
            alpha = 0.2
            zorder = 3
        else:
            color = HIGHLIGHT_COLOR
            alpha = 0.2
            zorder = 2

        ax.axvspan(rel_start, rel_end, color=color, alpha=alpha, zorder=zorder)

        if is_rbd:
            rbd_center_residue = (452 + 478) / 2
            rbd_center_rel = rbd_center_residue - TARGET_START
            ax.text(
                rbd_center_rel, n_rows + 0.6, region_name,
                ha='center', va='top',
                color=color, fontsize=12, fontweight='bold'
            )
        elif is_HVR1:
            ax.text(mid+5, -0.6, region_name, ha="center", va="bottom",
                    color=color, fontsize=12, fontweight="bold")

        else:
            ax.text(mid, -0.6, region_name, ha="center", va="bottom",
                    color=color, fontsize=12, fontweight="bold")

    # 坐标轴
    ax.set_yticks(range(len(cell_labels)))
    ax.set_yticklabels(cell_labels, fontweight="bold", fontsize=12)
    ax.set_xticks(np.linspace(0, len(seq_positions) - 1, 10, dtype=int))
    ax.set_xticklabels(seq_positions[np.linspace(0, len(seq_positions) - 1, 10, dtype=int)])
    ax.set_xlabel("Sequence Position", fontweight="bold", fontsize=12)
    # ax.set_title("Attention Weight Heatmap (Residues 400-650)",
                #  fontweight="bold", fontsize=16, y=1.08)

    # 色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Attention Weight", fontweight="bold", fontsize = 12)

    plt.tight_layout()
    plt.savefig(f"./figures/Attention Heatmap {TARGET_START}-{TARGET_END}.pdf", bbox_inches="tight", format="pdf")
    plt.close()

    print(f"注意力热图已生成：./figures/Attention Heatmap {TARGET_START}-{TARGET_END}.pdf")