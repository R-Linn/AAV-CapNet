# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 导入你的核心模型与工具
from aav_capnet import (
    load_and_merge_data,
    split_dataset,
    FourierAAVCapNet,
    train_model,
    evaluate_model,
    DEVICE,
    DATA_PATHS,
    export_attention_visualization
)

# 全局配置
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 消融实验配置
ABLATION_CONFIGS = {
    "Full Model": {"use_conv": True, "use_fft": True, "use_cross_attn": True, "use_cell_feat": True},
    "w/o Conv": {"use_conv": False, "use_fft": True, "use_cross_attn": True, "use_cell_feat": True},
    "w/o FFT": {"use_conv": True, "use_fft": False, "use_cross_attn": True, "use_cell_feat": True},
    "w/o Cross-Attention": {"use_conv": True, "use_fft": True, "use_cross_attn": False, "use_cell_feat": True},
    "w/o Cell Features": {"use_conv": True, "use_fft": True, "use_cross_attn": True, "use_cell_feat": False}
}

# 文件名映射
MODEL_FILENAMES = {
    "Full Model": "Full_Model.pth",
    "w/o Conv": "w_o_Conv.pth",
    "w/o FFT": "w_o_FFT.pth",
    "w/o Cross-Attention": "w_o_Cross_Attention.pth",
    "w/o Cell Features": "w_o_Cell_Features.pth"
}

# 运行消融实验
def run_ablation():
    # 创建目录
    os.makedirs("./ablation_results", exist_ok=True)
    os.makedirs("./visualization", exist_ok=True)
    os.makedirs("./figures", exist_ok=True)

    # 1. 加载统一数据
    print("=" * 60)
    print("Loading data for ablation study...")
    seq_feats, cell_feats, labels_cls, labels_reg, cell_lines = load_and_merge_data(DATA_PATHS)
    train_loader, val_loader, test_loader = split_dataset(seq_feats, cell_feats, labels_cls, labels_reg, cell_lines)

    # 2. 训练所有消融模型
    results = {}
    for name, cfg in ABLATION_CONFIGS.items():
        print("\n" + "=" * 60)
        print(f"Training: {name}")
        model = FourierAAVCapNet(**cfg).to(DEVICE)
        model = train_model(model, train_loader, val_loader)

        # 测试集评估
        metrics = evaluate_model(model, test_loader)
        results[name] = {
            "AUC": round(float(metrics["AUC"]), 4),
            "Accuracy": round(float(metrics["准确率"]), 4),
            "MSE": round(float(metrics["MSE"]), 6),
            "R²": round(float(metrics["R²分数"]), 6)
        }

        # 保存权重
        save_path = os.path.join("./ablation_results", MODEL_FILENAMES[name])
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

    # 3. 保存消融结果 npy + csv
    np.save("./ablation_results/ablation_summary.npy", results)
    df = pd.DataFrame(results).T
    df.to_csv("./tables/Ablation Result.csv", index_label="Model")
    print(f"Saved ablation summary: ./tables/Ablation Result.csv")

    # 4. 保存注意力权重（使用 Full Model）
    print("Generating attention weights...")
    full_model = FourierAAVCapNet().to(DEVICE)
    full_model.load_state_dict(torch.load("./ablation_results/Full_Model.pth", map_location=DEVICE))
    export_attention_visualization(full_model, test_loader)
    print(f"Saved attention weights: ./visualization/attention_weights.npy")

    # 5. 绘制消融柱状图
    plot_ablation_figure(df)
    print("\nAll ablation files generated successfully!")

# 绘图
def plot_ablation_figure(df):
    plt.rcParams["font.family"] = ["Arial", "SimHei"]
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    fig, ax1 = plt.subplots(figsize=(18, 10))

    models = df.index.tolist()
    auc = df["AUC"].values
    acc = df["Accuracy"].values
    mse = df["MSE"].values

    x = np.arange(len(models))
    width = 0.25

    # 柱子
    ax1.bar(x - width, auc, width, label='AUC', color='#3498DB', alpha=0.9)
    ax1.bar(x, acc, width, label='Accuracy', color='#2ECC71', alpha=0.9)
    ax1.set_ylabel("AUC or Accuracy", fontsize=13)
    ax1.set_ylim(0.65, 0.95)

    # 双轴 MSE
    ax2 = ax1.twinx()
    ax2.bar(x + width, mse, width, label='MSE', color='#E74C3C', alpha=0.9)
    ax2.set_ylabel("MSE", fontsize=13)
    ax2.set_ylim(0.000, 0.030)

    # 标注数值
    for i, (a, ac, m) in enumerate(zip(auc, acc, mse)):
        ax1.text(i-width, a+0.0008, f"{a:.3f}", ha='center', fontsize=12)
        ax1.text(i, ac+0.0008, f"{ac:.3f}", ha='center', fontsize=12)
        ax2.text(i+width, m+0.0004, f"{m:.4f}", ha='center', fontsize=12)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15)
    ax1.set_title("Ablation Study", fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("./figures/Ablation Study.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: ./figures/Ablation Study.pdf")

# 主程序
if __name__ == "__main__":
    run_ablation()