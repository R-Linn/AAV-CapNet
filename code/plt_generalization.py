# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 绘图全局配置
plt.rcParams['font.size'] = 12
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams['axes.linewidth'] = 1.2
os.makedirs("./figures", exist_ok=True)

# 多模型配色
COLOR_LIST = [
    '#E74C3C',   # AAV-CapNet   红
    '#3498DB',   # MLP          蓝
    '#2ECC71',   # CNN          绿
    '#F39C12',   # XGBoost      橙
    '#9B59B6',   # RandomForest 紫
    '#1ABC9C'    # LSTM         青
]

# 通用雷达图函数（支持任意多模型）
def radar_chart_all(categories, data_list, label_list, title, save_path):
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 循环绘制所有模型曲线
    for idx, values in enumerate(data_list):
        vals_closed = values + [values[0]]  # 闭合数据
        color = COLOR_LIST[idx % len(COLOR_LIST)]
        ax.plot(angles, vals_closed, 'o-', linewidth=2, label=label_list[idx], color=color, markersize=7)
        ax.fill(angles, vals_closed, color=color, alpha=0.15)

    # 坐标轴设置
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontweight='bold')
    ax.tick_params(axis='x', pad=25)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=25, fontweight='bold', fontsize=16, y=1.06, x = 0.5 + 0.13)

    # 图例外置避免遮挡
    ax.legend(loc='lower right', bbox_to_anchor=(1.32, -0.2), frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 主程序
if __name__ == "__main__":
    csv_path = "./tables/Model Comparison.csv"
    df = pd.read_csv(csv_path, index_col=0)
    # 过滤空行
    df = df.dropna(how="all")

    # CSV原始中文列（严格匹配你的表头：准确率，精确率，召回率，F1分数，AUC）
    csv_cols = ["准确率", "精确率", "召回率", "F1分数", "AUC"]
    # 绘图展示英文标签（与csv_cols一一对应）
    display_categories = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]

    # 全部模型顺序（和COLOR_LIST一一对应）
    # all_models = ["AAV-CapNet", "MLP", "CNN", "XGBoost", "RandomForest", "LSTM"]
    all_models = ["AAV-CapNet", "MLP", "CNN"]

    # 批量读取每个模型指标
    model_data = []
    for m_name in all_models:
        row_vals = df.loc[m_name, csv_cols].values.tolist()
        model_data.append(row_vals)

    # 绘制全模型对比雷达图
    radar_chart_all(
        categories=display_categories,
        data_list=model_data,
        label_list=all_models,
        title="Generalization Ability Comparison of All Models",
        save_path="./figures/All Models Radar.pdf"
    )

    print("All-model radar chart saved: ./figures/All Models Radar.pdf")
    print(f"Compared all models: {', '.join(all_models)}")