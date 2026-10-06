# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.fft import fft, fftfreq
import os

# 全局绘图配置
plt.rcParams['font.size'] = 12
plt.rcParams["font.family"] = ["Arial", "SimHei"]  # 兼容中文
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
os.makedirs("./other figures", exist_ok=True)

# 1. 注意力训练曲线
def plot_attention_training_curve(attention_loss_path="./attention_loss.csv", 
                                  save_path="./figures/Attention Training Curve.pdf"):
    df = pd.read_csv(attention_loss_path)
    epochs = df["epoch"].values
    attn_loss = df["attention_loss"].values
    total_loss = df["total_loss"].values

    fig, ax1 = plt.subplots(figsize=(10, 10))

    ax1.plot(epochs, attn_loss, 'b-o', linewidth=2, label='Attention Loss', markersize=4)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Attention Loss', color='b', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(epochs, total_loss, 'r-o', linewidth=2, label='Total Loss', markersize=4)
    ax2.set_ylabel('Total Loss', color='r', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Attention Mechanism Training Curve', fontweight='bold', fontsize=14)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"注意力训练曲线已保存：{save_path}")

# 2. 跨细胞系注意力对比热图
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cell_line_attention_heatmap(
    ccle_path="./data/ccle_features.csv",
    save_path="./figures/Cell Line Attention Heatmap.pdf",
    top_n=20
):
    df = pd.read_csv(ccle_path)
    key_genes = [
        "AAVR_expr_log2TPM_p1", "SLC35B2_expr_log2TPM_p1", "EXT1_expr_log2TPM_p1",
        "EXT2_expr_log2TPM_p1", "FGFR1_expr_log2TPM_p1", "ITGAV_expr_log2TPM_p1",
        "ITGB5_expr_log2TPM_p1", "HSPG2_expr_log2TPM_p1"
    ]
    df_clean = df[key_genes + ["cell_line_id"]].dropna().head(top_n)
    attention_matrix = df_clean[key_genes].T.values
    cell_names = df_clean["cell_line_id"].values

    clean_key_genes = [gene.replace("_", " ") for gene in key_genes]  # y轴标签处理
    clean_cell_names = [cell.replace("_", " ") for cell in cell_names]  # x轴标签处理

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attention_matrix,
        cmap="coolwarm",
        xticklabels=clean_cell_names,  # 使用替换后的x轴标签
        yticklabels=clean_key_genes,   # 使用替换后的y轴标签
        annot=False,
        center=0,
        linewidths=0.5
    )
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Cross Cell Line Attention Comparison Heatmap (Top 20 Genes)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"跨细胞系注意力对比热图已保存：{save_path}")

# 3. AAV时频特征箱线图
def plot_time_freq_boxplot(
    seq_data_path="./data/aav_sequence_base.csv",
    save_path="./figures/box plot.pdf"
):
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # 期刊极简样式
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # 读取数据
    df = pd.read_csv(seq_data_path)
    df_valid = df.dropna(subset=['full_amino_seq']).reset_index(drop=True)

    # 氨基酸编码
    amino_map = {'A':1,'R':2,'N':3,'D':4,'C':5,'Q':6,'E':7,'G':8,'H':9,'I':10,
                 'L':11,'K':12,'M':13,'F':14,'P':15,'S':16,'T':17,'W':18,'Y':19,'V':20}

    # 3分组规则
    WT_LIST = ['AAV1','AAV2','AAV3B','AAV4','AAV5','AAV7','AAV8','AAV9']

    feature_data = []
    for _, row in df_valid.iterrows():
        vid = row['variant_id'].strip()
        seq = row['full_amino_seq']
        val_seq = [amino_map[c] for c in seq if c in amino_map]

        # 计算特征
        time_var = np.var(val_seq)
        freq_dc = np.abs(np.fft.fft(val_seq))[0]

        # 1. 野生型
        if vid in WT_LIST:
            group = 'Wild-Type'
        # 2. 嵌合体（带 / 的 AAV2/x）
        elif '/' in vid:
            group = 'Chimera'
        # 3. 工程突变体（剩下所有：带- / 自定义后缀）
        else:
            group = 'Engineered'

        feature_data.append({
            'Serotype': vid, 'Group': group,
            'Time_Var': time_var, 'Freq_DC': freq_dc
        })

    df_plot = pd.DataFrame(feature_data)
    # Wild-Type（野生型）8 个：AAV1、AAV2、AAV3B、AAV4、AAV5、AAV7、AAV8、AAV9
    # Chimera（嵌合体）5 个：AAV2/1、AAV2/2、AAV2/4、AAV2/5、AAV2/8
    # Engineered（工程突变体）5 个：AAV-LK03、AAVzy9-3、AAV-PTP、AAV3B-DE5、AAV3B-V04
    group_order = ['Wild-Type', 'Chimera', 'Engineered']
    palette = ["#E74C3C", "#2ECC71", "#3498DB"]

    # 绘图：时域方差 + 频域DC分量
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：时域方差
    sns.boxplot(x='Group', y='Time_Var', data=df_plot, order=group_order,
                palette=palette, ax=ax1, width=0.6, showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    sns.stripplot(x='Group', y='Time_Var', data=df_plot, order=group_order,
                  color="black", size=5, alpha=0.7, ax=ax1)
    ax1.set_title('Time Domain | Sequence Variance', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Amino Acid Sequence Variance', fontweight='bold')
    ax1.set_xlabel('')

    # 右图：频域DC分量
    sns.boxplot(x='Group', y='Freq_DC', data=df_plot, order=group_order,
                palette=palette, ax=ax2, width=0.6, showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    sns.stripplot(x='Group', y='Freq_DC', data=df_plot, order=group_order,
                  color="black", size=5, alpha=0.7, ax=ax2)
    ax2.set_title('Frequency Domain | DC Component', fontweight='bold', fontsize=14)
    ax2.set_ylabel('FFT DC Component (0-frequency)', fontweight='bold')
    ax2.set_xlabel('')

    plt.tight_layout()
    plt.savefig("./figures/Time Domains and Sequence Domains.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # 打印分组统计
    print("3组样本统计：")
    print(df_plot['Group'].value_counts())
    print(f"3组全显示箱线图已保存：{save_path}")

# 4. 组织富集相关性热图
def plot_tissue_enrichment_corr(
    ccle_path="./data/ccle_features.csv",
    save_path="./figures/Tissue Enrichment Correlation Heatmap.pdf"
):
    # 1. 数据读取与预处理
    df = pd.read_csv(ccle_path)
    key_features = [
        "AAVR_expr_log2TPM_p1", "SLC35B2_expr_log2TPM_p1", "EXT1_expr_log2TPM_p1",
        "FGFR1_expr_log2TPM_p1", "ITGAV_expr_log2TPM_p1", "HSPG2_expr_log2TPM_p1"
    ]
    # 按组织分组求均值，计算相关系数矩阵
    tissue_group = df.groupby("tissue_origin")[key_features].mean().dropna()
    corr_matrix = tissue_group.corr()
    
    # 简化特征名（基因名），解决原图长名称重叠问题
    feature_names = [feat.split("_")[0] for feat in key_features]
    n = len(feature_names)
    
    # 2. 初始化画布
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111)
    cmap = plt.cm.plasma
    norm = plt.Normalize(0, 1)

    # 3. 遍历矩阵，分区域绘制：下三角热图,上三角饼图,对角线标注基因名
    for i in range(n):
        for j in range(n):
            corr_val = corr_matrix.iloc[i, j]
            # 对角线：标注基因名称
            if i == j:
                ax.text(j + 0.5, i + 0.5, feature_names[i], 
                        ha="center", va="center", fontsize=12, fontweight="bold")
                continue
            # 下三角 (i>j)：绘制标准热图色块 + 相关系数数值
            if i > j:
                # 绘制颜色块
                rect = plt.Rectangle((j, i), 1, 1, color=cmap(norm(corr_val)))
                ax.add_patch(rect)
                # 添加数值标注
                ax.text(j + 0.5, i + 0.5, f"{corr_val:.2f}", 
                        ha="center", va="center", fontsize=10, fontweight="bold")
            # 上三角 (i<j)：单元格内嵌饼图（展示相关强度or方向）
            else:
                import matplotlib.cm as cm
                import matplotlib.colors as mcolors
                size = abs(corr_val)  # 相关系数绝对值（饼图占比）
                sizes = [size, 1 - size]
                # 配色：红色=正相关，蓝色=负相关，白色=无相关部分
                # 输入：相关性系数（测试用，替换成你的corr_val即可）
                # 核心逻辑：bwr色图自动配色
                norm = mcolors.Normalize(vmin=0, vmax=1)
                rgba_color = cm.plasma(norm(corr_val))  # 获取bwr渐变的RGBA颜色
                hex_color = mcolors.to_hex(rgba_color)  # 转成十六进制（通用格式）

                # 最终结果：和你原代码格式完全一致
                colors = [hex_color, "white"]
                # 绘制内嵌饼图（控制大小避免重叠）
                ax.pie(sizes, colors=colors, radius=0.4, center=(j+0.5, i+0.5),
                       wedgeprops=dict(width=0.3, edgecolor="black", linewidth=0.5))

    # 4. 图表样式设置
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()  # 反转Y轴，匹配矩阵常规展示顺序
    
    # 设置坐标轴刻度
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(feature_names, fontsize=11)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Correlation Coefficient", fontsize=12, fontweight="bold")
    
    # 标题与布局
    plt.title('Tissue Enrichment Correlation Heatmap', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    # 5. 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"组织富集相关性热图（热图+饼图版）已保存：{save_path}")

# 主程序
if __name__ == "__main__":
    print("开始绘制图表...")
    # plot_attention_training_curve()
    plot_cell_line_attention_heatmap()
    plot_tissue_enrichment_corr()
    plot_time_freq_boxplot()
    print("所有图表绘制完成！已保存至 ./other_figures/ 目录")