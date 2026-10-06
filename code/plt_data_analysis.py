# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 路径配置
DATA_PATHS = {
    "aav_assay_pairs": "./data/aav_assay_pairs.csv",
    "aav_fit4function_support": "./data/aav_fit4function_support.csv",
    "aav_sequence_base": "./data/aav_sequence_base.csv",
    "ccle_features": "./data/ccle_features.csv"
}

# 绘图样式
plt.rcParams['font.size'] = 11
plt.rcParams["font.family"] = ["Arial", "SimHei"]
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300
os.makedirs("./figures/Data Analysis", exist_ok=True)

# 加载数据
def load_all_data():
    data = {}
    for name, path in DATA_PATHS.items():
        data[name] = pd.read_csv(path, low_memory=False)
    return data

# 生成数据集统计表 Table 1
def generate_dataset_stats(data):
    stats = {
        "AAV变异体数量": data["aav_sequence_base"]["variant_id"].nunique(),
        "平均序列长度": round(data["aav_sequence_base"]["seq_length"].mean(), 1),
        "血清型数量": data["aav_sequence_base"]["serotype_backbone"].nunique(),
        "细胞系数量": data["aav_assay_pairs"]["cell_line_original"].nunique(),
        "癌症类型数量": data["aav_assay_pairs"]["cancer_context"].nunique(),
        "功能实验记录数": len(data["aav_assay_pairs"]),
        "CCLE细胞系数": data["ccle_features"]["cell_line_id"].nunique()
    }
    df = pd.Series(stats).to_frame("数值")
    df.to_csv("./tables/Dataset Stats.csv", encoding='utf-8-sig')
    print("Table1 数据集统计已保存")

# 序列长度分布
def plot_sequence_length_dist(data):
    # 获取序列长度数据并去除空值
    seq_len = data["aav_sequence_base"]["seq_length"].dropna()
    
    # 统计每个长度出现的频次（核心：柱状图需要先统计计数）
    len_counts = seq_len.value_counts().sort_index()  # 按长度从小到大排序
    lengths = len_counts.index  # 长度值
    counts = len_counts.values  # 对应计数
    
    # 创建画布
    plt.figure(figsize=(8, 6))
    
    # 绘制柱状图（替换原直方图）
    plt.bar(lengths, counts, color='#3498db', alpha=0.7, edgecolor='black')
    
    # 添加平均值虚线
    plt.axvline(seq_len.mean(), color='red', linestyle='--', label=f'Average Length: {seq_len.mean():.3f}')
    
    # 坐标轴设置
    plt.xlabel("Sequence Length")
    plt.ylabel("Count")
    plt.xticks(np.arange(min(lengths), max(lengths)+1, 1))  # X轴刻度
    plt.ylim(0, max(counts) + 1)  # 自动适配高度，比固定10更灵活
    
    # 在柱子顶部添加数值标签
    for length, count in zip(lengths, counts):
        plt.text(length, count + 0.1, str(count), ha='center', fontsize=10)
    
    # 标题与图例
    plt.title("AAV Sequence Length Distribution", fontweight = "bold")
    plt.legend()
    plt.tight_layout()
    
    # 保存图片
    plt.savefig("./figures/Sequence Length Distribution.pdf")
    plt.close()
    print("序列长度柱状图已保存")

# 细胞系分布
def plot_cell_line_dist(data):
    cells = data["aav_assay_pairs"]["cell_line_original"].value_counts()
    plt.figure(figsize=(20, 6))
    cells.plot(kind='bar', color='#E74C3C', alpha=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Cell Line")
    plt.ylabel("Sample Count")
    plt.ylim (0, 25)
    plt.title("Cell Line Distribution", fontweight = "bold")
    for i in range (len (cells)):
        plt. text (i, cells[i] + 0.5, str (cells [i]), ha = "center")
    plt.tight_layout()
    plt.savefig("./figures/Cell Line Distribution.pdf")
    plt.close()
    print("细胞系分布图已保存")

# 功能实验值分布
# def plot_assay_value_dist(data):
#     val = pd.to_numeric(data["aav_assay_pairs"]["assay_value_raw"], errors='coerce').dropna()
    
#     if len(val) == 0:
#         print("assay_value_raw 无有效数值，跳过绘图")
#         return
    
#     plt.figure(figsize=(20, 4))
#     plt.boxplot(val, vert=False, widths=0.2)
#     plt.xlabel("Assay Value")
#     # plt. ylabel ("")
#     plt. yticks (color = "#ffffff")
#     plt.title("AAV Transduction Activity Distribution")
#     plt.tight_layout()
#     plt.savefig("./figures/Data Analysis/Assay Value Distribution.pdf")
#     plt.close()
#     print("功能实验值分布已保存")
def plot_assay_value_dist(data):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # 数据清洗
    val = pd.to_numeric(data["aav_assay_pairs"]["assay_value_raw"], errors='coerce').dropna()

    if len(val) == 0:
        print("assay_value_raw 无有效数值，跳过绘图")
        return

    # 截断设置
    left_start = -10        # 左侧区间起点
    left_end = 125        # 左侧区间终点
    right_start = 245    # 右侧区间起点

    # 计算左右区间的数值跨度，让子图宽度与跨度成正比
    left_range = left_end - left_start
    right_range = max(val) + 10 - right_start
    width_ratios = [left_range, right_range]  # 宽度与数值范围成正比

    # 创建画布与两个子图（实现截断）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # 绘制箱线图（带填充和样式）
    for ax in [ax1, ax2]:
        bp = ax.boxplot(val, vert=False, widths=0.3, patch_artist=True,
                        boxprops=dict(facecolor='#ffffff', alpha=0.7, edgecolor='black'),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='black', linewidth=1.2),
                        capprops=dict(color='black', linewidth=1.2),
                        flierprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=6))
        ax.set_yticks([])  # 隐藏y轴
        ax.tick_params(axis='x', labelsize=12)

    # 设置x轴范围
    ax1.set_xlim(left_start, left_end)
    ax2.set_xlim(right_start, max(val) + 10)

    # 移除中间共享的边框线
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # 绘制对称的截断斜线（标准样式）
    d = 0.015  # 斜线的角度控制参数
    # 左侧子图（ax1）的右侧截断线
    kwargs = dict(transform=ax1.transAxes, color='black', clip_on=False, linewidth=1.2)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)  # 补上上下两条斜线，形成完整截断效果

    # 右侧子图（ax2）的左侧截断线
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)

    # 调整子图间距，消除中间缝隙
    fig.subplots_adjust(wspace=0.05)

    # 统一标签与标题
    fig.supxlabel("Assay Value", fontsize=14)
    fig.suptitle("AAV Transduction Activity Distribution", fontsize=18, y=1.05, fontweight = "bold")

    plt.tight_layout()
    plt.savefig("./figures/Assay Value Distribution.pdf", bbox_inches='tight', dpi=300)
    plt.close()
    print("截断式功能实验值分布已保存")

# 细胞系富集相关性热图
def plot_enrichment_correlation(data):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    df = data["aav_fit4function_support"]
    enrich_cols = [c for c in df.columns if "log2_enrichment" in c]
    
    if len(enrich_cols) == 0:
        print("无富集数据，跳过相关性热图")
        return
    
    # 计算相关系数
    corr = df[enrich_cols].apply(pd.to_numeric, errors='coerce').corr()
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, cmap='bwr', vmin=-1, vmax=1)
    plt.colorbar(im, shrink=0.8)
    
    ticks = []
    for col in corr.columns:
        clean_name = col.replace("_", " ")  
        ticks.append(clean_name)


    # 设置坐标轴标签
    plt.xticks(range(len(corr)), ticks, rotation=45, ha='right')
    plt.yticks(range(len(corr)), ticks)
    plt.title("Enrichment Correlation Across Tissues or Cell Lines", fontweight = "bold")

    # 在每个方格中显示数值
    for i in range(len(corr)):
        for j in range(len(corr)):
            # 保留两位小数显示
            text = plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                           ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.savefig("./figures/Enrichment Correlation.pdf")
    plt.close()
    print("带数值的富集相关性热图已保存")

# CCLE 关键受体表达 PCA（带自动聚类上色）
# def plot_ccle_pca(data):
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     import numpy as np
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.decomposition import PCA
#     from sklearn.cluster import KMeans
#     from sklearn.metrics import silhouette_score
#     import os

#     # DEBUG 模式，看看数据到底啥样
#     print("=== 查看 ccle_features 数据信息 ===")
#     if "ccle_features" not in data:
#         print("错误：data 里没有 ccle_features")
#         return

#     df = data["ccle_features"]
#     print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
#     print("列名：", list(df.columns))
#     print("前5行数据：")
#     print(df.head())

#     # 直接取所有能转成数字的列，不限制任何名字
#     X = df.apply(pd.to_numeric, errors='coerce')
#     X = X.dropna(how='all', axis=1)  # 删除全空列
#     X = X.fillna(0)

#     # 最后检查：真的没数据才退出
#     if X.shape[1] == 0:
#         print("最终：没有任何可用数值列")
#         return

#     print(f"成功读取数据：{X.shape[0]} 行 {X.shape[1]} 列")

#     # 标准化 + PCA
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled)

#     # 自动聚类
#     print("正在自动寻找最佳聚类数...")
#     # 自动选择最优K（2~6类）
#     sil_scores = []
#     k_range = range(2, min(6, X_pca.shape[0]))  # 避免样本太少
#     for k in k_range:
#         kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', )
#         labels = kmeans.fit_predict(X_pca)
#         sil_scores.append(silhouette_score(X_pca, labels))

#     best_k = k_range[np.argmax(sil_scores)]
#     print(f"自动选择最佳聚类数：{best_k} 类")

#     # 最终聚类
#     kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
#     cluster_labels = kmeans.fit_predict(X_pca)

#     # 彩色聚类PCA图
#     from matplotlib.colors import ListedColormap
#     custom_cmap = ListedColormap(["#3498DB", "#2ECC71", "#E74C3C"])
#     plt.figure(figsize=(8,6))
#     scatter = plt.scatter(
#         X_pca[:,0], X_pca[:,1],
#         c=cluster_labels,       # 按聚类上色
#         cmap=custom_cmap,
#         s=50, alpha=0.8
#     )
#     plt.legend(*scatter.legend_elements(), title="Clusters")

#     plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
#     plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
#     plt.title(f"CCLE Receptor Expression PCA (Auto Clusters: {best_k})", fontweight = "bold")
#     plt.tight_layout()

#     # 保存（路径统一）
#     plt.savefig("./figures/CCLE Receptor PCA.pdf", bbox_inches='tight')
#     plt.close()
#     print("带自动聚类的PCA图已保存！")

# CCLE 关键受体表达 PCA（带自动聚类上色）
def plot_ccle_pca(data):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import os

    # DEBUG 模式，看数据到底啥样
    print("=== 查看 ccle_features 数据信息 ===")
    if "ccle_features" not in data:
        print("错误：data 里没有 ccle_features")
        return

    df = data["ccle_features"]
    print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
    print("列名：", list(df.columns))
    print("前5行数据：")
    print(df.head())

    # 直接取所有能转成数字的列，不限制任何名字
    X = df.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(how='all', axis=1)  # 删除全空列
    X = X.fillna(0)

    # 最后检查：真的没数据才退出
    if X.shape[1] == 0:
        print("最终：没有任何可用数值列")
        return

    print(f"成功读取数据：{X.shape[0]} 行 {X.shape[1]} 列")

    # 标准化 + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 自动聚类
    print("正在自动寻找最佳聚类数...")
    # 自动选择最优K（2~6类）
    sil_scores = []
    max_k = min(6, X_pca.shape[0])
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', )
        labels = kmeans.fit_predict(X_pca)
        sil_scores.append(silhouette_score(X_pca, labels))

    best_k = k_range[np.argmax(sil_scores)]
    print(f"自动选择最佳聚类数：{best_k} 类")

    # 最终聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_pca)

    # 彩色聚类PCA图
    from matplotlib.colors import ListedColormap
    # 扩充色板防止聚类数超过3报错
    full_color_list = ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12", "#9B59B6", "#1ABC9C"]
    custom_cmap = ListedColormap(full_color_list[:best_k])
    
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        X_pca[:,0], X_pca[:,1],
        c=cluster_labels,       # 按聚类上色
        cmap=custom_cmap,
        s=50, alpha=0.8
    )

    # 修改图例标签：Cluster {num}, n={样本数量}
    legend_handles, _ = scatter.legend_elements()
    cluster_counts = np.bincount(cluster_labels)
    legend_labels = [f"Cluster {i}, n={cluster_counts[i]}" for i in range(best_k)]
    plt.legend(legend_handles, legend_labels, title="Clusters")

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title(f"CCLE Receptor Expression PCA (Auto Clusters: {best_k})", fontweight = "bold")
    plt.tight_layout()

    # 保存（路径统一）
    os.makedirs("./figures", exist_ok=True)
    plt.savefig("./figures/CCLE Receptor PCA.pdf", bbox_inches='tight')
    plt.close()
    print("带自动聚类的PCA图已保存！")

# 主程序
if __name__ == "__main__":
    print("🔍 开始基础数据分析...")
    data = load_all_data()
    
    # generate_dataset_stats(data)
    # plot_sequence_length_dist(data)
    # plot_cell_line_dist(data)
    # plot_assay_value_dist(data)
    # plot_enrichment_correlation(data)
    plot_ccle_pca(data)
    
    print("全部基础数据分析完成！")
    print("保存路径：./figures")