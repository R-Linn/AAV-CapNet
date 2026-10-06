# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 项目全局配置与核心函数导入
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from aav_capnet import (
    DEVICE, SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE,
    DROPOUT_RATE, FourierAAVCapNet, compute_loss, evaluate_model,
    DATA_PATHS, CELL_FEAT_DIM, extract_aa_manual_features
)

# 全局复用工具函数
def simplify_serotype(serotype):
    # 血清型标准化合并规则，与原项目完全对齐
    if pd.isna(serotype) or serotype == 'Unknown':
        return 'Other'
    s = str(serotype)
    match = re.match(r'(AAV[0-9]+[A-Z]*)', s)
    if match:
        main = match.group(1)
        if main == 'AAV3B':
            return 'AAV3B'
        return main
    else:
        return 'Other'

def replace_bn_with_ln(model):
    # 递归替换网络所有BatchNorm1d为LayerNorm，适配小测试batch
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm1d):
            num_feat = child.num_features
            ln = nn.LayerNorm(num_feat, eps=child.eps, elementwise_affine=True)
            setattr(model, name, ln)
        else:
            replace_bn_with_ln(child)
    return model

def set_all_seeds(seed):
    # 锁定全部随机源，保证实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据清洗与加载函数
def clean_column_name(col_name):
    # 清除不可见字符、统一列名下划线格式
    col_name = col_name.replace('\ufeff', '').replace('\u200b', '')
    col_name = col_name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
    return col_name

def clean_numeric_value(val):
    # 清洗assay_value_raw实验数值：区间、大小于、空值统一处理
    if pd.isna(val) or val == '' or str(val).lower() in ['na', 'nan', 'none', '未知', '无']:
        return 0.0
    val_str = str(val).strip()
    range_pat = r'^(\d+\.?\d*)\s*-\s*(\d+\.?\d*)$'
    m = re.match(range_pat, val_str)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    sym_pat = r'^[<>≥≤]\s*(\d+\.?\d*)$'
    m = re.match(sym_pat, val_str)
    if m:
        return float(m.group(1))
    num_pat = r'[-+]?\d*\.\d+|\d+'
    nums = re.findall(num_pat, val_str)
    return float(nums[0]) if nums else 0.0

def read_csv_safe(file_path):
    # 多编码容错读取csv
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, sep=None, engine="python", encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        raise ValueError(f"文件读取失败: {file_path}")
    df.columns = [clean_column_name(c) for c in df.columns]
    return df

def load_stratified_kfold_data():
    # 加载与原项目完全一致的三张核心数据表，返回分层K折所需的全部数据
    # 与LOVO代码数据加载逻辑完全对齐，保证实验可比性
    # 返回：seq_feats, cell_feats, labels_cls, labels_reg, serotypes, variant_ids
    print("=== 加载三张核心数据表（与原项目完全一致） ===")
    # 1. 读取实验表
    assay_df = read_csv_safe(DATA_PATHS["aav_assay_pairs"])
    # 2. 读取序列&血清型表
    seq_df = read_csv_safe(DATA_PATHS["aav_sequence_base"])
    # 3. 读取CCLE细胞特征表
    ccle_df = read_csv_safe(DATA_PATHS["ccle_features"])

    # 字段映射
    col_mapping = {
        "variant_id": "variant_id",
        "cell_line_ccle_match": "cell_line_ccle_match",
        "assay_value_raw": "assay_value_raw",
        "full_amino_seq": "full_amino_seq",
        "serotype_backbone": "serotype_backbone",
        "cell_line_id": "cell_line_id"
    }
    for df in [assay_df, seq_df, ccle_df]:
        df.rename(columns={old: new for old, new in col_mapping.items() if old in df.columns}, inplace=True)

    # 清洗实验数值，仅保留>0有效实验记录
    assay_df['assay_value_raw'] = assay_df['assay_value_raw'].apply(clean_numeric_value).astype(np.float64)
    assay_df = assay_df[assay_df['assay_value_raw'] > 0].reset_index(drop=True)

    # 多表内连接
    merged = pd.merge(assay_df, seq_df, on="variant_id", how="inner")
    merged = pd.merge(merged, ccle_df, left_on="cell_line_ccle_match", right_on="cell_line_id", how="inner")

    # 过滤空氨基酸序列
    merged = merged[merged["full_amino_seq"].fillna("").str.len() > 0].reset_index(drop=True)

    # 提取并标准化血清型（用于复合分层）
    raw_sero = merged["serotype_backbone"].fillna("Unknown").tolist()
    serotypes = [simplify_serotype(s) for s in raw_sero]

    # 变体ID（用于后续分析）
    variant_ids = merged["variant_id"].tolist()

    # 细胞特征提取
    receptor_genes = [
        "aavr_expr_log2tpm_p1",
        "slc35b2_expr_log2tpm_p1",
        "ext1_expr_log2tpm_p1",
        "ext2_expr_log2tpm_p1",
        "fgfr1_expr_log2tpm_p1"
    ]
    rppa_proteins = [
        "rppa_ywhab",
        "rppa_ywhae",
        "rppa_ywhaz",
        "rppa_tp53bp1",
        "rppa_araf"
    ]
    cell_feat_cols = []
    # 先加入受体基因
    for c in receptor_genes:
        if c in merged.columns:
            cell_feat_cols.append(c)
    # 再补充RPPA蛋白，填满CELL_FEAT_DIM
    for c in rppa_proteins:
        if c in merged.columns and len(cell_feat_cols) < CELL_FEAT_DIM:
            cell_feat_cols.append(c)
    # 不足维度则补充其他数值特征（排除assay实验字段）
    if len(cell_feat_cols) < CELL_FEAT_DIM:
        num_cols = [
            c for c in merged.columns
            if merged[c].dtype in [np.float64, np.int64]
            and not c.startswith("assay_")
        ]
        cell_feat_cols += num_cols[:CELL_FEAT_DIM - len(cell_feat_cols)]
    cell_raw = merged[cell_feat_cols].fillna(0).values

    # 氨基酸序列特征
    seq_series = merged["full_amino_seq"]
    seq_feats = extract_aa_manual_features(seq_series)

    # 标签标准化
    label_raw = merged["assay_value_raw"].values
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    sc_cell = StandardScaler()
    cell_feats = sc_cell.fit_transform(cell_raw)
    sc_reg = MinMaxScaler()
    labels_reg = sc_reg.fit_transform(label_raw.reshape(-1, 1))
    median = np.median(label_raw)
    labels_cls = (label_raw >= median).astype(np.float32).reshape(-1, 1)

    print(f"数据集加载完成 | 总样本={len(seq_feats)} | 独立变体={len(set(variant_ids))} | 血清类别={len(set(serotypes))} | 分类标签正负比={np.sum(labels_cls)/len(labels_cls):.2f}")
    return seq_feats, cell_feats, labels_cls, labels_reg, serotypes, variant_ids

# 单折训练函数
def train_stratified_kfold_single_run(model, train_loader, val_loader):
    # 单折模型训练函数，与原项目LOVO训练逻辑完全一致
    # 包含早停、学习率调度、梯度裁剪、最优模型保存
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    best_auc = -1.0
    best_state = None
    stop_count = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for seq, cell, cls, reg in train_loader:
            seq, cell, cls, reg = seq.to(DEVICE), cell.to(DEVICE), cls.to(DEVICE), reg.to(DEVICE)
            pred_cls, pred_reg, _ = model(seq, cell)
            loss, _, _ = compute_loss(pred_cls, pred_reg, cls, reg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # 验证集评估与早停
        val_metric = evaluate_model(model, val_loader)
        scheduler.step(val_metric["AUC"])

        if val_metric["AUC"] > best_auc:
            best_auc = val_metric["AUC"]
            best_state = model.state_dict()
            stop_count = 0
        else:
            stop_count += 1
            if stop_count >= PATIENCE:
                break

    # 加载最优模型权重
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# 结果可视化与汇总函数
def plot_stratified_kfold_results(df_all, df_fold_summary, save_dir="./stratified_kfold_results"):
    # 分层K折结果可视化，包含：各折AUC柱状图、全局指标箱线图、重复实验均值±标准差图
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ["Arial"]
    plt.rcParams['axes.unicode_minus'] = False

    # 各折平均AUC柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_fold_summary["fold_id"].astype(str), df_fold_summary["avg_auc"], color="#4472c4", yerr=df_fold_summary["std_auc"], capsize=5)
    ax.axhline(y=0.5, linestyle="--", color="red", alpha=0.7, label="Random Baseline AUC = 0.5")
    ax.axhline(y=df_fold_summary["avg_auc"].mean(), linestyle="-.", color="blue", alpha=0.7, label=f"Overall Mean AUC = {df_fold_summary['avg_auc'].mean():.4f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fold ID")
    ax.set_ylabel("AUC")
    ax.set_title("Stratified K-Fold Cross Validation [Average AUC of each Fold]", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/stratified_kfold_auc_bar.png", dpi=300)
    plt.close()

    # 全局指标箱线图
    metric_cols = ["auc", "accuracy", "f1"]
    metric_names = ["AUC", "Accuracy", "F1 Score"]
    data_box = [df_all[m].values for m in metric_cols]
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data_box, labels=metric_names, patch_artist=True)
    # 美化箱线图
    for patch in bp['boxes']:
        patch.set_facecolor('#4472c4')
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)
    ax.set_ylim(0, 1.05)
    ax.set_title("Stratified K-Fold Cross Validation [Global Metric Distribution]", fontweight="bold")
    ax.grid(axis="y", alpha=0.6, linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/stratified_kfold_metric_boxplot.png", dpi=300)
    plt.close()

    # 重复实验AUC趋势图
    df_repeat_mean = df_all.groupby("repeat_id")["auc"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_repeat_mean["repeat_id"].astype(str), df_repeat_mean["auc"], marker='o', color='#4472c4', linewidth=2, markersize=8)
    ax.axhline(y=df_repeat_mean["auc"].mean(), linestyle="-.", color="red", alpha=0.7, label=f"Overall Mean AUC = {df_repeat_mean['auc'].mean():.4f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Repeat ID")
    ax.set_ylabel("Mean AUC")
    ax.set_title("Stratified K-Fold Cross Validation [Mean AUC of Each Repeat]", fontweight="bold")
    ax.legend()
    plt.grid(axis="y", alpha=0.6, linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/stratified_kfold_repeat_auc_trend.png", dpi=300)
    plt.close()

# 主执行入口
def main():
    # 全局随机种子锁定
    set_all_seeds(SEED)

    # 实验超参数配置
    K_FOLDS = 5                                         # 分层K折的折数，默认5折，可改为10折等
    REPEAT_TIMES = 3                                    # 重复实验次数，降低小样本随机波动
    SMALL_BS = min(BATCH_SIZE, 4)                       # 小batch适配
    VAL_SPLIT_RATIO = 0.1                               # 训练集中划分验证集的比例
    # OUT_DIR = "./stratified_kfold_cv_results"         # 结果保存目录
    OUT_DIR = "./stratified_group_kfold_cv_results"     # 结果保存目录

    # 创建结果保存目录
    os.makedirs(OUT_DIR, exist_ok=True)

    # 全局加载一次数据
    seq_feats, cell_feats, labels_cls, labels_reg, serotypes, variant_ids = load_stratified_kfold_data()

    # 分层核心：分类标签展平，用于StratifiedKFold
    labels_cls_flat = labels_cls.flatten()
    # 复合分层key：标签+血清型，保证每折血清型分布也均衡
    strat_key = np.array([f"{int(labels_cls_flat[i])}_{serotypes[i]}" for i in range(len(labels_cls_flat))])

    # 初始化分层K折划分器
    # skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    gsfk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    print(f"\n分层{K_FOLDS}折交叉验证实验开始")
    print(f"总样本数: {len(seq_feats)} | 折数: {K_FOLDS} | 重复实验次数: {REPEAT_TIMES}")
    print(f"每折训练样本数: {int(len(seq_feats) * (K_FOLDS-1)/K_FOLDS)} | 每折测试样本数: {int(len(seq_feats)/K_FOLDS)}")

    # 全局结果记录
    full_records = []

    # 多轮重复实验循环
    for repeat in range(REPEAT_TIMES):
        repeat_seed = SEED + repeat * 100
        set_all_seeds(repeat_seed)
        print(f"\n第 {repeat+1}/{REPEAT_TIMES} 轮完整{K_FOLDS}折实验，随机种子={repeat_seed}")

        # 每折循环
        # for fold_idx, (train_idx, test_idx) in enumerate(skf.split(seq_feats, strat_key)):
        for fold_idx, (train_idx, test_idx) in enumerate(gsfk.split(seq_feats, labels_cls_flat, variant_ids)):
            print(f"\n第 {fold_idx+1}/{K_FOLDS} 折")
            print(f"  训练集样本数: {len(train_idx)} | 测试集样本数: {len(test_idx)}")
            print(f"  训练集正负样本比: {np.sum(labels_cls_flat[train_idx])/len(train_idx):.2f} | 测试集正负样本比: {np.sum(labels_cls_flat[test_idx])/len(test_idx):.2f}")

            # 划分训练全集 & 完全隔离的测试集
            seq_tr, seq_te = seq_feats[train_idx], seq_feats[test_idx]
            cell_tr, cell_te = cell_feats[train_idx], cell_feats[test_idx]
            cls_tr, cls_te = labels_cls[train_idx], labels_cls[test_idx]
            reg_tr, reg_te = labels_reg[train_idx], labels_reg[test_idx]
            sero_tr = [serotypes[i] for i in train_idx]

            # 训练集中分层划分验证集，保证验证集分布均衡
            val_strat_key = np.array([f"{int(cls_tr[i][0])}_{sero_tr[i]}" for i in range(len(cls_tr))])
            try:
                tr_sub_idx, val_idx = train_test_split(
                    range(len(seq_tr)),
                    test_size=VAL_SPLIT_RATIO,
                    random_state=repeat_seed,
                    stratify=val_strat_key
                )
            except ValueError:
                # 分层失败兜底：仅用分类标签分层
                print(f"  复合分层失败，降级为仅分类标签分层")
                tr_sub_idx, val_idx = train_test_split(
                    range(len(seq_tr)),
                    test_size=VAL_SPLIT_RATIO,
                    random_state=repeat_seed,
                    stratify=cls_tr.flatten()
                )

            # 拆分训练子集、验证子集
            seq_train = seq_tr[tr_sub_idx]
            cell_train = cell_tr[tr_sub_idx]
            cls_train = cls_tr[tr_sub_idx]
            reg_train = reg_tr[tr_sub_idx]
            seq_val = seq_tr[val_idx]
            cell_val = cell_tr[val_idx]
            cls_val = cls_tr[val_idx]
            reg_val = reg_tr[val_idx]

            # 构建Dataset与DataLoader（适配小样本，drop_last=False）
            def build_dataloader(s, c, cl, r, shuffle=False):
                ds = TensorDataset(
                    torch.tensor(s, dtype=torch.float32),
                    torch.tensor(c, dtype=torch.float32),
                    torch.tensor(cl, dtype=torch.float32),
                    torch.tensor(r, dtype=torch.float32)
                )
                return DataLoader(ds, batch_size=SMALL_BS, shuffle=shuffle, drop_last=False)

            train_loader = build_dataloader(seq_train, cell_train, cls_train, reg_train, shuffle=True)
            val_loader = build_dataloader(seq_val, cell_val, cls_val, reg_val, shuffle=False)
            test_loader = build_dataloader(seq_te, cell_te, cls_te, reg_te, shuffle=False)

            # 模型初始化 + BN转LN
            model = FourierAAVCapNet().to(DEVICE)
            model = replace_bn_with_ln(model)

            # 训练过程异常捕获
            try:
                model = train_stratified_kfold_single_run(model, train_loader, val_loader)
            except Exception as e:
                print(f"  第{fold_idx+1}折训练失败: {str(e)}")
                continue

            # 测试集评估异常捕获
            try:
                test_metric = evaluate_model(model, test_loader)
            except Exception as e:
                print(f"  第{fold_idx+1}折测试评估失败: {str(e)}")
                continue

            # 保存单条实验记录
            full_records.append({
                "repeat_id": repeat + 1,
                "fold_id": fold_idx + 1,
                "n_train_sample": len(train_idx),
                "n_test_sample": len(test_idx),
                "auc": test_metric["AUC"],
                "accuracy": test_metric["准确率"],
                "precision": test_metric["精确率"],
                "recall": test_metric["召回率"],
                "f1": test_metric["F1分数"]
            })

            # 打印单折结果
            print(f"  第{fold_idx+1}折测试结果 | AUC: {test_metric['AUC']:.4f} | 准确率: {test_metric['准确率']:.4f} | F1: {test_metric['F1分数']:.4f}")

    # 结果汇总与保存
    print(f"\n实验完成，结果汇总")
    # 转换为DataFrame
    df_all = pd.DataFrame(full_records)
    # 保存原始全量结果
    df_all.to_csv(f"{OUT_DIR}/stratified_kfold_all_repeat_raw.csv", index=False, encoding="utf-8-sig")
    print(f"全量原始结果已保存至: {OUT_DIR}/stratified_kfold_all_repeat_raw.csv")

    # 每折汇总统计（多轮重复的均值±标准差）
    df_fold_summary = df_all.groupby("fold_id").agg(
        avg_auc=("auc", "mean"),
        std_auc=("auc", "std"),
        avg_acc=("accuracy", "mean"),
        std_acc=("accuracy", "std"),
        avg_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        avg_test_n=("n_test_sample", "mean")
    ).reset_index().round(4)
    # 保存每折汇总结果
    df_fold_summary.to_csv(f"{OUT_DIR}/stratified_kfold_fold_summary.csv", index=False, encoding="utf-8-sig")
    print(f"每折汇总结果已保存至: {OUT_DIR}/stratified_kfold_fold_summary.csv")

    # 全局整体汇总统计
    df_global_summary = df_all.agg(
        avg_auc=("auc", "mean"),
        std_auc=("auc", "std"),
        avg_acc=("accuracy", "mean"),
        std_acc=("accuracy", "std"),
        avg_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        min_auc=("auc", "min"),
        max_auc=("auc", "max")
    ).round(4).T
    # 保存全局汇总结果
    df_global_summary.to_csv(f"{OUT_DIR}/stratified_kfold_global_summary.csv", encoding="utf-8-sig")
    print(f"全局汇总结果已保存至: {OUT_DIR}/stratified_kfold_global_summary.csv")

    # 打印全局结果
    print(f"\n全局实验结果")
    print(f"整体平均AUC: {df_global_summary['avg_auc'].values[0]} ± {df_global_summary['std_auc'].values[0]}")
    print(f"整体平均准确率: {df_global_summary['avg_acc'].values[0]} ± {df_global_summary['std_acc'].values[0]}")
    print(f"整体平均F1分数: {df_global_summary['avg_f1'].values[0]} ± {df_global_summary['std_f1'].values[0]}")
    print(f"AUC范围: {df_global_summary['min_auc'].values[0]} ~ {df_global_summary['max_auc'].values[0]}")

    # 结果可视化
    plot_stratified_kfold_results(df_all, df_fold_summary, save_dir=OUT_DIR)
    print(f"\n结果可视化图表已保存至: {OUT_DIR}/")

if __name__ == "__main__":
    main()