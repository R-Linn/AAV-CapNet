# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 项目全局配置导入
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from aav_capnet import (
    DEVICE, SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE,
    DROPOUT_RATE, FourierAAVCapNet, compute_loss, evaluate_model,
    DATA_PATHS, CELL_FEAT_DIM, extract_aa_manual_features
)
# 全局复用工具函数
def simplify_serotype(serotype):
    # 血清型标准化合并规则
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
    # 锁定全部随机源
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据清洗基础函数
def clean_column_name(col_name):
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

def load_lovo_data():
    # 读取三张核心表：aav_assay_pairs / aav_sequence_base / ccle_features
    # 废弃aav_fit4function_support（无特征输入模型）
    # 返回：seq_feats, cell_feats, labels_cls, labels_reg, variant_ids, serotypes
    print("加载三张核心数据表")
    # 读取实验表
    assay_df = read_csv_safe(DATA_PATHS["aav_assay_pairs"])
    # 读取序列&血清型表
    seq_df = read_csv_safe(DATA_PATHS["aav_sequence_base"])
    # 读取CCLE细胞特征表
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
    # 清洗实验数值，仅保留大于0有效实验记录
    assay_df['assay_value_raw'] = assay_df['assay_value_raw'].apply(clean_numeric_value).astype(np.float64)
    assay_df = assay_df[assay_df['assay_value_raw'] > 0].reset_index(drop=True)
    # 多表内连接
    # 实验表 + 序列表 通过 variant_id
    merged = pd.merge(assay_df, seq_df, on="variant_id", how="inner")
    # 合并结果 + CCLE细胞表 通过 cell_line_ccle_match = cell_line_id
    merged = pd.merge(merged, ccle_df, left_on="cell_line_ccle_match", right_on="cell_line_id", how="inner")
    # 过滤空氨基酸序列
    merged = merged[merged["full_amino_seq"].fillna("").str.len() > 0].reset_index(drop=True)
    # 提取并标准化血清型（用于分层）
    raw_sero = merged["serotype_backbone"].fillna("Unknown").tolist()
    serotypes = [simplify_serotype(s) for s in raw_sero]
    # 分组主键：AAV变体ID
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
    print(f"数据集加载完成 | 总样本={len(seq_feats)} | 独立变体={len(set(variant_ids))} | 血清类别={len(set(serotypes))}")
    return seq_feats, cell_feats, labels_cls, labels_reg, variant_ids, serotypes

# LOVO单轮训练函数
def train_lovo_single_run(model, train_loader, val_loader):
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
        # 验证集评估早停
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
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# 结果可视化函数
def plot_lovo_results(df_all, save_dir="./lovo_results"):
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ["Arial"]
    plt.rcParams['axes.unicode_minus'] = False
    # 各变体平均AUC柱状图
    df_var_mean = df_all.groupby("variant")["auc"].mean().reset_index()
    df_var_mean = df_var_mean.sort_values("auc", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df_var_mean["variant"].astype(str), df_var_mean["auc"], color="#4472c4")
    ax.axhline(y=0.5, linestyle="--", color="red", alpha=0.7, label="Random Baseline AUC = 0.5")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("AAV Variant ID")
    ax.set_ylabel("AUC")
    ax.set_title("LOVO Cross-Validation [Average AUC of each Variant (Mean of 3 Repetitions)]", fontweight = "bold")
    ax.legend()
    for bar in bars:
        height = bar.get_height()
        ax.annotate('%.3f' % height,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/lovo_variant_auc_bar.png", dpi=300)
    plt.close()

# 主执行入口
def main():
    set_all_seeds(SEED)
    # 实验超参数
    REPEAT_TIMES = 3      # 重复3次降低小样本随机波动
    SMALL_BS = min(BATCH_SIZE, 4)
    VAL_SPLIT_RATIO = 0.1
    OUT_DIR = "./leave_one_variant_out_results"
    os.makedirs(OUT_DIR, exist_ok=True)
    # 全局加载一次数据（仅读取三张核心CSV）
    seq_feats, cell_feats, labels_cls, labels_reg, variant_ids, serotypes = load_lovo_data()
    groups = np.array(variant_ids)
    unique_vars = sorted(list(set(variant_ids)))
    print(f"LOVO待留出独立变体总数：{len(unique_vars)}")
    print(f"总样本数 seq_feats = {len(seq_feats)}, groups len={len(groups)}, labels_cls shape={labels_cls.shape}")
    full_records = []
    sample_pred_records = []

    # split()放到repeat循环里面，每轮重新生成索引迭代器
    for repeat in range(REPEAT_TIMES):
        repeat_seed = SEED + repeat * 100
        set_all_seeds(repeat_seed)
        print(f"\n第 {repeat+1}/{REPEAT_TIMES} 轮完整LOVO，种子={repeat_seed}")
        logo_splitter = LeaveOneGroupOut()
        # 每一轮重复，重新执行split生成全新索引
        for fold_idx, (train_idx, test_idx) in enumerate(logo_splitter.split(seq_feats, labels_cls.flatten(), groups=groups)):
            test_var = np.unique(groups[test_idx])[0]
            n_test_sample = len(test_idx)
            print(f"  当前留出变体 {test_var}，测试样本量 {n_test_sample}")

            # 划分训练全集 & 完全隔离的测试集 —— train_idx/test_idx是原始全集的全局索引
            seq_tr, seq_te = seq_feats[train_idx], seq_feats[test_idx]
            cell_tr, cell_te = cell_feats[train_idx], cell_feats[test_idx]
            cls_tr, cls_te = labels_cls[train_idx], labels_cls[test_idx]
            reg_tr, reg_te = labels_reg[train_idx], labels_reg[test_idx]
            sero_tr = [serotypes[i] for i in train_idx]

            # 双重分层：分类标签+血清型复合分层，保证验证集分布均衡
            strat_key = np.array([f"{int(cls_tr[i][0])}_{sero_tr[i]}" for i in range(len(cls_tr))])
            try:
                tr_sub_idx, val_idx = train_test_split(
                    range(len(seq_tr)),
                    test_size=VAL_SPLIT_RATIO,
                    random_state=repeat_seed,
                    stratify=strat_key
                )
            except ValueError:
                # 分层失败降级为仅标签分层
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

            # 构建Dataset与DataLoader（全部drop_last=False适配极小测试集）
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

            # 模型初始化 + BN全局替换LN
            model = FourierAAVCapNet().to(DEVICE)
            model = replace_bn_with_ln(model)
            # 训练容错
            try:
                model = train_lovo_single_run(model, train_loader, val_loader)
            except Exception as e:
                print(f"变体 {test_var} 训练失败: {str(e)}")
                continue
            # 测试集评估容错
            try:
                met = evaluate_model(model, test_loader)
            except Exception as e:
                print(f"变体 {test_var} 评估失败: {str(e)}")
                continue

            # 收集该变体所有测试样本真实值、预测回归值
            with torch.no_grad():
                for seq_feats_batch, cell_feats_batch, label_cls, label_reg in test_loader:
                    seq_feats_batch = seq_feats_batch.to(DEVICE)
                    cell_feats_batch = cell_feats_batch.to(DEVICE)
                    _, pred_reg, _ = model(seq_feats_batch, cell_feats_batch)
                    yt_reg = label_reg.detach().cpu().numpy().flatten()
                    yp_reg = pred_reg.detach().cpu().numpy().flatten()
                    for yt, yp in zip(yt_reg, yp_reg):
                        sample_pred_records.append({
                            "repeat_id": repeat + 1,
                            "variant": test_var,
                            "y_true_reg": float(yt),
                            "y_pred_reg": float(yp)
                        })

            full_records.append({
                "repeat_id": repeat + 1,
                "variant": test_var,
                "n_test": n_test_sample,
                "auc": met["AUC"],
                "accuracy": met["准确率"],
                "precision": met["精确率"],
                "recall": met["召回率"],
                "f1": met["F1分数"]
            })

    # 全部重复循环结束，汇总输出
    df_all = pd.DataFrame(full_records)
    df_all.to_csv(f"{OUT_DIR}/lovo_all_repeat_raw.csv", index=False, encoding="utf-8-sig")

    df_sample_pred = pd.DataFrame(sample_pred_records)
    df_sample_pred.to_csv(f"{OUT_DIR}/lovo_sample_level_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"\n预测文件输出完成 → {OUT_DIR}/lovo_sample_level_predictions.csv")

    var_summary = df_all.groupby("variant").agg(
        avg_auc=("auc", "mean"),
        std_auc=("auc", "std"),
        avg_acc=("accuracy", "mean"),
        std_acc=("accuracy", "std"),
        avg_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        avg_test_n=("n_test", "mean")
    ).reset_index().round(4)
    var_summary.to_csv(f"{OUT_DIR}/lovo_variant_summary.csv", index=False, encoding="utf-8-sig")

    global_avg = {
        "global_mean_auc": round(df_all["auc"].mean(), 4),
        "global_std_auc": round(df_all["auc"].std(), 4),
        "global_mean_acc": round(df_all["accuracy"].mean(), 4),
        "global_std_acc": round(df_all["accuracy"].std(), 4),
        "global_mean_f1": round(df_all["f1"].mean(), 4),
        "global_std_f1": round(df_all["f1"].std(), 4)
    }
    print("\nLOVO全局泛化性能（全部变体+3次重复）")
    for k, v in global_avg.items():
        print(f"{k}: {v}")
    pd.DataFrame([global_avg]).to_csv(f"{OUT_DIR}/lovo_global_summary.csv", index=False, encoding="utf-8-sig")

    plot_lovo_results(df_all, OUT_DIR)
    print(f"\n所有原始结果、汇总表、图表已保存至 {OUT_DIR}/")

if __name__ == "__main__":
    import re
    main()
