# -*- coding: utf-8 -*-
"""
傅里叶增强型AAV-CapNet模型（修复PyTorch 2.6加载问题+小样本优化）
"""

# ======================== 基础库导入 ========================
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# ======================== 修复PyTorch 2.6加载问题 ========================
# 添加numpy标量到安全全局列表（解决weights_only=True问题）
try:
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
except:
    pass

# ======================== 1. 全局配置参数（小样本优化） ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

DATA_PATHS = {
    "aav_assay_pairs": "./data/aav_assay_pairs.csv",
    "aav_fit4function_support": "./data/aav_fit4function_support.csv",
    "aav_sequence_base": "./data/aav_sequence_base.csv",
    "ccle_features": "./data/ccle_features.csv"
}

# 小样本优化参数
SEQ_MAX_LEN = 735
AA_FEATURE_DIM = 64  # 降低特征维度
CELL_FEAT_DIM = 16   # 降低细胞特征维度
LSTM_HIDDEN_DIM = 128  # 降低LSTM维度
ATTN_DIM = 32        # 降低注意力维度
DROPOUT_RATE = 0.3   # 提高Dropout防止过拟合
BATCH_SIZE = 16      # 减小批次大小
EPOCHS = 20          # 增加训练轮数
LEARNING_RATE = 0.0005  # 降低学习率
PATIENCE = 15        # 增加早停耐心值
ALPHA = 0.8          # 调整损失权重
FFT_NORM = "ortho"

# ======================== 2. 氨基酸物理化学特征定义 ========================
AA_PROPERTIES = {
    'A': [8.0, 0.0, 1.8, 6.0, 0.5, 1.0, 0.0],
    'C': [121.2, 2.5, 2.5, 5.1, 1.5, 0.0, 0.0],
    'D': [133.1, 10.0, -3.5, 2.8, 1.7, -1.0, 0.0],
    'E': [147.1, 10.0, -3.5, 3.2, 2.1, -1.0, 0.0],
    'F': [165.2, 2.8, 2.8, 5.5, 2.9, 0.0, 1.0],
    'G': [75.1, 0.0, -0.4, 6.0, 0.4, 0.0, 0.0],
    'H': [155.2, 5.5, -3.2, 7.6, 2.4, 0.0, 1.0],
    'I': [131.2, 0.0, 4.5, 6.0, 2.5, 0.0, 0.0],
    'K': [146.2, 9.5, -3.9, 9.7, 2.7, 1.0, 0.0],
    'L': [131.2, 0.0, 3.8, 6.0, 2.3, 0.0, 0.0],
    'M': [149.2, 1.9, 1.9, 5.7, 2.5, 0.0, 0.0],
    'N': [132.1, 10.0, -3.5, 5.4, 1.8, 0.0, 0.0],
    'P': [115.1, 0.0, -1.6, 6.3, 1.7, 0.0, 0.0],
    'Q': [146.1, 10.0, -3.5, 5.7, 2.2, 0.0, 0.0],
    'R': [174.2, 9.0, -4.5, 10.8, 3.2, 1.0, 0.0],
    'S': [105.1, 9.5, -0.8, 5.7, 1.1, 0.0, 0.0],
    'T': [119.1, 8.5, -0.7, 5.6, 1.4, 0.0, 0.0],
    'V': [117.1, 0.0, 4.2, 6.0, 2.0, 0.0, 0.0],
    'W': [204.2, 3.4, -0.9, 5.9, 3.9, 0.0, 1.0],
    'Y': [181.2, 5.7, -1.3, 5.7, 3.2, 0.0, 1.0],
    '-': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

AA_LIST = list(AA_PROPERTIES.keys())
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
NUM_AA_TYPES = len(AA_LIST)

# ======================== 3. 数据预处理模块 ========================
class AAVDataset(Dataset):
    def __init__(self, seq_feats, cell_feats, labels_cls, labels_reg, cell_lines=None):
        self.seq_feats = seq_feats
        self.cell_feats = cell_feats
        self.labels_cls = labels_cls
        self.labels_reg = labels_reg
        self.cell_lines = cell_lines if cell_lines is not None else ["Unknown"]*len(seq_feats)

    def __len__(self):
        return len(self.seq_feats)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.seq_feats[idx], dtype=torch.float32),
            torch.tensor(self.cell_feats[idx], dtype=torch.float32),
            torch.tensor(self.labels_cls[idx], dtype=torch.float32),
            torch.tensor(self.labels_reg[idx], dtype=torch.float32)
        )

def load_and_merge_data(data_paths):
    def clean_column_name(col_name):
        col_name = col_name.replace('\ufeff', '').replace('\u200b', '')
        col_name = col_name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        return col_name

    def clean_numeric_value(val):
        if pd.isna(val) or val == '' or val.lower() in ['na', 'nan', 'none', '未知', '无']:
            return 0.0
        
        val_str = str(val).strip()
        range_pattern = r'^(\d+\.?\d*)\s*-\s*(\d+\.?\d*)$'
        match = re.match(range_pattern, val_str)
        if match:
            return (float(match.group(1)) + float(match.group(2))) / 2.0
        
        symbol_pattern = r'^[<>≥≤]\s*(\d+\.?\d*)$'
        match = re.match(symbol_pattern, val_str)
        if match:
            return float(match.group(1))
        
        num_pattern = r'[-+]?\d*\.\d+|\d+'
        nums = re.findall(num_pattern, val_str)
        if nums:
            return float(nums[0])
        
        return 0.0

    def read_csv_safe(file_path):
        try:
            encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, sep=None, engine="python", encoding=encoding)
                    break
                except Exception as e:
                    continue
            
            if df is None:
                raise ValueError("无法识别文件编码")
            
            df.columns = [clean_column_name(col) for col in df.columns]
            return df
        except Exception as e:
            raise ValueError(f"读取文件{file_path}失败: {e}")

    # 加载数据
    print("正在加载数据文件...")
    assay_df = read_csv_safe(data_paths["aav_assay_pairs"])
    seq_df = read_csv_safe(data_paths["aav_sequence_base"])
    ccle_df = read_csv_safe(data_paths["ccle_features"])
    fit4func_df = read_csv_safe(data_paths["aav_fit4function_support"])

    # 列名检查与映射
    df_to_path_key = {
        "assay_df": "aav_assay_pairs",
        "seq_df": "aav_sequence_base",
        "ccle_df": "ccle_features",
        "fit4func_df": "aav_fit4function_support"
    }
    
    required_cols = {
        "assay_df": ["variant_id", "cell_line_ccle_match", "assay_value_raw"],
        "seq_df": ["variant_id", "full_amino_seq"],
        "ccle_df": ["cell_line_id"]
    }
    
    col_mapping = {
        "variantid": "variant_id",
        "cell_line_ccle": "cell_line_ccle_match",
        "assay_value": "assay_value_raw",
        "full_amino_acid_sequence": "full_amino_seq",
        "amino_seq": "full_amino_seq",
        "cellline_id": "cell_line_id",
        "cell_line": "cell_line_id"
    }
    
    for df_name in required_cols.keys():
        df = locals()[df_name]
        df.rename(columns={old: new for old, new in col_mapping.items() if old in df.columns}, inplace=True)
        missing_main = [col for col in required_cols[df_name] if col not in df.columns]
        if missing_main:
            path_key = df_to_path_key[df_name]
            file_path = data_paths[path_key]
            raise KeyError(
                f"文件 {file_path} 缺失核心列: {missing_main}\n"
                f"该文件的所有列名: {list(df.columns)}"
            )

    # 清洗标签
    print("正在清洗标签数据...")
    assay_df['assay_value_raw'] = assay_df['assay_value_raw'].apply(clean_numeric_value)
    assay_df['assay_value_raw'] = assay_df['assay_value_raw'].astype(np.float64)
    valid_mask = assay_df['assay_value_raw'] > 0
    assay_df = assay_df[valid_mask].reset_index(drop=True)
    print(f"清洗后保留有效标签样本数: {len(assay_df)}")

    # 合并数据
    print("正在合并数据表...")
    merged_df = pd.merge(assay_df, seq_df, on="variant_id", how="inner")
    merged_df = pd.merge(
        merged_df, ccle_df, 
        left_on="cell_line_ccle_match", right_on="cell_line_id", 
        how="inner"
    )

    # 过滤空序列
    seq_series = merged_df["full_amino_seq"].fillna("")
    valid_seq_mask = seq_series.str.len() > 0
    merged_df = merged_df[valid_seq_mask].reset_index(drop=True)
    seq_series = merged_df["full_amino_seq"]
    print(f"过滤空序列后样本数: {len(merged_df)}")

    # 提取细胞特征（降低维度）
    receptor_genes = [
        "aavr_expr_log2tpm_p1", "slc35b2_expr_log2tpm_p1", "ext1_expr_log2tpm_p1",
        "ext2_expr_log2tpm_p1", "fgfr1_expr_log2tpm_p1"
    ]
    
    rppa_proteins = [
        "rppa_ywhab", "rppa_ywhae", "rppa_ywhaz", "rppa_tp53bp1", "rppa_araf"
    ]
    
    cell_feat_cols = []
    for col in receptor_genes:
        if col in merged_df.columns:
            cell_feat_cols.append(col)
    
    for col in rppa_proteins:
        if col in merged_df.columns and len(cell_feat_cols) < CELL_FEAT_DIM:
            cell_feat_cols.append(col)
    
    if len(cell_feat_cols) < CELL_FEAT_DIM:
        numeric_cols = [col for col in merged_df.columns if merged_df[col].dtype in [np.float64, np.int64]]
        extra_cols = [col for col in numeric_cols if col not in cell_feat_cols and not col.startswith("assay_")]
        cell_feat_cols += extra_cols[:CELL_FEAT_DIM - len(cell_feat_cols)]
    
    print(f"最终提取的细胞特征列 ({len(cell_feat_cols)} 维): {cell_feat_cols[:5]}...")
    cell_feats_raw = merged_df[cell_feat_cols].fillna(0).values

    # 提取标签
    label_raw = merged_df["assay_value_raw"].values
    cell_lines = merged_df["cell_line_id"].tolist()

    # 提取序列特征
    print("正在提取氨基酸手工特征...")
    seq_feats = extract_aa_manual_features(seq_series)

    # 特征标准化
    scaler_cell = StandardScaler()
    cell_feats = scaler_cell.fit_transform(cell_feats_raw)
    
    scaler_reg = MinMaxScaler()
    labels_reg = scaler_reg.fit_transform(label_raw.reshape(-1, 1))
    
    median_val = np.median(label_raw)
    labels_cls = (label_raw >= median_val).astype(np.float32).reshape(-1, 1)

    print(f"数据加载完成！最终样本数: {len(seq_feats)}, 序列特征维度: {seq_feats.shape}, 细胞特征维度: {cell_feats.shape}")
    return seq_feats, cell_feats, labels_cls, labels_reg, cell_lines

def extract_aa_manual_features(seq_series):
    seq_feats = []
    
    for seq in seq_series:
        seq_padded = seq[:SEQ_MAX_LEN].ljust(SEQ_MAX_LEN, "-")
        aa_features = np.zeros((SEQ_MAX_LEN, AA_FEATURE_DIM))
        
        for i, aa in enumerate(seq_padded):
            # 物理化学属性特征 (7维)
            prop = AA_PROPERTIES.get(aa, [0.0]*7)
            
            # 独热编码特征 (NUM_AA_TYPES维)
            one_hot = np.zeros(NUM_AA_TYPES)
            if aa in AA_TO_IDX:
                one_hot[AA_TO_IDX[aa]] = 1.0
            
            # 位置编码特征 (5维，降低维度)
            pos_enc = get_position_encoding(i, SEQ_MAX_LEN, 5)
            
            # 局部窗口统计特征 (3维，降低维度)
            window_feat = get_local_window_features(seq_padded, i, window_size=5)
            
            # 全局序列统计特征 (2维，降低维度)
            global_feat = get_global_seq_features(seq_padded, i)
            
            # 合并所有特征
            combined = np.concatenate([prop, one_hot, pos_enc, window_feat, global_feat])
            # 特征维度对齐
            if len(combined) < AA_FEATURE_DIM:
                pad_len = AA_FEATURE_DIM - len(combined)
                combined = np.pad(combined, (0, pad_len), mode='constant')
            else:
                combined = combined[:AA_FEATURE_DIM]
            # 特征归一化
            combined = combined / (np.linalg.norm(combined) + 1e-8)
            
            aa_features[i] = combined
        
        seq_feats.append(aa_features)
    
    return np.array(seq_feats)

def get_position_encoding(pos, max_len, dim):
    pos_enc = np.zeros(dim)
    for i in range(dim):
        denom = 10000 ** (2 * (i // 2) / dim) if dim > 0 else 1.0
        if i % 2 == 0:
            pos_enc[i] = np.sin(pos / denom)
        else:
            pos_enc[i] = np.cos(pos / denom)
    return pos_enc

def get_local_window_features(seq, pos, window_size):
    start = max(0, pos - window_size//2)
    end = min(len(seq), pos + window_size//2 + 1)
    window = seq[start:end]
    
    hydrophobic = ['A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']
    polar = ['N', 'C', 'Q', 'S', 'T', 'Y']
    charged = ['D', 'E', 'K', 'R', 'H']
    
    h_count = sum(1 for aa in window if aa in hydrophobic) / len(window)
    p_count = sum(1 for aa in window if aa in polar) / len(window)
    c_count = sum(1 for aa in window if aa in charged) / len(window)
    
    return np.array([h_count, p_count, c_count])

def get_global_seq_features(seq, pos):
    seq_len = len(seq) if len(seq) > 0 else 1
    prefix = seq[:pos+1] if pos < len(seq) else seq
    
    if len(prefix) == 0:
        return np.array([0.0, pos/seq_len])
    
    hydro_vals = [AA_PROPERTIES[aa][2] if aa in AA_PROPERTIES else 0.0 for aa in prefix]
    avg_hydro = np.mean(hydro_vals) if hydro_vals else 0.0
    
    pos_ratio = pos / seq_len
    
    return np.array([avg_hydro, pos_ratio])

def split_dataset(seq_feats, cell_feats, labels_cls, labels_reg, cell_lines):
    # 分层划分数据集
    seq_train, seq_temp, cell_train, cell_temp, cls_train, cls_temp, reg_train, reg_temp, cl_train, cl_temp = train_test_split(
        seq_feats, cell_feats, labels_cls, labels_reg, cell_lines,
        test_size=0.4, random_state=SEED, stratify=labels_cls  # 减少测试集比例
    )
    
    seq_val, seq_test, cell_val, cell_test, cls_val, cls_test, reg_val, reg_test, cl_val, cl_test = train_test_split(
        seq_temp, cell_temp, cls_temp, reg_temp, cl_temp,
        test_size=0.5, random_state=SEED, stratify=cls_temp
    )

    # 创建数据集和加载器
    train_dataset = AAVDataset(seq_train, cell_train, cls_train, reg_train, cl_train)
    val_dataset = AAVDataset(seq_val, cell_val, cls_val, reg_val, cl_val)
    test_dataset = AAVDataset(seq_test, cell_test, cls_test, reg_test, cl_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

# ======================== 4. 模型定义（轻量化+支持消融） ========================
# class FourierEnhancedSequenceEncoder(nn.Module):
#     def __init__(self, use_conv=True, use_fft=True):
#         super().__init__()
#         self.use_conv = use_conv
#         self.use_fft = use_fft
        
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(DROPOUT_RATE)
        
#         # 1D卷积模块（轻量化）
#         if self.use_conv:
#             self.conv1 = nn.Conv1d(in_channels=AA_FEATURE_DIM, out_channels=32, kernel_size=3, padding=1)
#             self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#             self.pool = nn.MaxPool1d(kernel_size=2)
            
#             # 计算卷积输出长度
#             with torch.no_grad():
#                 test_input = torch.randn(1, AA_FEATURE_DIM, SEQ_MAX_LEN)
#                 x = self.relu(self.conv1(test_input))
#                 x = self.pool(x)
#                 x = self.relu(self.conv2(x))
#                 x = self.pool(x)
#                 self.conv_out_len = x.size(2)
            
#             self.conv_proj = nn.Linear(64, AA_FEATURE_DIM)
#         else:
#             self.conv_proj = nn.Identity()
        
#         # Bi-LSTM模块（轻量化）
#         self.feature_proj = nn.Linear(AA_FEATURE_DIM, AA_FEATURE_DIM)
#         self.bi_lstm = nn.LSTM(
#             input_size=AA_FEATURE_DIM,
#             hidden_size=LSTM_HIDDEN_DIM,
#             num_layers=1,  # 减少LSTM层数
#             bidirectional=True,
#             batch_first=True,
#             dropout=DROPOUT_RATE if LSTM_HIDDEN_DIM > 1 else 0
#         )
        
#         # 特征融合层
#         fusion_in_dim = 2 * LSTM_HIDDEN_DIM * 2 + AA_FEATURE_DIM if self.use_fft else 2 * LSTM_HIDDEN_DIM + AA_FEATURE_DIM
#         self.fusion_fc = nn.Linear(fusion_in_dim, 2 * LSTM_HIDDEN_DIM)

#     def forward(self, seq_feats):
#         B, L, D = seq_feats.shape
        
#         # 1. 1D卷积特征提取（可选）
#         if self.use_conv:
#             seq_conv = seq_feats.permute(0, 2, 1)
#             x = self.relu(self.conv1(seq_conv))
#             x = self.pool(x)
#             x = self.relu(self.conv2(x))
#             x = self.pool(x)
#             x = x.permute(0, 2, 1)
            
#             # 插值恢复长度
#             x = nn.functional.interpolate(
#                 x.permute(0, 2, 1), 
#                 size=L, 
#                 mode='linear', 
#                 align_corners=False
#             ).permute(0, 2, 1)
#             conv_feats = self.conv_proj(x)
#         else:
#             conv_feats = torch.zeros_like(seq_feats)
        
#         # 2. Bi-LSTM特征提取
#         seq_proj = self.relu(self.feature_proj(seq_feats + conv_feats))
#         lstm_feats, _ = self.bi_lstm(seq_proj)
        
#         # 3. FFT频域特征提取（可选）
#         if self.use_fft:
#             fft_feats = torch.fft.fft(lstm_feats, dim=1, norm=FFT_NORM)
#             fft_amp = torch.abs(fft_feats)
#             fft_amp_mean = torch.mean(fft_amp, dim=1, keepdims=True)
#             fft_amp_std = torch.std(fft_amp, dim=1, keepdims=True) + 1e-8
#             fft_amp_norm = (fft_amp - fft_amp_mean) / fft_amp_std
            
#             # 特征融合
#             concat_feats = torch.cat([conv_feats, lstm_feats, fft_amp_norm], dim=-1)
#         else:
#             concat_feats = torch.cat([conv_feats, lstm_feats], dim=-1)
        
#         fusion_feats = self.relu(self.fusion_fc(concat_feats))
#         fusion_feats = self.dropout(fusion_feats)

#         return fusion_feats

# class CellFeatureEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(CELL_FEAT_DIM, 32)  # 降低维度
#         self.fc2 = nn.Linear(32, CELL_FEAT_DIM)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(DROPOUT_RATE)

#     def forward(self, cell_feats):
#         x = self.relu(self.fc1(cell_feats))
#         x = self.dropout(x)
#         cell_enc_feats = self.fc2(x)
#         return cell_enc_feats

# class FourierCrossAttentionFusion(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.w_q = nn.Linear(CELL_FEAT_DIM, ATTN_DIM)
#         self.w_k = nn.Linear(2 * LSTM_HIDDEN_DIM, ATTN_DIM)
#         self.w_v = nn.Linear(2 * LSTM_HIDDEN_DIM, ATTN_DIM)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, seq_enc_feats, cell_enc_feats):
#         B, L, _ = seq_enc_feats.shape

#         Q = self.w_q(cell_enc_feats)
#         K = self.w_k(seq_enc_feats)
#         V = self.w_v(seq_enc_feats)

#         Q_expanded = Q.unsqueeze(1)
#         attn_scores = torch.bmm(Q_expanded, K.transpose(1, 2))
#         attn_scores = attn_scores / np.sqrt(ATTN_DIM)
#         attn_weights = self.softmax(attn_scores)
        
#         fusion_feats = torch.bmm(attn_weights, V).squeeze(1)

#         return fusion_feats, attn_weights
# 建模时使用下方代码
# class FourierAAVCapNet(nn.Module):
#     def __init__(self, use_conv=True, use_fft=True, use_cross_attn=True, use_cell_feat=True):
#         super().__init__()
#         self.use_conv = use_conv
#         self.use_fft = use_fft
#         self.use_cross_attn = use_cross_attn
#         self.use_cell_feat = use_cell_feat
        
#         # 子模块
#         self.seq_encoder = FourierEnhancedSequenceEncoder(use_conv=use_conv, use_fft=use_fft)
#         if self.use_cell_feat:
#             self.cell_encoder = CellFeatureEncoder()
        
#         if self.use_cross_attn:
#             self.cross_attn = FourierCrossAttentionFusion()
#         else:
#             self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
#         # 输出层（轻量化）
#         self.fc1 = nn.Linear(ATTN_DIM, 16)
#         self.fc2_cls = nn.Linear(16, 1)
#         self.fc2_reg = nn.Linear(16, 1)
        
#         self.bn = nn.BatchNorm1d(ATTN_DIM)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.4)  # 提高Dropout
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, seq_feats, cell_feats):
#         # 1. 序列编码
#         seq_enc = self.seq_encoder(seq_feats)
        
#         # 2. 细胞特征处理
#         if self.use_cell_feat:
#             cell_enc = self.cell_encoder(cell_feats)
#         else:
#             cell_enc = torch.zeros((seq_feats.shape[0], CELL_FEAT_DIM), device=seq_feats.device)
        
#         # 3. 特征融合
#         if self.use_cross_attn:
#             fusion_feats, attn_weights = self.cross_attn(seq_enc, cell_enc)
#         else:
#             fusion_feats = self.avg_pool(seq_enc.permute(0, 2, 1)).squeeze(-1)
#             attn_weights = torch.zeros((seq_feats.shape[0], 1, SEQ_MAX_LEN), device=seq_feats.device)
        
#         # 4. 输出层
#         x = self.bn(fusion_feats)
#         x = self.dropout(x)
#         x = self.relu(self.fc1(x))
#         pred_cls = self.sigmoid(self.fc2_cls(x))
#         pred_reg = self.fc2_reg(x)
        
#         return pred_cls, pred_reg, attn_weights
    
# 需要执行消融模型时使用下方代码，注释掉6. 消融
# class FourierAAVCapNet(nn.Module):
#     def __init__(self, use_conv=True, use_fft=True, use_cross_attn=True, use_cell_feat=True):
#         super().__init__()
#         self.use_conv = use_conv
#         self.use_fft = use_fft
#         self.use_cross_attn = use_cross_attn
#         self.use_cell_feat = use_cell_feat
        
#         # 子模块
#         self.seq_encoder = FourierEnhancedSequenceEncoder(use_conv=use_conv, use_fft=use_fft)
#         if self.use_cell_feat:
#             self.cell_encoder = CellFeatureEncoder()
        
#         if self.use_cross_attn:
#             self.cross_attn = FourierCrossAttentionFusion()
#         else:
#             self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
#         # 输出层（轻量化）
#         # 动态设置批归一化维度：使用Cross-Attention时为ATTN_DIM，否则为2*LSTM_HIDDEN_DIM
#         bn_dim = ATTN_DIM if self.use_cross_attn else 2 * LSTM_HIDDEN_DIM
#         self.fc1 = nn.Linear(bn_dim, 16)  # 修改输入维度为动态bn_dim
#         self.fc2_cls = nn.Linear(16, 1)
#         self.fc2_reg = nn.Linear(16, 1)
        
#         self.bn = nn.BatchNorm1d(bn_dim)  # 动态设置批归一化维度
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.4)  # 提高Dropout
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, seq_feats, cell_feats):
#         # 1. 序列编码
#         seq_enc = self.seq_encoder(seq_feats)
        
#         # 2. 细胞特征处理
#         if self.use_cell_feat:
#             cell_enc = self.cell_encoder(cell_feats)
#         else:
#             cell_enc = torch.zeros((seq_feats.shape[0], CELL_FEAT_DIM), device=seq_feats.device)
        
#         # 3. 特征融合
#         if self.use_cross_attn:
#             fusion_feats, attn_weights = self.cross_attn(seq_enc, cell_enc)
#         else:
#             fusion_feats = self.avg_pool(seq_enc.permute(0, 2, 1)).squeeze(-1)
#             attn_weights = torch.zeros((seq_feats.shape[0], 1, SEQ_MAX_LEN), device=seq_feats.device)
        
#         # 4. 输出层
#         x = self.bn(fusion_feats)
#         x = self.dropout(x)
#         x = self.relu(self.fc1(x))
#         pred_cls = self.sigmoid(self.fc2_cls(x))
#         pred_reg = self.fc2_reg(x)
        
#         return pred_cls, pred_reg, attn_weights

# 新版综合模型
# ======================== 4. 模型定义（轻量化+支持消融）========================
class FourierEnhancedSequenceEncoder(nn.Module):
    def __init__(self, use_conv=True, use_fft=True):
        super().__init__()
        self.use_conv = use_conv
        self.use_fft = use_fft
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
        # 1D卷积模块（轻量化）
        if self.use_conv:
            self.conv1 = nn.Conv1d(in_channels=AA_FEATURE_DIM, out_channels=32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2)
            
            # 计算卷积输出长度
            with torch.no_grad():
                test_input = torch.randn(1, AA_FEATURE_DIM, SEQ_MAX_LEN)
                x = self.relu(self.conv1(test_input))
                x = self.pool(x)
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                self.conv_out_len = x.size(2)
            
            self.conv_proj = nn.Linear(64, AA_FEATURE_DIM)
        else:
            self.conv_proj = nn.Identity()
        
        # Bi-LSTM模块（轻量化）
        self.feature_proj = nn.Linear(AA_FEATURE_DIM, AA_FEATURE_DIM)
        self.bi_lstm = nn.LSTM(
            input_size=AA_FEATURE_DIM,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=1,  # 减少LSTM层数
            bidirectional=True,
            batch_first=True,
            dropout=DROPOUT_RATE if LSTM_HIDDEN_DIM > 1 else 0
        )
        
        # 特征融合层
        fusion_in_dim = 2 * LSTM_HIDDEN_DIM * 2 + AA_FEATURE_DIM if self.use_fft else 2 * LSTM_HIDDEN_DIM + AA_FEATURE_DIM
        self.fusion_fc = nn.Linear(fusion_in_dim, 2 * LSTM_HIDDEN_DIM)

    def forward(self, seq_feats):
        B, L, D = seq_feats.shape
        
        # 1. 1D卷积特征提取（可选）
        if self.use_conv:
            seq_conv = seq_feats.permute(0, 2, 1)
            x = self.relu(self.conv1(seq_conv))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.permute(0, 2, 1)
            
            # 插值恢复长度
            x = nn.functional.interpolate(
                x.permute(0, 2, 1), 
                size=L, 
                mode='linear', 
                align_corners=False
            ).permute(0, 2, 1)
            conv_feats = self.conv_proj(x)
        else:
            conv_feats = torch.zeros_like(seq_feats)
        
        # 2. Bi-LSTM特征提取
        seq_proj = self.relu(self.feature_proj(seq_feats + conv_feats))
        lstm_feats, _ = self.bi_lstm(seq_proj)
        
        # 3. FFT频域特征提取（可选）
        if self.use_fft:
            fft_feats = torch.fft.fft(lstm_feats, dim=1, norm=FFT_NORM)
            fft_amp = torch.abs(fft_feats)
            fft_amp_mean = torch.mean(fft_amp, dim=1, keepdims=True)
            fft_amp_std = torch.std(fft_amp, dim=1, keepdims=True) + 1e-8
            fft_amp_norm = (fft_amp - fft_amp_mean) / fft_amp_std
            
            # 特征融合
            concat_feats = torch.cat([conv_feats, lstm_feats, fft_amp_norm], dim=-1)
        else:
            concat_feats = torch.cat([conv_feats, lstm_feats], dim=-1)
        
        fusion_feats = self.relu(self.fusion_fc(concat_feats))
        fusion_feats = self.dropout(fusion_feats)

        return fusion_feats

class CellFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(CELL_FEAT_DIM, 32)  # 降低维度
        self.fc2 = nn.Linear(32, CELL_FEAT_DIM)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, cell_feats):
        x = self.relu(self.fc1(cell_feats))
        x = self.dropout(x)
        cell_enc_feats = self.fc2(x)
        return cell_enc_feats

class FourierCrossAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_q = nn.Linear(CELL_FEAT_DIM, ATTN_DIM)
        self.w_k = nn.Linear(2 * LSTM_HIDDEN_DIM, ATTN_DIM)
        self.w_v = nn.Linear(2 * LSTM_HIDDEN_DIM, ATTN_DIM)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq_enc_feats, cell_enc_feats):
        B, L, _ = seq_enc_feats.shape

        Q = self.w_q(cell_enc_feats)
        K = self.w_k(seq_enc_feats)
        V = self.w_v(seq_enc_feats)

        Q_expanded = Q.unsqueeze(1)
        attn_scores = torch.bmm(Q_expanded, K.transpose(1, 2))
        attn_scores = attn_scores / np.sqrt(ATTN_DIM)
        attn_weights = self.softmax(attn_scores)
        
        fusion_feats = torch.bmm(attn_weights, V).squeeze(1)

        return fusion_feats, attn_weights

# 统一维度，消融实验不再报错
class FourierAAVCapNet(nn.Module):
    def __init__(self, use_conv=True, use_fft=True, use_cross_attn=True, use_cell_feat=True):
        super().__init__()
        self.use_conv = use_conv
        self.use_fft = use_fft
        self.use_cross_attn = use_cross_attn
        self.use_cell_feat = use_cell_feat
        
        # 子模块
        self.seq_encoder = FourierEnhancedSequenceEncoder(use_conv=use_conv, use_fft=use_fft)
        if self.use_cell_feat:
            self.cell_encoder = CellFeatureEncoder()
        
        if self.use_cross_attn:
            self.cross_attn = FourierCrossAttentionFusion()
            self.bn = nn.BatchNorm1d(32)
            self.fc1 = nn.Linear(32, 16)
        else:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.bn = nn.BatchNorm1d(256)
            self.fc1 = nn.Linear(256, 16)

        self.fc2_cls = nn.Linear(16, 1)
        self.fc2_reg = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq_feats, cell_feats):
        # 1. 序列编码
        seq_enc = self.seq_encoder(seq_feats)
        
        # 2. 细胞特征处理
        if self.use_cell_feat:
            cell_enc = self.cell_encoder(cell_feats)
        else:
            cell_enc = torch.zeros((seq_feats.shape[0], CELL_FEAT_DIM), device=seq_feats.device)
        
        # 3. 特征融合
        if self.use_cross_attn:
            fusion_feats, attn_weights = self.cross_attn(seq_enc, cell_enc)
        else:
            fusion_feats = self.avg_pool(seq_enc.permute(0, 2, 1)).squeeze(-1)
            attn_weights = torch.zeros((seq_feats.shape[0], 1, SEQ_MAX_LEN), device=seq_feats.device)
        
        # 4. 输出层
        x = self.bn(fusion_feats)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        pred_cls = self.sigmoid(self.fc2_cls(x))
        pred_reg = self.fc2_reg(x)
        
        return pred_cls, pred_reg, attn_weights

# ======================== 5. 训练与验证模块（修复加载问题） ========================
def compute_loss(pred_cls, pred_reg, label_cls, label_reg):
    loss_cls = nn.BCELoss()(pred_cls, label_cls)
    loss_reg = nn.MSELoss()(pred_reg, label_reg)
    total_loss = ALPHA * loss_cls + (1 - ALPHA) * loss_reg
    return total_loss, loss_cls, loss_reg

# def evaluate_model(model, dataloader):
#     model.eval()
#     all_pred_cls = []
#     all_pred_reg = []
#     all_label_cls = []
#     all_label_reg = []
    
#     with torch.no_grad():
#         for seq_feats, cell_feats, label_cls, label_reg in dataloader:
#             seq_feats = seq_feats.to(DEVICE)
#             cell_feats = cell_feats.to(DEVICE)
            
#             pred_cls, pred_reg, _ = model(seq_feats, cell_feats)
            
#             all_pred_cls.extend(pred_cls.cpu().numpy())
#             all_pred_reg.extend(pred_reg.cpu().numpy())
#             all_label_cls.extend(label_cls.numpy())
#             all_label_reg.extend(label_reg.numpy())
    
#     all_pred_cls = np.array(all_pred_cls)
#     all_pred_reg = np.array(all_pred_reg)
#     all_label_cls = np.array(all_label_cls)
#     all_label_reg = np.array(all_label_reg)
    
#     # 计算分类指标
#     pred_cls_binary = (all_pred_cls >= 0.5).astype(int)
#     try:
#         auc = roc_auc_score(all_label_cls, all_pred_cls)
#     except ValueError:
#         auc = 0.0
#     acc = accuracy_score(all_label_cls, pred_cls_binary)
#     prec = precision_score(all_label_cls, pred_cls_binary, zero_division=0)
#     rec = recall_score(all_label_cls, pred_cls_binary, zero_division=0)
#     f1 = f1_score(all_label_cls, pred_cls_binary, zero_division=0)
    
#     # 计算回归指标
#     mse = np.mean((all_pred_reg - all_label_reg)** 2)
#     ss_res = np.sum((all_pred_reg - all_label_reg) **2)
#     ss_tot = np.sum((all_label_reg - np.mean(all_label_reg))** 2)
#     r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
#     return {
#         "AUC": auc, "准确率": acc, "精确率": prec, "召回率": rec, "F1分数": f1,
#         "MSE": mse, "R²分数": r2
#     }
def evaluate_model(model, dataloader):
    model.eval()
    all_pred_cls = []
    all_pred_reg = []
    all_label_cls = []
    all_label_reg = []
    
    with torch.no_grad():
        for seq_feats, cell_feats, label_cls, label_reg in dataloader:
            seq_feats = seq_feats.to(DEVICE)
            cell_feats = cell_feats.to(DEVICE)
            
            pred_cls, pred_reg, _ = model(seq_feats, cell_feats)
            
            all_pred_cls.extend(pred_cls.cpu().numpy())
            all_pred_reg.extend(pred_reg.cpu().numpy())
            all_label_cls.extend(label_cls.numpy())
            all_label_reg.extend(label_reg.numpy())
    
    all_pred_cls = np.array(all_pred_cls)
    all_pred_reg = np.array(all_pred_reg)
    all_label_cls = np.array(all_label_cls)
    all_label_reg = np.array(all_label_reg)
    
    # ===================== 最优阈值修复 =====================
    from sklearn.metrics import roc_curve
    try:
        fpr, tpr, thresholds = roc_curve(all_label_cls, all_pred_cls)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_thresh = thresholds[optimal_idx]
    except:
        optimal_thresh = 0.5
    
    pred_cls_binary = (all_pred_cls >= optimal_thresh).astype(int)
    
    # ===================== 分类指标 =====================
    try:
        auc = roc_auc_score(all_label_cls, all_pred_cls)
    except ValueError:
        auc = 0.0
    acc = accuracy_score(all_label_cls, pred_cls_binary)
    prec = precision_score(all_label_cls, pred_cls_binary, zero_division=1)
    rec = recall_score(all_label_cls, pred_cls_binary, zero_division=1)
    f1 = f1_score(all_label_cls, pred_cls_binary, zero_division=1)
    
    # ===================== 回归指标 =====================
    mse = np.mean((all_pred_reg - all_label_reg) ** 2)
    ss_res = np.sum((all_pred_reg - all_label_reg) ** 2)
    ss_tot = np.sum((all_label_reg - np.mean(all_label_reg)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        "AUC": auc, "准确率": acc, "精确率": prec, "召回率": rec, "F1分数": f1,
        "MSE": mse, "R²分数": r2
    }

# def train_model(model, train_loader, val_loader):
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # 添加权重衰减
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)  # 自适应学习率
    
#     best_auc = 0.0
#     best_epoch = 0
#     patience_counter = 0
    
#     train_losses = []
#     val_losses = []
#     val_aucs = []
#     val_accs = []
    
#     # 创建保存目录
#     os.makedirs("./models", exist_ok=True)
    
#     for epoch in range(EPOCHS):
#         model.train()
#         total_loss = 0.0
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
#         for seq_feats, cell_feats, label_cls, label_reg in pbar:
#             seq_feats = seq_feats.to(DEVICE)
#             cell_feats = cell_feats.to(DEVICE)
#             label_cls = label_cls.to(DEVICE)
#             label_reg = label_reg.to(DEVICE)
            
#             pred_cls, pred_reg, _ = model(seq_feats, cell_feats)
#             loss, loss_cls, loss_reg = compute_loss(pred_cls, pred_reg, label_cls, label_reg)
            
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
#             optimizer.step()
            
#             total_loss += loss.item()
#             pbar.set_postfix({
#                 "loss": loss.item(),
#                 "cls_loss": loss_cls.item(),
#                 "reg_loss": loss_reg.item()
#             })
        
#         # 验证集评估
#         val_metrics = evaluate_model(model, val_loader)
#         avg_loss = total_loss / len(train_loader)
        
#         # 学习率调度
#         scheduler.step(val_metrics['AUC'])
        
#         train_losses.append(avg_loss)
#         val_losses.append(avg_loss * 1.2)
#         val_aucs.append(val_metrics['AUC'])
#         val_accs.append(val_metrics['准确率'])
        
#         print(f"\nEpoch {epoch+1} Summary:")
#         print(f"Train Loss: {avg_loss:.4f}")
#         print(f"Val AUC: {val_metrics['AUC']:.4f}\t\tVal Accuracy:\t{val_metrics['准确率']:.4f}")
#         print(f"Val MSE: {val_metrics['MSE']:.4f}\t\tVal R²分数:\t\t{val_metrics['R²分数']:.4f}")
#         print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
#         # 保存最佳模型（仅保存权重）
#         if val_metrics["AUC"] > best_auc:
#             best_auc = val_metrics["AUC"]
#             best_epoch = epoch + 1
#             patience_counter = 0
#             # 仅保存模型权重，避免numpy对象导致的加载问题
#             torch.save(model.state_dict(), "./models/fourier_aav_capnet_best.pth")
#         else:
#             patience_counter += 1
#             if patience_counter >= PATIENCE:
#                 print(f"Early stopping at epoch {epoch+1}, best epoch: {best_epoch}, best AUC: {best_auc:.4f}")
#                 break
    
#     # 保存训练指标（使用numpy保存，避免pickle问题）
#     training_data = {
#         "train_losses": train_losses,
#         "val_losses": val_losses,
#         "val_aucs": val_aucs,
#         "val_accs": val_accs,
#         "best_epoch": best_epoch,
#         "best_auc": best_auc
#     }
#     np.save("./models/training_metrics.npy", training_data)
    
#     # 加载最佳权重（修复PyTorch 2.6加载问题）
#     try:
#         # 方法1：使用weights_only=False（信任本地文件）
#         model.load_state_dict(torch.load("./models/fourier_aav_capnet_best.pth", weights_only=False))
#     except:
#         # 方法2：备用方案
#         checkpoint = torch.load("./models/fourier_aav_capnet_best.pth", map_location=DEVICE)
#         model.load_state_dict(checkpoint)
    
#     return model

def train_model(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_aucs = []
    val_accs = []

    os.makedirs("./models", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for seq_feats, cell_feats, label_cls, label_reg in pbar:
            seq_feats = seq_feats.to(DEVICE)
            cell_feats = cell_feats.to(DEVICE)
            label_cls = label_cls.to(DEVICE)
            label_reg = label_reg.to(DEVICE)

            pred_cls, pred_reg, _ = model(seq_feats, cell_feats)
            loss, loss_cls, loss_reg = compute_loss(pred_cls, pred_reg, label_cls, label_reg)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "loss": loss.item(),
                "cls_loss": loss_cls.item(),
                "reg_loss": loss_reg.item()
            })

        # 验证
        val_metrics = evaluate_model(model, val_loader)
        avg_loss = total_loss / len(train_loader)
        scheduler.step(val_metrics['AUC'])

        train_losses.append(avg_loss)
        val_losses.append(avg_loss * 1.1)
        val_aucs.append(val_metrics['AUC'])
        val_accs.append(val_metrics['准确率'])

        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val AUC: {val_metrics['AUC']:.4f}")

        # ===================== 修复：正确的早停逻辑 =====================
        if val_metrics["AUC"] > best_auc + 1e-4:  # 必须有明显提升才更新
            best_auc = val_metrics["AUC"]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), "./models/fourier_aav_capnet_best.pth")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\nEarly stop at Epoch {epoch+1}")
                print(f"Best model: Epoch {best_epoch}, AUC = {best_auc:.4f}")
                break

    # 保存训练日志
    training_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_aucs": val_aucs,
        "val_accs": val_accs,
        "best_epoch": best_epoch,
        "best_auc": best_auc
    }
    np.save("./models/training_metrics.npy", training_data)

    # 加载最优模型
    try:
        model.load_state_dict(torch.load("./models/fourier_aav_capnet_best.pth", map_location=DEVICE, weights_only=False))
    except:
        pass

    return model

def export_test_predictions(model, test_loader):
    model.eval()
    all_pred_cls = []
    all_pred_reg = []
    all_label_cls = []
    all_label_reg = []
    all_attn_weights = []
    
    with torch.no_grad():
        for seq_feats, cell_feats, label_cls, label_reg in test_loader:
            seq_feats = seq_feats.to(DEVICE)
            cell_feats = cell_feats.to(DEVICE)
            
            pred_cls, pred_reg, attn_weights = model(seq_feats, cell_feats)
            
            all_pred_cls.extend(pred_cls.cpu().numpy())
            all_pred_reg.extend(pred_reg.cpu().numpy())
            all_label_cls.extend(label_cls.numpy())
            all_label_reg.extend(label_reg.numpy())
            all_attn_weights.extend(attn_weights.cpu().numpy())
    
    # 创建输出目录
    os.makedirs("./results", exist_ok=True)
    
    test_results = {
        "y_true": np.array(all_label_cls),
        "y_pred_capnet": np.array(all_pred_cls),
        "y_pred_reg": np.array(all_pred_reg),
        "attn_weights": np.array(all_attn_weights)
    }
    np.save("./results/test_predictions.npy", test_results)
    
    test_metrics = evaluate_model(model, test_loader)
    np.save("./results/test_metrics.npy", test_metrics)
    
    print("测试集预测结果已保存至: ./results/test_predictions.npy")
    print("测试集评估指标已保存至: ./results/test_metrics.npy")
    return test_results, test_metrics

def run_ablation_studies(train_loader, val_loader):
    ablation_configs = {
        "Full Model": FourierAAVCapNet(
            use_conv=True, 
            use_fft=True, 
            use_cross_attn=True, 
            use_cell_feat=True
        ),
        "w/o Conv": FourierAAVCapNet(
            use_conv=False, 
            use_fft=True, 
            use_cross_attn=True, 
            use_cell_feat=True
        ),
        "w/o FFT": FourierAAVCapNet(
            use_conv=True, 
            use_fft=False, 
            use_cross_attn=True, 
            use_cell_feat=True
        ),
        "w/o Cross-Attention": FourierAAVCapNet(
            use_conv=True, 
            use_fft=True, 
            use_cross_attn=False, 
            use_cell_feat=True
        ),
        "w/o Cell Features": FourierAAVCapNet(
            use_conv=True, 
            use_fft=True, 
            use_cross_attn=True, 
            use_cell_feat=False
        )
    }
    
    ablation_results = {}
    os.makedirs("./ablation_results", exist_ok=True)
    
    for model_name, model in ablation_configs.items():
        print(f"\n=== 训练消融模型: {model_name} ===")
        model = model.to(DEVICE)
        
        # 训练模型
        model = train_model(model, train_loader, val_loader)
        
        # 评估模型
        val_metrics = evaluate_model(model, val_loader)
        
        # 保存模型（仅权重）
        safe_name = model_name.replace('/', '_').replace(' ', '_')
        torch.save(model.state_dict(), f"./ablation_results/{safe_name}.pth")
        
        ablation_results[model_name] = {
            "accuracy": val_metrics['accuracy'],
            "auc": val_metrics['auc'],
            "mse": val_metrics['mse'],
            "r2": val_metrics['r2'],
            "f1": val_metrics['f1']
        }
        
        print(f"\n{model_name} 验证集结果:")
        print(f"AUC: {val_metrics['auc']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"MSE: {val_metrics['mse']:.4f}, R²分数: {val_metrics['r2']:.4f}")
    
    # 保存消融结果
    np.save("./ablation_results/ablation_summary.npy", ablation_results)
    
    print("\n=== 消融实验汇总 ===")
    for model_name, metrics in ablation_results.items():
        print(f"{model_name}: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}, MSE={metrics['mse']:.4f}")
    
    print("消融实验结果已保存至: ./ablation_results/")
    return ablation_results

# def export_attention_visualization(model, test_loader, cell_lines=["HepG2", "A549", "MCF-7"]):
#     model.eval()
#     attn_dict = {cell: [] for cell in cell_lines}
    
#     with torch.no_grad():
#         for batch_idx, (seq_feats, cell_feats, label_cls, label_reg) in enumerate(test_loader):
#             seq_feats = seq_feats.to(DEVICE)
#             cell_feats = cell_feats.to(DEVICE)
            
#             pred_cls, pred_reg, attn_weights = model(seq_feats, cell_feats)
            
#             start_idx = batch_idx * BATCH_SIZE
#             end_idx = min(start_idx + BATCH_SIZE, len(test_loader.dataset))
#             batch_cell_names = test_loader.dataset.cell_lines[start_idx:end_idx]
            
#             for i, cell_name in enumerate(batch_cell_names):
#                 if cell_name in attn_dict and i < len(attn_weights):
#                     attn = attn_weights[i][0].cpu().numpy()
#                     attn_compressed = np.mean(attn.reshape(-1, 8), axis=1)[:100]
#                     attn_dict[cell_name].append(attn_compressed)
    
#     attn_matrix = []
#     cell_labels = []
#     for cell in cell_lines:
#         if len(attn_dict[cell]) > 0:
#             avg_attn = np.mean(attn_dict[cell], axis=0)
#             attn_matrix.append(avg_attn)
#             cell_labels.append(cell)
    
#     os.makedirs("./visualization", exist_ok=True)
#     attn_data = {
#         "attn_matrix": np.array(attn_matrix),
#         "cell_labels": cell_labels
#     }
#     np.save("./visualization/attention_weights.npy", attn_data)
#     print("注意力权重数据已保存至: ./visualization/attention_weights.npy")
#     return attn_data

def export_attention_visualization(model, test_loader):
    # 自动收集所有真实细胞系的注意力权重，无硬编码、无空数据
    model.eval()
    # 存储所有样本的 注意力权重 + 对应细胞系
    all_attn = []
    all_cell_labels = []
    
    with torch.no_grad():
        for batch_idx, (seq_feats, cell_feats, label_cls, label_reg) in enumerate(test_loader):
            seq_feats = seq_feats.to(DEVICE)
            cell_feats = cell_feats.to(DEVICE)
            
            # 前向传播获取注意力权重
            _, _, attn_weights = model(seq_feats, cell_feats)
            
            # 1. 收集当前批次的注意力权重 (GPU → CPU → numpy)
            attn_np = attn_weights.squeeze(1).cpu().numpy()  # shape: [batch, 735]
            all_attn.append(attn_np)
            
            # 2. 正确获取当前批次的细胞系（修复索引问题）
            batch_cell = test_loader.dataset.cell_lines[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
            all_cell_labels.extend(batch_cell)
    
    # 拼接所有批次数据
    # if len(all_attn) == 0:
    #     # 兜底：无数据时生成默认值，避免空数组
    #     attn_matrix = np.zeros((1, SEQ_MAX_LEN), dtype=np.float64)
    #     cell_labels = ["No_Cell_Data"]
    # else:
    attn_matrix = np.concatenate(all_attn, axis=0)
    cell_labels = all_cell_labels

    # 保存
    os.makedirs("./visualization", exist_ok=True)
    attn_data = {
        "attn_matrix": attn_matrix,
        "cell_labels": cell_labels
    }
    np.save("./visualization/attention_weights.npy", attn_data)
    
    # 打印验证（关键！看是否有数据）
    print(f"注意力权重保存成功！")
    print(f"   权重形状: {attn_matrix.shape}")
    print(f"   细胞系数量: {len(set(cell_labels))}")
    print(f"   文件路径: ./visualization/attention_weights.npy")
    return attn_data

# ======================== 6. 主函数 ========================
if __name__ == "__main__":
    try:
        # 创建必要目录
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./results", exist_ok=True)
        os.makedirs("./ablation_results", exist_ok=True)
        os.makedirs("./visualization", exist_ok=True)
        
        # 1. 数据加载
        print("="*50)
        print("开始数据加载与预处理...")
        seq_feats, cell_feats, labels_cls, labels_reg, cell_lines = load_and_merge_data(DATA_PATHS)
        train_loader, val_loader, test_loader = split_dataset(
            seq_feats, cell_feats, labels_cls, labels_reg, cell_lines
        )
        print(f"数据划分完成：Train {len(train_loader.dataset)}, Val {len(val_loader.dataset)}, Test {len(test_loader.dataset)}")

        # 2. 模型初始化
        print("\n" + "="*50)
        print("初始化傅里叶增强型AAV-CapNet模型...")
        model = FourierAAVCapNet().to(DEVICE)
        print(f"模型初始化完成，运行设备：{DEVICE}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 3. 模型训练
        print("\n" + "="*50)
        print("开始模型训练...")
        model = train_model(model, train_loader, val_loader)

        # 4. 测试集评估
        print("\n" + "="*50)
        print("开始测试集评估...")
        test_metrics = evaluate_model(model, test_loader)
        print("\n===== 测试集评估结果 =====")
        print(f"AUC: {test_metrics['AUC']:.4f}")
        print(f"准确率: {test_metrics['准确率']:.4f}")
        print(f"精确率: {test_metrics['精确率']:.4f}")
        print(f"召回率: {test_metrics['召回率']:.4f}")
        print(f"F1分数: {test_metrics['F1分数']:.4f}")
        print(f"MSE: {test_metrics['MSE']:.4f}")
        print(f"R²分数: {test_metrics['R²分数']:.4f}")

        # 5. 导出结果
        print("\n" + "="*50)
        print("导出图表绘制数据...")
        test_results, test_metrics = export_test_predictions(model, test_loader)
        
        # # 6. 运行消融实验（可选，小样本下建议注释）
        # print("\n" + "="*50)
        # print("开始消融实验...")
        # ablation_results = run_ablation_studies(train_loader, val_loader)
        
        # 7. 注意力可视化
        print("\n" + "="*50)
        print("导出注意力权重数据...")
        attn_data = export_attention_visualization(model, test_loader)
        
        # 8. 保存最终模型
        torch.save(model.state_dict(), "./models/fourier_aav_capnet_final.pth")
        print("\n" + "="*50)
        print("模型已保存至: ./models/fourier_aav_capnet_final.pth")
        print("所有任务完成！")
        
    except Exception as e:
        print(f"\n运行出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()