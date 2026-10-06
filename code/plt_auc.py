# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

warnings. filterwarnings ("ignore")

# 主文件名
from aav_capnet import (
    load_and_merge_data,
    split_dataset,
    FourierAAVCapNet,
    train_model,
    evaluate_model,
    DEVICE,
    DATA_PATHS
)

# 全局统一配置
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 1. 特征处理（统一所有模型输入）
def flatten_features(seq_feats, cell_feats):
    n = seq_feats.shape[0]
    return np.concatenate([seq_feats.reshape(n, -1), cell_feats], axis=1)

# 2. 统一评价函数
def compute_metrics(y_true_cls, y_pred_cls, y_prob,
                    y_true_reg=None, y_pred_reg=None):
    res = {}
    res["准确率"] = round(accuracy_score(y_true_cls, y_pred_cls), 4)
    res["精确率"] = round(precision_score(y_true_cls, y_pred_cls, zero_division=0), 4)
    res["召回率"] = round(recall_score(y_true_cls, y_pred_cls, zero_division=0), 4)
    res["F1分数"] = round(f1_score(y_true_cls, y_pred_cls, zero_division=0), 4)

    try:
        res["AUC"] = round(roc_auc_score(y_true_cls, y_prob), 4)
    except:
        res["AUC"] = 0

    if y_true_reg is not None:
        res["MSE"] = round(np.mean((y_true_reg - y_pred_reg) ** 2), 4)
        res["R²分数"] = round(r2_score(y_true_reg, y_pred_reg), 4)
    else:
        res["MSE"] = None
        res["R²分数"] = None

    return res

# 3. 统一 sklearn 模型训练
def train_ml_models(X_train, X_test, y_cls_train, y_cls_test,
                    y_reg_train, y_reg_test):
    results = {}
    models = {
        "RandomForest": (RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
                         RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED)),
        "XGBoost": (XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED),
                    XGBRegressor(random_state=RANDOM_SEED)),
        "MLP": (MLPClassifier(hidden_layer_sizes=(256,128), max_iter=500, random_state=RANDOM_SEED),
                MLPRegressor(hidden_layer_sizes=(256,128), max_iter=500, random_state=RANDOM_SEED))
    }

    for name, (clf, reg) in models.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_cls_train.ravel())
        reg.fit(X_train, y_reg_train.ravel())

        prob = clf.predict_proba(X_test)[:, 1]
        pred_cls = (prob >= 0.5).astype(int)
        pred_reg = reg.predict(X_test)

        results[name] = compute_metrics(y_cls_test, pred_cls, prob, y_reg_test, pred_reg)
    return results

# 4. 统一 DL 模型训练（20轮 + 细胞特征）
class LSTMModel(nn.Module):
    def __init__(self, seq_input_dim, cell_input_dim):
        super().__init__()
        self.lstm = nn.LSTM(seq_input_dim, 128, batch_first=True)
        self.fc_cell = nn.Linear(cell_input_dim, 64)
        self.fc = nn.Linear(128 + 64, 1)

    def forward(self, x_seq, x_cell):
        _, (h, _) = self.lstm(x_seq)
        c = F.relu(self.fc_cell(x_cell))
        return self.fc(torch.cat([h[-1], c], dim=-1))

class CNNModel(nn.Module):
    def __init__(self, seq_input_dim, cell_input_dim):
        super().__init__()
        self.conv = nn.Conv1d(seq_input_dim, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc_cell = nn.Linear(cell_input_dim, 64)
        self.fc = nn.Linear(128 + 64, 1)

    def forward(self, x_seq, x_cell):
        x = x_seq.transpose(1, 2)
        x = F.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        c = F.relu(self.fc_cell(x_cell))
        return self.fc(torch.cat([x, c], dim=-1))

def train_dl_model(model, train_loader, test_loader, epochs=20):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for seq, cell, y_cls, _ in train_loader:
            seq, cell, y_cls = seq.to(DEVICE), cell.to(DEVICE), y_cls.to(DEVICE)
            pred = model(seq, cell)
            loss = F.binary_cross_entropy_with_logits(pred, y_cls)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    probs, preds, y_true = [], [], []
    with torch.no_grad():
        for seq, cell, y_cls, _ in test_loader:
            seq, cell = seq.to(DEVICE), cell.to(DEVICE)
            out = torch.sigmoid(model(seq, cell)).cpu().numpy()
            probs.extend(out)
            preds.extend((out >= 0.5).astype(int))
            y_true.extend(y_cls.numpy())

    return compute_metrics(np.array(y_true), np.array(preds), np.array(probs))

# 5. 绘图
def plot_accuracy_auc_together(results_df, save_path):
    models = results_df.index.tolist()
    acc = results_df["准确率"].values
    auc = results_df["AUC"].values

    x = np.arange(len(models))
    width = 0.35
    plt.rcParams['font.family'] = ["Arial", "SimHei"]
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, acc, width, label="Accuracy", color = "#E74C3C")
    plt.bar(x + width/2, auc, width, label="AUC", color = "#3498DB")

    for i in range(len(x)):
        plt.text(x[i] - width/2, acc[i], f"{acc[i]:.4f}", ha='center', va='bottom', fontsize=9)
        plt.text(x[i] + width/2, auc[i], f"{auc[i]:.4f}", ha='center', va='bottom', fontsize=9)

    plt.xticks(x, models, rotation=30)
    plt.ylabel("Score")
    plt.title("Model Comparison", fontweight = "bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 6. 主程序
if __name__ == "__main__":
    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./tables", exist_ok=True)

    # 全局唯一一次数据加载与划分
    print("Loading data...")
    seq_feats, cell_feats, labels_cls, labels_reg, cell_lines = load_and_merge_data(DATA_PATHS)
    train_loader, val_loader, test_loader = split_dataset(seq_feats, cell_feats, labels_cls, labels_reg, cell_lines)

    # 从 dataloader 提取统一的 numpy 数据（所有模型共用）
    def extract_numpy_data(loader):
        seqs, cells, ys_cls, ys_reg = [], [], [], []
        for s, c, yc, yr in loader:
            seqs.append(s.numpy())
            cells.append(c.numpy())
            ys_cls.append(yc.numpy())
            ys_reg.append(yr.numpy())
        return (np.concatenate(seqs), np.concatenate(cells),
                np.concatenate(ys_cls), np.concatenate(ys_reg))

    seq_train, cell_train, y_cls_train, y_reg_train = extract_numpy_data(train_loader)
    seq_test, cell_test, y_cls_test, y_reg_test = extract_numpy_data(test_loader)

    X_train = flatten_features(seq_train, cell_train)
    X_test = flatten_features(seq_test, cell_test)

    # 训练所有模型
    print("Training ML models...")
    ml_results = train_ml_models(X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test)

    print("Training LSTM...")
    lstm_res = train_dl_model(LSTMModel(seq_feats.shape[-1], cell_feats.shape[-1]), train_loader, test_loader, epochs=20)

    print("Training CNN...")
    cnn_res = train_dl_model(CNNModel(seq_feats.shape[-1], cell_feats.shape[-1]), train_loader, test_loader, epochs=20)

    print("Training AAV-CapNet...")
    capnet = FourierAAVCapNet().to(DEVICE)
    capnet = train_model(capnet, train_loader, val_loader)
    
    # 修复加载
    try:
        capnet.load_state_dict(torch.load("./models/fourier_aav_capnet_best.pth", weights_only=False, map_location=DEVICE))
    except:
        pass
    
    # 正确评估
    capnet_res = evaluate_model(capnet, test_loader)

    # 汇总结果（只保留分类指标，回归去掉）
    results = {}
    for k, v in ml_results.items():
        results[k] = {
            "准确率": v["准确率"],
            "精确率": v["精确率"],
            "召回率": v["召回率"],
            "F1分数": v["F1分数"],
            "AUC": v["AUC"],
            "MSE": "",
            "R²分数": ""
        }

    for name, res in [("LSTM", lstm_res), ("CNN", cnn_res), ("AAV-CapNet", capnet_res)]:
        results[name] = {
            "准确率": round(res["准确率"],4),
            "精确率": round(res["精确率"],4),
            "召回率": round(res["召回率"],4),
            "F1分数": round(res["F1分数"],4),
            "AUC": round(res["AUC"],4),
            "MSE": "",
            "R²分数": ""
        }

    df = pd.DataFrame(results).T
    # df. drop(["MSE", "R²分数"])
    csv_path = "./tables/Model Comparison.csv"
    df.to_csv(csv_path)
    print(f"Saved CSV: {csv_path}")

    pdf_path = "./figures/Accuracy AUC Comparison.pdf"
    plot_accuracy_auc_together(df, pdf_path)
    print(f"Saved plot: {pdf_path}")
    print("全部模型已对比完成！")