# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

# 导入模型与数据工具
from aav_capnet import (
    load_and_merge_data,
    split_dataset,
    FourierAAVCapNet,
    train_model,
    DEVICE,
    DATA_PATHS
)
plt.rcParams['font.family'] = ["Arial", "SimHei"]

# 工具函数
def extract_numpy_data(loader):
    seqs, cells, ys_cls = [], [], []
    for s, c, yc, _ in loader:
        seqs.append(s.numpy())
        cells.append(c.numpy())
        ys_cls.append(yc.numpy())
    return np.concatenate(seqs), np.concatenate(cells), np.concatenate(ys_cls)

def flatten_features(seq_feats, cell_feats):
    n = seq_feats.shape[0]
    return np.concatenate([seq_feats.reshape(n, -1), cell_feats], axis=1)

# 机器学习模型
def get_ml_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(256,128), max_iter=500, random_state=42)
    }

# 深度学习模型
def get_dl_preds(model, loader):
    model.eval()
    probs, y_true = [], []
    with torch.no_grad():
        for seq, cell, yc, _ in loader:
            seq, cell = seq.to(DEVICE), cell.to(DEVICE)
            out = model(seq, cell)
            if isinstance(model, FourierAAVCapNet):
                out = out[0]
            out = torch.sigmoid(out).cpu().numpy()
            probs.extend(out)
            y_true.extend(yc.numpy())
    return np.array(probs), np.array(y_true)

# 绘制ROC曲线
if __name__ == "__main__":
    os.makedirs("./figures", exist_ok=True)

    # 加载统一数据（所有模型共用）
    seq_feats, cell_feats, labels_cls, labels_reg, cell_lines = load_and_merge_data(DATA_PATHS)
    train_loader, val_loader, test_loader = split_dataset(seq_feats, cell_feats, labels_cls, labels_reg, cell_lines)
    seq_train, cell_train, y_cls_train = extract_numpy_data(train_loader)
    seq_test, cell_test, y_cls_test = extract_numpy_data(test_loader)

    X_train = flatten_features(seq_train, cell_train)
    X_test = flatten_features(seq_test, cell_test)

    # 准备画布
    plt.figure(figsize=(8, 6))

    # ML 模型 ROC
    print("Training ML models for ROC...")
    ml_models = get_ml_models()
    colors = ["#F7A24F", "#FBEB66", "#4EA660"]
    for idx, (name, clf) in enumerate(ml_models.items()):
        clf.fit(X_train, y_cls_train.ravel())
        y_score = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_cls_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})", color=colors[idx])

    # LSTM
    print("Training LSTM...")
    from torch import nn
    class LSTMModel(nn.Module):
        def __init__(self, seq_dim, cell_dim):
            super().__init__()
            self.lstm = nn.LSTM(seq_dim, 128, batch_first=True)
            self.fc_cell = nn.Linear(cell_dim, 64)
            self.fc = nn.Linear(128+64, 1)
        def forward(self, x_seq, x_cell):
            _, (h, _) = self.lstm(x_seq)
            c = torch.relu(self.fc_cell(x_cell))
            return self.fc(torch.cat([h[-1], c], dim=-1))

    lstm = LSTMModel(seq_feats.shape[-1], cell_feats.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    for _ in range(20):
        lstm.train()
        for seq, cell, yc, _ in train_loader:
            seq, cell, yc = seq.to(DEVICE), cell.to(DEVICE), yc.to(DEVICE)
            loss = nn.functional.binary_cross_entropy_with_logits(lstm(seq, cell), yc)
            opt.zero_grad()
            loss.backward()
            opt.step()
    y_pred, y_true = get_dl_preds(lstm, test_loader)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, lw=2, label=f"LSTM (AUC = {auc(fpr,tpr):.3f})", color = "#79CAFB")

    # CNN
    print("Training CNN...")
    class CNNModel(nn.Module):
        def __init__(self, seq_dim, cell_dim):
            super().__init__()
            self.conv = nn.Conv1d(seq_dim, 128, 3, padding=1)
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc_cell = nn.Linear(cell_dim, 64)
            self.fc = nn.Linear(128+64, 1)
        def forward(self, x_seq, x_cell):
            x = x_seq.transpose(1,2)
            x = torch.relu(self.conv(x))
            x = self.pool(x).squeeze(-1)
            c = torch.relu(self.fc_cell(x_cell))
            return self.fc(torch.cat([x,c], dim=-1))

    cnn = CNNModel(seq_feats.shape[-1], cell_feats.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    for _ in range(20):
        cnn.train()
        for seq, cell, yc, _ in train_loader:
            seq, cell, yc = seq.to(DEVICE), cell.to(DEVICE), yc.to(DEVICE)
            loss = nn.functional.binary_cross_entropy_with_logits(cnn(seq, cell), yc)
            opt.zero_grad()
            loss.backward()
            opt.step()
    y_pred, y_true = get_dl_preds(cnn, test_loader)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, lw=2, label=f"CNN (AUC = {auc(fpr,tpr):.3f})", color = "#AA77E9")

    # AAV-CapNet
    print("Training AAV-CapNet...")
    capnet = FourierAAVCapNet().to(DEVICE)
    capnet = train_model(capnet, train_loader, val_loader)
    try:
        capnet.load_state_dict(torch.load("./models/fourier_aav_capnet_best.pth", weights_only=False, map_location=DEVICE))
    except:
        pass
    y_pred, y_true = get_dl_preds(capnet, test_loader)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, lw=2, color='#CC247C', label=f"AAV-CapNet (AUC = {auc(fpr,tpr):.3f})")

    # 画图样式
    plt.plot([0,1],[0,1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False-Positive Rate (FPR)")
    plt.ylabel("True-Positive Rate (TPR)")
    plt.title("ROC Curve - AAV Model Comparison", fontweight = "bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    # 保存
    save_path = "./figures/ROC Curve.pdf"
    plt.savefig(save_path)
    plt.close()
    print(f"ROC 曲线已保存：{save_path}")