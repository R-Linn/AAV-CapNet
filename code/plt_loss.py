# -*- coding: utf-8 -*-
# 请在运行本程序之前重新运行原始模型，保证所测模型为原始模型
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.size']  = 12
plt.rcParams ["font.family"] = ["Arial", "SimHei"]
plt.rcParams['axes.linewidth'] = 1.2
os.makedirs("./figures", exist_ok=True)

if __name__ == "__main__":
    metric_path = "./models/training_metrics.npy"
    data = np.load(metric_path, allow_pickle=True).item()

    train_loss = data["train_losses"]
    val_loss   = data["val_losses"]
    val_auc    = data["val_aucs"]
    best_epoch = data["best_epoch"]
    best_auc   = data["best_auc"]
    epochs     = np.arange(1, len(train_loss)+1)

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.set_xticks(range(0, 20 + 1, 1))

    # 左轴 Loss
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#c0392b")
    ax1.plot(epochs, train_loss, marker="o", markersize=3, label="Train Loss", color="#e74c3c")
    ax1.plot(epochs, val_loss, marker="o", markersize=3, label="Validation Loss", color="#2ECC71")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.legend(loc="upper left")
    ax1.set_ylim (0.3, 1)

    # 右轴 AUC
    ax2 = ax1.twinx()
    ax2.set_ylabel("AUC", color="#2980b9")
    ax2.plot(epochs, val_auc, marker="o", markersize=3, label="Validation AUC", color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax2.set_ylim(0.0, 1)
    ax2.legend(loc="upper right")

    # 只在合理轮数标注早停
    if best_epoch > 2:
        ax1.axvline(x=best_epoch, color="black", linestyle="--", linewidth=1.2)
        best_train_loss = train_loss[best_epoch]
        best_val_loss = val_loss[best_epoch]
        ax1.text(best_epoch+1, max(train_loss)*0.7,
                 f"Best Epoch {best_epoch}\
                    \nAUC = {best_auc:.3f}\
                    \nTrain Loss = {best_train_loss:.3f}\
                    \nValidation Loss = {best_val_loss:.3f}",
                 fontsize=10, fontweight="bold")

    plt.title("Training Loss & AUC Curve", {'fontweight': 'bold'})
    plt.tight_layout()
    plt.savefig("./figures/Training Loss & AUC Curve.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"训练曲线已保存")
    print(f"最佳模型在 Epoch {best_epoch}")