import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_curves(log_csv, out_png):
    try:
        df = pd.read_csv(log_csv)
        if df.empty or "round" not in df.columns:
            print(f"[BỎ QUA] File rỗng hoặc thiếu cột 'round': {log_csv}")
            return
        plt.figure()
        for col in ["mae", "rmse", "acc", "macro_f1", "recall"]:
            if col in df.columns:
                plt.plot(df["round"], df[col], label=col.upper())
        plt.xlabel("Round"); plt.ylabel("Metric value")
        plt.title(os.path.basename(log_csv))
        plt.legend(); plt.tight_layout()
        plt.savefig(out_png, dpi=180); plt.close()
    except Exception as e:
        print(f"[LỖI] plot_curves: {log_csv} → {e}")

def plot_confusion_matrix(cm_csv, out_png):
    try:
        cm = pd.read_csv(cm_csv, header=None).values
        if cm.size == 0:
            print(f"[BỎ QUA] Confusion matrix rỗng: {cm_csv}")
            return
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
        fig, ax = plt.subplots()
        disp.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title(os.path.basename(cm_csv).replace(".csv", ""))
        fig.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)
    except Exception as e:
        print(f"[LỖI] plot_confusion_matrix: {cm_csv} → {e}")

def plot_trust_heatmap(csv_path, out_png):
    try:
        df = pd.read_csv(csv_path)
        if df.empty or not {"round", "client", "trust"}.issubset(df.columns):
            print(f"[BỎ QUA] File trust không hợp lệ: {csv_path}")
            return
        pivot = df.pivot(index="client", columns="round", values="trust").values
        plt.figure()
        plt.imshow(pivot, aspect="auto")
        plt.colorbar(label="Trust")
        plt.xlabel("Round"); plt.ylabel("Client")
        plt.title("TAR-Fed-Fuzzy: Trust dynamics")
        plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()
    except Exception as e:
        print(f"[LỖI] plot_trust_heatmap: {csv_path} → {e}")

def save_ck_wide(trust_csv, out_csv):
    try:
        df = pd.read_csv(trust_csv)
        if df.empty or not {"round", "client", "trust"}.issubset(df.columns):
            print(f"[BỎ QUA] File trust không hợp lệ: {trust_csv}")
            return
        wide = df.pivot(index="round", columns="client", values="trust").sort_index()
        wide.columns = [f"C{c}" for c in wide.columns]
        wide["mean"] = wide.mean(axis=1)
        wide["std"] = wide.std(axis=1)
        wide.reset_index().to_csv(out_csv, index=False)
    except Exception as e:
        print(f"[LỖI] save_ck_wide: {trust_csv} → {e}")

def plot_trust_dynamics(trust_csv, out_png_line, out_png_heat):
    try:
        df = pd.read_csv(trust_csv)
        if df.empty or not {"round", "client", "trust"}.issubset(df.columns):
            print(f"[BỎ QUA] File trust không hợp lệ: {trust_csv}")
            return

        # Line plot
        plt.figure()
        for cid, g in df.groupby("client"):
            plt.plot(g["round"], g["trust"], alpha=0.4)
        pivot = df.pivot(index="round", columns="client", values="trust").sort_index()
        mean = pivot.mean(axis=1); std = pivot.std(axis=1)
        plt.plot(mean.index, mean.values, linewidth=2.5, label="mean Ck")
        plt.fill_between(mean.index, (mean - std).values, (mean + std).values, alpha=0.2, label="±1 std")
        plt.xlabel("Round"); plt.ylabel("Trust Ck"); plt.title("TAR-Fed-Fuzzy trust dynamics")
        plt.legend(); plt.tight_layout(); plt.savefig(out_png_line, dpi=180); plt.close()

        # Heatmap
        plt.figure()
        plt.imshow(pivot.values.T, aspect="auto", origin="lower")
        plt.colorbar(label="Trust Ck")
        plt.xlabel("Round"); plt.ylabel("Client"); plt.title("Trust heatmap (client × round)")
        plt.tight_layout(); plt.savefig(out_png_heat, dpi=180); plt.close()
    except Exception as e:
        print(f"[LỖI] plot_trust_dynamics: {trust_csv} → {e}")

def plot_compare_four(log_dir, out_prefix):
    methods = ["fedavg", "fedprox", "fedtrim", "tarfed_fuzzy"]
    labels = {
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "fedtrim": "FedTrim",
        "tarfed_fuzzy": "TAR-Fed-Fuzzy"
    }
    metrics = ["mae", "rmse", "acc", "macro_f1", "recall"]
    data = {}

    for m in methods:
        p = os.path.join(log_dir, f"logs_{m}.csv")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if not df.empty and "round" in df.columns:
                    data[m] = df
            except Exception as e:
                print(f"[LỖI] Đọc file {p} → {e}")

    for met in metrics:
        plt.figure()
        for m, df in data.items():
            if met in df.columns:
                plt.plot(df["round"], df[met], label=labels[m])
        plt.xlabel("Round"); plt.ylabel(met.upper()); plt.title(f"Comparison ({met.upper()})")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"{out_prefix}_{met}.png"), dpi=180); plt.close()
