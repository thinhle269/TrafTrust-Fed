#import sys, pathlib, argparse
#sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
"""
import os, argparse, pandas as pd
from src.plot_utils import plot_curves, plot_confusion_matrix, plot_trust_heatmap, save_ck_wide, plot_trust_dynamics, plot_compare_four

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results")
    args = ap.parse_args()
    res = args.results_dir; os.makedirs(res, exist_ok=True)

    # Curves từng phương pháp
    for m in ["fedavg","fedprox","fedtrim","tarfed_fuzzy"]:
        csv_path = os.path.join(res, f"logs_{m}.csv")
        if os.path.exists(csv_path):
            plot_curves(csv_path, os.path.join(res, f"plot_{m}_curves.png"))

    # Confusion matrices
    for m in ["fedavg","fedprox","fedtrim","tarfed_fuzzy"]:
        cm_csv = os.path.join(res, f"confusion_matrix_{m}.csv")
        if os.path.exists(cm_csv):
            plot_confusion_matrix(cm_csv, os.path.join(res, f"plot_{m}_cm.png"))

    # Trust (Ck) cho TAR-Fed-Fuzzy
    tcsv = os.path.join(res, "trust_tarfed_fuzzy.csv")
    if os.path.exists(tcsv):
        save_ck_wide(tcsv, os.path.join(res, "ck_values_tarfed_fuzzy.csv"))
        plot_trust_dynamics(tcsv,
            os.path.join(res, "plot_tarfed_fuzzy_trust_dynamics.png"),
            os.path.join(res, "plot_tarfed_fuzzy_trust_heatmap.png")
        )

    # So sánh last round + overlay 4 phương pháp
    rows = []
    for m in ["fedavg","fedprox","fedtrim","tarfed_fuzzy"]:
        csv_path = os.path.join(res, f"logs_{m}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            rows.append({"method": m, **df.iloc[-1].to_dict()})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(res, "comparison_last_round.csv"), index=False)
        plot_compare_four(res, out_prefix="compare_four")
"""
import sys
import pathlib
import os
import argparse
import pandas as pd

# Thêm thư mục src vào sys.path để import các hàm vẽ
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.plot_utils import (
    plot_curves,
    plot_confusion_matrix,
    plot_trust_heatmap,
    save_ck_wide,
    plot_trust_dynamics,
    plot_compare_four
)

def safe_read_csv(path):
    """Đọc CSV an toàn, trả về None nếu file rỗng hoặc lỗi."""
    try:
        if os.path.getsize(path) == 0:
            print(f"[BỎ QUA] File rỗng: {path}")
            return None
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"[LỖI] Không thể đọc file CSV: {path}")
        return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results")
    args = ap.parse_args()

    res = args.results_dir
    os.makedirs(res, exist_ok=True)

    methods = ["fedavg", "fedprox", "fedtrim", "tarfed_fuzzy"]

    # Vẽ curves từng phương pháp
    for m in methods:
        csv_path = os.path.join(res, f"logs_{m}.csv")
        if os.path.exists(csv_path):
            df = safe_read_csv(csv_path)
            if df is not None:
                plot_curves(csv_path, os.path.join(res, f"plot_{m}_curves.png"))

    # Vẽ confusion matrix
    for m in methods:
        cm_csv = os.path.join(res, f"confusion_matrix_{m}.csv")
        if os.path.exists(cm_csv):
            df = safe_read_csv(cm_csv)
            if df is not None:
                plot_confusion_matrix(cm_csv, os.path.join(res, f"plot_{m}_cm.png"))

    # Vẽ trust dynamics cho TAR-Fed-Fuzzy
    trust_csv = os.path.join(res, "trust_tarfed_fuzzy.csv")
    if os.path.exists(trust_csv):
        df = safe_read_csv(trust_csv)
        if df is not None:
            save_ck_wide(trust_csv, os.path.join(res, "ck_values_tarfed_fuzzy.csv"))
            plot_trust_dynamics(
                trust_csv,
                os.path.join(res, "plot_tarfed_fuzzy_trust_dynamics.png"),
                os.path.join(res, "plot_tarfed_fuzzy_trust_heatmap.png")
            )

    # So sánh kết quả vòng cuối
    rows = []
    for m in methods:
        csv_path = os.path.join(res, f"logs_{m}.csv")
        if os.path.exists(csv_path):
            df = safe_read_csv(csv_path)
            if df is not None and not df.empty:
                rows.append({"method": m, **df.iloc[-1].to_dict()})

    if rows:
        compare_df = pd.DataFrame(rows)
        compare_df.to_csv(os.path.join(res, "comparison_last_round.csv"), index=False)
        plot_compare_four(res, out_prefix="compare_four")
    else:
        print("[THÔNG BÁO] Không có dữ liệu để so sánh.")
