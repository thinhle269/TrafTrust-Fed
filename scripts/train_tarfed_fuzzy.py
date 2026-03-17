import os
# Tối ưu cho CPU 10C/20T
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["TF_NUM_INTRAOP_THREADS"] = "10"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # giảm log TF

import sys, pathlib, argparse
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.train_fed import train_federated

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", type=str, default="data/metr_la.npy")
    ap.add_argument("--coords_file", type=str, default="data/sensor_locations_la.csv")
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--input_steps", type=int, default=12)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--engine", choices=["tf","sklearn"], default="tf")
    ##
    ap.add_argument("--alpha", type=float, default=0.9, help="EMA smoothing factor")

    args = ap.parse_args()
    train_federated(args.data_file, args.coords_file, "results",
                    method="tarfed_fuzzy", K=args.clients, R=args.rounds, E=args.local_epochs, B=args.batch_size,
                    lr=args.lr, input_steps=args.input_steps, engine=args.engine,alpha=args.alpha) ##
