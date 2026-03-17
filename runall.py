import os, sys, subprocess, argparse
## phan nay la 0.1 co chinh code lai 
def run(cmd):
    print(">>", " ".join(cmd)); subprocess.check_call(cmd)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["tf","sklearn"], default="tf")
    ap.add_argument("--data_file", type=str, default="data/metr_la.npy")
    ap.add_argument("--coords_file", type=str, default="data/sensor_locations_la.csv")
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--input_steps", type=int, default=12)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mu_prox", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.9, help="EMA smoothing factor")
    args = ap.parse_args()

     run([sys.executable, "scripts/train_fedavg.py",
          "--data_file", args.data_file, "--coords_file", args.coords_file,
          "--rounds", str(args.rounds), "--clients", str(args.clients),
          "--input_steps", str(args.input_steps), "--local_epochs", str(args.local_epochs),
          "--batch_size", str(args.batch_size), "--lr", str(args.lr),
          "--engine", args.engine])

     run([sys.executable, "scripts/train_fedprox.py",
          "--data_file", args.data_file, "--coords_file", args.coords_file,
          "--rounds", str(args.rounds), "--clients", str(args.clients),
          "--input_steps", str(args.input_steps), "--local_epochs", str(args.local_epochs),
          "--batch_size", str(args.batch_size), "--lr", str(args.lr),
          "--mu_prox", str(args.mu_prox), "--engine", args.engine])
     run([sys.executable, "scripts/train_fedtrim.py",
          "--data_file", args.data_file, "--coords_file", args.coords_file,
          "--rounds", str(args.rounds), "--clients", str(args.clients),
          "--input_steps", str(args.input_steps), "--local_epochs", str(args.local_epochs),
          "--batch_size", str(args.batch_size), "--lr", str(args.lr),
          "--engine", args.engine])



    run([sys.executable, "scripts/train_tarfed_fuzzy.py",
         "--data_file", args.data_file, "--coords_file", args.coords_file,
         "--rounds", str(args.rounds), "--clients", str(args.clients),
         "--input_steps", str(args.input_steps), "--local_epochs", str(args.local_epochs),
         "--batch_size", str(args.batch_size), "--lr", str(args.lr),
         "--engine", args.engine,
         "--alpha", str(args.alpha)])

    run([sys.executable, "scripts/plot_all.py"])
    print("All done. Check the 'results/' folder.")
