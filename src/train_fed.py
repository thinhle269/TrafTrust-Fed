import os, numpy as np, pandas as pd
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
from sklearn.neural_network import MLPRegressor

from .data_utils import set_global_seed, load_metr_la_npy, load_coords, partition_clients
from .models import build_mlp
from .metrics import regression_metrics, classification_metrics
from .agg_fedavg import aggregate_fedavg as agg_fedavg
from .agg_fedprox import aggregate_fedavg as agg_fedavg_fp, prox_update
from .agg_fedtrim import aggregate_fedtrim

# [SỬA ĐỔI 1]: Import class FuzzyEMATrust chứa logic EMA thực sự thay vì hàm dummy
from .trust import FuzzyEMATrust  

from .plot_utils import plot_curves, plot_confusion_matrix, plot_trust_dynamics

def evaluate_global(model_or_models, client_windows, splits, split="test", engine="tf"):
    y_true_all, y_pred_all = [], []
    for (Xc, yc), (n_tr, n_val, n_te) in zip(client_windows, splits):
        lo, hi = (n_tr, n_tr+n_val) if split=="val" else (n_tr+n_val, n_tr+n_val+n_te)
        if hi-lo <= 0: continue
        Xs, ys = Xc[lo:hi], yc[lo:hi]
        if engine=="tf":
            y_hat = model_or_models.predict(Xs, verbose=0).flatten()
        else:
            preds = [m.predict(Xs).reshape(-1) for m in model_or_models]
            y_hat = np.mean(preds, axis=0)
        y_true_all.append(ys); y_pred_all.append(y_hat)
    if not y_true_all:
        return (np.nan,)*5 + (None,)
    y_true = np.concatenate(y_true_all); y_pred = np.concatenate(y_pred_all)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    acc, macro_f1, recall, cm = classification_metrics(y_true, y_pred)
    return mae, rmse, acc, macro_f1, recall, cm

# [SỬA ĐỔI 2]: Thêm tham số beta=0.7 vào hàm train_federated
def train_federated(
    data_file, coords_file, results_dir,
    method="fedavg", K=20, R=150, E=2, B=128, lr=1e-3, seed=42, input_steps=12,
    mu=0.02, engine="tf", trim_rate=0.1, 
    alpha=0.9 
):
    os.makedirs(results_dir, exist_ok=True)
    set_global_seed(seed)

    data = load_metr_la_npy(data_file)
    coords = load_coords(coords_file)
    if data.shape[0] != coords.shape[0]:
        raise ValueError(f"sensors mismatch: metr_la.npy has {data.shape[0]}, coords has {coords.shape[0]}")

    client_windows, splits = partition_clients(data, coords, n_clients=K, input_steps=input_steps, seed=seed)
    loss_name = "huber" if method!="fedavg" else "mae"
    use_sklearn = (engine=="sklearn") or (not TF_AVAILABLE)

    if not use_sklearn:
        global_model = build_mlp(input_dim=input_steps, lr=lr, loss_name=loss_name)
        global_weights = global_model.get_weights()
    else:
        global_model = None; global_weights = None

    log_rows, trust_rows = [], []

    # [SỬA ĐỔI 3]: Khởi tạo Object Trust Evaluator ở NGOÀI vòng lặp để nó CÓ TRÍ NHỚ (EMA)
    trust_evaluator = None
    if method == "tarfed_fuzzy" and not use_sklearn:
        # Nếu trong file trust.py bạn để biến khởi tạo là `alpha` thì dùng alpha=beta, 
        # nếu bạn đã đổi tên thành `beta` trong trust.py thì dùng beta=beta
        trust_evaluator = FuzzyEMATrust(num_clients=K, alpha= 0.9) 
      ##  print(f"[*] Đã khởi tạo hệ thống Fuzzy-EMA với Beta = {beta}")

    for r in range(1, R+1):
        print(f"[{method}] Round {r}/{R} ...")

        local_weights, sample_counts = [], []
        val_mae_list, grad_norm_list = [], []
        sklearn_models = [] if use_sklearn else None

        for k, ((Xc,yc), (n_tr,n_val,n_te)) in enumerate(zip(client_windows, splits)):
            if n_tr<=0:
                local_weights.append(None if use_sklearn else global_weights)
                sample_counts.append(1); val_mae_list.append(1.0); grad_norm_list.append(0.0)
                if use_sklearn:
                    sklearn_models.append(MLPRegressor(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=1, warm_start=True, random_state=seed))
                continue

            if use_sklearn:
                model = MLPRegressor(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=1, warm_start=True, random_state=seed)
            else:
                model = build_mlp(input_dim=input_steps, lr=lr, loss_name=loss_name)
                model.set_weights(global_weights)

            Xtr, ytr = Xc[:n_tr], yc[:n_tr]
            Xva, yva = (Xc[n_tr:n_tr+n_val], yc[n_tr:n_tr+n_val]) if n_val>0 else (None, None)

            if method=="fedprox" and (not use_sklearn):
                prox_update(model, global_weights, mu=mu)

            if use_sklearn:
                model.fit(Xtr, ytr)
            else:
                model.fit(Xtr, ytr, epochs=E, batch_size=B, verbose=0)

            if use_sklearn:
                sklearn_models.append(model)
                local_weights.append(None)
                flat_norm = 0.0
            else:
                lw = model.get_weights()
                local_weights.append(lw)
                flat = np.concatenate([ (lw_i-gw_i).ravel() for lw_i,gw_i in zip(lw, global_weights) ], axis=0)
                flat_norm = float(np.linalg.norm(flat))

            sample_counts.append(len(Xtr))

            if Xva is not None and len(Xva)>0:
                yhat_val = (model.predict(Xva).reshape(-1) if use_sklearn else model.predict(Xva, verbose=0).flatten())
                val_mae = float(np.mean(np.abs(yva - yhat_val)))
            else:
                val_mae = 1.0
            val_mae_list.append(val_mae)
            grad_norm_list.append(flat_norm)

        # [SỬA ĐỔI 4]: Gọi instance trust_evaluator vừa khởi tạo bên trên
        this_round_trust = None
        if (method == "tarfed_fuzzy") and (not use_sklearn):
            this_round_trust = trust_evaluator(local_weights, global_weights, val_mae_list, grad_norm_list)
            for k, t in enumerate(this_round_trust):
                trust_rows.append({"round": r, "client": k, "trust": t})

        if use_sklearn:
            new_weights = global_weights
        else:
            if method == "fedavg":
                new_weights = agg_fedavg(local_weights, sample_counts)
            elif method == "fedprox":
                new_weights = agg_fedavg_fp(local_weights, sample_counts)
            elif method == "fedtrim":
                new_weights = aggregate_fedtrim(local_weights, sample_counts, trim=trim_rate)
            elif method == "tarfed_fuzzy":
                weights = np.array(this_round_trust, dtype=float)
                nks = np.array(sample_counts, dtype=float); nks = np.maximum(nks, 1.0)
                w = (weights * nks); w = w/(w.sum()+1e-12)
                new_weights = []
                for i in range(len(global_weights)):
                    stacked = np.stack([lw[i] for lw in local_weights], axis=0)
                    w_exp = w.reshape((-1,) + (1,)*(stacked.ndim-1))
                    new_weights.append(np.sum(w_exp * stacked, axis=0))
            else:
                raise ValueError(f"Unknown method: {method}")

        if not use_sklearn:
            global_weights = new_weights
            if global_model is None:
                global_model = build_mlp(input_dim=input_steps, lr=lr, loss_name=loss_name)
            global_model.set_weights(global_weights)

        model_for_eval = sklearn_models if use_sklearn else global_model
        mae, rmse, acc, macro_f1, recall, cm = evaluate_global(model_for_eval, client_windows, splits, split="test", engine=("sklearn" if use_sklearn else "tf"))
        log_rows.append({"round": r, "mae": mae, "rmse": rmse, "acc": acc, "macro_f1": macro_f1, "recall": recall})

        if r == R and cm is not None:
            np.savetxt(os.path.join(results_dir, f"confusion_matrix_{method}.csv"), cm, fmt="%d", delimiter=",")

        pd.DataFrame(log_rows).to_csv(os.path.join(results_dir, f"logs_{method}.csv"), index=False)
        if trust_rows:
            pd.DataFrame(trust_rows).to_csv(os.path.join(results_dir, f"trust_{method}.csv"), index=False)

    # [SỬA ĐỔI 5]: Dọn dẹp phần vẽ biểu đồ bị lặp code
    log_path = os.path.join(results_dir, f"logs_{method}.csv")
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        plot_curves(log_path, os.path.join(results_dir, f"plot_{method}_curves.png"))

    cm_path = os.path.join(results_dir, f"confusion_matrix_{method}.csv")
    if os.path.exists(cm_path) and os.path.getsize(cm_path) > 0:
        plot_confusion_matrix(cm_path, os.path.join(results_dir, f"plot_{method}_cm.png"))

    trust_path = os.path.join(results_dir, f"trust_{method}.csv")
    if os.path.exists(trust_path) and os.path.getsize(trust_path) > 0:
        plot_trust_dynamics(
            trust_path,
            os.path.join(results_dir, f"plot_{method}_trust_dynamics.png"),
            os.path.join(results_dir, f"plot_{method}_trust_heatmap.png")
        )