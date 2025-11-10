# File: trust.py
# PHIÊN BẢN VÁ LỖI (HOTFIX): Giữ Class EMA MỚI + Thêm lại hàm CŨ (dummy)

import numpy as np
from .fuzzy.fis import SimpleMamdaniFIS

def _norm01(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0: return x
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)

def _cosine_distance(u, v):
    u = u.ravel(); v = v.ravel()
    nu = np.linalg.norm(u)+1e-12; nv = np.linalg.norm(v)+1e-12
    cos = np.dot(u, v)/(nu*nv)
    return 1.0 - np.clip(cos, -1.0, 1.0)


# === PHẦN 1: LOGIC MỚI CHO TrafTrust-Fed ===
# (Class này được dùng bởi train_tarfed_fuzzy.py)

class FuzzyEMATrust:
    """
    Class này đóng gói logic Fuzzy-EMA (4-input, smoothed).
    """
    def __init__(self, num_clients, alpha=0.7):
        """
        Khởi tạo bộ lọc.
        """
        self.K = num_clients
        self.alpha = alpha
        self.fis = SimpleMamdaniFIS()
        
        # Khởi tạo C(t-1)
        self.C_previous = np.ones(self.K) / self.K
        print(f"Khởi tạo FuzzyEMATrust (K={self.K}, alpha={self.alpha})")

    def __call__(self, local_weights, global_weights, val_mae_list, grad_norm_list):
        """
        Đây là hàm được gọi mỗi round bởi train_tarfed_fuzzy.py
        """
        
        # --- 1. Tính toán 4 chỉ số ---
        has_weights = (
            (local_weights is not None) and (global_weights is not None) and
            (len(local_weights) == self.K) and (local_weights[0] is not None)
        )

        val_mae = np.array(val_mae_list, dtype=float)
        val_perf = 1.0 - _norm01(val_mae)  # 1=Tốt

        if has_weights:
            drift_vals = []
            deltas = []
            g_flat = np.concatenate([w.ravel() for w in global_weights], axis=0)
            g_norm = np.linalg.norm(g_flat) + 1e-12

            for k in range(self.K):
                d_k = [local_weights[k][i] - global_weights[i] for i in range(len(global_weights))]
                d_k_flat = np.concatenate([x.ravel() for x in d_k], axis=0)
                drift_vals.append(np.linalg.norm(d_k_flat) / g_norm)
                deltas.append(d_k_flat)

            delta_stack = np.stack(deltas, axis=0)
            mean_delta = delta_stack.mean(axis=0)
            align_vals = [ _cosine_distance(delta_stack[k], mean_delta) for k in range(self.K) ]
            gnorm_vals = np.array(grad_norm_list if grad_norm_list is not None else [0.0]*self.K, dtype=float)
        else:
            drift_vals = np.zeros(self.K, dtype=float)
            align_vals = np.zeros(self.K, dtype=float)
            gnorm_vals = np.zeros(self.K, dtype=float)

        drift_n = _norm01(drift_vals) # 1=Tệ
        align_n = _norm01(align_vals) # 1=Tệ
        gnorm_n = _norm01(gnorm_vals) # 1=Tệ

        # --- 2. Tính toán Điểm thô (Raw Score) ---
        C_raw = np.array([
            self.fis(val_perf[k], drift_n[k], align_n[k], gnorm_n[k]) 
            for k in range(self.K)
        ], dtype=float)
        
        C_raw = np.clip(C_raw, 0.02, 1.0)
        
        # --- 3. Áp dụng Bộ lọc EMA (Logic MỚI) ---
        C_smoothed = (self.alpha * self.C_previous) + ((1 - self.alpha) * C_raw)
        
        # --- 4. Chuẩn hóa và Trả về ---
        C_normalized = C_smoothed / (C_smoothed.sum() + 1e-12)
        
        # Lưu lại trạng thái cho round sau
        self.C_previous = C_normalized
        
        return C_normalized.tolist()


# === PHẦN 2: BẢN VÁ (HOTFIX) CHO FedAvg/FedProx/FedTrim ===
# (Các file CŨ import hàm này nhưng không bao giờ gọi nó)

def compute_trust_fuzzy(local_weights, global_weights, val_mae_list, grad_norm_list):
    """
    HÀM GIẢ (DUMMY FUNCTION) - Chỉ để tương thích ngược (backward compatibility).
    FedAvg/FedProx/FedTrim import hàm này (do lỗi import của train_fed.py) 
    nhưng sẽ KHÔNG BAO GIỜ GỌI (CALL) nó.
    """
    print("CRITICAL WARNING: compute_trust_fuzzy (hàm CŨ) đã được gọi. Lỗi logic!")
    # Trả về một phân bổ đều (giống như FedAvg)
    K = len(val_mae_list)
    trust = np.ones(K) / K
    return trust.tolist()