"""
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

def compute_trust_fuzzy(local_weights, global_weights, val_mae_list, grad_norm_list):
  
    fis = SimpleMamdaniFIS()
    K = len(val_mae_list)

    has_weights = (
        (local_weights is not None) and (global_weights is not None) and
        (len(local_weights) == K) and (local_weights[0] is not None)
    )

    # 1) Validation performance: higher is better
    val_mae = np.array(val_mae_list, dtype=float)
    val_perf = 1.0 - _norm01(val_mae)   # normalize rồi đảo chiều

    # 2) Drift / Align / Grad-norm
    if has_weights:
        # drift: ||Δw|| / ||w||
        drift_vals = []
        deltas = []
        g_flat = np.concatenate([w.ravel() for w in global_weights], axis=0)
        g_norm = np.linalg.norm(g_flat) + 1e-12

        for k in range(K):
            d_k = [local_weights[k][i] - global_weights[i] for i in range(len(global_weights))]
            d_k_flat = np.concatenate([x.ravel() for x in d_k], axis=0)
            drift_vals.append(np.linalg.norm(d_k_flat) / g_norm)
            deltas.append(d_k_flat)

        delta_stack = np.stack(deltas, axis=0)  # (K, P)
        mean_delta = delta_stack.mean(axis=0)
        align_vals = [ _cosine_distance(delta_stack[k], mean_delta) for k in range(K) ]

        gnorm_vals = np.array(grad_norm_list if grad_norm_list is not None else [0.0]*K, dtype=float)
    else:
        # Không có weights → các thành phần này = 0, nhưng val_perf vẫn tạo trust động
        drift_vals = np.zeros(K, dtype=float)
        align_vals = np.zeros(K, dtype=float)
        gnorm_vals = np.zeros(K, dtype=float)

    # Chuẩn hóa về 0..1
    drift_n = _norm01(drift_vals)
    align_n = _norm01(align_vals)
    gnorm_n = _norm01(gnorm_vals)

    # 3) FIS ra trust_k
    trust = [fis(val_perf[k], drift_n[k], align_n[k], gnorm_n[k]) for k in range(K)]
    trust = np.clip(np.array(trust, dtype=float), 0.02, 1.0)
    trust = trust / (trust.sum() + 1e-12)
    return trust.tolist()
"""
import numpy as np
from .fuzzy.fis import SimpleMamdaniFIS

def _norm01(x):
    """Chuẩn hóa mảng về [0, 1] dùng percentile 5-95."""
    x = np.asarray(x, dtype=float)
    if x.size == 0: return x
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)

def _cosine_distance(u, v):
    """Tính 1 - cosine similarity. (Hàm này không còn được sử dụng
    trong phiên bản 2-đầu-vào, nhưng giữ lại để tránh lỗi import)."""
    u = u.ravel(); v = v.ravel()
    nu = np.linalg.norm(u)+1e-12; nv = np.linalg.norm(v)+1e-12
    cos = np.dot(u, v)/(nu*nv)
    return 1.0 - np.clip(cos, -1.0, 1.0)

def compute_trust_fuzzy(local_weights, global_weights, val_mae_list, grad_norm_list=None):
    """
    Trả về: list trust per-client (chuẩn hóa tổng=1).

    PHIÊN BẢN ĐÃ SỬA (CHO KHỚP VỚI PDF):
    Chỉ sử dụng 2 đầu vào cho FIS:
    1. val_perf (từ val_mae_list)
    2. drift_n (từ local_weights và global_weights)
    
    Các tham số 'align' và 'gnorm' đã bị loại bỏ khỏi logic.
    """
    
    # Khởi tạo FIS (sẽ gọi file fis.py 2-đầu-vào mà chúng ta đã sửa)
    fis = SimpleMamdaniFIS()
    K = len(val_mae_list)

    has_weights = (
        (local_weights is not None) and (global_weights is not None) and
        (len(local_weights) == K) and (local_weights[0] is not None)
    )

    # 1) Input 1: Validation performance (val_perf)
    # (Đảo ngược MAE: 1.0 = Tốt/Lỗi thấp, 0.0 = Tệ/Lỗi cao)
    val_mae = np.array(val_mae_list, dtype=float)
    val_perf = 1.0 - _norm01(val_mae) 

    # 2) Input 2: Gradient Drift (drift_n)
    if has_weights:
        # drift: ||Δw_k|| / ||w_global||
        drift_vals = []
        g_flat = np.concatenate([w.ravel() for w in global_weights], axis=0)
        g_norm = np.linalg.norm(g_flat) + 1e-12

        for k in range(K):
            d_k = [local_weights[k][i] - global_weights[i] for i in range(len(global_weights))]
            d_k_flat = np.concatenate([x.ravel() for x in d_k], axis=0)
            drift_vals.append(np.linalg.norm(d_k_flat) / g_norm)
        
        # [ĐÃ XÓA] Tính toán 'align_vals' và 'gnorm_vals' không cần thiết
    
    else:
        # Nếu không có weights, drift = 0 (stable)
        drift_vals = np.zeros(K, dtype=float)
        
        # [ĐÃ XÓA] Tính toán 'align_vals' và 'gnorm_vals' không cần thiết

    # Chuẩn hóa về 0..1
    # (0.0 = Stable, 1.0 = Unstable)
    drift_n = _norm01(drift_vals)
    
    # [ĐÃ XÓA] Chuẩn hóa 'align_n' và 'gnorm_n'

    # 3) FIS ra trust_k (SỬA TỪ 4-ĐẦU-VÀO XUỐNG CÒN 2-ĐẦU-VÀO)
    trust = [fis(val_perf[k], drift_n[k]) for k in range(K)]
    
    # Chuẩn hóa trust cuối cùng
    trust = np.clip(np.array(trust, dtype=float), 0.02, 1.0) # Đặt mức sàn tối thiểu
    trust = trust / (trust.sum() + 1e-12)
    
    return trust.tolist()