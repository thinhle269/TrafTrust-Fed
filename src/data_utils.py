import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def set_global_seed(seed=42):
    import random
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    random.seed(seed); np.random.seed(seed)

def load_metr_la_npy(npy_path):
    v = np.load(npy_path).astype("float32")  # expected [N, T]
    if v.ndim != 2:
        raise ValueError("metr_la.npy must be 2D [N, T]")
    vv = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(vv, 0.0), np.percentile(vv, 99.5)
    vv = np.clip(vv, lo, hi)
    vv = (vv - vv.min()) / (vv.max() - vv.min() + 1e-12)
    return vv

def load_coords(csv_path):
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path)
    # Chuẩn hoá tên cột về chữ thường, bỏ khoảng trắng
    colmap = {c.lower().strip(): c for c in df.columns}

    # Các biến thể tên cột có thể gặp
    lat_key = next((k for k in ["lat","latitude","sensor_lat","y","vĩ độ","vi do","lat_dd"] if k in colmap), None)
    lon_key = next((k for k in ["lon","lng","longitude","sensor_lon","x","kinh độ","kinh do","lon_dd","long"] if k in colmap), None)

    if lat_key is None and "latitude" in colmap:
        lat_key = "latitude"
    if lon_key is None and "longitude" in colmap:
        lon_key = "longitude"

    if lat_key is None or lon_key is None:
        raise KeyError(
            f"Cannot find lat/lon columns. Found columns: {list(df.columns)}. "
            "Expected something like 'latitude'/'longitude' or 'lat'/'lon'."
        )

    lat = pd.to_numeric(df[colmap[lat_key]], errors="coerce").to_numpy(dtype="float32")
    lon = pd.to_numeric(df[colmap[lon_key]], errors="coerce").to_numpy(dtype="float32")

    # Xử lý NaN đơn giản nếu có
    for arr in (lat, lon):
        if np.isnan(arr).any():
            idx = np.arange(len(arr))
            ok = ~np.isnan(arr)
            if ok.any():
                arr[~ok] = np.interp(idx[~ok], idx[ok], arr[ok])
            if np.isnan(arr).any():
                arr[np.isnan(arr)] = np.nanmedian(arr)

    return np.stack([lat, lon], axis=1).astype("float32")


def create_windows(series, n_in=12, n_out=1):
    T = len(series); N = T - n_in - n_out + 1
    if N <= 0:
        return np.zeros((0, n_in), dtype="float32"), np.zeros((0,), dtype="float32")
    X = np.lib.stride_tricks.sliding_window_view(series, n_in)[:N]
    y = series[n_in:n_in+N]
    return X.astype("float32"), y.astype("float32")

def partition_clients(data, coords, n_clients=10, input_steps=12, seed=42):
    kmeans = KMeans(n_clusters=n_clients, random_state=seed).fit(coords)
    labels = kmeans.labels_
    client_sets = [np.where(labels==k)[0] for k in range(n_clients)]
    client_windows, splits = [], []
    for sensors in client_sets:
        sensors = list(sensors) if len(sensors)>0 else [np.random.randint(0, data.shape[0])]
        X_list, y_list = [], []
        for s in sensors:
            Xs, ys = create_windows(data[s], n_in=input_steps, n_out=1)
            if len(Xs): X_list.append(Xs); y_list.append(ys)
        if not X_list:
            Xc = np.zeros((0, input_steps), dtype="float32"); yc = np.zeros((0,), dtype="float32")
        else:
            Xc = np.concatenate(X_list, axis=0); yc = np.concatenate(y_list, axis=0)
        n = len(Xc); n_tr=int(n*0.7); n_val=int(n*0.15); n_te=n-n_tr-n_val
        client_windows.append((Xc, yc)); splits.append((n_tr, n_val, n_te))
    return client_windows, splits

def speed_to_class(v):
    if v < 0.2: return 0
    if v < 0.4: return 1
    if v < 0.6: return 2
    if v < 0.8: return 3
    return 4
