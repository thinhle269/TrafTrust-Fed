# File: .fuzzy/fis.py
import numpy as np
from .memberships import trimf

class SimpleMamdaniFIS:
    """
    PHIÊN BẢN SỬA LỖI (4-INPUT, logic "AND")
    Hệ thống Mamdani FIS 4-đầu-vào, được thiết kế lại để
    cân bằng giữa Hiệu năng (F1/Recall) và Ổn định (Sigma).

    Inputs in [0,1]:
    1. val_perf (Hiệu năng, 1=Tốt)
    2. drift (Drift, 1=Tệ)
    3. align (Align, 1=Tệ - xa trung bình)
    4. gnorm (Norm, 1=Tệ - bùng nổ gradient)

    Outputs in [0,1]: trust
    """
    def __init__(self):
        # 1. Định nghĩa các hàm thành viên (MFs)
        self.low = lambda x: trimf(x, 0.0, 0.0, 0.5)
        self.med = lambda x: trimf(x, 0.25, 0.5, 0.75)
        self.high = lambda x: trimf(x, 0.5, 1.0, 1.0)
        
        # 2. Định nghĩa các hàm thành viên cho OUTPUT (Trust)
        self.defuzz_xs = np.linspace(0, 1, 101)
        self.c_low_shape = trimf(self.defuzz_xs, 0.0, 0.0, 0.5)
        self.c_med_shape = trimf(self.defuzz_xs, 0.25, 0.5, 0.75)
        self.c_high_shape = trimf(self.defuzz_xs, 0.5, 1.0, 1.0)

    def __call__(self, val_perf, drift, align, gnorm):
        # Bước 1: Fuzzify 4 giá trị đầu vào
        vp_h = self.high(val_perf); vp_m = self.med(val_perf); vp_l = self.low(val_perf)
        dr_h = self.high(drift);    dr_m = self.med(drift);    dr_l = self.low(drift)
        al_h = self.high(align);    al_m = self.med(align);    al_l = self.low(align)
        gn_h = self.high(gnorm);    gn_m = self.med(gnorm);    gn_l = self.low(gnorm)

        # Bước 2: Áp dụng BỘ LUẬT MỚI (logic "AND")
        
        # --- Trust High Rules ---
        r1_high = min(vp_h, dr_l, al_l, gn_l)

        # --- Trust Low Rules ---
        r2_low = vp_l  # Hiệu năng validation quá thấp
        r3_low = dr_h  # Drift quá cao (cập nhật khác biệt)
        r4_low = al_h  # Align quá tệ (cập nhật đi ngược xu thế)
        r5_low = gn_h  # Gnorm quá cao (bùng nổ gradient)
        
        act_low = max(r2_low, r3_low, r4_low, r5_low) # Dùng 'max' (OR) ở đây là ĐÚNG

        # --- Trust Medium Rules ---
        r6_med = min(vp_m, dr_l, al_l) # Hiệu năng trung bình, nhưng ổn định
        r7_med = min(vp_h, dr_m, al_m) # Hiệu năng cao, nhưng hơi nhiễu
        r8_med = vp_m # Quy tắc chung cho hiệu năng trung bình
        
        act_med = max(r6_med, r7_med, r8_med)
        act_high = r1_high

        # Bước 3: Defuzzify (Centroid)
        if act_low > act_med:
           act_med = 0.0
        if act_low > act_high:
           act_high = 0.0

        mu_high = np.fmin(act_high, self.c_high_shape)
        mu_med = np.fmin(act_med, self.c_med_shape)
        mu_low = np.fmin(act_low, self.c_low_shape)

        mu_combined = np.fmax(mu_high, np.fmax(mu_med, mu_low))
        
        num = (self.defuzz_xs * mu_combined).sum()
        den = mu_combined.sum() + 1e-12

        if den == 0:
            return 0.5  # Dự phòng
        
        return float(num / den)