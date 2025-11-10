
#import numpy as np
#from .memberships import trimf
"""
class SimpleMamdaniFIS:
    
    def __init__(self):
        # Triangular sets over [0,1]
        self.low  = lambda x: trimf(x, 0.0, 0.0, 0.5)
        self.med  = lambda x: trimf(x, 0.25,0.5,0.75)
        self.high = lambda x: trimf(x, 0.5, 1.0, 1.0)

    def __call__(self, val_perf, drift, align, gnorm):
        # fuzzify
        vp_h = self.high(val_perf); vp_m = self.med(val_perf); vp_l = self.low(val_perf)
        dr_h = self.high(drift);    dr_m = self.med(drift);    dr_l = self.low(drift)
        al_h = self.high(align);    al_m = self.med(align);    al_l = self.low(align)
        gn_h = self.high(gnorm);    gn_m = self.med(gnorm);    gn_l = self.low(gnorm)
        # rules
        r1 = min(vp_h, dr_l, al_l)
        r2 = max(dr_h, al_h, gn_h)
        r3 = max(min(vp_m, dr_m), min(vp_h, al_m))  # a soft "otherwise"
        # aggregate output sets as piecewise levels: Low(0.2), Med(0.5), High(0.85)
        # centroid defuzzification on discrete samples
        xs = np.linspace(0,1,101)
        mu_low  = r2 * np.maximum(1 - (xs/0.2), 0)  # spike around 0.1~0.2
        mu_med  = r3 * np.maximum(1 - np.abs(xs-0.5)/0.25, 0)
        mu_high = r1 * np.maximum(1 - (1-xs)/0.15, 0)  # spike near 0.85~1.0
        mu = np.maximum.reduce([mu_low, mu_med, mu_high])
        num = (xs*mu).sum(); den = mu.sum()+1e-12
        return float(num/den)
"""
import numpy as np
from .memberships import trimf

class SimpleMamdaniFIS:
    """
    Hệ thống Mamdani FIS 2-đầu-vào, cài đặt lại cho đúng với
    phương pháp luận trong bản thảo (PDF Mục 3.3, Bảng 1).

    Inputs:
    1. val_perf [0,1]: Hiệu năng validation (Đã chuẩn hóa, 1 = Tốt nhất/Lỗi thấp, 0 = Tệ nhất/Lỗi cao)
    2. drift [0,1]:     Gradient drift (Đã chuẩn hóa, 0 = Ổn định/Stable, 1 = Không ổn định/Unstable)

    Output:
    - trust [0,1]:     Độ tin cậy (Defuzzified)
    """
    def __init__(self):
        # 1. Định nghĩa các hàm thành viên (Membership Functions) cho 2 INPUTS
        
        # Input 1: Validation Performance (vp)
        # (PDF 'e Low' = 'vp High' | 'e Med' = 'vp Med' | 'e High' = 'vp Low')
        self.vp_low = lambda x: trimf(x, 0.0, 0.0, 0.5)    # Tương đương 'e High'
        self.vp_med = lambda x: trimf(x, 0.25, 0.5, 0.75)  # Tương đương 'e Medium'
        self.vp_high = lambda x: trimf(x, 0.5, 1.0, 1.0)   # Tương đương 'e Low'

        # Input 2: Gradient Drift (g)
        # (PDF 'g Stable' = 'g Stable' | 'g Mod' = 'g Mod' | 'g Unstable' = 'g Unstable')
        self.g_stable = lambda x: trimf(x, 0.0, 0.0, 0.5)
        self.g_mod = lambda x: trimf(x, 0.25, 0.5, 0.75)
        self.g_unstable = lambda x: trimf(x, 0.5, 1.0, 1.0)

        # 2. Định nghĩa các hàm thành viên cho OUTPUT (Trust)
        # (Chúng ta sẽ cắt (clip) các hình dạng này)
        self.defuzz_xs = np.linspace(0, 1, 101)
        self.c_low_shape = trimf(self.defuzz_xs, 0.0, 0.0, 0.5)
        self.c_med_shape = trimf(self.defuzz_xs, 0.25, 0.5, 0.75)
        self.c_high_shape = trimf(self.defuzz_xs, 0.5, 1.0, 1.0)

    def __call__(self, val_perf, drift):
        # Bước 1: Fuzzify 2 giá trị đầu vào
        vp_l = self.vp_low(val_perf)
        vp_m = self.vp_med(val_perf)
        vp_h = self.vp_high(val_perf)

        g_s = self.g_stable(drift)
        g_m = self.g_mod(drift)
        g_u = self.g_unstable(drift)

        # Bước 2: Áp dụng 9 LUẬT (R1-R9) từ Bảng 1 (dùng 'min' cho toán tử 'AND')
        # Logic được dịch: 'e Low' = vp_h, 'e High' = vp_l, 'g Stable' = g_s, v.v.

        # Trust High Rules
        r1 = min(vp_h, g_s)  # R1: IF (e Low) AND (g Stable) THEN C High
        r2 = min(vp_h, g_m)  # R2: IF (e Low) AND (g Moderate) THEN C High
        r4 = min(vp_m, g_s)  # R4: IF (e Medium) AND (g Stable) THEN C High
        
        # Trust Medium Rules
        r3 = min(vp_h, g_u)  # R3: IF (e Low) AND (g Unstable) THEN C Medium
        r5 = min(vp_m, g_m)  # R5: IF (e Medium) AND (g Moderate) THEN C Medium
        r7 = min(vp_l, g_s)  # R7: IF (e High) AND (g Stable) THEN C Medium
        
        # Trust Low Rules
        r6 = min(vp_m, g_u)  # R6: IF (e Medium) AND (g Unstable) THEN C Low
        r8 = min(vp_l, g_m)  # R8: IF (e High) AND (g Moderate) THEN C Low
        r9 = min(vp_l, g_u)  # R9: IF (e High) AND (g Unstable) THEN C Low
        
        # Bước 3: Tổng hợp các luật (dùng 'max' cho toán tử 'OR' giữa các luật)
        act_high = max(r1, r2, r4)
        act_med = max(r3, r5, r7)
        act_low = max(r6, r8, r9)

        # Bước 4: Defuzzify (Centroid)
        # Cắt (clip) các hình dạng output tại mức kích hoạt (activation)
        mu_high = np.fmin(act_high, self.c_high_shape)
        mu_med = np.fmin(act_med, self.c_med_shape)
        mu_low = np.fmin(act_low, self.c_low_shape)

        # Kết hợp các hình dạng đã cắt
        mu_combined = np.fmax(mu_high, np.fmax(mu_med, mu_low))
        
        # Tính toán centroid
        num = (self.defuzz_xs * mu_combined).sum()
        den = mu_combined.sum() + 1e-12

        # Trả về giá trị trust cuối cùng
        if den == 0:
            return 0.5  # Trường hợp dự phòng nếu không có luật nào kích hoạt
        
        return float(num / den)