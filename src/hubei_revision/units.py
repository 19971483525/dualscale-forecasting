"""唯一物理单位转换入口；模型空间保留原始数值 /1000，避免旧缩放叠加。"""
import numpy as np
RAW_TO_ENGINEERING = 0.01
MODEL_TO_ENGINEERING = 10.0
def power_raw_to_gw(x): return np.asarray(x) * RAW_TO_ENGINEERING
def energy_raw_to_gwh(x): return np.asarray(x) * RAW_TO_ENGINEERING
def energy_model_to_gwh(x): return np.asarray(x) * MODEL_TO_ENGINEERING
def power_model_to_gw(x): return np.asarray(x) * MODEL_TO_ENGINEERING
def raw_to_model(x): return np.asarray(x) / 1000.0
def integrate_raw_power(x): return np.asarray(x).sum(axis=-1) * 0.25
def metrics_gwh(y, p, space):
    if space not in ("model", "GWh"): raise ValueError("必须明确指定 model 或 GWh")
    y,p=np.asarray(y,dtype=np.float64),np.asarray(p,dtype=np.float64)
    if space=="model": y,p=energy_model_to_gwh(y),energy_model_to_gwh(p)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all(): raise ValueError("非有限或不等形预测")
    if np.any(np.abs(y)<1e-9): raise ValueError("日总电量接近零，MAPE无定义")
    return {"mae_gwh":float(np.mean(np.abs(y-p))),"rmse_gwh":float(np.sqrt(np.mean((y-p)**2))),"mape_pct":float(np.mean(np.abs((y-p)/y))*100),"n":int(y.size),"absolute_error_unit":"GWh"}
