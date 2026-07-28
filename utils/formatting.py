from __future__ import annotations
import math

def fmt(value, digits=1, suffix=""):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        return f"{value:,.{digits}f}{suffix}"
    except Exception:
        return "—"

def status_from_utilization(value):
    if value is None: return "No Data"
    if value >= 0.95: return "Alarm"
    if value >= 0.80: return "Warning"
    if value >= 0.65: return "Watch"
    return "Normal"
