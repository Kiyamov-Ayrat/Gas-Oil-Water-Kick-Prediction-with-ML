import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


# ЗАГРУЗКА
folder_path = r"C:/Users/User/Desktop/analyse/dataset/56-32_Pason_Archive"
files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

dfs = []
for f in files:
    print(f"Загружаю: {f}")
    chunk = pd.read_csv(f, sep=",", header=0, low_memory=False,
                        na_values=[-999.25, "-999.25"])
    dfs.append(chunk)

df = pd.concat(dfs, ignore_index=True)
df = df.copy()  # убираем фрагментацию
print(f"Объединено строк: {len(df):,}")

# DATETIME
df["datetime"] = pd.to_datetime(
    df["YYYY/MM/DD"].astype(str) + " " + df["HH:MM:SS"].astype(str),
    format="%Y/%m/%d %H:%M:%S", errors="coerce"
)
df = df.sort_values("datetime").reset_index(drop=True)

# ДЕКОДИРОВАНИЕ БИТОВЫХ ФЛАГОВ ALARM STATE
# Значения: 128=нет тревоги (базовый), 193/194=тревога gain/loss, 129/130=предупреждение
# Битовая маска: бит 6 (64) = gain alarm, бит 0 (1) = gain warn, бит 1 (2) = loss warn
alarm_col = "Mud G/L Alarm State (unitless)"
df["alarm_int"] = df[alarm_col].fillna(0).astype(int)

print("\n=== Alarm State — все значения и их интерпретация ===")
alarm_map = {
    0:   "Нет данных",
    128: "Норма (насосы остановлены)",
    129: "Предупреждение GAIN",
    130: "Предупреждение LOSS",
    193: "ТРЕВОГА GAIN (приток!)",
    194: "ТРЕВОГА LOSS (потеря)",
}
for val, count in df["alarm_int"].value_counts().items():
    label = alarm_map.get(val, f"Неизвестно ({val})")
    print(f"  {val:5d}  →  {label:30s}  : {count:>10,}")

# Alarm gain = значение 193 (это и есть газопроявление по системе PVT)
cond_alarm_gain = df["alarm_int"] == 193

# ИСПРАВЛЕННАЯ РАЗМЕТКА
pvt_col  = "PVT Total Mud Gain/Loss (barrels)"
spp_col  = "Standpipe Pressure (psi)"
pump_col = "Total Pump Output (gal_per_min)"
flow_col = "Flow 1 Gain/Loss (percent)"

# Фильтр: насосы работают (более надёжен чем SPP)
cond_pumping = (df[pump_col].fillna(0) > 50) | (df[spp_col].fillna(0) > 100)

# Сигнал 1: PVT gain — используем более высокий порог чтобы убрать шум
# Из статистики: p90=21, p95=51 — порог 15 барр разумен
cond_pvt = df[pvt_col].fillna(0) > 15

# Сигнал 2: Alarm State 193 = подтверждённая тревога gain
# (это наш самый надёжный сигнал)
cond_alarm = cond_alarm_gain

# Сигнал 3: PVT Monitor как подтверждение (6.98% NaN — почти полный)
pvtmon_col = "PVT Monitor Mud Gain/Loss (barrels)"
cond_pvtmon = df[pvtmon_col].fillna(0) > 15

# Итоговая метка: alarm ИЛИ (pvt + pvtmon оба > порога) + насосы работают
df["kick_raw"] = (
    (cond_alarm | (cond_pvt & cond_pvtmon)) & cond_pumping
).astype(int)

# Сглаживание: 60 сек окно, минимум 15 срабатываний
df["kick_label"] = (
    df["kick_raw"]
    .rolling(window=60, min_periods=1, center=True)
    .sum() >= 15
).astype(int)


# СТАТИСТИКА
print("\n" + "="*55)
print("СТАТИСТИКА МЕТОК")
print("="*55)
for label in ["kick_raw", "kick_label"]:
    n1 = df[label].sum()
    n0 = len(df) - n1
    print(f"\n{label}:")
    print(f"  Класс 0 (норма)       : {n0:>10,}  ({n0/len(df)*100:.2f}%)")
    print(f"  Класс 1 (газопроявл.) : {n1:>10,}  ({n1/len(df)*100:.2f}%)")
    print(f"  Дисбаланс             : 1 к {n0//max(n1,1)}")

print(f"\n  Alarm 193 строк : {cond_alarm.sum():,}")
print(f"  PVT > 15 строк  : {cond_pvt.sum():,}")
print(f"  Насосы активны  : {cond_pumping.sum():,}")


# ВИЗУАЛИЗАЦИЯ — весь датасет + зум на события
fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

axes[0].plot(df["datetime"], df[pvt_col], color="steelblue", lw=0.4, alpha=0.8)
axes[0].axhline(15, color="red", ls="--", lw=1, label="Порог 15 барр")
axes[0].set_ylabel("PVT Gain/Loss (барр)")
axes[0].set_title("Диагностика газопроявления — полный датасет")
axes[0].legend(fontsize=8)

axes[1].fill_between(df["datetime"], df["alarm_int"].eq(193).astype(int),
                     alpha=0.6, color="purple", label="Alarm 193 (GAIN)")
axes[1].set_ylabel("Alarm State 193")
axes[1].set_ylim(-0.1, 1.3)
axes[1].legend(fontsize=8)

axes[2].fill_between(df["datetime"], df["kick_raw"],
                     alpha=0.5, color="orange", label="kick_raw")
axes[2].fill_between(df["datetime"], df["kick_label"],
                     alpha=0.7, color="crimson", label="kick_label (сглажено)")
axes[2].set_ylabel("Метка kick")
axes[2].set_ylim(-0.1, 1.3)
axes[2].legend(fontsize=8)

axes[3].plot(df["datetime"], df[pump_col].fillna(0),
             color="green", lw=0.4, label="Насосы (gal/min)")
axes[3].set_ylabel("Pump Output")
axes[3].set_xlabel("Время")
axes[3].legend(fontsize=8)

plt.tight_layout()
plt.savefig("kick_labels_v2.png", dpi=150, bbox_inches="tight")
print("\nГрафик сохранён: kick_labels_v2.png")

# СОХРАНЕНИЕ
# df.to_parquet("C:/Users/User/Desktop/ML/курсовая/drilling_labeled.parquet", index=False)
# print(f"Датасет сохранён: drilling_labeled.parquet ({len(df):,} строк)")