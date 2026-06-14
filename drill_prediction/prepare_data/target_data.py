import pandas as pd
import numpy as np
import glob
import os

# ===================== НАСТРОЙКИ =====================
folder_path = r"C:/Users/User/Desktop/analyse/dataset/56-32_Pason_Archive"   # ←←← ИЗМЕНИ, если файлы не в текущей папке

# Загрузка файлов
csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))

print(f"Найдено файлов: {len(csv_files)}")
for f in csv_files:
    print(f" → {os.path.basename(f)}")

df_list = []
for file in csv_files:
    print(f"Загружаю: {os.path.basename(file)}")
    temp = pd.read_csv(file, low_memory=False)
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)
print(f"\nВсего строк: {len(df):,}")

# ===================== ВАЖНО: СМОТРИМ РЕАЛЬНЫЕ НАЗВАНИЯ КОЛОНОК =====================
print("\nПервые 30 колонок в датасете:")
print(df.columns.tolist()[:30])

# ===================== ОБРАБОТКА =====================
df.replace(-999.25, np.nan, inplace=True)

# Сортировка по времени
time_col = 'YYYY/MM/DD,HH:MM:SS'
if time_col in df.columns:
    df[time_col] = pd.to_datetime(df[time_col], format='%Y/%m/%d,%H:%M:%S', errors='coerce')
    df = df.sort_values(by=time_col).reset_index(drop=True)

# ===================== ТОЧНЫЕ НАЗВАНИЯ КОЛОНОК =====================
# Используем точные названия из твоего датасета
df['target_kick'] = 0

conditions = (
    (df['PVT Total Mud Gain/Loss (barrels)'] > 15) |
    (df['PVT Monitor Mud Gain/Loss (barrels)'] > 15) |
    (df['Smart Mud G/L (barrels)'] > 10) |
    (df['Smart Flow G/L (percent)'] > 15)
)

df.loc[conditions, 'target_kick'] = 1

# ===================== СТАТИСТИКА =====================
print("\n" + "="*60)
print("=== СТАТИСТИКА ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ===")
print("="*60)
print(df['target_kick'].value_counts())
print(f"Процент класса 1: {df['target_kick'].mean()*100:.4f}%")
print(f"Всего событий: {df['target_kick'].sum():,}")

# Дополнительно
print("\nРаспределение по условиям:")
print(f"PVT Total > 25      : {(df['PVT Total Mud Gain/Loss (barrels)'] > 25).sum():,}")
print(f"Monitor > 25        : {(df['PVT Monitor Mud Gain/Loss (barrels)'] > 25).sum():,}")
print(f"Smart Mud G/L > 15  : {(df['Smart Mud G/L (barrels)'] > 15).sum():,}")
print(f"Smart Flow G/L > 20 : {(df['Smart Flow G/L (percent)'] > 20).sum():,}")

# ===================== СОХРАНЕНИЕ =====================
# output_file = 'C:/Users/User/Desktop/ML/курсовая/drilling_data_with_target.csv'
# df.to_csv(output_file, index=False)
# print(f"\nФайл сохранён: {output_file}")