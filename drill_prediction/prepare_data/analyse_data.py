import pandas as pd
import numpy as np
import glob
import os



folder_path = r"C:/Users/User/Desktop/analyse/dataset/56-32_Pason_Archive"  # ←←← ИЗМЕНИ НА СВОЙ ПУТЬ


# Названия колонок
key_columns = [
    'Pason Gas (percent)',
    'PVT Total Mud Gain/Loss (barrels)',
    'PVT Monitor Mud Gain/Loss (barrels)',
    'Smart Mud G/L (barrels)',
    'Smart Flow G/L (percent)',
    'Flow 1 Gain/Loss (percent)'
]


# Находим все csv файлы в папке
csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))

print(f"Найдено файлов: {len(csv_files)}")
for f in csv_files:
    print(f" → {os.path.basename(f)}")

# Загружаем и объединяем все файлы
df_list = []
for file in csv_files:
    print(f"Загружаю: {os.path.basename(file)}")
    temp = pd.read_csv(file, low_memory=False)
    df_list.append(temp)

# Объединяем в один DataFrame
df = pd.concat(df_list, ignore_index=True)
print(f"\nОбъединено строк: {len(df):,}")

# ОБРАБОТКА
# Заменяем -999.25 на NaN
df.replace(-999.25, np.nan, inplace=True)

# Преобразуем время в datetime
time_col = 'YYYY/MM/DD,HH:MM:SS'
if time_col in df.columns:
    df[time_col] = pd.to_datetime(df[time_col], format='%Y/%m/%d,%H:%M:%S', errors='coerce')
    df = df.sort_values(by=time_col).reset_index(drop=True)
    print("Данные отсортированы по времени")

# СТАТИСТИКА
print("\nСТАТИСТИКА ПО КЛЮЧЕВЫМ КОЛОНКАМ\n")

for col in key_columns:
    if col in df.columns:
        print(f"Колонка: {col}")
        print(f"  Минимум     : {df[col].min():.4f}")
        print(f"  Максимум    : {df[col].max():.4f}")
        print(f"  Среднее     : {df[col].mean():.4f}")
        print(f"  Медиана     : {df[col].median():.4f}")
        print(f"  Std         : {df[col].std():.4f}")
        print(f"  NaN         : {df[col].isna().sum():,} "
              f"({df[col].isna().mean() * 100:.2f}%)")

        # Перцентили — самые важные для выбора порогов
        print("  Перцентили:")
        perc = df[col].quantile([0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999]).round(4)
        print(perc)
        print("-" * 70, "\n")
    else:
        print(f"Колонка {col} НЕ НАЙДЕНА!")

# Сохраняем объединённый и обработанный датасет (по желанию)
# df.to_csv('C:/Users/User/Desktop/ML/курсовая/combined_drilling_data.csv', index=False)