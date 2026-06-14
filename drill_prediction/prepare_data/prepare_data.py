import pandas as pd

# Загрузка файла
df = pd.read_csv("C:/Users/User/Desktop/ML/курсовая/drilling_data_with_target.csv", low_memory=False)

print(df.shape)
print(df['target_kick'].value_counts())
print(df['target_kick'].value_counts(normalize=True) * 100)

# Посмотреть первые строки
print(df.head())

# Сохранить небольшую часть для Excel (например, 100 000 строк)
df_sample = df.sample(n=100_000, random_state=42)
df_sample.to_csv('C:/Users/User/Desktop/ML/курсовая/drilling_sample_for_excel.csv', index=False)
print("Сэмпл сохранён для Excel")