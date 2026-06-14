import glob
import os
import pandas as pd
import numpy as np
import config


def load_raw() -> pd.DataFrame:
    """Загружает все CSV из DATA_DIR, объединяет, сортирует по времени."""
    files = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(f"CSV не найдены в {config.DATA_DIR}")

    dfs = []
    for f in files:
        print(f"Загрузка: {os.path.basename(f)}")
        dfs.append(pd.read_csv(
            f,
            low_memory=False,
            na_values=[-999.25, "-999.25"],
        ))

    df = pd.concat(dfs, ignore_index=True).copy()
    print(f"Объединено строк: {len(df):,}")

    df["datetime"] = pd.to_datetime(
        df["YYYY/MM/DD"].astype(str) + " " + df["HH:MM:SS"].astype(str),
        format="%Y/%m/%d %H:%M:%S",
        errors="coerce",
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт целевую колонку kick_label."""
    L = config.LABEL

    df["alarm_int"] = df["Mud G/L Alarm State (unitless)"].fillna(0).astype(int)

    cond_alarm = df["alarm_int"] == L["alarm_gain_code"]
    cond_pvt = df["PVT Total Mud Gain/Loss (barrels)"].fillna(0)    > L["pvt_threshold"]
    cond_pvtmon = df["PVT Monitor Mud Gain/Loss (barrels)"].fillna(0)  > L["pvt_threshold"]
    cond_pump = (df["Total Pump Output (gal_per_min)"].fillna(0)     > L["pump_min"]) \
                | (df["Standpipe Pressure (psi)"].fillna(0)            > L["spp_min"])

    df["kick_raw"] = ((cond_alarm | (cond_pvt & cond_pvtmon)) & cond_pump).astype(int)

    df["kick_label"] = (
        df["kick_raw"]
        .rolling(window=L["smooth_window"], min_periods=1, center=True)
        .sum() >= L["smooth_min_hits"]
    ).astype(int)

    n1 = df["kick_label"].sum()
    n0 = len(df) - n1
    print(f"  Метки: 0={n0:,} ({n0/len(df)*100:.1f}%)  1={n1:,} ({n1/len(df)*100:.1f}%)")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: базовые, скользящие, производные признаки."""

    # Базовые
    df["pvt_total"] = df["PVT Total Mud Gain/Loss (barrels)"].fillna(0)
    df["pvt_monitor"] = df["PVT Monitor Mud Gain/Loss (barrels)"].fillna(0)
    df["spp"] = df["Standpipe Pressure (psi)"].fillna(0)
    df["flow_gl"] = df["Flow 1 Gain/Loss (percent)"].fillna(0)
    df["pump_out"] = df["Total Pump Output (gal_per_min)"].fillna(0)
    df["wob"] = df["Weight on Bit (klbs)"].fillna(0)
    df["rop"] = df["Rate Of Penetration (ft_per_hr)"].fillna(0)
    df["torque"] = df["Rotary Torque (kft_lb)"].fillna(0)
    df["diff_press"] = df["Differential Pressure (psi)"].fillna(0)
    df["hookload"] = df["Hook Load (klbs)"].fillna(0)
    df["rpm"] = df["Rotary RPM (RPM)"].fillna(0)
    df["mud_volume"] = df["Total Mud Volume (barrels)"].fillna(0)
    df["on_bottom"] = df["On Bottom (unitless)"].fillna(0)
    df["alarm_state"] = df["alarm_int"].fillna(0)
    df["hole_depth"] = df["Hole Depth (feet)"].ffill()
    df["bit_depth"] = df["Bit Depth (feet)"].ffill()

    # Скользящие статистики
    for w in config.ROLLING_WINDOWS:
        s = str(w)
        df[f"pvt_mean_{s}s"] = df["pvt_total"].rolling(w, min_periods=1).mean()
        df[f"pvt_std_{s}s"] = df["pvt_total"].rolling(w, min_periods=1).std().fillna(0)
        df[f"pvt_max_{s}s"] = df["pvt_total"].rolling(w, min_periods=1).max()
        df[f"spp_mean_{s}s"] = df["spp"].rolling(w, min_periods=1).mean()
        df[f"spp_std_{s}s"] = df["spp"].rolling(w, min_periods=1).std().fillna(0)
        df[f"flow_mean_{s}s"] = df["flow_gl"].rolling(w, min_periods=1).mean()

    # Производные (скорость изменения)
    df["pvt_delta_10s"] = df["pvt_total"].diff(10).fillna(0)
    df["pvt_delta_60s"] = df["pvt_total"].diff(60).fillna(0)
    df["spp_delta_10s"] = df["spp"].diff(10).fillna(0)
    df["mud_delta_10s"] = df["mud_volume"].diff(10).fillna(0)
    df["mud_delta_60s"] = df["mud_volume"].diff(60).fillna(0)

    # Физические индикаторы
    df["pvt_divergence"] = df["pvt_total"] - df["pvt_monitor"]
    df["depth_diff"] = df["hole_depth"] - df["bit_depth"]
    df["is_pumping"] = (df["pump_out"] > config.LABEL["pump_min"]).astype(int)
    df["is_on_bottom"] = (df["on_bottom"] > 0.5).astype(int)

    # Временные
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek

    return df


def get_train_test():
    """
    Главная функция: возвращает X_train, X_test, y_train, y_test.
    Кэширует parquet чтобы не пересчитывать каждый раз.
    """
    if os.path.exists(config.PARQUET_FILE):
        print(f"Загружаю кэш: {config.PARQUET_FILE}")
        df = pd.read_parquet(config.PARQUET_FILE)
    else:
        print("Кэш не найден, обрабатываю CSV...")
        df = load_raw()
        df = make_labels(df)
        df = make_features(df)
        df.to_parquet(config.PARQUET_FILE, index=False)
        print(f"  Кэш сохранён: {config.PARQUET_FILE}")

    cols = config.FEATURE_COLS + [config.TARGET]
    df = df[cols].dropna()

    split = int(len(df) * config.TRAIN_SIZE)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train = train[config.FEATURE_COLS]
    y_train = train[config.TARGET]
    X_test  = test[config.FEATURE_COLS]
    y_test  = test[config.TARGET]

    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test