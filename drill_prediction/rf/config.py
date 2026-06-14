import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

PARQUET_FILE = os.path.join(OUTPUT_DIR, "drilling_labeled.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL = {
    "pvt_threshold": 15.0,   # барр — порог PVT Gain/Loss
    "pump_min": 50.0,   # gal/min — признак работы насосов
    "spp_min": 100.0,   # psi — минимальное давление циркуляции
    "alarm_gain_code": 193,     # код тревоги GAIN в Mud G/L Alarm State
    "smooth_window": 60,     # сек — окно сглаживания меток
    "smooth_min_hits": 15,     # минимум срабатываний в окне
}

ROLLING_WINDOWS = [30, 300, 600]   # секунды для скользящих статистик

FEATURE_COLS = [
    # Основные сигналы
    "pvt_total", "pvt_monitor", "spp", "flow_gl", "pump_out",
    "wob", "rop", "torque", "diff_press", "hookload", "rpm",
    "mud_volume", "on_bottom",
    # Скользящие статистики
    "pvt_mean_30s",  "pvt_mean_300s",  "pvt_mean_600s",
    "pvt_std_30s",   "pvt_std_300s",   "pvt_std_600s",
    "pvt_max_30s",   "pvt_max_300s",   "pvt_max_600s",
    "spp_mean_30s",  "spp_mean_300s",
    "spp_std_30s",   "spp_std_300s",
    "flow_mean_30s", "flow_mean_300s",
    # Производные
    "pvt_delta_10s", "pvt_delta_60s",
    "spp_delta_10s", "mud_delta_10s", "mud_delta_60s",
    # Физические
    "pvt_divergence", "depth_diff", "is_pumping", "is_on_bottom",
    # Временные
    "hour", "dayofweek",
    # Глубина
    "hole_depth", "bit_depth",
]

TARGET = "kick_label"

TRAIN_SIZE = 0.80

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 20,
    "class_weight":  "balanced_subsample",
    "n_jobs": -1,
    "random_state": 42,
}