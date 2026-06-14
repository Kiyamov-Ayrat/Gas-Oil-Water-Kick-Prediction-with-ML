import os
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    ConfusionMatrixDisplay,
)

import config
from data_loader import get_train_test


def train(X_train, y_train, X_test, y_test) -> xgb.XGBClassifier:
    print("XGBoost")
    # scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        # scale_pos_weight=scale_pos,   # балансировка дисбаланса
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        **config.XGB_PARAMS,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )
    print("Готово")
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\nМетрики")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Норма", "Флюидопроявление"]))
    return y_pred, y_prob


def save_plots(model, X_test, y_test, y_pred, y_prob):
    out = config.OUTPUT_DIR
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, lw=2, color="darkorange")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title(f"ROC  (AUC={roc_auc_score(y_test, y_prob):.3f})")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].grid(alpha=0.3)

    # PR
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, lw=2, color="darkorange")
    axes[1].set_title(f"Precision-Recall  (AP={average_precision_score(y_test, y_prob):.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].grid(alpha=0.3)

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Норма", "Kick"],
        ax=axes[2], colorbar=False,
    )
    axes[2].set_title("Confusion Matrix")

    plt.tight_layout()
    path = os.path.join(out, "xgb_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  График сохранён: {path}")

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=config.FEATURE_COLS)
    top20 = imp.nlargest(20).sort_values()

    fig2, ax = plt.subplots(figsize=(8, 7))
    top20.plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title("Top-20 важных признаков — XGBoost")
    ax.set_xlabel("Важность")
    plt.tight_layout()
    path2 = os.path.join(out, "xgb_feature_importance.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  График сохранён: {path2}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test()
    model = train(X_train, y_train, X_test, y_test)
    y_pred, y_prob = evaluate(model, X_test, y_test)
    save_plots(model, X_test, y_test, y_pred, y_prob)