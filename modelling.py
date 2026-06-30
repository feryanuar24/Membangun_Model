import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "hotel-booking_preprocessing.csv"
)

# ==========================================
# SPLIT DATA
# ==========================================

X = df.drop(
    "is_canceled",
    axis=1
)

y = df["is_canceled"]

X_train,X_test,y_train,y_test = \
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# MLFLOW (AUTO LOGGING)
# ==========================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Hotel Booking Auto Logging"
)

mlflow.autolog()

with mlflow.start_run():

    # ==========================================
    # PARAMETER
    # ==========================================

    n_estimators = 250
    max_depth = 8
    learning_rate = 0.1

    # ==========================================
    # MODEL TRAINING
    # ==========================================

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    # ==========================================
    # ACCURACY
    # ==========================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"Skor Akurasi: {accuracy}")

    # ==========================================
    # CONFUSION MATRIX
    # ==========================================

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6,4))

    plt.imshow(cm)

    plt.title(
        "Confusion Matrix"
    )

    plt.colorbar()

    plt.savefig(
        "artifacts/auto/confusion_matrix.png"
    )

    print("Confusion matrix berhasil disimpan ke artifacts/auto/confusion_matrix.png")

    plt.close()

    # ==========================================
    # CLASSIFICATION REPORT
    # ==========================================

    report = classification_report(
        y_test,
        predictions
    )

    with open(
        "artifacts/auto/classification_report.txt",
        "w"
    ) as f:
        f.write(report)

    print("Classification report berhasil disimpan ke artifacts/auto/classification_report.txt")

    # ======================================
    # FEATURE IMPORTANCE
    # ======================================

    importance = model.feature_importances_

    plt.figure(
        figsize=(12,6)
    )

    plt.bar(
        X.columns,
        importance
    )

    plt.xticks(
        rotation=90
    )

    plt.tight_layout()

    plt.savefig(
        "artifacts/auto/feature_importance.png"
    )

    plt.close()

    print("Feature importance berhasil disimpan ke artifacts/auto/feature_importance.png")