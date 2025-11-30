import os
import warnings

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# parametri
TRAIN_CSV_PATH = "/Classificazione/train.csv"
TEST_CSV_PATH  = "/Classificazione/test.csv"
PLOTS_DIR = "/Classificazione/plot_scenario1"

os.makedirs(PLOTS_DIR, exist_ok=True)

# caricamento dati
df_train = pd.read_csv(TRAIN_CSV_PATH)
df_train = df_train.drop(columns=[c for c in ["Unnamed: 0", "id"] if c in df_train.columns])


df_train["satisfaction_binary"] = (df_train["satisfaction"] == "satisfied").astype(int)
TARGET_COL = "satisfaction_binary"

# feature = tutte tranne target e satisfaction
feature_cols = [c for c in df_train.columns if c not in [TARGET_COL, "satisfaction"]]

X_train = df_train[feature_cols]
y_train = df_train[TARGET_COL].values

print("[INFO] TRAIN shape:", X_train.shape)
print("[INFO] Feature usate nel TRAIN:", feature_cols)

categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols     = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

print("[INFO] Categorical:", categorical_cols)
print("[INFO] Numeric:", numeric_cols)

# Per i modelli lineari escludo 'Arrival Delay in Minutes' dallo scaling, è molto correlato a Departure Delay
numeric_cols_linear = [c for c in numeric_cols if c != "Arrival Delay in Minutes"]

preprocess_linear = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler())
        ]), numeric_cols_linear),
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh",  OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols),
    ],
    remainder="drop"
)

preprocess_non_linear = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ]), categorical_cols),
    ],
    remainder="drop"
)


models = {
    "LogisticRegression": Pipeline([
        ("prep", preprocess_linear),
        ("clf", LogisticRegression(
            C=0.1,
            penalty="l2",
            max_iter=1000,
            solver="liblinear"
        ))
    ]),

    "LinearSVC": Pipeline([
        ("prep", preprocess_linear),
        ("clf", LinearSVC(C=0.1, dual=False))
    ]),

    "KNN": Pipeline([
        ("prep", preprocess_linear),
        ("clf", KNeighborsClassifier(
            n_neighbors=11,
            weights="distance"
        ))
    ]),

    "RandomForest": Pipeline([
        ("prep", preprocess_non_linear),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "GradientBoosting": Pipeline([
        ("prep", preprocess_non_linear),
        ("clf", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])
}

print(f"\n[INFO] Numero modelli da valutare: {len(models)}")

# 4. Cross validation

cv_scores = {}
print("\n================== Cross validation sul trainset ==================\n")

for name, model in models.items():
    print(f">>> Cross per modello: {name}")
    scores = cross_val_score(model, X_train, y_train, cv=3,
                             scoring="accuracy", n_jobs=-1)
    cv_scores[name] = scores.mean()
    print(f"Cross Accuracy: {scores.mean():.4f}\n")


df_test = pd.read_csv(TEST_CSV_PATH)
df_test = df_test.drop(columns=[c for c in ["Unnamed: 0", "id"] if c in df_test.columns])

if "satisfaction" not in df_test.columns:
    raise ValueError("Nel test.csv mi aspetto la colonna 'satisfaction' per valutare le predizioni.")

df_test["satisfaction_binary"] = (df_test["satisfaction"] == "satisfied").astype(int)

# uso esattamente le stesse feature del train
X_test = df_test[feature_cols]
y_test = df_test["satisfaction_binary"].values

print("[INFO] TEST shape:", X_test.shape)

def get_scores_for_roc(model, X):

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        return None

def plot_confusion_matrix(cm, model_name, save_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["dissatisfied", "satisfied"],
                yticklabels=["dissatisfied", "satisfied"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, scores, model_name, save_path):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ============================================================
# 7. TRAIN SU TUTTO IL TRAIN + EVAL SU TEST + PLOT
# ============================================================

print("\n================== EVALUATION SU TEST ==================\n")

for name, model in models.items():
    print(f"=========== TEST METRICS: {name} ===========")

    # fit sul train completo
    model.fit(X_train, y_train)

    # predizione su test
    y_pred = model.predict(X_test)

    # metriche base
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f"CV Accuracy: {cv_scores[name]:.4f}")
    print(f"Accuracy : {acc}")
    print(f"Precision: {prec}")
    print(f"Recall   : {rec}")
    print(f"F1       : {f1}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # salvataggio immagine conf matrix
    cm_path = os.path.join(PLOTS_DIR, f"cm_{name}.png")
    plot_confusion_matrix(cm, name, cm_path)


    # ROC / AUC (se possibile)
    scores = get_scores_for_roc(model, X_test)
    if scores is not None:
        auc = roc_auc_score(y_test, scores)
        print(f"AUC      : {auc}")
        roc_path = os.path.join(PLOTS_DIR, f"roc_{name}.png")
        plot_roc_curve(y_test, scores, name, roc_path)
    else:
        print("[WARN] Modello senza probabilità/decision_function: AUC/ROC non calcolata.")

    print()  # riga vuota tra modelli

