import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

# ============================================================
# PATH
# ============================================================

TRAIN_CSV_PATH = "/Classificazione/train.csv"
TEST_CSV_PATH  = "/Classificazione/test.csv"
PLOTS_DIR = "/Classificazione/plot_scenario2"
os.makedirs(PLOTS_DIR, exist_ok=True)


# caricamento dati
print("[INFO] Carico train e test, li unisco e lavoro con un unico dataset...")

df_train = pd.read_csv(TRAIN_CSV_PATH)
df_test  = pd.read_csv(TEST_CSV_PATH)

# Drop colonne inutili
cols_to_drop = ["Unnamed: 0", "id"]
df_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
df_test  = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])

# Creo target binaria su entrambi
for df in (df_train, df_test):
    df["satisfaction_binary"] = (df["satisfaction"] == "satisfied").astype(int)

# Allineo colonne ed unisco
common_cols = sorted(set(df_train.columns) & set(df_test.columns))
df_all = pd.concat(
    [df_train[common_cols], df_test[common_cols]],
    axis=0,
    ignore_index=True
)


# Target
TARGET_COL = "satisfaction_binary"
y_all = df_all[TARGET_COL].values

# Feature = tutte tranne target + colonna testuale 'satisfaction'
feature_cols = [c for c in df_all.columns if c not in [TARGET_COL, "satisfaction"]]
X_all = df_all[feature_cols]

print("[INFO] Colonne feature usate:", feature_cols)

# Preprocessing

categorical_cols = X_all.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols     = X_all.select_dtypes(exclude=["object", "category"]).columns.tolist()

print("[INFO] Categorical:", categorical_cols)
print("[INFO] Numeric:", numeric_cols)

# per i modelli lineari escludo Arrival Delay dallo scaling
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

# Modelli

models = []

# Logistic Regression
models.append((
    "Logistic_Regression",
    Pipeline([
        ("prep", preprocess_linear),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="liblinear"
        ))
    ]),
    {
        "clf__C": [0.1, 1.0],
        "clf__penalty": ["l2"]
    }
))

# Linear SVC
models.append((
    "Linear_SVC",
    Pipeline([
        ("prep", preprocess_linear),
        ("clf", LinearSVC(dual=False))
    ]),
    {
        "clf__C": [0.1, 1.0]
    }
))

# KNN
models.append((
    "KNN",
    Pipeline([
        ("prep", preprocess_linear),
        ("clf", KNeighborsClassifier())
    ]),
    {
        "clf__n_neighbors": [11, 31],
        "clf__weights": ["uniform", "distance"]
    }
))

# Random Forest
models.append((
    "Random_Forest",
    Pipeline([
        ("prep", preprocess_non_linear),
        ("clf", RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ))
    ]),
    {
        "clf__n_estimators": [200],
        "clf__max_depth": [8, 12],
        "clf__min_samples_leaf": [3]
    }
))

# Gradient Boosting
models.append((
    "Gradient_Boosting",
    Pipeline([
        ("prep", preprocess_non_linear),
        ("clf", GradientBoostingClassifier(random_state=42))
    ]),
    {
        "clf__n_estimators": [150],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3]
    }
))

print(f"[INFO] Numero modelli: {len(models)}")

# Nested Cross Validation

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

nested_results = {}
y_true_all   = {}
y_pred_all   = {}
y_score_all  = {}

def get_scores(estimator, X):
    """Restituisce score continui per ROC (probabilità o decision_function)."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None

print("\n================== Nested Cross Validation ==================\n")

for name, pipeline, param_grid in models:
    print(f">>> MODEL: {name}")
    outer_scores = []

    y_true_all[name]  = []
    y_pred_all[name]  = []
    y_score_all[name] = []

    fold_idx = 1
    for train_idx, test_idx in outer_cv.split(X_all, y_all):
        X_tr, X_te = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1
        )

        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_

        y_pred = best.predict(X_te)
        score  = accuracy_score(y_te, y_pred)
        outer_scores.append(score)

        y_true_all[name].extend(y_te)
        y_pred_all[name].extend(y_pred)

        s = get_scores(best, X_te)
        if s is not None:
            y_score_all[name].extend(s.tolist())

        fold_idx += 1

    mean_acc = np.mean(outer_scores)
    std_acc  = np.std(outer_scores)
    nested_results[name] = (mean_acc, std_acc)
    print(f"   Nested Acc: {mean_acc:.4f} ± {std_acc:.4f}\n")

print("\n=========== RISULTATI NESTED ===========")
for name, (mean, std) in nested_results.items():
    print(f"{name:20s} | {mean:.4f} ± {std:.4f}")
print()


# 5. METRICHE FINALI + PLOT CONFUSION MATRIX & ROC PER OGNI MODELLO

for name, _, _ in models:
    print(f"\n================ METRICHE GLOBALI (nested OUT-OF-FOLD) - {name} ================")

    y_true = np.array(y_true_all[name])
    y_pred = np.array(y_pred_all[name])
    y_score = np.array(y_score_all[name]) if len(y_score_all[name]) == len(y_true) else None

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")

    # AUC se abbiamo uno score continuo
    auc_val = None
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            auc_val = roc_auc_score(y_true, y_score)
            print(f"AUC      : {auc_val:.4f}")
        except Exception:
            pass

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, f"cm_{name}.png")
    plt.savefig(cm_path)
    plt.close()

    # ROC curve
    if y_score is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(PLOTS_DIR, f"roc_{name}.png")
        plt.savefig(roc_path)
        plt.close()
    else:
        print(" Nessuno score continuo disponibile -> niente ROC per questo modello.")

