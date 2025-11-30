import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import os

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_CSV_PATH = "/Classificazione/train.csv"
TEST_CSV_PATH  = "/Classificazione/test.csv"
BASE_PLOT_DIR = "/Classificazione/plot_scenario3"

def plot_confusion_matrix(cm, model_name, scenario_name, out_dir):

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                cbar=True, xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix - {model_name}")
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"cm_{model_name}.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"[INFO] Salvata confusion matrix in: {fname}")


def plot_roc_curve_global(y_true, scores, model_name, scenario_name, out_dir):

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_val = roc_auc_score(y_true, scores)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"roc_{model_name}.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def get_scores_for_roc(model, X):

    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        try:
            return model.decision_function(X)
        except Exception:
            return None


print("[INFO] Carico train e test, li unisco e lavoro con un unico dataset...")
df_train = pd.read_csv(TRAIN_CSV_PATH)
df_test  = pd.read_csv(TEST_CSV_PATH)

# droppo colonne inutili se presenti
cols_to_drop = ["Unnamed: 0", "id"]
df_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
df_test  = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])

# Target binario
df_train["satisfaction_binary"] = (df_train["satisfaction"] == "satisfied").astype(int)
df_test["satisfaction_binary"]  = (df_test["satisfaction"] == "satisfied").astype(int)

# concateno
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)
print("[INFO] Shape df_all:", df_all.shape)

# feature candidate generali
all_feature_cols = [
    "Gender", "Customer Type", "Age", "Type of Travel", "Class",
    "Flight Distance", "Inflight wifi service",
    "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Inflight service",
    "Cleanliness", "Departure Delay in Minutes", "Arrival Delay in Minutes"
]

# Definizione scenari
scenarios = {
    # Scenario 1: rimuovo solo ritardi -> niente leakage temporale
    "scenario1_no_delays": [
        c for c in all_feature_cols
        if c not in ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
    ],

    # Scenario 2: rimuovo tutti i rating di servizio, tengo solo info "oggettive"
    "scenario2_no_service_scores": [
        "Gender", "Customer Type", "Age", "Type of Travel", "Class",
        "Flight Distance", "Departure Delay in Minutes",
        "Arrival Delay in Minutes"
    ],

    # Scenario 3: minimalista - solo info presenti prima della prenotazione/volo
    "scenario3_minimalist": [
        "Gender", "Customer Type", "Type of Travel", "Class",
        "Age", "Flight Distance"
    ]
}

print("[INFO] Scenari definiti:")
for k, v in scenarios.items():
    print(f"  - {k}: {len(v)} feature")


# Loop sugli scenari
os.makedirs(BASE_PLOT_DIR, exist_ok=True)

for scenario_name, feat_cols in scenarios.items():
    print("\n" + "=" * 80)
    print(f"[SCENARIO] {scenario_name}")
    print(f"[INFO] Numero feature: {len(feat_cols)}")
    print("[INFO] Feature:", feat_cols)

    # X, y per questo scenario
    X = df_all[feat_cols].copy()
    y = df_all["satisfaction_binary"].values

    # suddivisione per tipo
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols     = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    print("[INFO] Categorical:", categorical_cols)
    print("[INFO] Numeric:", numeric_cols)

    # preprocessore lineare (per modelli che beneficiano dello scaling)
    preprocess_linear = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols),
        ],
        remainder="drop"
    )

    # preprocessore per alberi / boosting (no scaling necessario, uso OrdinalEncoder)
    preprocess_tree = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline([
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
    models = {
        "Logistic_Regression": (
            Pipeline([
                ("prep", preprocess_linear),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    solver="liblinear"
                ))
            ]),
            {
                "clf__C": [0.1, 1.0]
            }
        ),

        "Linear_SVC": (
            Pipeline([
                ("prep", preprocess_linear),
                ("clf", LinearSVC(dual=False))
            ]),
            {
                "clf__C": [0.1, 1.0]
            }
        ),

        "KNN": (
            Pipeline([
                ("prep", preprocess_linear),
                ("clf", KNeighborsClassifier())
            ]),
            {
                "clf__n_neighbors": [11, 31]
            }
        ),

        "Random_Forest": (
            Pipeline([
                ("prep", preprocess_tree),
                ("clf", RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            {
                "clf__n_estimators": [200],
                "clf__max_depth": [10, None],
                "clf__min_samples_leaf": [3]
            }
        ),

        "Gradient_Boosting": (
            Pipeline([
                ("prep", preprocess_tree),
                ("clf", GradientBoostingClassifier(random_state=42))
            ]),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3]
            }
        ),
    }

    print(f"[INFO] Numero modelli: {len(models)}")

    # Nested Cross-Validation
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scenario_results = {}

    print("\n================== Nested Cross Validation ==================\n")

    for model_name, (pipeline, param_grid) in models.items():
        print(f">>> MODEL: {model_name}")

        oof_pred = np.zeros_like(y)
        oof_scores = np.zeros_like(y, dtype=float)
        have_scores = True

        outer_scores = []

        for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            gs = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=inner_cv,
                scoring="accuracy",
                n_jobs=-1
            )
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_

            # predizioni sul fold esterno
            y_pred_fold = best_model.predict(X_test)
            oof_pred[test_idx] = y_pred_fold

            # scores per ROC/AUC
            scores_fold = get_scores_for_roc(best_model, X_test)
            if scores_fold is not None:
                oof_scores[test_idx] = scores_fold
            else:
                have_scores = False

            # accuracy del fold esterno
            outer_scores.append(accuracy_score(y_test, y_pred_fold))

        outer_scores = np.array(outer_scores)
        print(f"   Nested Acc: {outer_scores.mean():.4f} ± {outer_scores.std():.4f}\n")

        # metriche globali su tutte le OOF
        acc = accuracy_score(y, oof_pred)
        prec = precision_score(y, oof_pred)
        rec = recall_score(y, oof_pred)
        f1 = f1_score(y, oof_pred)

        if have_scores:
            auc_val = roc_auc_score(y, oof_scores)
        else:
            auc_val = None

        scenario_results[model_name] = {
            "nested_mean_acc": outer_scores.mean(),
            "nested_std_acc": outer_scores.std(),
            "acc": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc_val,
        }

        print(f"================ METRICHE GLOBALI -  {model_name} ================")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1       : {f1:.4f}")
        if auc_val is not None:
            print(f"AUC      : {auc_val:.4f}")

        # plot CM + ROC
        scenario_dir = os.path.join(BASE_PLOT_DIR, scenario_name)
        cm = confusion_matrix(y, oof_pred)
        plot_confusion_matrix(cm, model_name, scenario_name, scenario_dir)

        if have_scores:
            plot_roc_curve_global(y, oof_scores, model_name, scenario_name, scenario_dir)

        print()

    # riepilogo scenario
    print("\n=========== RISULTATI NESTED  -", scenario_name, "===========")
    for model_name, res in scenario_results.items():
        print(f"{model_name:18s} | {res['nested_mean_acc']:.4f} ± {res['nested_std_acc']:.4f}")

