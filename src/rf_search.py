from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV


# ============================
# 1. CHARGEMENT DES DONNÉES
# ============================

DATA_PATH = Path("data/weatherAUS_encoded.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ Fichier introuvable : {DATA_PATH.resolve()}")

print("📂 Chargement des données...")
df = pd.read_csv(DATA_PATH)
print("✅ Données chargées :", df.shape)

if "RainTomorrow" not in df.columns:
    raise ValueError("❌ La colonne cible 'RainTomorrow' n'existe pas dans le fichier.")

X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]

print("🎯 Variable cible : 'RainTomorrow'")
print("   Nombre de features :", X.shape[1])

# ============================
# 2. TRAIN / TEST SPLIT
# ============================

print("\n✂️ Découpage train / test...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("   X_train :", X_train.shape)
print("   X_test  :", X_test.shape)

# ============================
# 3. RANDOM FOREST DE BASE
# ============================

print("\n🤖 Entraînement du RandomForest de base...")

rf_base = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

acc_base = accuracy_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base)
prec_base = precision_score(y_test, y_pred_base)
rec_base = recall_score(y_test, y_pred_base)

print("\n--- Performances du modèle de base ---")
print(f"Accuracy       : {acc_base:.3f}")
print(f"F1 (pluie=1)   : {f1_base:.3f}")
print(f"Précision (1)  : {prec_base:.3f}")
print(f"Rappel (1)     : {rec_base:.3f}")

# ============================
# 4. RANDOMIZED SEARCH
# ============================

print("\n🔍 Lancement du RandomizedSearchCV sur RandomForest...")

param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
    "bootstrap": [True, False],
}

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
)

rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=20,          # 20 combinaisons testées
    scoring="f1",       # on optimise le F1 de la classe 1 (pluie)
    cv=3,               # validation croisée 3-fold
    n_jobs=-1,
    verbose=2,
    random_state=42,
)

rf_search.fit(X_train, y_train)

print("\n✅ RandomizedSearch terminé.")
print("Meilleurs hyperparamètres trouvés :")
print(rf_search.best_params_)
print("Meilleur F1 (cross-validation) :", rf_search.best_score_)

# ============================
# 5. ÉVALUATION DU MEILLEUR MODÈLE
# ============================

best_rf = rf_search.best_estimator_

print("\n📊 Évaluation du RandomForest optimisé sur le test :")
y_pred_best = best_rf.predict(X_test)

acc_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
prec_best = precision_score(y_test, y_pred_best)
rec_best = recall_score(y_test, y_pred_best)

print("\n--- Performances du modèle optimisé ---")
print(f"Accuracy       : {acc_best:.3f}")
print(f"F1 (pluie=1)   : {f1_best:.3f}")
print(f"Précision (1)  : {prec_best:.3f}")
print(f"Rappel (1)     : {rec_best:.3f}")

print("\nClassification report :\n")
print(classification_report(y_test, y_pred_best))
import joblib
print("\nClassification report :\n")
print(classification_report(y_test, y_pred_best))

import joblib
import os

# Création du dossier "models" si nécessaire
os.makedirs("models", exist_ok=True)

# Sauvegarde du modèle optimisé
joblib.dump(best_rf, "models/random_forest_optimized.joblib")

print("\n✨ Résumé :")
print(f"F1 base      : {f1_base:.3f}")
print(f"F1 optimisé  : {f1_best:.3f}")
print("Si le F1 optimisé est plus élevé, Random Search a apporté un gain.")
