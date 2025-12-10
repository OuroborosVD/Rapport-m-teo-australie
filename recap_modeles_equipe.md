# Récapitulatif des modèles de l'équipe

Ce document résume les modèles testés dans le cadre du projet météo
et leurs performances principales.

## 1. Modèles de Louis

Louis a travaillé sur les premières étapes de modélisation :

- Mise en place de modèles de base (régression logistique, arbres de décision, etc.)
- Établissement d'une baseline de performance sur la variable `RainTomorrow`.

Ces modèles ont servi de point de départ pour comparer les approches plus avancées.

## 2. Modèle de Jean-Paul – XGBoost

Jean-Paul a testé un modèle **XGBoost** sur les données encodées.

Objectifs :
- Explorer un modèle de gradient boosting plus complexe,
- Comparer ses performances avec les autres modèles de l'équipe.

Les résultats n'ont pas permis d'améliorer suffisamment le F1-score sur la classe *pluie*,
mais ce modèle a été important dans la démarche comparative.
## 3. Modèle d'Iris – RandomForest (base vs optimisé)

### Modèle de base

- F1 pluie : 0.600
- Rappel pluie : 0.487
- Précision pluie : 0.782

### Modèle optimisé (RandomizedSearchCV)

- F1 pluie : 0.611
- Rappel pluie : 0.520
- Précision pluie : 0.741

L'optimisation a permis :
- d'améliorer le rappel (meilleure détection des jours de pluie),
- d'obtenir un meilleur équilibre entre précision et rappel.

## 4. Modèle final retenu

Le modèle **RandomForest optimisé** :

- il offre le meilleur compromis F1 / rappel sur la classe pluie,
- il est stable et robuste,
- il reste simple à déployer et interpréter.

Ce modèle servira de base pour la suite du projet (présentation, Streamlit, etc.).
