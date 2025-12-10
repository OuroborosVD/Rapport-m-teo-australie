## 🌦️ Description du projet

L’objectif est de construire un modèle capable de prédire s’il pleuvra le lendemain à partir des données météo du dataset officiel **WeatherAUS**.

## 📌 Étapes principales du projet

### 1. Nettoyage & prétraitement
- Gestion des valeurs manquantes  
- Encodage des variables  
- Reconstruction de la dimension temporelle  

### 2. Découpage chronologique
- **Train** = toutes les données sauf les 12 derniers mois  
- **Test** = les 12 derniers mois  

### 3. Traitement du déséquilibre de classe
- Undersampling pour obtenir un meilleur équilibre pluie / non-pluie  

## 🤖 Modélisation

Les modèles testés incluent :

- Régression Logistique  
- RandomForest (base + optimisation)  
- Gradient Boosting  
- **HistGradientBoosting (modèle final)**  

## ⚙️ Optimisation

- Recherche d’hyperparamètres (**RandomizedSearchCV**)  
- Sélection du **seuil de décision**  
- Ajout de **variables mémoire** (lag features)  
- Validation chronologique  
## 🏆 Modèle final retenu : HistGradientBoostingClassifier

Ce modèle a offert :

- Le **meilleur score F1 sur la classe pluie**  
- Un excellent compromis **précision / rappel**  
- Une **rapidité de calcul supérieure**  
- Une **accuracy globale d'environ 0.86**  

Ce modèle est donc celui retenu pour le rendu final du projet.
