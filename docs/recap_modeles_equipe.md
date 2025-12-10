 Récapitulatif des modèles de l'équipe
Ce document résume les modèles testés dans le cadre du projet de prévision météo ainsi que leurs performances principales.

1. Modèles explorés (baseline)

Plusieurs modèles simples ont d'abord été développés pour établir une baseline :

Régression logistique

Arbre de décision

RandomForest (version non optimisée)

Ces premiers modèles ont permis de définir des performances de référence sur la variable RainTomorrow.

2. Modèle XGBoost

Un modèle XGBoost a été testé afin d'explorer un algorithme de gradient boosting plus avancé.

Objectifs :

Comparer un modèle plus complexe aux modèles classiques

Évaluer s'il permettait d'améliorer la détection des jours de pluie

Résultat :
Même si les performances n'ont pas surpassé les autres modèles, XGBoost a joué un rôle important dans la démarche comparative.

3. Modèle RandomForest (base vs optimisé)
🔹 Modèle de base

F1 pluie : 0.600

Rappel pluie : 0.487

Précision pluie : 0.782

🔹 Modèle optimisé (RandomizedSearchCV)

F1 pluie : 0.611

Rappel pluie : 0.520

Précision pluie : 0.741

💡 L’optimisation a permis :

d’améliorer la détection des jours de pluie (rappel plus élevé),

d’obtenir un meilleur équilibre précision / rappel,

d’augmenter légèrement le score F1.

4. Modèle final retenu : HistGradientBoosting

Après comparaison de l’ensemble des modèles (baseline, XGBoost, RandomForest optimisé),
le modèle HistGradientBoostingClassifier a été choisi comme modèle final.

🔹 Performances finales :

Seuil optimisé : 0.69

F1 pluie : 0.69

Précision pluie : 0.71

Rappel pluie : 0.67

Accuracy globale : 0.86

🏁 Conclusion

Le RandomForest optimisé constitue un excellent modèle intermédiaire et interprétable.

Le HistGradientBoostingClassifier est retenu comme meilleur modèle final, offrant les meilleures performances globales sur la variable pluie.

L'ensemble des modèles testés a permis d'aboutir à une sélection justifiée et cohérente avec les objectifs du projet.
