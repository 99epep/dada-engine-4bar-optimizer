# dada-engine-4bar-optimizer
DADA Engine 4-Bar Optimizer

Présentation

Ce projet développe un solveur Python destiné à optimiser un mécanisme à quatre barres utilisé dans le projet DADA Engine.

L'objectif est de déterminer automatiquement les dimensions optimales du mécanisme afin d'obtenir la trajectoire souhaitée d'un point de travail E, tout en respectant les contraintes géométriques et cinématiques.

Contrairement aux optimiseurs classiques, le point E n'est pas limité au segment CD : il peut être situé à une distance quelconque de cette bielle, ce qui permet d'explorer un espace de conception beaucoup plus vaste.

Objectifs de la version 0.1

- Modéliser un quadrilatère articulé plan.
- Résoudre sa cinématique complète.
- Définir librement la position du point E.
- Calculer la trajectoire de E sur un cycle complet.
- Fournir une fonction objectif pour l'optimisation.
- Préparer l'intégration des contraintes thermodynamiques dans les versions suivantes.

Technologies

- Python 3.12+
- NumPy
- SciPy
- Matplotlib

Architecture

- "geometry.py" : géométrie du mécanisme.
- "kinematics.py" : résolution cinématique.
- "objective.py" : fonction coût.
- "constraints.py" : contraintes géométriques.
- "optimizer.py" : algorithme d'optimisation.
- "main.py" : exemple d'utilisation.

Feuille de route

V0.1

Solveur géométrique et cinématique.

V0.2

Optimisation automatique des longueurs des barres et de la position du point E.

V0.3

Ajout des contraintes mécaniques.

V0.4

Couplage avec le modèle thermodynamique du DADA Engine.

Licence

À définir.