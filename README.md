# Optimiseur de quadrilatères articulés pour le DADA Engine

Ce programme recherche et affine les dimensions d'un mécanisme plan à quatre
barres destiné à commander les pistons du [DADA Engine](https://dada-engine.org).
L'objectif cinématique est d'obtenir une phase rapide entre les deux extrema de
la course, une fermeture lente et un plateau suffisamment immobile du côté du
piston vide.

Le calcul est organisé en deux niveaux :

1. une exploration globale discrète de nombreuses géométries et, pour F,
   positions de point d'attache ;
2. un raffinement local continu de chaque candidat retenu.

Les deux familles de point d'attache sont classées séparément :

- **E** : mécanisme classé directement à partir du déplacement angulaire du
  culbuteur `CD`. Après raffinement, `CDE` est calculé avec `DE = CD` afin que
  DE soit orthogonal à X au milieu de la course angulaire ;
- **F** : point solidaire du coupleur `BC`, défini par ses coordonnées locales
  `(local_x, local_y)` dans le repère de `BC`.

Le bâti `AD` est fixe. Les dimensions optimisées sont la manivelle `AB`, le
coupleur `BC` et le culbuteur `CD`.

## Installation

Le projet déclare Python 3.12 ou plus récent et dépend de NumPy, SciPy,
Matplotlib et Pillow.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Sous Debian, Pillow peut aussi être fourni par le paquet `python3-pil`.
Pillow est utilisé directement par Matplotlib pour produire les GIF ; aucun
paquet nommé `pillowwriter` n'est nécessaire.

## Utilisation rapide

Les recherches E et F sont volontairement disjointes. Chaque commande enchaîne
automatiquement l'exploration globale et le raffinement local :

```bash
python3 src/main.py E
python3 src/main.py F
```

La chaîne E produit `results_E.npz` puis `refined_results_E.npz`. La chaîne F
produit `results_F.npz` puis `refined_results_F.npz`. Le premier fichier est un
intermédiaire conservé pour la traçabilité ; le second est le résultat final à
consulter.

Pour une validation rapide de toute la chaîne :

```bash
python3 src/main.py E --low-def --limit-refinement 1
python3 src/main.py F --low-def --limit-refinement 1
```

Les noms des fichiers peuvent être choisis explicitement :

```bash
python3 src/main.py E \
    --level1-output calcul-E.npz \
    --output calcul-E-raffine.npz
```

Une recherche en haute définition peut durer plusieurs heures. La réduction du
disque F à `1,5 × BC` divise déjà fortement le nombre de supports par rapport à
l'ancienne recherche.

Le point d'entrée `refine.py` reste disponible pour reprendre séparément un
fichier intermédiaire compatible, mais il n'est plus nécessaire dans le flux
normal.

### Raffinement local

Le niveau 2 affine uniquement la géométrie :

- E : `AB`, `BC` et `CD` ;
- F : `AB`, `BC`, `CD`, `local_x` et `local_y`.

`AD` reste fixe. La recherche est une descente par coordonnées : les paramètres
sont mélangés à chaque tour, les déplacements positifs et négatifs sont testés,
et toute amélioration est acceptée immédiatement. Après convergence à un pas
donné, tous les pas sont divisés par cinq.

Chaque essai emprunte exactement la chaîne cinématique, les filtres et le score
du niveau 1. Le raffinement est continu, mais reste à l'intérieur du domaine
géométrique exploré au niveau 1 : bornes des longueurs et disque de recherche
de F. Cela évite les solutions dégénérées obtenues en
laissant, par exemple, `AB` tendre vers zéro.

Après raffinement de E, CDE est calculé sans participer au score. Le programme
choisit le côté `+Y` : lorsque le culbuteur se trouve au milieu de ses deux
angles extrêmes, DE est vertical vers le haut et `DE = CD`.

Le score final reste calculé à la résolution angulaire du niveau 1 afin de
rester comparable au score source. Après optimisation, une courbe à 0,1° est
calculée séparément pour les métriques physiques et l'affichage.

## Visualisation des résultats raffinés

Le même visualiseur final accepte les fichiers E et F. Avec un nom de fichier
seul, il affiche la liste classée :

```bash
python3 src/refined_visualizer.py refined_results_E.npz
```

Pour utiliser un autre fichier ou tracer un candidat :

```bash
python3 src/refined_visualizer.py refined_results_E.npz E 0
python3 src/refined_visualizer.py refined_results_F.npz F 0
```

Le graphique superpose :

- le déplacement normalisé `f(theta)` ;
- le mécanisme opposé `f(-theta mod 360)` ;
- les repères idéaux A1, A2 et A3 employés par le score ;
- les limites des plateaux physiques des deux pistons ;
- les intervalles d'échange pendant lesquels aucun piston n'est sur son
  plateau vide.

Il rappelle également la géométrie, le score avant et après raffinement, la
course utile, la largeur du plateau, les échanges et les précompressions.

## Animation GIF

Une animation d'un candidat raffiné peut être produite avec :

```bash
python3 src/animate_mechanism.py refined_results_E.npz E 0
python3 src/animate_mechanism.py refined_results_F.npz F 0 --output candidat-F0.gif
```

Elle représente les deux quadrilatères montés en opposition, leur manivelle et
leur bâti communs, les deux points d'attache, les bielles de piston, les pistons,
les cylindres ouverts et les deux courbes de déplacement. Le dernier photogramme
précède le premier du même pas angulaire que tous les autres, ce qui assure une
boucle régulière.

Options disponibles :

```text
--output FICHIER
--fps VALEUR
--angle-step DEGRÉS
--piston-rod-length LONGUEUR
--cylinder-1-width LARGEUR
--cylinder-2-width LARGEUR
--show-traces
--hide-symmetric
```

Les valeurs par défaut destinées à uniformiser tous les GIF se trouvent au
début de `src/animate_mechanism.py`. Les réglages actuels donnent une image de
500 × 618 pixels, avec le mécanisme sur les 309 pixels supérieurs et le graphe
sur les 309 pixels inférieurs, 25 images par seconde et un pas de 4°.

## Cinématique et machine symétrique

Le quadrilatère est défini par :

```text
A : pivot de la manivelle sur le bâti
B : articulation manivelle-coupleur
C : articulation coupleur-culbuteur
D : pivot du culbuteur sur le bâti
```

L'axe X suit `AD`. La cinématique choisit une branche cohérente de
l'intersection des cercles définissant C.

Le score étudie un seul mécanisme. Sa variable est l'angle du culbuteur pour E
et la projection X du point d'attache pour F :

```text
x1(theta) = f(theta)
```

Le second mécanisme n'intervient jamais dans le score. Il est construit après
l'optimisation par opposition exacte :

```text
x2(theta) = f(-theta mod 360)
```

Cette symétrie sert uniquement à décrire la machine complète, ses deux plateaux,
ses phases d'échange et ses précompressions.

## Filtres du niveau 1

Les paramètres sont centralisés dans `src/config.py`.

### Géométrie

Les mécanismes impossibles et ceux qui ne satisfont pas la condition de Grashof
pour un montage manivelle-culbuteur sont rejetés avant le calcul des supports.

Pour F, le support reste dans un disque de rayon `1,5 × BC` centré sur le
milieu de BC. Son déplacement doit également vérifier :

```text
course_Y(F) <= course_X(F)
```

Cette condition est appliquée sur la grille globale et à chaque essai du
raffinement.

### Plateau cinématique

Le filtre recherche, autour de l'extrémum attendu près de 90° ou 270°, un
intervalle continu dont l'amplitude de la variable notée reste sous une
fraction configurée de sa course. Cette variable est l'angle du culbuteur pour
E et la projection X de F. Sa largeur et la position de son centre doivent
respecter les bornes du fichier de configuration.

Ce plateau de filtrage ne doit pas être confondu avec :

- le plateau de la loi idéale utilisé pour noter la forme ;
- le plateau physique réel, recalculé à haute résolution après raffinement.

### Phase rapide et A3

A3 est l'extrémum opposé au plateau, détecté sur la courbe réelle. La phase
rapide doit être centrée au voisinage de 0° ou 180°.

A3 doit en outre rester proche de `+40°` ou `-40°`, avec une tolérance de 25° :

```text
A3 dans [15°, 65°] U [295°, 345°]
```

Cette contrainte maintient une véritable phase rapide et empêche A1 et A2 de se
confondre lorsque A3 dérive vers 90° ou 270°.

## Loi idéale et score

A1 et A2 ne sont pas optimisés. Ils sont déduits de l'A3 réel du candidat :

```text
A1 = (180° + A3) mod 360°
A2 = (360° - A3) mod 360°
```

Le sens de lecture dépend du demi-cycle contenant A3 :

```text
A3 <= 180° : A2 -> A3 -> A1, puis plateau A1 -> A2
A3 >  180° : A1 -> A3 -> A2, puis plateau A2 -> A1
```

La variable réelle — angle du culbuteur pour E, projection X pour F — est
normalisée puis comparée à la même loi idéale sur trois zones : plateau, phase
rapide et phase lente. Les erreurs RMS donnent une note de forme. Une note
d'accélération compare séparément les trois transitions ; la transition
adjacente à la phase lente reçoit un poids double.

Le score de base est :

```text
score_base = 0.35 * shape_quality + 0.65 * acceleration_quality
```

Une note supplémentaire mesure l'antisymétrie centrale de la phase rapide,
après normalisation du déplacement entre -1 et 1. Une symétrie parfaite vérifie
`f(theta) = -f(-theta)`. La dissymétrie est convertie en décalage angulaire
équivalent ; le score de symétrie devient nul à 12° par défaut.

```text
score_final = score_base * symmetry_quality
```

Les composantes détaillées du score sont conservées dans les deux fichiers NPZ.

## Métriques physiques du niveau 2

Après le raffinement, le plateau réellement utilisable est détecté localement
autour de l'extrémum du piston vide sur la courbe à haute résolution. Sa bande
d'amplitude est indépendante du filtre du niveau 1.

Le fichier raffiné conserve notamment :

- la course utile ;
- le début, la fin et la largeur du plateau réel ;
- le plateau symétrique exact ;
- les deux durées d'échange ;
- les deux précompressions.

Pour chaque échange, la précompression mesure la fermeture déjà effectuée par
le piston plein au moment où l'autre piston termine son plateau vide :

```text
precompression_ratio = 1 - volume_normalized_at_exchange_start
```

Le volume normalisé vaut zéro lorsque le piston est vide et un lorsqu'il est
plein. Aucun volume mort n'est encore pris en compte.

Les métriques vérifient géométriquement :

```text
2 * largeur_plateau + échange_1 + échange_2 = 360°
```

## Diversité et déduplication

E et F utilisent des voisinages distincts. Les longueurs `AB`, `BC` et `CD`
sont comparées relativement à leur taille, et non plus avec une tolérance fixe
en millimètres. Pour F, les coordonnées du support sont d'abord normalisées par
BC avant comparaison :

```text
local_x / BC
local_y / BC
```

Après le niveau 1, un voisinage assez large évite qu'une seule famille de
géométries remplisse le classement. Après le raffinement, une seconde passe
plus fine fusionne les candidats qui ont réellement convergé. Dans chaque
voisinage, le meilleur score est toujours conservé.

Les seuils actuels sont tous regroupés dans `src/config.py` : 8 % sur les
longueurs et 0,25 sur les coordonnées normalisées de F au niveau 1 ; 2 % et
0,05 après raffinement.

## Configuration

Les principaux réglages modifiables dans `src/config.py` sont :

- mode haute ou basse définition ;
- résolution angulaire des deux niveaux ;
- grilles de `AB`, `BC` et `CD` ;
- nombre maximal de solutions conservées ;
- seuils et largeurs du plateau ;
- fenêtre autorisée pour A3 ;
- rayon et pas de la grille F ;
- tolérances de diversité propres aux deux niveaux ;
- graine aléatoire du raffinement.

Toute modification des grilles, des filtres ou de la formule du score nécessite
de régénérer le fichier `results_*.npz` avant de lancer le niveau 2. Un ancien
fichier NPZ ne doit pas être raffiné avec une configuration d'évaluation
différente : le programme contrôle que le score source peut être reproduit
numériquement.

Les fichiers produits avant le passage de E au déplacement angulaire sont donc
incompatibles avec cette version et doivent être recalculés.

## Tests

Depuis la racine du dépôt :

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Les tests couvrent notamment le déplacement angulaire de E, le calcul final de
CDE, les relations A1/A2/A3, le disque et la contrainte verticale de F, les
deux déduplications, le domaine du raffinement, `DE = CD`, la symétrie exacte
`f(-theta)`, la précompression et la lecture/écriture du fichier final.

## Organisation du code

```text
src/config.py               paramètres des deux niveaux
src/mechanism_generator.py  génération des quadrilatères
src/kinematics.py           résolution et trajectoires E/F
src/support_search.py       grilles des points d'attache
src/objective.py            filtres, loi idéale et score
src/deduplication.py        diversité géométrique E/F aux deux niveaux
src/optimizer.py            explorations globales disjointes E/F
src/result_io.py            formats intermédiaires et raffinés NPZ
src/refinement.py           descente locale et métriques physiques
src/main.py                 enchaînement des deux niveaux pour E ou F
src/refine.py               reprise manuelle du niveau 2
src/refined_visualizer.py   inspection de la machine raffinée
src/animate_mechanism.py    génération des animations GIF
tests/                      tests ciblés
```

## État du projet

Le solveur couvre actuellement la recherche géométrique, le raffinement local,
la description cinématique de deux mécanismes opposés et la production de
graphiques et d'animations. Les contraintes mécaniques détaillées et le modèle
thermodynamique complet ne sont pas encore intégrés.

## Licence

Ce programme est placé sous **CC0 1.0 Universel** : son auteur renonce, dans
toute la mesure permise par la loi, à ses droits d'auteur et droits voisins sur
le projet. Le code peut donc être copié, modifié, redistribué et utilisé, y
compris commercialement, sans demander d'autorisation.

Voir le fichier [`LICENSE`](LICENSE) et le
[texte officiel CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
