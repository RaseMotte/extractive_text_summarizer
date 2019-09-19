 Notice pour les résumés.

    Created by Sebastien GADIOUX, last modified on Aug 23, 2019


# Setup

```
pip install -r requirements.txt
```

# TODO

- [ ] Wtf is there a linear kernel applied to tfidf matrix ?

- [ ] Burutal tokenizer to 10 - meilleur résultat avec 2 askip
- [ ] html + meta open graph
- [ ] order of sentences in document
- [ ] RageRank Cython

- [ ] HTML balise
- [ ] bias to matrice


# Resume_Interface.ipynb

Le fichier principal est Resume_Interface.ipynb.

La première cellule est la cellule d'import des librairies et d'exécution.

La deuxième est une cellule conçue pour afficher une version html du texte à
résumer avec une coloration différentes pour les phrases choisies.

Les deux cellules suivantes sont conçues pour permettre d'extraire les données
des corpus à résumer ainsi que  les résumés gold.



La cellule suivante est celle pour choisir les méthodes utilisées pour
calculer les résumés.

Vient ensuite la cellule pour initialiser le calcul des scores.

La cellule suivante est la cellule de calcul principale

La cellule suivante est celle pour l'utilisation de la librairies sumy.

La dernière cellule est celle pour l'affichage graphique des résultats.



 # Process_Summary.ipynb


Le fichier secondaire Process_Summary.ipynb contient toutes les méthodes pour
générer les résumés à partir des résultats des méthodes de résumés.


Les méthodes de calculs sont toutes dans le dossier Summary_Processes

Les méthodes de calcul de résumé sont des classes devant avoir au moins ces
deux fonctions :

preprocess(corpus) : où l'intégralité du corpus est passé pour faire du
préprocessing (TFIDF...), ne renvoie rien.

summarize(corpus) : où juste le document est passé sous la forme d'une liste
de phrases. La méthode renvoie une liste de nombre de même taille que le
nombre d'élément que le corpus où chaque nombre est un score de
représentativité du document (le + le mieux).
