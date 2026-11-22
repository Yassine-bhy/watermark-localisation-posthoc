# Résumé de la recherche — localisation de watermark (dim. 22 nov. 2026)

Rapport complet : `research/REPORT.md`. État des lieux : `research/00_state.md`. Scripts de test : `research/tests/`.
Notes bibliographiques : `research/research_notes/Post hoc watermark localisation routes/`.

Rien n'a été soumis au leaderboard ; aucun fichier existant ni job en cours n'a été modifié.

## Verdict

**0.80 n'est pas plausible depuis l'état actuel (CV 0.653, public 0.6456) ; 0.70–0.72 l'est**, par deux routes qui
ont passé leur test. Les deux n'agissent que sur le régime partial. Le régime full (la moitié des images, à
0.60–0.67) n'a eu aucun test réussi : même avec partial à 0.90, on plafonne vers 0.76. Une équipe à 0.78 a forcément
full vers 0.70, et nous n'avons pas trouvé comment.

## Top 3

1. **Inférence zoomée pour les rectangles d'outpainting.** Le watermark y a été rétréci avec la photo d'origine
   (facteur 3 à 6), donc le segmenteur ne le voit pas à la résolution native. En découpant l'image en tuiles agrandies
   ×3, l'AUC sur les rectangles passe de 0.68 à 0.91 pour TrustMark et de 0.61 à 0.81 pour StegaStamp. Le zoom dégrade
   les objets et les full : il faut le router, pas le moyenner.
2. **SAM cliqué au pic de notre carte, pour les objets transplantés.** La carte sait où est le watermark
   (AUC 0.94–0.98) mais son contour est flou ; SAM le dessine. Gain mesuré sur ces images : PixelSeal 0.643 → 0.730,
   TrustMark 0.682 → 0.793, StegaStamp 0.708 → 0.804. Ce gain n'est pas en validation croisée (le seuil de référence
   est ajusté sur les mêmes images).
3. **Régime full : modèle spécialisé « trous » (non testé).** Le docstring de `wmloc/data.py` dit que les trous sont
   des pixels de l'original propre restaurés, pas du contenu généré. D'où l'idée d'entraîner un modèle uniquement sur
   ce type de trous, en sortie 128×128. C'est une hypothèse, sans mesure.

En complément du point 1 : les pics de la carte zoomée devraient servir de graines à la recherche de boîtes
StegaStamp / PixelSeal, avec des pas de 4 px. Le pic du score fait environ 4 px de large et `seedrect.py` utilise 8 px.

## Tests de vérification

| test | résultat |
|---|---|
| Zoom par tuiles (`test_zoom.py`, TrustMark et StegaStamp ; PixelSeal non lancé) | **réussi** |
| SAM au pic de la carte (`test_samprompt.py`) | **réussi** sur les objets, échoué sur full (−0.03) |
| TruFor pré-entraîné sur les trous (`test_trufor.py`) | échoué (AUC 0.44–0.50) |
| Couture visible aux bords des rectangles (`test_seam.py`) | échoué |
| Rectangle de somme maximale sur les cartes actuelles (`test_maxrect.py`) | échoué (à refaire sur cartes zoomées) |
| Pics de la carte actuelle comme graines (`test_seedhit.py`) | échoué (pic dans le rectangle 23–37 % du temps) |
| Ré-embedding du message complémentaire + régression (`test_counterfactual.py`) | échoué (AUC 0.56–0.67 contre 0.9 pour notre carte) |
| Corrélation locale avec le motif ré-embarqué (`pattern_test.py`, 30 images par méthode) | échoué (AUC 0.49–0.53) ; absent de `REPORT.md` |

Les seuils de réussite étaient écrits dans le docstring de chaque script avant l'exécution.

## Points à savoir

- **Origine des données : inconnue.** Aucun script de génération public. Le benchmark « What Breaks Local
  Watermarks? » existe (arXiv 2609.16832) mais n'est pas la source : pas de StegaStamp, uniquement SD 1.5.
- **Contradiction avec le brief.** Il disait « StegaStamp 0 % de décodage, recherche plate ». Les logs montrent
  l'inverse : la vraie boîte bat la boîte trouvée dans 66 % des cas ; c'est la recherche qui rate un pic très étroit.
- **Fiabilité des sources.** Les chiffres des papiers ont été lus par des sous-agents via un outil de résumé ; à
  revérifier avant de les citer ailleurs.
- **Effectifs faibles.** Le test zoom repose sur 15 à 18 rectangles tenus à l'écart par méthode.

## Prochaine étape

Lancer `research/tests/test_zoom.py pixelseal` (environ 12 min sur un GPU), puis intégrer le zoom dans `e2e.py` et
`seedrect.py` comme décrit en section 3.1 de `REPORT.md`.
