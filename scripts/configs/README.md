# Konfiguration der Stufenskripte

Dieser Ordner ist eine unabhaengige Kopie von `configs/` fuer die Skripte
`stage_01` bis `stage_08`. Aenderungen hier beeinflussen die normale
`adascope`-Pipeline nicht.

Die wichtigsten Dateien fuer die Bildverarbeitung sind:

- `lane.yaml`: ROI, Weissmaske, Canny, Hough, Clustering und Linienfit
- `bev.yaml`: Auswahl der Richtungsfahrbahn, Homographie-Ziel und Peaks
- `pipeline.yaml`: Halten und Glaetten der Homographie
- `windows.yaml`: alternative Spurgrenzensuche in der BEV
- `scripts.yaml`: Quelle, Framezahl, Snapshots, Videos und Ausgabeordner

Die neuen Geometrie-Schutzschichten werden hier kalibriert:

- `lane.cluster_method: union_find` und `lane.cluster_max_*` steuern die
  paarweise Hough-Kompatibilitaet;
- `pipeline.homography_max_point_jump`, `*_width_change_ratio` und
  `*_vanishing_jump` entscheiden, ob ein neues Paar `fresh` wird oder die
  vorige Homographie als `held` bestehen bleibt;
- Stage 3 zaehlt Ablehnungsgruende pro Segmentpaar, Stage 4 zeigt abgelehnte
  Kandidaten rot und schreibt den Grund in die CSV.

Die Stufenskripte verwenden diesen Ordner standardmaessig. Die Hauptkonfiguration
kann weiterhin explizit ausgewaehlt werden:

```powershell
python scripts/stage_04_homography.py --config configs
```

Wenn eine Verbesserung aus den Stufen in die Hauptpipeline uebernommen werden
soll, muss der betreffende Wert bewusst nach `configs/` uebertragen werden.
