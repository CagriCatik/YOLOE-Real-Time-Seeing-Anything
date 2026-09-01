# Einzelschritt-Skripte der Spurpipeline

`adascope scenarios` laesst die ganze Kette laufen und sagt, wie gut es lief.
Wenn eine Stufe schlecht arbeitet, sagt es nicht **welche**. Diese Skripte
schneiden je eine Stufe heraus — gleiche Quelle, gleiche Konfiguration, Ausgabe
genau eines Zwischenstands als Bild, Video und CSV.

| Skript | Stufe | Stellschrauben in `configs/` |
|---|---|---|
| `stage_01_mask.py` | Weissmaske und ROI | `lane.white_l_min/max`, `roi_polygon` |
| `stage_02_segments.py` | Kanten und Hough-Segmente | `lane.canny_*`, `hough_*` |
| `stage_03_lines.py` | Cluster, Linienfit, Rollen | `lane.cluster_*`, `robust_*` |
| `stage_03a_cluster_compare.py` | Greedy gegen Union-Find | `lane.cluster_*` |
| `stage_04_homography.py` | Randlinienpaar und H | `pipeline.homography_*`, `bev.min_pair_separation` |
| `stage_04a_temporal_gate.py` | Rohkandidat gegen temporal akzeptierte H | `pipeline.homography_*` |
| `stage_05_bev.py` | gewarpte Spurmaske | `bev.width/height`, `x_left/x_right` |
| `stage_06_histogram.py` | Spaltenhistogramm, Peaks | `bev.peak_*`, `histogram_blur` |
| `stage_07_corridors.py` | Korridore, eigene Fahrbahn, Footprints | `indexing.*` |
| `stage_08_events.py` | State Machine | `events.*` |
| `stage_09_perception_eval.py` | manuelle Wahrnehmungs-Ground-Truth | `ground_truth/*.yaml` |

Annotationen fuer Stufe 9 werden bewusst manuell erzeugt:

```powershell
python scripts/annotate_perception.py lane_departure_3_lanes --frames 0,60,120,180,245
python scripts/stage_09_perception_eval.py --source scenarien/lane_departure_3_lanes
```

Der erste Befehl schreibt nur einen `*.perception_draft.yaml`-Entwurf. Er wird
nicht automatisch als Ground Truth aktiviert; erst nach Sichtung werden die
Eintraege in die kanonische `ground_truth/<szenario>.yaml` uebernommen.

## Vergleichsstufen fuer die neuen Geometrie-Verfahren

Die beiden `a`-Stufen sind bewusst getrennte Diagnoseprogramme. Sie rufen
dieselben Funktionen wie die Produktions-Pipeline in `adascope` auf, damit
Darstellung und reales Ergebnis nicht auseinanderlaufen.

```powershell
python scripts/stage_03a_cluster_compare.py --frames 120
python scripts/stage_04a_temporal_gate.py --frames 300
```

- `stage_03a`: Links steht das alte, reihenfolgeabhaengige Greedy-Clustering,
  rechts das deterministische Union-Find-Clustering. Beide Seiten erhalten
  exakt dieselben Hough-Segmente. Farben markieren Cluster, weisse Linien die
  daraus behaltenen Fits.
- `stage_04a`: Links steht der ungefilterte Homographie-Kandidat des aktuellen
  Frames, rechts die vom zeitlichen Gate verwendete Geometrie. Ein roter
  Kandidat links und `HELD` rechts zeigt direkt, dass ein Geometriesprung
  verworfen wurde.

Neben Bildern und Video schreiben beide Stufen eine CSV mit den relevanten
Metriken. Damit lassen sich Grenzwerte in `scripts/configs/` nachvollziehen,
ohne nur auf das fertige Overlay zu vertrauen.

`stage_03` und `stage_03a` zeigen absichtlich alle Bildlinien als Kandidaten.
Das bedeutet nicht, dass jede Linie weitergegeben wird. In `stage_05` zeigt
`03_direction_filter_overlay` gruen weitergegebene und rot ausgeschlossene
Pixel; `04_driving_area_mask` ist der verbindliche Downstream-Eingang. Nur
Pixel zwischen den akzeptierten Grenzen der eigenen Richtungsfahrbahn werden
nach BEV gewarpt und vom Histogramm gezaehlt.

Jedes Skript nennt beim Start die Schluessel, die es beeinflussen. Die
Stufenskripte verwenden standardmaessig ihre unabhaengige Konfigurationskopie
in `scripts/configs/`. Damit koennen Parameter ausprobiert werden, ohne die
Hauptkonfiguration in `configs/` zu veraendern. Mit `--config configs` laesst
sich weiterhin bewusst die Hauptkonfiguration verwenden.

```powershell
python scripts/stage_06_histogram.py --source scenarien/acc_plus_3 --frames 200
```

## Voreinstellungen statt Tipparbeit

Alle Skriptparameter stehen in **`scripts/configs/scripts.yaml`** — Quelle, Framezahl,
Ausgabeort, Video ja/nein, dazu die Schwellen der Modell-Testskripte.

```
Rangfolge:  CLI-Flag  >  Abschnitt der Stufe  >  defaults  >  eingebauter Wert
```

Ein Flag gewinnt also immer. Was aus der Datei kam, sagt das Skript beim Start:

```
  Frames        60
  (aus scripts/configs/scripts.yaml: config, fps, source)
```

Damit laesst sich die uebliche Quelle einmal eintragen und danach nur noch
`python scripts/stage_06_histogram.py` aufrufen. Je Stufe sind abweichende
Werte moeglich — `stage_04` laeuft laenger, weil `basis_delta` erst ueber
Laufzeit aussagekraeftig wird; `stage_08` laeuft ueber alle Frames, weil ein
Manoever in Frame 300 sonst durchfaellt.

Fehlt die Datei, laufen die Skripte mit den eingebauten Werten weiter.

Gemeinsame Argumente: `--source`, `--config` (Standard `scripts/configs`), `--frames`
(0 = alle), `--out`, `--no-video`.

`stage_05` bis `stage_08` laden zusaetzlich das Detektionsmodell. Stufe 5 und 6
brauchen es, weil die echte Pipeline Fahrzeugboxen vor dem BEV-Warp aus der
Weissmaske entfernt. Mit `--no-detect` zeigen sie bewusst die Rohmaske. Bei
`stage_07` laesst sich das mit `--no-detect` abschalten — dann laeuft es ohne
Modell, aber **auch ohne Ego-Bezug**: ohne Ego-Footprint faellt die Auswahl der
eigenen Fahrbahn auf einen Ersatzpunkt zurueck, und die Korridore sind nicht
mehr dieselben, die der Szenariolauf sieht. Zum Kalibrieren von `indexing.*`
also mit Detektion arbeiten.

Ausgabe je Stufe in `outputs/stages/<stufe>/`: die ersten Frames als PNG,
benannte Zwischenbilder, ein Video, eine CSV mit den Kennzahlen und am Ende
Median und Spanne je Spalte. Die Zahl der PNG-Frames steuert `snapshots` in
`scripts/configs/scripts.yaml` oder `--snapshots` (0 = aus).

## Reihenfolge beim Suchen

Von unten nach oben. Eine Stufe kann nicht besser sein als ihr Eingang:

1. **`stage_01`** — ist ueberhaupt etwas in der Maske? Wenige Prozent der ROI
   sind richtig; nahe null oder zweistellig ist beides falsch.
2. **`stage_04`** — steht die Homographie? `basis_delta` ist das Wackeln in
   Zahlen: der Abstand der beiden Stuetzpunkte, auf den die BEV-Skala normiert
   ist. Springt er, atmet die gesamte Bodenebene. `ablehnungsgrund` erklaert,
   warum ein Kandidat rot verworfen und die vorige Geometrie gehalten wurde.
3. **`stage_06`** — stimmt die Zahl der Peaks? `abstand_min` deutlich unter der
   Spurbreite heisst: `peak_min_distance` zu klein.
4. **`stage_07`** — `n_korridore` gegen `n_spuren`. Grosse Differenz heisst,
   dass viel jenseits der eigenen Fahrbahn liegt oder verworfen wird.
   `n_synthetisch` > 0 heisst, dass Spuren aus verschmolzenen Korridoren
   **erfunden** wurden — eine Annahme, keine Messung.

   Die Footprints (gelb das Ego, cyan die Fremdfahrzeuge) sind die auf den
   Boden projizierten Bbox-Unterkanten. Nur sie liegen in der Bodenebene;
   deshalb ein Segment und kein Rechteck. Liegt ein Footprint neben allen
   Korridoren, ist entweder die Homographie schief (`stage_04`) oder die
   Spurbreitenschaetzung falsch.
