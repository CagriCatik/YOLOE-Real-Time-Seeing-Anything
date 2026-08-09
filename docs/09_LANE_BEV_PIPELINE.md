# Spur-/BEV-Pipeline: Erkennung, Indizierung, Cut-In-Ereignisse

Proof of Concept auf **einem** Einzelbild der HCP3-HMI-Ansicht.
Stand: Einzelbild verifiziert, **nicht videofähig**. Die blockierenden Lücken
sind unten quantifiziert und über `adascope lane-sensitivity` reproduzierbar.

## Struktur

```text
adascope/lanes/detection.py   Stufe 1: Spurlinien in der Bildebene (Hough)
adascope/lanes/bev.py         Stufe 2: Bodenebene, Korridore, Footprints
adascope/lanes/indexing.py    Stufe 3: ego-relative Spurnummern
adascope/lanes/events.py      Stufe 4: temporale Ereignisse (Cut-In/Cut-Out)
adascope/lanes/pipeline.py    Sequenz-Pipeline: verkettet alle Stufen
adascope/detection/tracking.py  YOLO11 + ByteTrack, Bildebene
adascope/render/debug_views.py  die Renderer je Perspektive
adascope/render/camera.py       virtuelle Kamera auf die Bodenebene
adascope/cli/lane_debug.py    Debug-Videos aus mehreren Perspektiven
adascope/cli/sensitivity.py   Robustheitsmessungen (Belege für alle Zahlen)
config/lane.yaml · bev.yaml · ...   die gesamte Kalibrierung
test_images/test_frame_masked.png   Referenz-Frame
outputs/debug/                      erzeugte Visualisierungen
```

Ausführen:

```powershell
adascope lane-sensitivity --frame test_images/test_frame_masked.png
adascope lane-debug --source scenarien/lane_departure_3_lanes.mp4 --views all
```

Installation: `pip install -e .` — Abhängigkeiten stehen in `pyproject.toml`.

## YOLO11-Fahrzeugtracking

`adascope.detection.tracking` kapselt YOLO11 und ByteTrack. Das COCO-Modell erkennt
`car`, `motorcycle`, `bus` und `truck`; ByteTrack erzeugt persistente Track-IDs.
Das sichtbare Ego-Fahrzeug wird beim ersten Auftreten über eine normalisierte
Ankerzone im unteren Bildzentrum gewählt und anschließend über seine Track-ID
beibehalten. Alle übrigen Tracks sind Co-Fahrzeuge.

Schneller Test auf den vorbereiteten Frames:

```powershell
adascope track --source data/frames/cropped_lane_departure_warning --device 0 --max-frames 100
```

Ausgaben:

- `outputs/vehicle_tracks.mp4` — Boxen, Ego-/Co-Rolle und Track-ID
- `outputs/vehicle_tracks.csv` — pro Frame Box, Rolle, Confidence und
  Bbox-Unterkante als Eingang für `project_footprint()`

`yolo11n.pt` ist die schnelle PoC-Variante. Für die finale Auswertung kleiner
Fernfahrzeuge ist `yolo11s.pt` der empfohlene Genauigkeits-/Latenz-Kompromiss;
der Tracker benötigt kein eigenes neuronales Modell.

## Debug-Videos (`adascope lane-debug`)

Erzeugt pro Perspektive ein MP4, dazu `debug_metrics.csv` (Homographie-Zustand,
Korridor- und Spuranzahl, Spurbreite je Frame — die Messung aus „Nächste
Schritte" Punkt 1) und `debug_events.csv` mit den Ereignissen der State Machine.

```powershell
adascope lane-debug --views all --device 0
adascope lane-debug --source <...> --views dash --max-frames 300 --device 0
adascope lane-debug --source <...> --views bev,hist --no-detect   # ohne YOLO
adascope lane-debug --source <...> --scenario <name>   # Szenario-Kalibrierung
```

| Ansicht | Perspektive | wofür |
|---|---|---|
| `front` | Bildebene | Hough-Linien mit Rolle und Support, Tracks, **Bbox-Unterkante hervorgehoben**, Ereignislog |
| `mask` | Bildebene | Weißmaske, Canny, Hough-Segmente nach Cluster; verworfene Cluster grau |
| `bev` | Bodenebene von oben | Rohkorridore **gegen** die rekonstruierten ego-relativen Spuren |
| `hist` | Bodenebene + Plot | BEV-Maske über dem Spaltenhistogramm mit Peaks und `peak_min_pixels` |
| `smear` | Bodenebene von oben | **magenta** = Bbox-Fläche nach BEV gewarpt, **gelb** = projizierte Unterkante |
| `oblique` | virtuelle Schrägkamera | dieselbe Bodenebene aus 24° Neigung, Footprints als Bodenquader |
| `shoulder` | virtuelle Kamera rechts | Blick vom Standstreifen, Längsversatz sichtbar |
| `dash` | Komposit | front + bev + oblique + hist, darunter Zeitverlauf und Ereignisse |

Durchgängige Farbcodierung:

- **Grenzen** — grau = Rohpeak aus dem Histogramm, weiß = übernommene
  Spurgrenze, orange gestrichelt = virtuelle Grenze aus `split_corridors()`
- **Zustand** — grün `outside`, orange `encroaching`, rot `inside`,
  grau `invalid` (Footprint unplausibel, wird der State Machine vorenthalten)

`oblique` und `shoulder` sind keine neuen Messungen. Die Bodenebene ist durch
`H` bereits festgelegt; jede weitere Ansicht davon ist eine reine Homographie
und kostet einen zusätzlichen `warpPerspective`. Sie zeigen dieselben Daten aus
einem Winkel, in dem Fehler auffallen, die in der Draufsicht flach wirken —
vor allem die Footprint-Divergenz im Fernfeld.

`smear` animiert das Kernprinzip: die magentafarbene Fläche überdeckt mehrere
Spuren, die gelbe Unterkante bleibt in ihrer. Das ist `bev_smear_problem.png`
über die ganze Sequenz.

`bev` stellt die beiden Nummerierungen direkt gegenüber: die grauen Linien sind,
was das Spaltenhistogramm liefert, die weiß/orangen sind, was `lanes.indexing`
daraus macht. Der Standstreifen fällt sichtbar heraus — seine Breite ist kein
ganzzahliges Vielfaches der Spurbreite.

Der Zeitstreifen unter `dash` ist der Wirksamkeitsnachweis: Homographie-Zustand
als Farbband (grün `fresh`, orange `held`, rot `none`), darunter `len(corridors)`
in Weiß gegen die Anzahl ego-relativer Spuren in Cyan. Orange Marken am oberen
Plotrand sind Sprünge des **positionsbasierten** Ego-Index; springt Weiß, ohne
dass Cyan springt, hat die Indizierung einen Ausfall aufgefangen. Unten stehen
die Ereignisse der State Machine.

## Kernprinzip

Die Homographie ist **ausschließlich in der Bodenebene gültig**. Fahrzeuge
haben Bauhöhe — beim Warpen zerlaufen sie radial vom Kamerapunkt weg und
überdecken in BEV mehrere Spuren (siehe `outputs/bev_smear_problem.png`).
Eine Flächensegmentierung des Fahrzeugs im BEV ist deshalb strukturell falsch.

| Ebene | Zuständigkeit | Homographie |
|---|---|---|
| BEV | Fahrbahnbereiche, Spurkorridore | gültig |
| Bildebene | Fahrzeugdetektion (YOLOv8n Bbox) | ungültig |
| Brücke | **nur** die Bbox-Unterkante (Radaufstandslinie) | liegt in der Bodenebene |

Aus der projizierten Unterkante entsteht ein laterales Footprint-Segment.
Dessen Überlappung mit dem Ego-Korridor ergibt eine **kontinuierliche**
Einscher-Rate — der stetige Eingang für die temporale State Machine, kein
binäres Flag.

## Ablauf

1. Bildebene: die beiden durchgezogenen Randlinien per Hough → Homographie `H`.
2. Fahrzeug-Bboxen aus der Weiß-Maske ausstanzen (Detektor räumt die
   Spurmaske auf → kooperierende Module).
3. Maske nach BEV warpen, Spaltenhistogramm → **alle** Spurgrenzen.
   Gestrichelte Linien akkumulieren hier zu Säulen; genau die kurzen
   Fern-Dashes, die Hough in der Bildebene verliert, werden so stabil.
4. Bbox-Unterkanten projizieren → Footprint-Segment.
5. Überlappung Footprint × Ego-Korridor → Einscher-Rate → Status.

## Ergebnis auf dem Referenz-Frame

```
Spurgrenzen (BEV x): [81, 191, 266, 342, 420]
Korridorbreiten:     [110, 75, 76, 78]     # 110 = Standstreifen
Ego-Spur = L2

left_car   L1  ego_ratio=0.00  clear
truck      L3  ego_ratio=0.00  invalid   (Fernfeld-Artefakt)
far_car    L1  ego_ratio=0.33  invalid   (Fernfeld-Artefakt)
```

## Befunde

**Die Geometrie ist stabil.** Bei ±3 px Rauschen auf den Linienstützpunkten
bleiben Ego-Spur-Index und Korridoranzahl über 300 Durchläufe konstant.
Die Homographie ist nicht der schwache Punkt.

**Hough in der Bildebene verliert Spurlinien.** Die erste Version filterte
über `min_cluster_support=2` eine echte Grenze weg (Fern-Dashes bei
x≈563). Der Ego-Korridor war dadurch zwei Spuren breit und meldete das linke
Fahrzeug fälschlich als `in_ego_lane`. Deshalb liegt die Spurfindung jetzt im
BEV-Spaltenhistogramm.

**Die Rückprojektion divergiert im Fernfeld.** Gemessene Footprint-Breiten
gegen 76 px Spurbreite:

| Objekt | BEV y | Breite | Anteil Spur | |
|---|---|---|---|---|
| ego | 669 | 42.3 px | 0.56 | ok |
| left_car | 590 | 48.4 px | 0.64 | ok |
| truck | 273 | 73.6 px | 0.97 | Artefakt |
| far_car | 205 | 97.6 px | 1.28 | Artefakt |

Ein Fahrzeug kann keine 1.28 Spurbreiten belegen. `footprint_is_plausible()`
verwirft solche Samples als `invalid`, statt zu raten. Das definiert die
**nutzbare Detektionsreichweite** — deutlich kürzer, als das BEV-Bild suggeriert.

## Referenzszenario: lane_departure_3_lanes

```powershell
adascope lane-debug --source scenarien/lane_departure_3_lanes.mp4 `
                      --scenario lane_departure_3_lanes --views all --device 0
```

246 Frames, durchgehend dieselbe dreispurige Fahrbahn. Auf diesem Material
arbeitet die Geometrie sauber — es eignet sich als Regressionsreferenz:

| Größe | Wert |
|---|---|
| Homographie | 95 % `fresh`, 5 % `held`, 0 % `none` |
| Korridore | 4 in 245 von 246 Frames |
| Spuren (ego-relativ) | 3 in **allen** 246 Frames, keine virtuellen Grenzen |
| Spurbreite | Median 77 px (min 76) |
| Ego-Index-Sprünge | 2 = 0.8 % der Übergänge |
| Ereignisse | keine |

**Keine Ereignisse ist hier korrekt.** Pro Frame sind 2–5 Co-Fahrzeuge mit
plausiblem Footprint sichtbar, aber alle liegen durchgehend auf `rel=-1`. Kein
Fahrzeug wechselt je in die Ego-Spur — es gibt schlicht kein Einscheren.

**Was das Szenario prüft, kennt `lanes.events` noch nicht.** Das *Ego* verlässt
seine Spur, nicht ein Fremdfahrzeug. `EventKind` hat dafür keinen Wert, und
`_detect_ego_shift()` greift nicht: das Ego driftet innerhalb seiner Spur, die
ego-relativen Nummern der anderen verschieben sich also gar nicht.

Das Signal ist aber vorhanden und wird jetzt gemessen: `FrameAnalysis.ego_in_lane`
ist derselbe `ego_overlap()`, nur auf den Ego-Footprint angewandt. Er fällt in
**65 von 246 Frames** unter 1.00, Minimum 0.88 — genau dort, wo die HMI ihre rote
Warnlinie zeigt. In `front`, `bev`, `oblique` und `shoulder` färbt sich der
Ego-Footprint orange, sobald er die Spurgrenze berührt; im Zeitstreifen ist er
ein eigenes Band. Klassifiziert wird nichts — die Schwelle und der EventKind
dafür sind eine Design-Entscheidung, keine Messung.

## Befunde aus dem Videobetrieb

Gemessen über 2117 Frames von `data/frames/cropped_lane_departure_warning`
mit `adascope lane-debug`; alle Zahlen stehen in `outputs/debug/debug_metrics.csv`.

**Die Indexverschiebung ist real und `lanes.indexing` halbiert sie.** Sprünge
des positionsbasierten Ego-Index zwischen aufeinanderfolgenden gültigen Frames:

| Konfiguration | Index-Sprünge |
|---|---|
| `peak_min_distance=25` (vorher) | 167 = 12.0 % der Übergänge |
| `peak_min_distance=55` + `lanes.indexing` | 58 = 4.5 % der Übergänge |

**`peak_min_distance=25` war zu klein.** Der Wert ist ein Drittel der
schmalsten echten Spurbreite (75 px). Auf dem Referenz-Frame fällt das nicht
auf, im Video überlebten dadurch Kleinstkorridore von 25…42 px. Auf 55 angehoben
— der Referenz-Frame liefert bis `peak_min_distance=70` identische Grenzen,
das Einzelbild-Ergebnis ändert sich also nicht.

**`estimate_lane_width()` bricht an genau diesen Fehlpeaks.** Sie nimmt die
kleinste wiederkehrende Korridorbreite; mit den Kleinstkorridoren schätzte sie
30 statt 76 px, und in der Folge wurde jede echte Spur als „verschmolzen"
zerlegt. Nur 53 % der Frames hatten dann eine verwertbare Spurliste, nach der
Korrektur 100 % auf demselben Abschnitt.

**Die BEV-Skala ist nicht metrisch.** `x_left=81` / `x_right=419` bilden die
zwei gefundenen durchgezogenen Linien auf feste 338 px ab — unabhängig davon,
wie viele Spuren dazwischen liegen. Die Sequenz wechselt den Fahrbahntyp, und
die geschätzte Spurbreite folgt exakt diesem Quotienten:

| Korridore | Frames | Median Spurbreite | 338 / Spurbreite |
|---|---|---|---|
| 2 | 508 | 168 px | 2.0 |
| 3 | 101 | 72 px | 4.7 |
| 4 | 631 | 77 px | 4.4 |

Dieselben 76 px bedeuten je nach Abschnitt eine andere reale Breite. Deshalb ist
ein festes `--lane-width` über die ganze Sequenz falsch, und `multiple_tolerance`
kann einen Faktor 2 nicht auffangen. **Das ist die eigentliche Kalibrierlücke:**
die longitudinale Strichperiodik (18 m) liefert eine Skala, die nicht davon
abhängt, wie viele Spuren die Randlinien einschließen.

**Die Homographie ist die knappste Ressource.** Nur 39 % der Frames liefern
beide durchgezogenen Randlinien (`fresh`), 46 % laufen auf der gehaltenen
Homographie (`held`), 15 % haben keine (`none`). Ursachen im Video: zweispurige
Abschnitte ohne linke Randlinie, HMI-Popups („Emergency assist"), die die
Fahrbahn verdecken. Damit ist die Spurerkennung der Bildebene — nicht die
Belegungsstufe — der begrenzende Faktor.

**YOLO detektiert das Kombiinstrument.** Das stilisierte Fahrzeug-Icon im
unteren HMI-Bereich wird stabil als `car` erkannt und bekommt eine eigene
Track-ID — ein Dauer-Cut-In-Kandidat aus einem Bildschirmelement.
`road_vehicles()` verwirft Detektionen, deren Bbox-Unterkante unter der
Fahrbahn-Referenzlinie liegt.

**`build_homography()` ist videountauglich.** `{L.role: L for L in lines}`
wirft `KeyError`, sobald eine der beiden Rollen fehlt (63 % der Frames), und
behält bei mehreren Linien derselben Rolle stumm die *innere* — im ersten Frame
x=318 statt x=64. `outer_solid_pair()` in `lanes/pipeline.py` wählt explizit die
äußerste je Seite; die Einzelbild-Funktion bleibt unverändert.

## Befunde aus den synthetischen Szenen

Die Trajektorie vorzugeben und das Bild daraus zu erzeugen hat **drei Defekte
der State Machine** aufgedeckt, die auf unannotiertem Material als „keine
Ereignisse" erschienen und damit unsichtbar waren.

**Ein zügiger Spurwechsel fiel stumm aus.** `reached_encroaching` wurde nur
gesetzt, wenn `encroaching` als Zustand *bestätigt* war — also über
`confirm_frames` Frames hinweg. Ein Fahrzeug, das das Band zwischen
`thr_encroaching` und `thr_inside` in ein bis zwei Frames durchquert, erreichte
den bestätigten Zwischenzustand nie und erzeugte **kein cut_in**. Gemessen an
der synthetischen Fahrt: Überlappung 0.00 → 0.36 → 0.65 → 1.00, `rel` −1 → 0,
Ereignisse: keine. Jetzt zählt `encroach_frames` die *beobachtete* Anfahrt; die
Absicherung gegen „taucht mitten in der Ego-Spur auf" bleibt erhalten.

**Ein langsames Ausscheren wurde als Abbruch gemeldet.** Der `cut_out` verlangte
`prev == "inside"` beim bestätigten Übergang nach `outside`. Wer über
`encroaching` ausschert, hat dort aber `prev == "encroaching"` — und wurde als
`aborted` gemeldet, obwohl das Fahrzeug nachweislich in der Ego-Spur war. Jetzt
entscheidet `was_inside`.

**Zwei gleichsinnige Wechsel lösten einen falschen Ego-Spurwechsel aus.**
`_detect_ego_shift()` erkennt einen eigenen Spurwechsel daran, dass sich alle
ego-relativen Nummern gleichsinnig verschieben. Wechseln zwei Fremdfahrzeuge
gleichzeitig in dieselbe Richtung, sieht das identisch aus — und die
Unterdrückung löscht dann genau deren beide echten Ereignisse. Jetzt muss der
Ego-Footprint dabei auch wirklich seine Spurgrenze berühren
(`events.ego_departure_max`).

## Verbleibende Lücken

**Die Ereignisausbeute ist noch klein.** Über die Sequenz feuert die State
Machine zweimal `cut_out` und keinmal `cut_in`. Das ist plausibel — bei einer
Homographie-Verfügbarkeit von 39 % `fresh` fehlt schlicht die Grundlage — aber
es ist kein Nachweis, dass die Schwellen stimmen. Dafür braucht es einen
Abschnitt mit gesichertem Einscher-Ereignis als Referenz.

**Der eigene Spurwechsel kann nicht erkannt werden.** `_detect_ego_shift()`
verlangt `ego_shift_min_tracks=2` gleichsinnig verschobene Tracks. In 1090 von
2117 Frames ist außer dem Ego überhaupt kein Fahrzeug im Bild, in 398 genau
eines. Die Szenenebene kann damit auf diesem Material praktisch nie greifen —
der eigene Spurwechsel müsste stattdessen aus der Bewegung der Spurgrenzen
relativ zum Ego-Footprint abgeleitet werden, nicht aus den Fremdfahrzeugen.

**Das Spaltenhistogramm setzt gerade Spuren voraus.** In BEV projizieren sich
Kurven nicht auf vertikale Säulen; das Histogramm verschmiert, Peaks fallen aus.
Für Kurven: Sliding Windows plus Polynomfit 2. Grades.

**Frame-spezifische Konstanten.** `roi_polygon`, `y_bottom=295`, `y_top=55`
sind auf genau diesen Screenshot getuned, inklusive Letterbox-Balken.

## Nächste Schritte

Erledigt: Messung über die Videosequenz (`adascope lane-debug`, Zahlen oben),
YOLO11 + ByteTrack (`vehicle_tracking.py`), ego-relative Nummerierung statt
Boundary-IDs (`lane_indexing.py`), Homographie halten bei Ausfall
(`HomographyTracker`, Gültigkeitszähler `held_frames`).

1. **Metrische Skala aus der Strichperiodik.** Der stärkste Hebel: er löst die
   Skalen-Mehrdeutigkeit, macht `estimate_lane_width()` überflüssig und damit
   `lanes.indexing` unabhängig vom Fahrbahntyp.
2. **Homographie-Verfügbarkeit erhöhen.** 39 % `fresh` ist die Obergrenze für
   alles Nachgelagerte. Randlinien auf zweispurigen Abschnitten sauber
   zuordnen, HMI-Popup-Bereiche aus der ROI nehmen.
3. **Eigenen Spurwechsel aus der Spurgeometrie** statt aus Fremdfahrzeugen
   ableiten — siehe „Verbleibende Lücken".
4. Homographie zeitlich glätten (bisher wird nur gehalten, nicht gefiltert).
5. Sliding Windows für Kurven.
6. Referenzabschnitt mit gesichertem Cut-In annotieren, um `thr_encroaching`,
   `thr_inside` und `confirm_frames` gegen Grundwahrheit zu prüfen.

## Kalibrierung

Die gesamte Kalibrierfläche steckt in `config/*.yaml` (siehe ADR-0002). Zwei
Größen lassen sich aus dem Bild selbst ableiten, ohne Kamerahöhe oder
Nickwinkel zu kennen:

- **lateral:** die gemessene Spurbreite (hier 76 px)
- **longitudinal:** die Strichperiodik (deutsche Autobahn: 6 m Strich +
  12 m Lücke = 18 m Periode)
