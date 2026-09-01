# Review der Bildverarbeitungsstufen

Stand: 2026-08-09. Untersucht wurde zuerst
`scenarien/lane_departure_3_lanes`, jeweils an den ersten drei Frames. Die
Messungen sind ein Diagnose-Sample, noch kein Datensatz-Benchmark.

## Wichtigste Befunde

### 1. Debug und Produktionspipeline hatten verschiedene Masken

`SequencePipeline` entfernt erkannte Fahrzeugboxen vor dem BEV-Warp. Die
Stufen 5 und 6 riefen dagegen `build_lane_mask(..., [])` auf. Autos wurden so
zu hellen Strukturen und potentiellen Histogramm-Peaks, obwohl die echte
Pipeline sie entfernt.

Die Stufen 5 und 6 verwenden nun standardmaessig denselben Fahrzeugfilter.
`--no-detect` liefert weiterhin bewusst die Rohvariante. Stufe 5 schreibt beide
Kameramasken als benannte PNGs und die CSV enthaelt `n_fahrzeuge` sowie
`entfernte_pixel`.

Gemessen an drei Frames:

| Variante | BEV-Pixel (Median) | belegte Spalten | Peaks |
|---|---:|---:|---:|
| bilinear, ohne Boxen | 8.372 | 149 | 5 |
| binaer/Nearest, ohne Boxen | 5.751 | 138 | 5 |
| binaer/Nearest, mit Boxen | 2.750 | 54 | 5 |

Der Peak-Ausgang blieb in diesem Sample stabil. Das ist gut, bedeutet aber
nicht, dass die Stoerungen harmlos sind: in schwierigeren Frames koennen sie
die Schwelle ueberschreiten oder den staerkeren von zwei nahen Peaks stellen.

### 2. Eine binaere Maske wurde bilinear interpoliert

`cv2.warpPerspective` nutzte den Default `INTER_LINEAR`. An jeder Kante
entstanden Grauwerte, die `lane_histogram` mit `> 0` als voll belegte Pixel
zaehlte. `warp_lane_mask` verwendet jetzt `INTER_NEAREST`; die Ausgabe bleibt
nachweislich in `{0, 255}`. Im Sample sank die Pixelzahl um rund 31 Prozent,
ohne dass eine Grenze verloren ging.

### 3. Die dokumentierten Canny-Parameter existierten nicht

Die Stufentexte verwiesen auf `lane.canny_*`, der Code verwendete fest
`50, 150`. `canny_low` und `canny_high` sind jetzt echte, validierte
`LaneConfig`-Werte. Die Defaults bilden das alte Verhalten exakt ab.

### 4. HLS-L allein ist keine Linienklassifikation

Die neue Schwellenansicht zeigt deutlich: weisse Fahrzeuge und das runde
Tempolimit-Schild bestehen dieselbe Helligkeitsschwelle wie Markierungen.
Eine zusaetzliche S-/Farbschwelle loest das nur teilweise, denn graue und
weisse Fahrzeuge haben ebenfalls geringe Saettigung. Fahrzeugboxen sind
deshalb der belastbarere erste Filter.

Im Hough-Debug ist ausserdem eine lange Diagonale vom Fahrzeugbereich in
Richtung Schild sichtbar. Der reine Winkelfilter akzeptiert sie zu Recht nach
seiner heutigen Definition: sie ist nicht horizontal. Der Fehler wird erst in
der geometrischen Bedeutung sichtbar.

### 5. Die aeusserste Linie kann zur Gegenfahrbahn gehoeren

Im Referenzframe lagen links zwei kontinuierliche Kandidaten bei etwa x=42
und x=301. Die historische Auswahl nahm immer den aeussersten Kandidaten x=42
und spannte damit beide Fahrtrichtungen auf. Der korrekte Richtungsteiler ist
x=301; die rechte Begrenzung liegt bei x=855.

`bev.boundary_pair_strategy: nearest_continuous` waehlt nun standardmaessig
die dem Ego naechste kontinuierliche Nicht-Ego-Linie je Seite. Die alte
Topologie bleibt als `outermost` konfigurierbar. Stage 4 zeichnet angebotene,
aber verworfene Solid-Kandidaten grau ein.

Die Korrektur wirkt wie erwartet durch alle Folgestufen:

| Messung | alte Aussenkante | Richtungsfahrbahn |
|---|---:|---:|
| Homographie-Basis | ca. 813 px | ca. 553 px |
| Histogramm-Peaks | 5 | 4 |
| erkannte Korridore | 4 | 3 |
| Median Peak-Abstand | ca. 78 px | ca. 112 px |

Damit wird die Gegenfahrbahn nicht mehr als zusaetzlicher Korridor gezaehlt.

## Neue Debugbilder

Mit `snapshots: 3` in `configs/scripts.yaml` entstehen fuer die ersten drei
Frames zusaetzliche, semantisch benannte PNGs:

- Stufe 1: Original + ROI, HLS-L, rohe Weissschwelle, ROI-Maske, Endmaske
- Stufe 2: Canny-Kanten und Hough akzeptiert/verworfen
- Stufe 3: farbige Cluster und rohe/behaltene Linienfits
- Stufe 4: Homographie-Quellgitter und seine vier Stuetzpunkte
- Stufe 5: Rohmaske, fahrzeugbereinigte Maske und binaere BEV-Maske
- Stufe 6: BEV mit Peaks und separater Histogrammplot

`--snapshots 0` schaltet die PNGs ab. Die Methode ist vom Video getrennt,
damit mehrere Zwischenbilder nicht die Videozeitachse verfaelschen.

## Empfohlene naechste Algorithmen

### Prioritaet A: Fahrzeugboxen auch vor Hough verwenden

Die BEV-Maske wird bereinigt, `detect_lanes` sieht Fahrzeuge und Schild aber
weiterhin. Als naechster kontrollierter Versuch sollte `detect_lanes` optionale
Boxen erhalten und Stufe 2 zwei Ergebnisse darstellen: mit und ohne Boxfilter.
Bewertet werden nicht nur Segmentzahl, sondern Homographie-Verfuegbarkeit,
`basis_delta` und falsche Aussenlinien. Wichtig: verdeckte Markierungen werden
innerhalb einer Box ebenfalls entfernt; deshalb muss der Sequenztracker kurze
Aussetzer halten koennen.

### Umgesetzt: geometrische Segment-Kompatibilitaet statt Greedy-Cluster

`lane.cluster_method: union_find` ersetzt das reihenfolgeabhaengige Greedy-
Verfahren. Jedes Segmentpaar muss auf derselben Ego-Seite liegen und bei
Steigung, Projektion an `y_bottom`, lokalem Querabstand, vertikaler Luecke und
Projektion im Fluchtpunktband `y_top` zusammenpassen. Zusammenhangskomponenten
werden danach deterministisch per Union-Find gebildet. `greedy` bleibt fuer
A/B-Vergleiche verfuegbar.

Auf 60 Referenzframes blieb die Anzahl verwendbarer Fits stabil: 280 mit
Union-Find gegen 279 mit Greedy. Stage 3 protokolliert nun kompatible Paare und
Ablehnungen wegen Steigung, lokalem Abstand und Fluchtprojektion.

### Umgesetzt: Homographie zeitlich plausibilisieren

Neue Paare werden vor der Glaettung gegen den letzten akzeptierten Zustand
geprueft: Eckpunktsprung, Breitenveraenderung oben/unten, Fluchtpunktsprung,
Perspektivverhaeltnis, Continuity und Support. Bei einem Ausreisser bleibt die
letzte Homographie als `held` aktiv. Stage 4 zeichnet den abgelehnten Kandidat
rot und das tatsaechlich verwendete Gitter blau/orange; die CSV nennt Grund und
Messwert.

Auf `lane_departure_3_lanes` fing der Gate in 246 Frames 28 `point_jump`-
Ausreisser und einen echten Paarausfall ab. Die verwendete Basis blieb in
552.8..600.0 px, obwohl Kandidaten bis 875.6 px angeboten wurden. Im
detektorbereinigten 60-Frame-Test lieferte Stufe 6 danach in jedem Frame genau
vier Grenzen fuer drei Fahrspuren.

### Prioritaet D: kompakte Nicht-Linien in BEV nur diagnostisch filtern

Das Tempolimit-Schild bleibt nach dem Fahrzeugfilter sichtbar. Ein
Connected-Components-Filter nach Laenglichkeit, Orientierung und vertikaler
Ausdehnung kann solche Objekte markieren. Er sollte zuerst als alternative
Debugmaske laufen, nicht sofort produktiv: kurze, ferne Fahrbahnstriche sind
ebenfalls kleine Komponenten. Erfolgskriterium ist Peak-Stabilitaet ueber alle
Szenarien, nicht ein optisch saubereres Einzelbild.

## Bewertungsplan

Jede Algorithmusvariante sollte mit identischer Quelle gegen die Basis laufen.
Mindestens zu erfassen sind:

- Maske: Anteil in ROI, entfernte Pixel, Komponenten nach Formklasse
- Hough/Fit: Kandidaten, Winkel-Ablehnungen, Cluster, Support, Fit-Residuum
- Homographie: fresh/held/none, Stuetzpunktgeschwindigkeit, `basis_delta`
- BEV: binaere Pixel, belegte Spalten, Peak-Prominenz und Peak-Abstand
- Sequenz: Grenz-ID-Wechsel und Anzahl synthetischer Korridore

Erst wenn eine Variante ueber mehrere Szenarien besser ist, sollte sie Default
werden. Ein saubereres Debugbild allein ist kein Qualitaetsbeweis.

## Automatische Kalibrierung

Der Kalibrator verwendet dieselbe Fahrtrichtungs-Auswahl, binaere
Nearest-Warp-Funktion, Fahrzeugbereinigung und Crop-Logik wie die Pipeline.
Ohne `--apply` ist der Lauf immer ein Trockenlauf:

```powershell
adascope calibrate `
  --source scenarien/lane_departure_3_lanes `
  --config-dir scripts/configs `
  --max-frames 300 --stride 3 `
  --report outputs/calibration/lane_departure.yaml
```

Automatisches Anwenden ist eine ausdrueckliche zweite Entscheidung:

```powershell
adascope calibrate `
  --source scenarien/lane_departure_3_lanes `
  --config-dir scripts/configs `
  --max-frames 300 --stride 3 --apply
```

Dabei gelten folgende Gates:

- ab 20 Frames wird ein Messvorschlag angezeigt;
- ab 50 Frames kann ein stabiler Wert automatisch angewandt werden;
- `y_top`, `y_bottom` und andere Geometrie bleiben manuell zu pruefen;
- instabile Spurbreiten und Peak-Abstaende werden nicht geschrieben;
- vor jeder Aenderung entsteht eine timestamp-basierte `.bak-*`-Datei;
- nach dem Schreiben wird die gesamte Konfiguration erneut geladen und
  validiert; bei einem Fehler werden die Backups wiederhergestellt.
