# adascope

Analyse von ADAS-HMI-Videos: Spurerkennung, Bodenebene, Fahrzeugtracking und
Cut-In-Ereignisse — sichtbar gemacht aus mehreren Perspektiven.

```text
Video ─▶ Spurlinien (Hough) ─▶ Homographie ─▶ Spurkorridore ─▶ ego-relative
        Bildebene              Bodenebene      Histogramm       Nummerierung
                                                                     │
        Debug-Videos ◀── Renderer ◀── Ereignisse (State Machine) ◀───┘
```

---

## 🚀 Einstiegspunkt

**Ein Kommando für alles: `adascope`.**

```powershell
pip install -e ".[dev]"      # einmalig

adascope --help              # alle Kommandos
python -m adascope --help    # gleichwertig, funktioniert immer
```

> **`adascope` wird nicht gefunden?** Dann liegt pips Skriptverzeichnis nicht im
> PATH. `python -m adascope <command>` funktioniert immer und ist funktional
> identisch — es ist derselbe Einstiegspunkt.
>
> Dauerhaft beheben (Pfad ggf. an die Python-Version anpassen, danach neues
> Terminal öffnen):
>
> ```powershell
> $dir = "$env:APPDATA\Python\Python313\Scripts"
> [Environment]::SetEnvironmentVariable(
>     "Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";$dir", "User")
> ```

## 🎬 Szenarien auswerten

Der übliche Weg: Aufnahme in `scenarien/` ablegen, ein Kommando, Ergebnisse in
`results/`.

```powershell
adascope scenarios --list                  # was liegt in scenarien/?
adascope scenarios                         # alle auswerten
adascope scenarios lane_departure_3_lanes  # nur dieses eine
adascope scenarios --quick                 # schneller Blick, 100 Frames je Aufnahme
```

Drei Orte, verbunden über den Dateinamen:

```text
scenarien/lane_departure_3_lanes.mp4          die Aufnahme (oder ein Frame-Ordner)
configs/scenarios/lane_departure_3_lanes.yaml  optional: NUR die Abweichungen
results/lane_departure_3_lanes/               entsteht beim Lauf
    debug_dash.mp4        Komposit mit Zeitverlauf und Ereignislog
    debug_bev.mp4         Bodenebene von oben
    debug_oblique.mp4     virtuelle Schrägkamera
    debug_shoulder.mp4    Blick vom Standstreifen
    debug_front.mp4       Bildebene mit Spurlinien und Tracks
    debug_mask.mp4        Hough-Cluster und Weißmaske
    debug_hist.mp4        Spaltenhistogramm mit Peaks
    debug_smear.mp4       warum Fahrzeuge im BEV nicht flächig gelten
    debug_metrics.csv     jede Kennzahl je Frame
    debug_events.csv      Cut-In / Cut-Out / Spurwechsel
    summary.txt           die Kennzahlen des Laufs
results/index.md          Vergleichstabelle über ALLE Läufe
```

Findet sich `configs/scenarios/<name>.yaml`, wird sie automatisch als
Kalibrier-Überlagerung verwendet — sonst gilt die Basiskalibrierung.

## ✅ Nachweis statt Vermutung

Ohne Annotation ist **„keine Ereignisse" nicht von „nichts erkannt" zu
unterscheiden**. Genau diese Verwechslung hat drei Defekte in der State Machine
verdeckt. Deshalb zwei Ebenen:

**1. Synthetische Szenen — beweisen, dass die Logik stimmt.** Die Trajektorie
wird vorgegeben, das Bild daraus erzeugt, und die Pipeline muss die
vorgegebenen Ereignisse zurückliefern. Das prüft die ganze Kette (Hough →
Homographie → Korridore → Indizierung → State Machine) ohne Modell, ohne
Annotation, in Millisekunden:

```powershell
pytest tests/lanes/test_events_end_to_end.py -v
```

Abgedeckt: sauberes Einscheren, Ausscheren, abgebrochener Wechsel, zwei
Fahrzeuge gleichzeitig, Einscheren bei ausgefallenem Fernbereich, Einscheren
unter Rauschen und Störstrichen, Fernfeldartefakte, Zeitpunkt des Ereignisses.

**2. Annotierte Aufnahmen — beweisen, dass es auf echtem Material greift.**

```yaml
# ground_truth/<szenario>.yaml
tolerance: 8
events:
  - {frame: 142, kind: cut_in,  track: any}
  - {frame: 210, kind: cut_out, track: ID3}
```

Eine **leere** Liste ist die stärkste Aussage: „hier passiert nachweislich
nichts", jedes gemeldete Ereignis ist damit ein Falschalarm. Die Tabelle
bekommt eine Spalte `Bewertung` (`2/2 ok (±3f)`, `1/2 1 fehlt`), und
`adascope scenarios` liefert **Rückgabewert 0 nur, wenn jede vorhandene
Annotation erfüllt ist** — damit taugt der Lauf als Regressionstest in CI.
Vorlage: `ground_truth/VORLAGE.yaml`.

Positive Ereignisse sind nicht Voraussetzung fuer eine sinnvolle Regression:
`events: []` misst Falschalarme und weist den positiven Recall ausdruecklich als
N/A aus. Zusaetzlich koennen ausgewaehlte Frames die Wahrnehmung selbst pruefen:

```yaml
perception:
  - frame: 18
    driving_area: [[290, 295], [855, 295], [660, 55], [535, 55]]
    boundaries_bev: [81, 190, 302, 419]
    lane_count: 3
    ego_lane_position: 1
```

Die Klickhilfe `scripts/annotate_perception.py` erzeugt dafuer einen manuellen
Entwurf. `adascope scenarios` bewertet Richtungsflaechen-IoU, Grenz-Recall und
-Fehler, Spurzahl, Ego-Spur und optional Fahrzeugspuren und schreibt
`debug_perception.csv`. Nicht annotierte Groessen bleiben N/A.

**Zuschnitt entscheidet sich je Aufnahme.** `lane.yaml` ist in Pixeln eines
Referenzausschnitts notiert (`reference_size`). Am ersten Frame wird geprüft:

| Aufnahme | Verhalten |
|---|---|
| passt schon (1209×457) | unverändert |
| gleiche Proportion, andere Größe (1428×534) | Kalibrierung wird mitskaliert |
| Vollbild (1920×1080) | auf `detection.crop_box` zugeschnitten, wird gemeldet |
| passt so oder so nicht | klare Fehlermeldung, übrige Szenarien laufen weiter |

Mit `--no-crop` lässt sich das Zuschneiden abschalten.

Für beliebige Quellen und Ausgabeorte gibt es `adascope lane-debug --source …`;
beide Wege laufen durch denselben Code und liefern vergleichbare Zahlen.

---

## 📚 Documentation

| Document | Purpose | Read Time |
| ---------- | --------- | ----------- |
| **[01_UNIFIED_GUIDE.md](docs/01_UNIFIED_GUIDE.md)** | Unified project guide | 8 min |
| **[02_GETTING_STARTED.md](docs/02_GETTING_STARTED.md)** | Setup & first run | 15 min |
| **[03_USER_GUIDE.md](docs/03_USER_GUIDE.md)** | Workflows & tools | 30 min |
| **[04_ARCHITECTURE.md](docs/04_ARCHITECTURE.md)** | System design & data flow | 20 min |
| **[05_DESIGN_AND_CONCEPTS.md](docs/05_DESIGN_AND_CONCEPTS.md)** | YOLOE concepts & theory | 30 min |
| **[06_API_REFERENCE.md](docs/06_API_REFERENCE.md)** | Tool options & parameters | reference |
| **[07_YOLOE_CONCEPTS.md](docs/07_YOLOE_CONCEPTS.md)** | YOLOE background | 30 min |
| **[08_ASSISTED_LANE_CHANGE_CASE_STUDY.md](docs/08_ASSISTED_LANE_CHANGE_CASE_STUDY.md)** | Worked example | 20 min |
| **[09_LANE_BEV_PIPELINE.md](docs/09_LANE_BEV_PIPELINE.md)** | Spur-/BEV-Pipeline, Befunde und Messungen | 25 min |
| ⭐ **[10_PIPELINE_UND_ROBUSTHEIT.md](docs/10_PIPELINE_UND_ROBUSTHEIT.md)** | **Jeder Schritt mit Diagrammen + Robustheits-Fahrplan** | 20 min |
| ⭐ **[11_ANFORDERUNGSABDECKUNG.md](docs/11_ANFORDERUNGSABDECKUNG.md)** | **FR-1 bis FR-7: Status, Codestelle, Test** | 10 min |

**→ Wie die Pipeline arbeitet und wo sie bricht:
[10_PIPELINE_UND_ROBUSTHEIT.md](docs/10_PIPELINE_UND_ROBUSTHEIT.md)**

---

## 📁 What's Inside

```text
adascope/
├── config/                      # ⭐ Kalibrierung — eine YAML je Domäne
│   ├── detection.yaml           #   YOLOE: Modell, ROIs, Carpet, HUD
│   ├── lane.yaml                #   Spurerkennung Bildebene (Hough)
│   ├── bev.yaml                 #   Bodenebene, Belegungsschwellen
│   ├── tracking.yaml            #   YOLO11 + ByteTrack
│   ├── indexing.yaml            #   ego-relative Spurnummern
│   ├── events.yaml              #   Cut-In-/Cut-Out-State-Machine
│   ├── pipeline.yaml            #   Zustand über Frames hinweg
│   ├── debug.yaml               #   virtuelle Kameras, Farben, Layout
│   └── scenarios/               #   nur Abweichungen je Aufnahmesituation
├── models/                      # YOLOE- und YOLO11-Gewichte
├── adascope/                    # ⭐ Das Paket — flach, kein src/
│   ├── config/                  #   typisierte Configs + YAML-Loader
│   ├── io/                      #   Frames, Videos, Tabellen
│   ├── detection/               #   Modell-Adapter (einziger ultralytics-Ort)
│   ├── vision/                  #   YOLOE-Domäne: ROIs, Carpet, HUD, Analyse
│   ├── lanes/                   #   CV-Domäne: Spuren, BEV, Indizierung, Events
│   ├── render/                  #   Overlays, virtuelle Kameras, Debug-Views
│   ├── cli/                     #   `adascope <command>`
│   └── tool/                    #   PyQt6-Werkzeuge (Extra [gui])
├── tests/                       # 203 Tests, <1 s — kein Test lädt ein Modell
├── data/ · test_images/         # Frames und Referenzbilder (gitignored)
├── scenarien/                   # ⭐ Eingang: eine Aufnahme je Fahrsituation
├── results/                     # ⭐ Ausgang: results/<szenario>/ (gitignored)
├── outputs/                     # Ergebnisse der YOLOE-Pipeline (gitignored)
└── docs/                        # Dokumentation und ADRs ⭐
```

Die Schichten kennen nur die jeweils inneren: Domänencode (`vision`, `lanes`)
importiert weder `ultralytics` noch `argparse` und schreibt keine Dateien.
Details in [ADR-0002](docs/decisions/ADR-0002-unified-layout-and-domain-configs.md)
und [ADR-0003](docs/decisions/ADR-0003-rename-and-flat-layout.md).

---

## 🎯 Typische Abläufe

**Szenario auswerten** — der Normalfall:

```powershell
adascope scenarios <name>
```

**Neue Aufnahme kalibrieren**, wenn die Spurerkennung nichts findet:

```powershell
adascope lane-sensitivity --frame test_images/test_frame_masked.png
# lane.yaml anpassen, dann erneut:
adascope scenarios <name> --quick --views front,mask
```

`front` zeigt die gefundenen Linien mit Rolle, `mask` die Hough-Cluster —
zusammen sagen sie, ob ROI, Helligkeitsschwelle oder Clustering schuld ist.

**YOLOE-Analyse** (Carpet, HUD, ROI-Zählung):

```powershell
adascope detect --frames data/frames/cropped
adascope probe --image test_images/test_frame.png --prompts "green road"
```

---

## 🛠️ Kommandos

Alles läuft über ein Kommando: `adascope <command>` (oder
`python -m adascope <command>`).

| Kommando | Zweck |
| --- | --- |
| `download` · `extract` · `crop` · `assemble` | Datenaufbereitung |
| `roi-editor` · `crop-box` | Kalibrierung, schreibt nach `configs/detection.yaml` |
| `detect` | YOLOE-Analyse: Fahrzeuge, Carpet, HUD → `states.csv` + Debug-Video |
| `probe` | YOLOE-Prompts auf einem Einzelbild ausprobieren |
| `track` | Fahrzeugtracking in der Bildebene → Video + Track-CSV |
| `scenarios` | ⭐ Alles aus `scenarien/` auswerten → `results/<name>/` |
| `lane-debug` | Debug-Videos für EINE beliebige Quelle |
| `calibrate` | ⭐ Spurgeometrie automatisch aus dem Material bestimmen (FR-6) |
| `lane-sensitivity` | Robustheitsmessungen der BEV-Geometrie auf einem Einzelbild |
| `tool` | PyQt6-Datenwerkzeuge (`pip install -e .[gui]`) |

```powershell
# Debug-Videos aus allen Perspektiven, mit Szenario-Kalibrierung
adascope lane-debug --source scenarien/lane_departure_3_lanes.mp4 `
                      --scenario lane_departure_3_lanes --views all --device 0

adascope lane-debug --list-views     # front, mask, bev, hist, smear, oblique, shoulder, dash
adascope lane-debug --views bev,hist --no-detect --source data/frames/xyz   # ohne YOLO
```

Details zur Spur-/BEV-Pipeline in [09_LANE_BEV_PIPELINE.md](docs/09_LANE_BEV_PIPELINE.md),
zum YOLOE-Workflow in [03_USER_GUIDE.md](docs/03_USER_GUIDE.md).

---

## 📊 Output Format

**CSV (states.csv):** `veh_*` count **other** vehicles (ego excluded); `state_*`
is the per-lane drivable state (available/blocked/drivable/clear).

```
frame,veh_left,veh_ego,veh_right,state_left,state_ego,state_right
0,1,0,1,clear,drivable,clear
1,1,0,1,available,drivable,clear
2,0,0,1,blocked,drivable,clear
```

**Video (debug.mp4):**

- ROI overlays (colored polygons)
- Vehicle detections (boxes per lane)
- Carpet status (green/red/clear indicators)

See [06_API_REFERENCE.md](docs/06_API_REFERENCE.md#output-formats) for details.

---

## ⚙️ Configuration

**Eine YAML je Domäne, eine Datei entspricht genau einer Dataclass.** Jede ist
kommentiert und beschreibt, was der Wert bewirkt und woran man merkt, dass er
falsch steht.

Drei Regeln gelten überall:

1. **Der Default steht im Code.** Eine fehlende Datei heißt „Defaults", kein
   Fehler — Tests und Bibliotheksnutzung brauchen kein Dateisystem.
2. **Die YAML überschreibt nur, was sie nennt.**
3. **Validiert wird beim Laden, einmal.** Ein Tippfehler im Schlüssel ist ein
   Fehler, kein stilles Ignorieren.

```yaml
# configs/bev.yaml — Auszug
peak_min_distance: 55   # Mindestabstand zweier Histogramm-Peaks.
                        #   Deutlich UNTER der schmalsten echten Spurbreite,
                        #   deutlich ÜBER der Breite der Fehlpeaks. Bei 25
                        #   überlebten Kleinstkorridore von 25..42 px.
```

**Szenarien** überlagern die Basiskalibrierung und enthalten ausschließlich
Abweichungen:

```yaml
# configs/scenarios/lane_departure_3_lanes.yaml
indexing:
  lane_width: 77        # durchgehend dreispurig -> BEV-Skala konstant
events:
  confirm_frames: 2
```

```powershell
adascope lane-debug --scenario lane_departure_3_lanes --source scenarien/...
adascope roi-editor      # schreibt interaktiv nach configs/detection.yaml
adascope crop-box
```

**Neue Perspektive ohne Codeänderung:** ein Eintrag unter `cameras:` in
`configs/debug.yaml` ist danach als `--views <name>` verfügbar — BEV und
Schrägsicht sind Bilder derselben Ebene, die Abbildung dazwischen ist wieder
nur eine Homographie.

---

## 🆘 Troubleshooting

| Issue | Solution |
| ------- | ---------- |
| Setup problems | → [02_GETTING_STARTED.md](docs/02_GETTING_STARTED.md#troubleshooting) |
| How do I...? | → [03_USER_GUIDE.md](docs/03_USER_GUIDE.md#-common-tasks) |
| Tool options | → [06_API_REFERENCE.md](docs/06_API_REFERENCE.md) |
| Why this design? | → [05_DESIGN_AND_CONCEPTS.md](docs/05_DESIGN_AND_CONCEPTS.md) |
| Understanding output | → [06_API_REFERENCE.md](docs/06_API_REFERENCE.md#-output-formats) |

---

## 🔑 Key Concepts

### Open-Vocabulary Detection (YOLOE)

- Prompts can change at inference time (no retraining)
- Text, visual, or free-prompting modes
- Works across domains and videos

### Perspective Polygons (ROI)

- Lane regions as 3D perspective-aware polygons
- Top points narrow (vanishing point), bottom points wide (near edge)
- Resolution-independent with 0..1 normalized coordinates

### HSV Color Thresholding (Carpet)

- Green (lane available), Red (blocked), White (normal road)
- Reliable for synthetic overlays
- Tunable sensitivity with `min_frac` parameter

### Modular Pipeline

- Independent tools that compose together
- Can run steps individually or orchestrated
- Easy to integrate into your own systems

---

## 📈 Performance

| Task | GPU | CPU |
| ------ | ----- | ----- |
| Frame extraction (100 frames) | — | 30 sec |
| YOLOE inference per frame | 0.1 s | 0.5 s |
| Carpet detection per frame | 0.02 s | 0.02 s |
| **Full 300-frame run** | **15-20 min** | **30-45 min** |

**Optimization:**

- Use GPU if available (10x faster)
- Use smaller model (`-m` or `-pf` variants)
- Use `--stride 5` for fast preview

---

## 🌟 Example Use Cases

### Autonomous Driving

- Vehicle detection in surrounding lanes
- Lane availability assessment
- Decision-making support

### Traffic Analysis

- Vehicle counts per lane
- Lane congestion detection
- Traffic pattern analysis

### Video Annotation

- Auto-label frames with detections
- Export structured data (CSV)
- Generate debug videos for review

### Research

- Study lane change scenarios
- Benchmark detection algorithms
- Validate safety features

---

## 📝 License & Attribution

This project uses:

- **YOLOE** (Ultralytics) — Open-vocabulary detection
- **OpenCV** — Computer vision
- **Supervision** — Detection utilities
- **PyYAML** — Configuration
- **yt-dlp** + **ffmpeg** — Video handling

See `requirements.txt` for full dependency list.

---

## 🚀 Next Steps

1. **[Read the Getting Started guide](docs/02_GETTING_STARTED.md)** (15 min)
2. **Szenarien auswerten:** `adascope scenarios`
3. **Explore tools:** `adascope detect --frame data/frames/raw/frame_000100.jpg`
4. **Deep dive:** [04_ARCHITECTURE.md](docs/04_ARCHITECTURE.md) for design details

---

**Questions?** Start with [docs/01_UNIFIED_GUIDE.md](docs/01_UNIFIED_GUIDE.md) for the complete documentation map.
