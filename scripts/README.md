# Modell-Testskripte

Drei eigenständige Skripte, um YOLO-Modelle auf einzelnen Bildern, Videos oder
Frame-Ordnern auszuprobieren — **ohne** die Spurpipeline.

## Abgrenzung

Diese Skripte beantworten *„was sieht dieses Modell auf diesem Bild"*.
Sie kennen keine Kalibrierung, keine Homographie, keine Zustandslogik.

| Frage | Werkzeug |
|---|---|
| Findet das Modell die Fahrzeuge? Welche Konfidenz? Wie schnell? | **diese Skripte** |
| Welcher Prompt trifft die Fläche, die ich meine? | **`test_yoloe.py`** |
| Funktioniert Spurerkennung, Indizierung, Cut-In? | `adascope scenarios` |

## Die drei Modelle

| Skript | Modell | Vokabular | Masken | Größe |
|---|---|---|---|---|
| `test_yolo11n.py` | yolo11n | 80 COCO-Klassen | nein | 5 MB |
| `test_yolo11l_seg.py` | yolo11l-seg | 80 COCO-Klassen | **ja** | 50 MB |
| `test_yoloe.py` | yoloe-11l-seg | **frei per Textprompt** | ja | 71 MB |

Fehlende Gewichte werden beim ersten Lauf nach `models/` geladen — nicht ins
Arbeitsverzeichnis, wie es ultralytics von sich aus tut.

## Aufrufe

```powershell
# Einzelbild, nur Fahrzeugklassen
python scripts/test_yolo11n.py --source test_images/test_frame.png --vehicles

# Video mit ByteTrack-IDs über Frames
python scripts/test_yolo11n.py --source scenarien/acc_plus_1_vid.mp4 --track --device 0

# Segmentierungsmasken statt Rechtecke
python scripts/test_yolo11l_seg.py --source test_images/test_frame.png --vehicles

# Offenes Vokabular: eigene Prompts
python scripts/test_yoloe.py --source test_images/test_frame_arrow.png `
    --prompts "gray trapezoid on the road" "car" --conf 0.01 --max-area 0.5

# Genau die Prompts testen, die die Pipeline benutzt
python scripts/test_yoloe.py --source test_images/test_frame.png --from-config --conf 0.02

# Ohne Prompts — was findet das Modell von sich aus?
python scripts/test_yoloe.py --source test_images/test_frame.png --prompt-free
```

Gemeinsame Schalter: `--conf --iou --imgsz --device --stride --max-frames
--outdir --no-labels --top`. Klassen ansehen: `--list-classes`.

## Ausgaben

Je Lauf nach `outputs/model_tests/`:

- `<quelle>_<modell>.png` bzw. `.mp4` — annotiert, Farbe je Klasse stabil
- `<quelle>_<modell>.csv` — jede Detektion mit Box, Konfidenz, Flächenanteil, Track-ID
- `<quelle>_<modell>.json` — dasselbe plus Laufparameter

Die Zusammenfassung auf der Konsole nennt Detektionen je Frame, Inferenzzeit
(Mittel und p90), und je Klasse Anzahl, Median-Konfidenz und Median-Flächenanteil.
Klassen mit über 50 % Bildfläche sind mit `<- Ganzbild?` markiert — das ist der
häufigste Müll bei offenem Vokabular.

## Was man beim Prompten wissen sollte

Gemessen auf `test_frame_arrow.png` gegen die per Helligkeit ermittelte
Zielfläche:

| Prompt | conf | IoU |
|---|---|---|
| `gray trapezoid on the road` | 0.019 | **0.82** |
| `gray road area` | 0.007 | 0.26 |
| `gray overlay on the lane` | 0.007 | 0.07 (Ganzbild) |
| `gray patch/shape/polygon/area`, `shaded area`, `driving area` | — | kein Treffer |

**Das Formwort entscheidet, nicht die Farbe.** „trapezoid" trifft, „area",
„patch", „shape", „polygon" nicht — bei identischem Farbwort.

Zwei Folgen für die Bedienung:

- `--conf` muss klein sein (**0.005…0.03**). Der Standard 0.25 verwirft auf
  synthetischen HMI-Grafiken praktisch alles.
- `--max-area 0.5` filtert Ganzbildtreffer, den häufigsten Fehlalarm.

YOLOE trifft **diskrete Objekte** gut (Schilder 0.038, Fahrzeuge 0.69) und
**flächige Farbbereiche** schlecht. Für Letztere ist die HSV-Schwelle unter
`carpet:` in `config/detection.yaml` der zuverlässigere Weg.

## Zur Segmentierungsmaske

Die Maske aus `test_yolo11l_seg.py` ist für die **Spurpipeline nicht
verwendbar**: die Homographie gilt nur in der Bodenebene, ein Fahrzeugumriss hat
Bauhöhe und zerläuft beim Warpen über mehrere Spuren. Nur die Bbox-Unterkante
darf zwischen den Ebenen wechseln — siehe
[docs/10_PIPELINE_UND_ROBUSTHEIT.md](../docs/10_PIPELINE_UND_ROBUSTHEIT.md).

Nützlich ist sie für anderes: Fahrzeuge präziser aus der Weißmaske ausstanzen,
als ein Rechteck es kann, und Verdeckungen beurteilen.

## Aufbau

`_common.py` trägt alles Gemeinsame: Argumente, Gewichte-Auflösung, Quelle,
Zeichnen, Ausgabe, Zusammenfassung. Ein Skript enthält nur, was sein Modell
besonders macht — genau die Trennung, deren Fehlen den früheren `scripts/`-Ordner
unbrauchbar gemacht hat (siehe
[ADR-0002](../docs/decisions/ADR-0002-unified-layout-and-domain-configs.md)).
Geteilt wird zusätzlich nur die I/O-Schicht von `adascope`.
