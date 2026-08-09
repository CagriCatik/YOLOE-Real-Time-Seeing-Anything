# ADR-0002: Ein Projekt, Schichten nach Zuständigkeit, eine Config je Domäne

## Status

Accepted (2026-08-08). Erweitert [ADR-0001](ADR-0001-package-layout-and-ports.md);
die dort getroffene Ports-&-Adapters-Entscheidung gilt unverändert.

> **Hinweis:** Paket und Kommando hiessen damals `yoloe_lane` bzw.
> `yoloe-lane`, das Layout war `src/`-basiert. Beides geändert mit
> [ADR-0003](ADR-0003-rename-and-flat-layout.md).

## Kontext

ADR-0001 hat `src/yoloe_lane` als installierbares Paket etabliert. Danach sind
zwei weitere Codebestände daneben gewachsen, ohne diese Struktur zu übernehmen:

- **`lane-detection/`** — die klassische CV-Spur-/BEV-Pipeline. Flache Skripte
  mit `sys.path.insert`-Hacks, eigenem README, eigener `requirements.txt`,
  eigenem `data/` und `outputs/`, ohne einen einzigen Test.
- **`tool/`** — eine PyQt6-Datenaufbereitung mit eigenen Requirements, eigenen
  JSON-Configs und eigenem Testordner.
- **`scripts/`** — zwei Einzelskripte für YOLOE-Prompt-Experimente, jedes mit
  einer eigenen Kopie von Modell-Setup, Zeichnen und JSON-Ausgabe.

Daraus folgten konkrete Doppelungen: Frame- und Video-I/O existierte dreimal mit
je anderer Menge erlaubter Endungen und anderem Verhalten bei defekten Dateien;
CSV-Schreiben zweimal; zwei Modell-Adapter ohne gemeinsames Dach; drei
`requirements.txt`.

Schwerer wog die **Kalibrierung im Code**. Sechs Dataclasses (`LaneConfig`,
`BevConfig`, `IndexConfig`, `FsmConfig`, `VehicleTrackerConfig`, die
`VirtualCam`-Presets) trugen ihre Werte als Defaults im Quelltext. Sie sind
frame-, kamera- und szenenspezifisch — eine Anpassung war jedes Mal eine
Codeänderung, und ein Szenario ließ sich nicht neben einem anderen ablegen.

## Entscheidung

**Ein Paket, Schichten nach Zuständigkeit.** `lane-detection/`, `tool/` und
`scripts/` werden aufgelöst und nach `src/yoloe_lane/` überführt:

```text
config/     typisierte Kalibrierung, eine YAML je Domäne
io/         Frames, Videos, Tabellen; kein Domänenwissen
detection/  Modell-Adapter; einziger Ort mit `ultralytics`
vision/     YOLOE-Domäne: ROIs, Carpet, Driving Area, HUD
lanes/      klassische CV-Domäne: Spuren, Bodenebene, Ereignisse
render/     Overlays, virtuelle Kameras, Debug-Ansichten
cli/        dünne Adapter, ein Subkommando je Aufgabe
tool/       optionale PyQt6-Werkzeuge (Extra `[gui]`)
```

Jede Schicht kennt nur die inneren. Domänencode importiert weder `ultralytics`
noch `argparse` und schreibt keine Dateien.

**Eine Config-Datei je Domäne, Defaults im Code.** `config/lane.yaml`,
`bev.yaml`, `tracking.yaml`, `indexing.yaml`, `events.yaml`, `pipeline.yaml`,
`debug.yaml`, `detection.yaml`. Eine Datei entspricht genau einer Dataclass.
Drei Regeln:

1. Jede Dataclass ist **ohne Datei** konstruierbar — Tests und Bibliotheks-
   nutzung brauchen kein Dateisystem, eine fehlende Datei heißt „Defaults".
2. Die YAML überschreibt **nur, was sie nennt**.
3. Validiert wird **beim Laden, einmal**; danach sind die Objekte frozen. Ein
   unbekannter Schlüssel ist ein Fehler, kein stilles Ignorieren.

**Szenarien als Overlay.** `config/scenarios/<name>.yaml` enthält je Domänen-
Sektion ausschließlich Abweichungen und wird über die Basis gemischt. Damit
beschreibt eine Datei eine Aufnahmesituation vollständig, ohne die
Basiskalibrierung zu kopieren.

**Virtuelle Kameras sind Konfiguration, nicht Code.** Weil BEV und Schrägsicht
Bilder derselben Ebene sind, ist die Abbildung dazwischen wieder eine
Homographie. Eine neue Perspektive entsteht durch einen Eintrag in `debug.yaml`
und ist danach als `--views <name>` verfügbar.

**Harter Schnitt bei den Aufrufwegen.** `python lane-detection/src/X.py` und
`python scripts/X.py` entfallen ersatzlos; alles läuft über
`yoloe-lane <command>`.

## Konsequenzen

**Aufgelöste Doppelungen.** Frame-/Video-I/O liegt einmal in `io.frames`,
inklusive `VideoWriter`, der ungerade Kantenlängen explizit auf gerade
zuschneidet — mp4v schnitt sie sonst stumm ab. Beide Modell-Adapter liegen unter
`detection/`. Eine einzige Abhängigkeitsquelle: `pyproject.toml` mit den Extras
`[dev]`, `[gui]`, `[download]`.

**Zwei stille Fehler sind dabei verschwunden.** `build_homography()` nahm
`{L.role: L for L in lines}` und behielt bei mehreren Linien derselben Rolle die
*innere* statt der Fahrbahnkante (in 41 % der Videoframes), bzw. warf `KeyError`,
wenn eine Rolle fehlte (63 %). `outer_solid_pair()` wählt jetzt explizit die
äußerste je Seite, und `HomographyTracker` hält die letzte gültige Abbildung über
Ausfälle hinweg. Zweitens gab es zwei Plausibilitätsprüfungen für Footprints mit
verschiedenen Annahmen; die verbliebene nimmt die Spurbreite aus
`lanes.indexing` statt einen Median über `corridors[1:]`, der voraussetzte, dass
Korridor 0 immer der Standstreifen ist.

**Testbarkeit ohne Modell.** Die Suite läuft in unter einer Sekunde, weil kein
Test ein Modell lädt: der Detektor liegt hinter einem Port, `SequencePipeline`
bekommt die Fahrzeugliste übergeben, und jede Config ist ohne Datei
konstruierbar. Neu abgedeckt sind Config-Schicht, I/O, Spur-Domäne
(`bev`, `indexing`, `events`, `pipeline`) und die View-Registry.

**Verhaltensparität ist geprüft.** `lane-sensitivity` reproduziert die
README-Zahlen des Einzelbilds unverändert, und `lane-debug` liefert auf
`lane_departure_3_lanes.mp4` dieselben Werte wie vor dem Umbau (95 % `fresh`,
2 Index-Sprünge, `ego_in_lane` min 0.88).

**Offen.** Die Fließtext-Dokumentation unter `docs/01`–`docs/08` beschreibt
stellenweise noch den Workflow vor diesem Umbau; angepasst wurden bisher nur die
Pfad- und Modulverweise. `docs/09_LANE_BEV_PIPELINE.md` (früher
`lane-detection/README.md`) ist inhaltlich aktuell.
