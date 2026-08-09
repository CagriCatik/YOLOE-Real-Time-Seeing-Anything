# ADR-0003: Umbenennung zu `adascope`, flaches Layout, Szenario-Runner

## Status

Accepted (2026-08-08). Ersetzt die Namens- und Layoutfestlegung aus
[ADR-0001](ADR-0001-package-layout-and-ports.md) und
[ADR-0002](ADR-0002-unified-layout-and-domain-configs.md); die
Ports-&-Adapters-Struktur und die Config-Schichtung gelten unverändert.

## Kontext

Drei Probleme, die zusammen auffielen:

**Der Name log.** `yoloe_lane` benannte ein Modell, das inzwischen der kleinere
Teil ist. Das Projekt besteht aus klassischer CV-Spurerkennung, Bodenebenen-
Geometrie, YOLO11-Tracking, ego-relativer Spurindizierung und einer temporalen
Ereignis-State-Machine. YOLOE liefert davon einen Baustein — die
open-vocabulary-Detektion für Carpet und HUD.

**Die Verschachtelung verdeckte den Einstiegspunkt.** `src/yoloe_lane/cli/…`
ist für eine Bibliothek richtig (das src-Layout verhindert, dass beim Testen
versehentlich der Quellordner statt des installierten Pakets importiert wird).
Dieses Projekt ist aber eine Anwendung: es wird aus dem Repo heraus benutzt, nicht
als Abhängigkeit installiert. Der Nutzen wog den Preis nicht auf — es war nicht
mehr erkennbar, wo man anfängt.

**Es gab keinen Weg von `scenarien/` zu Ergebnissen.** Aufnahmen lagen da, aber
jede Auswertung war ein handgeschriebener Aufruf mit Pfaden, Ansichtsliste und
Ausgabeordner. Ein Vergleich mehrerer Aufnahmen bedeutete, CSVs von Hand
nebeneinanderzulegen.

## Entscheidung

**Name: `adascope`.** ADAS + Blickwinkel. Trägt kein Modell im Namen und bleibt
gültig, wenn HMI-Elemente, Warnungen und weitere Assistenzfunktionen dazukommen.
Paketname, Konsolenkommando und Ordnername sind identisch.

**Flaches Layout.** `adascope/` liegt direkt im Wurzelverzeichnis, kein `src/`.
Die innere Schichtung aus ADR-0002 bleibt unverändert.

**Ein sichtbarer Einstiegspunkt.** `adascope <command>`, ersatzweise
`python -m adascope <command>` — letzteres funktioniert immer, auch wenn das
Skriptverzeichnis von pip nicht im PATH liegt. Die Hilfe nennt den Schnellstart
zuerst.

**Szenarien sind ein Erstklassen-Konzept.** Ein Szenario ist eine Aufnahme in
`scenarien/`; der Dateistamm ist sein Name und verbindet drei Orte:

```text
scenarien/<name>.mp4              die Aufnahme (oder ein Ordner mit Frames)
config/scenarios/<name>.yaml      optional: nur die Abweichungen
results/<name>/                   Debug-Videos, CSVs, summary.txt
```

`adascope scenarios` findet alles, führt es aus und schreibt zusätzlich
`results/index.md` mit einer Vergleichstabelle über alle Läufe.

**Die Spurkalibrierung passt sich der Aufnahme an, nicht umgekehrt.**
`lane.yaml` ist in Pixeln eines Referenzausschnitts notiert
(`reference_size`). Am ersten Frame jeder Quelle wird entschieden:

- gleiche Proportion, andere Größe → `LaneConfig.scaled_to()` skaliert alle
  Längen mit; Winkel, Helligkeitsschwellen und Stückzahlen bleiben.
- Vollbild, das nach `detection.crop_box` passen würde → zuschneiden, gemeldet.
- sonst → Abbruch mit einer Meldung, die den Ausweg nennt.

Die Entscheidung fällt je QUELLE, nicht je Aufruf. Ein globaler Schalter war der
erste Entwurf und ist daran gescheitert, dass dieselbe Sammlung fertig
zugeschnittene Ausschnitte *und* Vollbilder enthält — er hätte die einen richtig
und die anderen doppelt beschnitten.

## Konsequenzen

**Aufrufwege ändern sich.** `yoloe-lane X` → `adascope X`, `import yoloe_lane` →
`import adascope`. Alles andere — Konfigurationsformat, Kommandos, Ausgaben —
bleibt gleich. Die Standardausgabe von `lane-debug` wanderte von
`outputs/debug/` nach `results/lane-debug/`, damit alle Ergebnisse an einem Ort
liegen.

**Ein Lauf, ein Codepfad.** `lane-debug` und `scenarios` rufen beide
`runner.run_debug()`. Ein Einzelaufruf und ein Szenario-Durchlauf liefern
damit garantiert vergleichbare Zahlen.

**Ein Fehlschlag beendet den Durchlauf nicht.** Scheitert ein Szenario, steht
der Grund in seiner Tabellenzeile und die übrigen laufen weiter. Auf dem
aktuellen Material betrifft das die zehn `adjusting_speed_scenario_*`-Ordner:
sie liegen in 1920×1080 statt im kalibrierten 1209×457-Zuschnitt und werden
jetzt automatisch zugeschnitten. Danach laufen alle 13 Aufnahmen durch.

**Ein Fehler wurde dabei gefunden und behoben.** Der erste Entwurf des
Szenario-Runners setzte den Tracker zwischen Aufnahmen über ein `persist=False`
am nächsten Frame zurück. Ultralytics registriert seinen Tracking-Callback aber
beim *ersten* `track()`-Aufruf und backt den damaligen `persist`-Wert hinein —
der Tracker fing danach in jedem Frame neu an. Messbar auf
`lane_departure_3_lanes`: Ego-Index-Sprünge von 0.8 % auf 16.7 %, dazu 45
erfundene Ereignisse. `YoloVehicleTracker.reset()` benutzt jetzt
`BYTETracker.reset()`; die Referenzwerte werden exakt reproduziert.
