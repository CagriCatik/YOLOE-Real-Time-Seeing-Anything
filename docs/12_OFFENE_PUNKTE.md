# Offene Punkte

Stand nach dem Lauf über alle 21 Aufnahmen, 346 Tests grün.
Priorisiert nach gemessener Wirkung, nicht nach Aufwand.

**Zwei Defekte behoben** (Punkte 2 und 5). Über alle 21 Aufnahmen zusammen
fallen jetzt **0,8 %** der Frames aus, vorher waren es vier praktisch tote
Aufnahmen plus 3,4 % Ego-Ausfall im Rest.

---

## Zwei aktuelle Beobachtungen, nachgemessen

### Das Flackern im `debug_dash.mp4` ist real und quantifiziert

Gemessen als Anteil der Frameübergänge, bei denen sich die Anzahl der Korridore
bzw. der ego-relativen Spuren ändert:

| Aufnahme | Flackern Korridore | Flackern Spuren |
|---|---|---|
| `adjusting_speed_scenario_8` | **34 %** | **36 %** |
| `acc_plus_1` | 10 % | 10 % |
| `lane_departure_1_lane` | 8 % | 7 % |

In jedem dritten Frameübergang von `scenario_8` ändert sich die Korridorzahl —
bei durchgehend `fresh` Homographie. **Die Ursache liegt nicht in der
Homographie, sondern in der Grenzensuche:** das Spaltenhistogramm wird in jedem
Frame vollständig neu ausgewertet, ohne jedes Gedächtnis. Eine Strichlücke lässt
eine Grenze verschwinden, der nächste Frame findet sie wieder.

Dasselbe Flackern erklärt die 25,4 % Index-Sprünge in dieser Aufnahme und die
springenden Grenzen-IDs (`2, 6, 2, 2, 2, 3, 3, …`). Der `BoundaryTracker`
arbeitet dabei korrekt — er vergibt neue Kennungen, weil tatsächlich neue
Grenzen auftauchen.

> **Sliding Windows sind nicht die Abhilfe.** Gemessen auf `acc_plus_1`:
> Flackern der Korridore 10 % → **40 %**, der Spuren 10 % → **71 %**. Sie
> bringen ihre eigene Instabilität mit.

### Die kurvige Straße in `lane_departure_1_lane`

Sie **ist** messbar krummer als der Rest — aber das ist dort nicht das Problem.

| Messung | `lane_departure_1_lane` | Rest des Materials |
|---|---|---|
| Sagitta (Median) | 6,7 px | 4,7 px |
| Sagitta (p90) | 43,2 px | 26–41 px |
| Residuum Grad 1 → Grad 2 | 3,61 → 3,34 px (**+7 %**) | +5 % |

Ein Polynom 2. Grades senkt das Residuum um 7 %. Zum Vergleich: der getrimmte
robuste Fit senkt es auf dem kritischen Dezil um 84 %. **Auch hier sind es
Ausreißer, nicht Krümmung.**

Und der Praxistest bestätigt es: mit `method: windows` fällt die verwertbare
Spurliste dort von **42 % auf 12 %**. Die Fenstersuche macht es deutlich
schlechter, weil sie im Nahbereich genug Pixel je Fenster braucht — und die hat
diese Aufnahme nicht.

**Der eigentliche Blocker dort ist Stufe 1:** nur **22 %** der Frames liefern
überhaupt eine Homographie. Ohne Bodenebene ist jede Diskussion über Splines
gegenstandslos.

---

## Offene Punkte, priorisiert

### 1. ~~Spurgrenzen zeitlich verfolgen~~ — umgesetzt, gemessen, **verworfen**

**Umgesetzt** als `lanes/stabilize.py` (halten + bestätigen, 24 Tests).
**Gemessen — und die Diagnose war falsch.**

Die Annahme lautete: die Korridorzahl schwankt, weil eine stabile Struktur kurz
ausfällt. Der Test der Annahme widerlegt sie:

| Korridorbreite | `scenario_8` | `acc_plus_1` |
|---|---|---|
| zu schmal (< 0,55 Spurbreiten) | **0,0 %** | **0,0 %** |
| genau eine Spur (0,8–1,25) | 70,4 % | 77,9 % |
| verschmolzen (> 1,25) | 19,8 % | 22,1 % |

**Keine einzige Scheingrenze.** Die Korridore sind durchweg legitim. Die Zahl
schwankt, weil am Rand des BEV echte Markierungen sichtbar werden und wieder
verschwinden — kein Ausfall, sondern wechselnde Sicht.

Halten füllt dort keine Lücke, es sammelt an:

| `adjusting_speed_scenario_8` | aus | nur bestätigen | nur halten | beides |
|---|---|---|---|---|
| Spurliste | **93 %** | 76 % | 70 % | 59 % |
| Flackern | **34 %** | 59 % | 30 % | 46 % |
| Korridore (Mittel) | **4,26** | 3,41 | 6,84 | 5,90 |

Jede Variante ist schlechter als der Ausgangszustand.

**Konsequenz:** `boundaries.stabilize: false` als Standard. Der Code bleibt —
auf `lane_departure_1_lane`, wo die Struktur wirklich steht, senkt er das
Flackern von 8 % auf 2 % und hebt die Spurliste von 42 % auf 44 %. Er ist je
Aufnahme einschaltbar, nicht global.

> **Nachgeprüft: die ego-relative Indizierung hält.** Die Frage war, ob das
> Flackern in die Auswertung durchschlägt. Gemessen an der Frage, ob die
> **Ego-Spur physisch springt**, ohne dass sich das Ego bewegt:
>
> | | `scenario_8` | `acc_plus_1` |
> |---|---|---|
> | Ego-Spurmitte springt > ½ Spurbreite | 7,8 % | 10,4 % |
> | davon **ohne** Ego-Bewegung (Defekt) | **4,3 %** | **0,0 %** |
>
> Auf `acc_plus_1` kein einziger Fall. Auf `scenario_8` fünf Übergänge von 116,
> vier davon zeitgleich mit einer geänderten Grenzenzahl.
>
> Ein Zwischenergebnis war irreführend und ist hiermit zurückgezogen: dass sich
> die *Spanne* der Spurliste bei fast jeder Grenzenänderung mitbewegt, ist
> **korrektes** Verhalten — eine neu sichtbare Randspur verlängert die Liste,
> ohne die Ego-Zuordnung anzutasten.
>
> Damit ist das Flackern zum größten Teil ein Darstellungsthema: Grenzen
> außerhalb der Ego-Spanne schwächer zeichnen, statt sie mitzuzählen.

### 2. ~~Homographie-Verfügbarkeit in Stufe 1~~ — **behoben**

Vier Aufnahmen lagen bei 0–22 % `fresh`. Die Ursache war weder ROI noch
Helligkeitsschwelle noch Clustering — die Trichtermessung zeigte auf
`adjusting_speed_scenario_9` gesunde 2,8 % Weißanteil, 19,7 Segmente und 5,5
Cluster je Frame, aber **`left_solid` = 0,00 in jedem einzelnen Frame** bei
gleichzeitig 2,2 `right_solid`.

**Der Defekt:** `classify_lanes` vergibt `left_solid` nur an Linien, die *weiter
außen* liegen als die ego-nächste. Fährt das Ego links außen, ist die
Fahrbahnkante zugleich die ego-nächste Linie — sie heißt dann `left_dashed`, und
`outer_solid_pair` findet nie ein Paar. Die Rollennamen sind an dieser Stelle
positionell, nicht gemessen: **keine Stufe prüft je, ob eine Linie wirklich
durchgezogen ist.**

**Behoben** durch einen Rückfall auf die äußerste Linie je Seite, wenn die Rolle
fehlt — je Seite einzeln, bestehende `*_solid`-Rollen behalten Vorrang. Dazu
`MIN_PAIR_SEPARATION`, damit zwei fast deckungsgleiche Linien abgelehnt werden
statt still eine entartete Homographie zu erzeugen.

| Aufnahme | `fresh` | Spurliste |
|---|---|---|
| `adjusting_speed_scenario_9` | 0 % → **100 %** | 0 % → **98 %** |
| `adjusting_speed_scenario_10` | 2 % → **100 %** | 40 % → **100 %** |
| `adjusting_speed_10` | 5 % → **100 %** | → **100 %** |
| `lane_departure_1_lane` | 22 % → **97 %** | 42 % → **95 %** |

**Keine Regression:** alle übrigen Aufnahmen unverändert. Über alle 21 liegt
`fresh` jetzt bei ≥ 96 % und die verwertbare Spurliste bei ≥ 75 %.

Vier Tests in `test_detection.py` halten den Fall fest, einschließlich der
Gegenprobe, dass eine vorhandene `*_solid`-Rolle weiterhin Vorrang hat.

### 2b. ~~Einspurige Fahrbahn wurde als Ausfall behandelt~~ — **behoben**

Im vollständigen Lauf (nicht in den 150-Frame-Stichproben) fiel
`lane_departure_1_lane` mit **36 %** verwertbarer Spurliste auf, bei 99 %
`fresh`. Direkt gemessen — ohne Fahrzeugdetektion — waren es 99,6 %.

**Der Unterschied war die Detektion, und sie arbeitete korrekt.**

| | Frames | Ergebnis |
|---|---|---|
| genau **2** Grenzen | 312 | verworfen: „nur 1 Korridor" |
| 3+ Grenzen | 175 | ausgewertet |

Die Aufnahme heißt nicht zufällig `1_lane`: eine einspurige Fahrbahn hat zwei
Grenzen und einen Korridor — eine vollständige, gültige Szene. Die Pipeline
forderte zwei Korridore und verwarf sie.

Ohne Detektion erzeugten Fahrzeugpixel Scheingrenzen, die auf drei kamen. **Die
Aufnahme lief also nur, weil die Spurmaske verschmutzt war** — die Detektion
aufzuräumen hat den Defekt erst sichtbar gemacht.

**Behoben** über `indexing.min_corridors` (Standard 1). Bei einem Korridor gibt
es keine Nachbarspur und damit kein `cut_in`/`cut_out`; das Spurverlassen des
Ego bleibt messbar, und genau darum geht es in dieser Aufnahme (`ego_in_lane`
fällt dort auf 0,28).

**`lane_departure_1_lane`: 36 % → 100 %.** Alle übrigen 20 Aufnahmen unverändert,
die annotierte weiterhin `0/0 ok`.

> **Methodischer Nachtrag:** Diesen Defekt hatten meine 150-Frame-Stichproben
> nicht gezeigt — sie liefen ohne Detektion und meldeten 99 %. Kennzahlen aus
> einem verkürzten Pfad sind kein Ersatz für den vollständigen Lauf.

### 3. Metrische Skala aus der Strichperiodik

**Problem:** Die BEV-Skala ist auf den Abstand der zwei gefundenen Randlinien
normiert, nicht metrisch. Die geschätzte Spurbreite streut über die Aufnahmen
zwischen 62 und 112 px.

**Ansatz:** Autokorrelation des BEV-Spaltenprofils *längs* liefert die
Strichperiodik (deutsche Autobahn: 6 m Strich + 12 m Lücke = 18 m) und daraus
px/m — unabhängig davon, wie viele Spuren die Randlinien einschließen.

**Wirkung:** macht `estimate_lane_width()` überflüssig statt sie zu verbessern,
und die Footprint-Plausibilität absolut statt relativ.

### 4. Kurven: der Fit in der **Bildebene** fehlt

> **Korrektur:** Die hier dokumentierte Krümmungsgrenze von ~220 war ein
> **Artefakt derselben Rollenzuweisung** (siehe Punkt 2), nicht des
> Geradenfits. In der Kurve wandert die äußere Linie aus dem Bild, die
> verbleibende links vom Ego heißt `left_dashed`, und es gab kein Paar. Mit dem
> Rückfall findet Stufe 1 über den gesamten geprüften Bereich (150 … 2000) ein
> Paar. Die Fenster kommen jetzt zum Zug — was sie dort taugen, ist damit erst
> messbar geworden.

Sliding Windows sind gebaut und lösen die Bodenebene.

**Offen:** derselbe Modellsprung in der Bildebene, also Polynom statt Gerade in
`lanes/detection.fit_lanes`. Das ist ein echter Eingriff — die Rollenzuordnung,
die Homographie-Stützpunkte und `x_bottom` hängen alle an der Geradenform.

**Vorher messen:** ob echtes Kurvenmaterial das überhaupt braucht. Auf dem
aktuellen Material bringt Grad 2 zwischen 5 % und 7 %.

### 5. ~~„Ego-Footprint in keiner plausiblen Spur"~~ — **behoben**

Der häufigste Ausfall des ganzen Systems: 3,4 % aller Frames, bis zu 25,4 % auf
`acc_plus_7`. Der Verdacht lag auf `tracking.ego_zone` — falsch.

**Die Aufschlüsselung der Ausfälle:**

| | `acc_plus_7` | `acc_plus_3` | `adj_speed_5` |
|---|---|---|---|
| Ego außerhalb der Spanne | 4 | 2 | 3 |
| **Ego in einem Loch** | **17** | **20** | **17** |
| zu wenig Korridore | 11 | 5 | 0 |

**Der Defekt:** `split_corridors` verwirft Korridore, die zu schmal oder zu
breit für eine Spur sind — Standstreifen, Randflächen. Die Prüfung kennt nur
Geometrie. Fällt ausgerechnet der Korridor durch sie hindurch, **in dem das Ego
steht**, entsteht ein Loch in der Spurliste, und `build_lane_index` meldet
danach „Ego-Footprint liegt in keiner plausiblen Spur" — über einen Korridor,
in dem das Ego nachweislich fährt.

**Behoben:** `split_corridors(…, keep_x=…)` schützt den Korridor, der die
Ego-Position enthält. Dass das Ego dort fährt, ist die stärkste verfügbare
Evidenz für Befahrbarkeit — stärker als eine Breitenschwelle. Fährt es
tatsächlich auf dem Standstreifen, ist es ebenfalls richtig, das zu zeigen,
statt den ganzen Frame zu verwerfen.

| Aufnahme | Spurliste |
|---|---|
| `adjusting_speed_scenario_5` | 83 % → **100 %** |
| `acc_plus_3` | 82 % → **97 %** |
| `acc_plus_7` | 75 % → **91 %** |
| `adjusting_speed_scenario_8` | 93 % → **100 %** |
| `acc_plus_4` | 91 % → **99 %** |

**Der Ausfallgrund existiert nicht mehr** — 3,4 % → 0,0 %. Über alle 21
Aufnahmen bleiben zusammen 0,8 % Ausfall (0,4 % „nur 0 Korridore", 0,3 % „keine
Homographie", 0,1 % „nur 1 Korridor").

Drei Tests in `test_indexing.py`, einschließlich der Gegenprobe, dass ein
Nicht-Spur-Korridor **ohne** Ego weiterhin verworfen wird und dass der
geschützte Korridor nicht fälschlich als `synthetic` markiert wird (das würde
FR-4.2 die Mapping-Schicht grundlos verstummen lassen).

### 5b. Ego-Auswahl über Persistenz — weiterhin offen

`tracking.ego_zone` bleibt ein festes normiertes Rechteck, auf den kalibrierten
Zuschnitt getunt. Es fällt jetzt nicht mehr auf, weil der Folgefehler behoben
ist — die Ankerzone selbst ist dadurch nicht robuster geworden.

**Ansatz:** Das Ego ist das Fahrzeug, dessen Footprint über viele Frames im
selben Korridor bleibt und dessen Bildposition minimal schwankt — nicht das in
einem festen Rechteck.

### 6. Homographie zeitlich glätten

Aktuell wird die letzte gültige Abbildung **gehalten**, nicht gefiltert. Ein
Ausreißer-Frame prägt die Geometrie bis zu `max_hold` Frames. Ein gleitendes
Mittel über die vier Stützpunkte, gewichtet nach Support der Linien, kostet
wenig.

### 7. Annotationen — der Nachweis fehlt fast überall

**1 von 21 Aufnahmen** ist annotiert. Ohne sie ist „keine Ereignisse" nicht von
„nichts erkannt" zu unterscheiden — genau die Verwechslung, die drei Defekte in
der State Machine verdeckt hatte.

Offen sind insbesondere die beiden TC-Szenarien aus FR-7.1 (EGO 2→4, CO 2→1).
Dafür braucht es zwei Dinge, die nur ihr liefern könnt:

1. **Welche** der 21 Aufnahmen sind die TCs? Keine ist gekennzeichnet.
2. Die Annotation dazu — Frame, Richtung, Fahrzeug. Vorlage in
   `ground_truth/VORLAGE.yaml`.

### 8. Abnahmegate für FR-3.2 (Kurven-Confounder)

Die Trennung Translation/Drehung ist gegen **synthetische** Kurven geprüft
(`test_fr32_*`). Für das Gate fehlt eine reale Aufnahme mit echter Kurve **und**
bekanntem Ego-Verhalten. Ich kann Kurven erzeugen, aber nicht bezeugen, dass
eine reale Aufnahme keinen Spurwechsel enthält.

### 9. Ausgegliedert: Fine-Tuning (FR-2.1, FR-2.5)

Es läuft YOLO11n mit COCO-Gewichten. Inferenz ist mit 21 ms/Frame gemessen; das
Training ist als eigenes Projekt vereinbart.

---

## Was ausdrücklich **nicht** offen ist

Damit die Liste nicht mit Erledigtem verwechselt wird:

- Richtung, Grenzen-ID und Frame-Spanne im Ereignis (FR-1.2, 2.3, 5.1)
- Per-Frame-Zustand mit fünf Feldern, EGO und CO gleich (FR-1.4) —
  `debug_states.csv`
- Ego-Wechsel aus der Linienstruktur inkl. Kurven-Trennung und
  Unsicherheitsmarkierung (FR-3.1–3.3)
- Optionale Mapping-Schicht, strikt getrennt (FR-4)
- Auto-Kalibrierung (FR-6) — `adascope calibrate`
- Richtungsprüfung in der Ground-Truth-Bewertung (FR-7.2)
- Robuster getrimmter Linienfit (−84 % Residuum auf dem kritischen Dezil)
- Sliding Windows für die Bodenebene, konfigurierbar je Aufnahme

Details je Anforderung in
[11_ANFORDERUNGSABDECKUNG.md](11_ANFORDERUNGSABDECKUNG.md).
