# Anforderungsabdeckung FR-1 bis FR-7

Stand: alle Anforderungen außer Transfer Learning / Fine-Tuning umgesetzt
(FR-2.1 und der Trainingsteil von FR-2.5 sind als eigenes Projekt ausgegliedert).

Jede Zeile nennt die Codestelle und den Test, der sie festhält. Die Tests laufen
mit `pytest` in unter fünf Sekunden — **kein Test lädt ein Modell**.

---

## Übersicht

| | Anforderung | Status | Nachweis |
|---|---|---|---|
| **FR-1.1** | Grenzübergang je Fahrzeug, ohne Spurnummer | ✅ | `lanes/events.py` · `test_events_end_to_end.py` |
| **FR-1.2** | Richtung LINKS / RECHTS | ✅ | `lanes/tracking_ids.crossing_direction` · `test_fr12_*` |
| **FR-1.3** | Kern unabhängig von 1–4-Zuordnung | ✅ | `lanes/indexing.py` ego-relativ · `test_fr43_*` |
| **FR-1.4** | Per-Frame-Zustand mit 5 Feldern | ✅ | `FrameAnalysis.states()` · `test_fr14_*` |
| **FR-2.1** | Fine-getuntes YOLOv8n | ⛔ | **ausgegliedert** — läuft mit YOLO11n/COCO |
| **FR-2.2** | Laterale Position relativ zur Linie | ✅ | `lanes/indexing.locate` · `test_indexing.py` |
| **FR-2.3** | CO-Event mit Richtung | ✅ | `test_fr23_*` (beide Richtungen) |
| **FR-2.4** | Fehlende Detektion ≠ Wechsel | ✅ | `EventConfig.max_missing` · `test_events.py` |
| **FR-2.5** | Trainier-/inferierbar auf Zielhardware | 🟡 | Inferenz gemessen (21 ms/Frame); **Training ausgegliedert** |
| **FR-3.1** | Ego-Wechsel aus der Linienstruktur | ✅ | `lanes/egomotion.py` · `test_fr31_*` |
| **FR-3.2** | Translation vs. Kurvenrotation | ✅ | `EgoMotionConfig.max_spread` · `test_fr32_*` |
| **FR-3.3** | Unsicher markieren statt schweigen | ✅ | `Event.certain` · `test_fr33_*` |
| **FR-4.1** | Optionale absolute Spurnummern | ✅ | `lanes/mapping.py` · `test_fr41_*` |
| **FR-4.2** | `None` bei unvollständiger Struktur | ✅ | `test_fr42_*` (zwei Fälle) |
| **FR-4.3** | Strikt vom Kern getrennt | ✅ | `test_fr43_*` prüft, dass der Kern sie nicht importiert |
| **FR-4.4** | Erfüllt den Wortlaut des Kriteriums | ✅ | siehe FR-4.1 |
| **FR-5.1** | Event {fahrzeug, richtung, grenze_id, frames} | ✅ | `lanes/events.Event` · `test_fr51_*` |
| **FR-5.2** | Zwischenzustand als eigener Zustand | ✅ | `encroaching` · `test_events.py` |
| **FR-6.1** | Auto-Kalibrierung je Fahrzeugprojekt | ✅ | `calibration.py` · `adascope calibrate` |
| **FR-6.2** | Voraussetzung fürs Ausrollen | ✅ | siehe FR-6.1 |
| **FR-7.1** | Ground Truth für die TC-Szenarien | 🟡 | Format + synthetische TCs; **echte Aufnahmen fehlen** |
| **FR-7.2** | Prüfung gegen Ground Truth inkl. Richtung | ✅ | `ground_truth.py` · `test_fr72_*` |
| **FR-7.3** | Kurven-Confounder-Varianten | ✅ | `SyntheticRoad(curvature=…)` · `test_windows.py` |

---

## FR-1 — Kern: Grenzübergangs-Erkennung

**Die Richtung kommt ohne jede Spurnummer zustande.** In BEV-Koordinaten wächst
x nach rechts; das Vorzeichen der lateralen Bewegung *ist* die Richtung:

```python
crossing_direction(before=300.0, after=200.0)  # -> "links"
```

**Der Per-Frame-Zustand (FR-1.4)** liegt als `FrameAnalysis.states()` vor und
umfasst EGO und CO im selben Satz:

```python
{'frame_id': 0, 'fahrzeug': 'EGO', 'lateral_pos': 249.77,
 'aktive_grenze_id': 1, 'confidence': 0.0}
{'frame_id': 0, 'fahrzeug': 'ID1', 'lateral_pos': 136.87,
 'aktive_grenze_id': 0, 'confidence': 0.125}
```

> **Zur Grenzen-ID — hier lag ein Widerspruch zur Architektur.**
> FR-1.4 und FR-5.1 verlangen eine `grenze_id`. Frühere Arbeit hatte
> Boundary-IDs bewusst *abgeschafft*, weil positionsbasierte Nummern bei einem
> Grenzausfall rutschen und Falschalarme ohne Szenenänderung erzeugen.
>
> Aufgelöst über die Trennung zweier Fragen. `lanes/indexing` beantwortet
> weiterhin *„die wievielte Spur relativ zum Ego"* — ego-relativ, ohne absolute
> Zählung. `lanes/tracking_ids` beantwortet neu *„ist das dieselbe Linie wie im
> letzten Frame"*. Nur Letzteres ist die `grenze_id`. Die Belegungsauswertung
> bleibt unberührt.

**Die `confidence`** stammt aus der Beobachtungsgeschichte, nicht aus der
Peakhöhe: eine Grenze, die seit vielen Frames an derselben Stelle liegt, ist
verlässlicher als ein starker Peak in einem Einzelframe.

---

## FR-3 — Der Ego-Pfad, neu gebaut

**Der Mechanismus wurde ausgetauscht.** Vorher schloss `_detect_ego_shift()`
aus der gleichsinnigen Verschiebung der *Fremdfahrzeuge*. Gemessen auf dem
Projektmaterial: in **1090 von 2117 Frames** ist außer dem Ego kein Fahrzeug im
Bild — die Szenenebene konnte dort nie greifen.

`lanes/egomotion.py` wertet stattdessen die **Linienstruktur** aus, die immer da
ist, wenn die Homographie steht:

```
Translation (Wechsel)          Drehung (Kurve)
||||  ->  ||||                 ||||  ->  \\\\
alle gleich weit               fern mehr als nah
```

**FR-3.2 ist als Streuungsmaß umgesetzt.** Die Standardabweichung der
Einzelverschiebungen, geteilt durch die mittlere Verschiebung. Bei echter
Translation nahe null, in einer Kurve groß. Über `egomotion.max_spread` gilt der
Wechsel als nicht belegt.

**FR-3.3 meldet, statt zu schweigen.** Ein nicht belegter Wechsel wird als
`Event(certain=False)` ausgegeben und in der Textform als `[UNSICHER]` markiert:

```
[f030-050] ego_lane_change ego nach links ueber B2 [UNSICHER]
    Verschiebung 0.80 Spurbreiten, Streuung 0.90
```

> **Das bleibt die schwächste Annahme des Systems** — so wie ihr sie selbst
> markiert habt. Die Trennung ist gegen **synthetische** Kurven geprüft
> (`test_fr32_*`), nicht gegen echtes Kurvenmaterial. Für das Abnahmegate fehlt
> eine Aufnahme mit echter Kurve **und** bekanntem Ego-Verhalten.

---

## FR-4 — Die Mapping-Schicht ist wirklich getrennt

`lanes/mapping.py` liest `FrameAnalysis` und gibt Nummern zurück. Es schreibt
nichts, und der Kern importiert es nicht — `test_fr43_the_core_runs_without_the_mapping_layer`
prüft genau das.

Drei Bedingungen, jede für sich hinreichend zum Ablehnen (FR-4.2):

1. keine verwertbare Spurliste
2. eine Spur wurde aus einem verschmolzenen Korridor **rekonstruiert** — eine
   virtuelle Grenze ist eine Annahme, keine Beobachtung
3. die erwartete Spurzahl stimmt nicht

Dann kommt `None` mit Begründung, keine plausible Zahl.

---

## FR-6 — Auto-Kalibrierung

```powershell
adascope calibrate --source scenarien/acc_plus_1_vid.mp4
```

Bestimmt aus dem Material selbst: `y_bottom`, `y_top`, `white_l_min` (Otsu-
Trennwert Asphalt/Markierung), `lane_width` und daraus `peak_min_distance`.

**Jeder Wert kommt mit Herkunft:**

```
  white_l_min    121.0  aus 60 Frames  Streuung 0.6  130 -> 121 (-9)
  lane_width      63.1  aus 60 Frames  Streuung 4.4
```

Werte aus zu wenigen Frames werden als `ZU WENIG DATEN` markiert und nicht
vorgeschlagen. **Geschrieben wird nichts** — eine Kalibrierung, die sich selbst
einspielt, ändert das Verhalten ohne Spur in der Historie.

---

## Was noch aussteht

**FR-2.1 / FR-2.5 (Training)** — ausgegliedert wie vereinbart. Es läuft
YOLO11n mit COCO-Gewichten; Inferenz ist mit 21 ms/Frame gemessen.

**FR-7.1 — die echten TC-Aufnahmen.** Format, Bewertung und Richtungsprüfung
stehen. Die beiden TCs sind **synthetisch** nachgebaut (`test_fr71_*`), weil die
Richtung nur gegen eine bekannte Wahrheit prüfbar ist. Für die Abnahme auf
echtem Material braucht es zwei Dinge, die ich nicht liefern kann:

1. **Welche Aufnahmen** sind die TCs? In `scenarien/` liegen 21 Aufnahmen; keine
   ist als „EGO 2→4" oder „CO 2→1" gekennzeichnet.
2. **Die Annotation** dazu — Frame, Richtung, Fahrzeug. Sie erfordert das
   Sichten des Debug-Videos; die Vorlage liegt in `ground_truth/VORLAGE.yaml`.

Sobald beides vorliegt:

```powershell
adascope scenarios <tc-name>     # Rückgabewert 0 = Annotation erfüllt
```

**Das Kurven-Abnahmegate für FR-3.2** — siehe Kasten oben.
