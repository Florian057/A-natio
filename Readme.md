# nagelsmannGrrr — Moneyball für die deutsche Nationalmannschaft

Wendet das Factor-Investing-Muster aus [`../moneyball`](../moneyball) (dort: MLB-Roster)
auf die Auswahl des DFB-Kaders an: aus einem Faktor-Zoo von Spielerstats herausfinden,
welche Kennzahlen am stärksten mit Gewinnen korrelieren, dann alle deutschen Spieler der
Top-5-Ligen danach bewerten.

## Pipeline

```
python buildDatabase.py             # data/ befüllen (einmalig, danach idempotent)
python factorPerformanceFootball.py # Korrelationen, OLS, PCA pro Position
python factorPerformanceMLFootball.py  # RF/Lasso/Ridge/Ensemble, CV-R²
python rankPlayers.py               # Rangliste deutscher Spieler pro Position
```

## Datenquellen

- **Club-Ebene** (`scrapeFootballData.py`): [JaseZiv/worldfootballR_data](https://github.com/JaseZiv/worldfootballR_data),
  ein archivierter, statischer Mirror von FBref-Stats (Big-5-Ligen, 2010–2022/23) und
  Spielergebnissen. FBref selbst blockt Scraper (auch echten Headless-Chrome) mit einer
  Cloudflare-Challenge — dieser Mirror ist der Weg, der ohne Residential-IP funktioniert.
  **Datenstand-Obergrenze: Saison 2022/23** (Spieler-Tabellen wurden davor archiviert).
- **Länderspiel-Ebene** (`scrapeInternationalResults.py`): [martj42/international_results](https://github.com/martj42/international_results),
  alle Länderspiele seit 1872 — dient nur als Validierungs-/Kontextquelle, nicht als
  Trainingsdaten (zu wenige Spiele pro Nation für eine eigene Regression).

## Architektur

Vier Positions-Blöcke (`GK`, `DF`, `MF`, `FW`) statt moneyballs zwei (Batting/Pitching) —
siehe `factorPerformanceFootball.py` für die genaue Faktorenliste pro Block. KPIs
(`win_pct`, `pts_per_game`, `goal_diff_per_game`) kommen aus echten Match-Ergebnissen,
nicht aus den Stat-Seiten selbst.

`rankPlayers.py` ist der Schritt, den `moneyball` nie gebaut hat (dort nur als TODO
im Readme): Score pro Spieler = z-standardisierte Faktoren (relativ zu allen Spielern
derselben Position/Saison) gewichtet mit den aus der Team-Regression gelernten
OLS-Koeffizienten. Eine Rangliste pro Position, kein ILP-Kaderoptimierer (bewusste
Entscheidung für diese Iteration).

## Bekannte Limitierungen

- Datenstand 2022/23 — keine aktuellen Formkurven, keine Spieler die erst danach
  durchgebrochen sind.
- Zwei Faktoren (`Prog_Carries`, `CPA_Carries`) sind für die Saison 2022/23 im Mirror
  komplett leer (defekter Scrape kurz vor der Archivierung); `rankPlayers.py` weicht
  deshalb automatisch auf die letzte vollständige Saison (2021/22) aus.
- Kein Kosten-/Marktwert-Faktor (Analog zu `payroll_per_win` in moneyball) — könnte über
  die im selben Mirror vorhandenen Transfermarkt-Marktwerte (`data/tm_player_vals`)
  nachgerüstet werden.
