# English intermediate presentation draft — experiments completed so far

Team #14 · Six slides · 5 minutes + 3 minutes Q&A. Submit a viewable slide link by **Monday, November 16, 23:59**; present **Thursday, November 19**.

**Scope rule:** show completed work and preliminary results. Use “we tested” only for the 40-dong mobility pilot or the teammate's separate search/restaurant experiment. Use “next” for the 427-dong mobility panel and search/restaurant features in the mobility model.

## Slide 1 — Research question and team contributions (25 seconds)

### On slide

**Can we predict Seoul's next hot places?**

*A preliminary forecast of monthly destination mobility by administrative dong*

`[Name]  Mapped neighborhoods; normalized 30 Naver search series → align with mobility`

`[Name]  Analyzed restaurant openings and closures → use them as predictors`

`[Name]  Prepared mobility data → complete the citywide dataset`

`[Name]  Applied models learned in class → study one dong in depth`

Small line: **Team #14 HEAT · Intermediate presentation**

### Visual

Large question at the top. Put each member's **work so far → next step** on one line. The four rows are **workstream slots, not verified person assignments**: combine or split them to match the actual team. No model names or error metrics on this slide. Confirm names, completed work, and ownership of next steps with the team before submission.

### Speaker notes

“Our long-term question is whether we can identify neighborhoods gaining activity before they become established hot places. We use monthly destination mobility as a proxy, not a direct count of leisure visitors. The team has divided the work across location mapping, supporting signals, mobility data, and forecasting. Each stream has a concrete next step.”

## Slide 2 — Proposal feedback: definition and hypotheses (50 seconds)

### On slide

**We revised what a “hot place” means**

```
Main mobility pilot       40 dongs · monthly · 2023–25 data
                          Predict one month using observations through m−2

Supporting analysis       Search + restaurants · 2022–25 development
                          Predict the next three-month average / total
```

Bottom labels under three small time-series sketches:

**High activity**     **Sustained growth**     **Short surge: review separately**

### Visual

Two horizontal timelines. Keep the bottom labels attached to three simple line shapes. No formula on the slide.

### Speaker notes

“Feedback asked us to define the forecast windows and what counts as a spike. We now distinguish high activity from sustained growth; a short surge needs a data-quality check. The supporting search and restaurant study starts in 2022 because search data start then. Its three-month target summarizes the next three months. Our mobility pilot predicts one month using data through two months earlier. The 2024 mobility surge is only a candidate spike until we understand its cause.”

**Working hypotheses, not results:** (H1) past mobility has useful persistence or seasonality; (H2) lagged search and restaurant activity may add predictive information; (H3) neighborhoods flagged as rising may also change in external business or footfall measures. H2 and H3 have not been tested against the mobility target.

**Do not claim** that the initial reason for choosing three months has been verified. The teammate report compares 3, 6, and 12 months but does not explicitly state why 3 was first selected.

## Slide 3 — Data cleaning and feature engineering completed (55 seconds)

### On slide

**One monthly mobility target; aligned geography**

```
Purpose-coded mobility ──> monthly target y
2025 dong nameplate ──────> 427-code location map
Daytime living population ─> lagged input X for the pilot

External check (planned) ─> commercial footfall / sales; not model inputs
```

Bottom line: **427-dong map built · 40-dong model panel tested**

### Visual

One left-to-right data flow with the target visually prominent. Use the 427-dong map as a small spatial thumbnail only if it is already available and accurate; otherwise use the code-matching flow.

### Speaker notes

“We aggregated the Seoul destination-mobility data by administrative dong and month. The target is the daily average of purposes 4, 5, and 7. ‘Other’ is broad, so this is a proxy for non-routine mobility, not a count of leisure visitors. We built a 427-dong nameplate crosswalk, but the model results on the next slides are from an earlier 40-dong pilot. The only external input tested in that pilot is lagged daytime living population. Search and restaurant series have been analyzed separately by a teammate; they have not yet been tested as features for this mobility target. Commercial footfall and sales would be used later as corroborating evidence, kept out of the model in that test.”

## Slide 4 — Preliminary forecasting experiment (45 seconds)

### On slide

**Same 40 dongs. Same months. Six forecasts.**

```
Fit: 2023 Mar ───────── 2024 Nov    Evaluate: 2025 Jan ── Aug

Baselines: last available value · same month last year
Models:    ARIMA · SARIMA · ARIMAX · SARIMAX
```

Small line: **Two-step monthly forecast · 320 dong-month observations**

### Visual

One timeline. Below it, two rows for baselines and models. Keep model orders and equations off the slide.

### Speaker notes

“At the start of target month m, we assume mobility observations are available only through m minus two. We fitted fixed-order ARIMA and SARIMA models to mobility history. ARIMAX and SARIMAX used the same target plus daytime living population with a two-month lag. Each model was scored on the same 40 dongs and the same January–August 2025 months. This is a preliminary backtest, not a final train-validation-test result: we did not select the orders in a separate validation period.”

## Slide 5 — Initial results and a negative finding (80 seconds)

### On slide

**Adding living population did not improve WAPE**

| Forecast | WAPE ↓ |
|---|---:|
| Last available value | 11.96% |
| Same month last year | 17.20% |
| ARIMA | 10.91% |
| SARIMA | **10.89%** |
| ARIMAX + living population | 11.54% |
| SARIMAX + living population | 11.75% |

Footer: **40 dongs · Jan–Aug 2025 · 320 observations · search not included**

### Visual

Replace this table with one editable horizontal bar chart in the actual deck. Gray for the two baselines, one color for ARIMA/SARIMA, another for models with living population. Put values directly at bar ends. Use only the January–August subset; do not include the 2025 full-year chart on this slide.

### Speaker notes

“SARIMA has the lowest WAPE at 10.89%, but it is almost tied with ARIMA. The two models with living population have higher error. This is a useful negative result: an additional signal does not automatically improve the forecast. It does not tell us whether search or restaurant data will help, because those variables were not used here. The short training history and a broad mobility surge in August–October 2024 may affect these estimates, so we treat the ranking as provisional.”

## Slide 6 — Limits and next experiments (45 seconds)

### On slide

**Next: diagnose one dong, check external evidence, then scale**

```
One dong   ADF / KPSS · ACF / PACF → detailed course-model comparison
Validation   Compare rising places with external footfall or sales
Citywide   Complete 427-dong panel; test lagged search and restaurant signals
```

Small line: **2024 surge unexplained · source difference reported from Sep 2025**

### Visual

Three short steps across one line. Label all three as **planned**. No promised accuracy gain or unrun 427-dong result. A small calendar mark may indicate the planned 2026 holdout if data quality permits.

### Speaker notes

“Next we will examine one dong in depth: ADF and KPSS for stationarity, ACF and PACF for dependence and seasonality, then compare models learned in class. We plan to check rising places against external footfall or sales without using those measures as model inputs. Seoul's commercial service already ranks observed business activity; our question is whether future monthly mobility can be forecast and evaluated. Then we will extend to 427 dongs and test search and restaurant signals. These diagnostics and external checks are planned. The source-quality issues need attention before a final test.”

## Submission checks

- Replace every `[Name]` placeholder with a verified team role and concrete output.
- If the team's actual presentation language is English, use the English slide copy above and rehearse the speaker notes; do not paste the notes onto the slide.
- Rebuild Slide 5 as a **single January–August chart**, with a visible 40-dong label.
- Do not show the 427-dong model or search-to-mobility results as completed.
- Do not label ADF/KPSS, PACF, external validation, or one-dong deep modeling as completed.
- Verify view access on the submitted link.

## Evidence used

- Mobility model and baselines: `arima_family_summary.json`, `pilot_summary.json` in this output folder.
- 2024 mobility surge: `EDA_REVIEW_2026-11-11.md` in this output folder.
- Supporting search and restaurant experiment: user-provided `서울 행정동 검색량·음식점 개폐업 시계열 예측 결과 리포트.pdf`.
- [Seoul Commercial Analysis Service introduction](https://golmok.seoul.go.kr/introduce.do): rankings by store counts, sales, footfall, and residential population.
