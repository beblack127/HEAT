# Mid-presentation EDA: Openings and Closures

## Figures

All figure titles, labels, captions, and filenames are in English. Each SVG is a separate 16:9, presentation-ready chart.

1. `01_district_jungnang_openings_closures_increase.svg`
2. `02_district_jung_decrease.svg`
3. `03_neighborhood_samseong_1_increase.svg`
4. `04_neighborhood_yeoksam_2_decrease.svg`

The charts show monthly openings and closures separately from January 2023 through December 2025, plus annual totals and a short interpretation. They use raw monthly counts; no fitted trend line is drawn.

## Selection method

The starting point was the full monthly district/neighborhood table in `EDA_동별분석/월별_구동_개폐업_2023-2025.csv`. For each region, openings and closures were combined only to rank the overall event-volume change between the 2023 and 2025 annual totals. The two largest increases and two largest decreases were reviewed at each spatial level, then checked against the monthly series. The final set includes one increasing and one decreasing example at the district and neighborhood levels:

| Level | Region | 2023 total events | 2024 total events | 2025 total events | 2023–2025 change |
|---|---|---:|---:|---:|---:|
| District | Jungnang-gu | 970 | 872 | 1,399 | +44.2% |
| District | Jung-gu | 2,372 | 1,850 | 1,925 | −18.8% |
| Neighborhood | Samseong 1-dong, Gangnam-gu | 729 | 844 | 1,021 | +40.1% |
| Neighborhood | Yeoksam 2-dong, Gangnam-gu | 376 | 297 | 192 | −48.9% |

## Interpretation and limits

- Total events mean openings plus closures. This is an event-volume measure, not a count of active stores, a net-growth rate, or a direct measure of commercial prosperity or decline.
- In Jungnang-gu, the 2025 increase is dominated by closures (460 in 2023 to 930 in 2025), while openings fall (510 to 469). The size and timing of the closure increase should be checked against source reporting changes before drawing a market conclusion.
- In Jung-gu, June 2023 has 311 closures. Replacing that month with the mean of the other eleven 2023 months changes the 2023–2025 total-event comparison from −18.8% to approximately −10.9%.
- In Yeoksam 2-dong, June 2023 has 105 closures, compared with 5–19 in the other 2023 months. Replacing that month with the mean of the other eleven months changes the comparison from −48.9% to approximately −32.0%. The chart retains the observed data and calls out the spike.
- Dong assignment follows the existing EDA mapping, which uses current registry addresses and coordinates. Historical events may not reflect a location's administrative dong at the time of the event.

## Regenerate

From the project root:

```bash
python3 EDA_중간발표용/create_presentation_charts.py
```
