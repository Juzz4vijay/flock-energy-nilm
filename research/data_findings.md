# Data Understanding & Research Findings

> First-pass findings from direct file inspection + literature review.  
> Date: 2026-10-08

---

## 1. Dataset at a Glance

| Item | Fact |
|---|---|
| Source | University of Strathclyde / Loughborough, UK |
| Coverage | 20 houses (numbered 1–21, **House 14 missing**), Oct 2013 – Jul 2015 |
| Channels | 1 aggregate clamp + 9 IAMs per house |
| Units | Active power in **Watts only** (no reactive power) |
| Raw sampling | ~2–12 s between logged rows (logs on **change**, not fixed interval) |
| Format | `Time, Unix, Aggregate, Appliance1…9 [, Issues]` |

---

## 2. Raw vs. Clean — Key Differences

### 2.1 BST → UTC correction
The cleaned data shifted **both** the datetime string and the Unix timestamp by −3600 s.  
Raw row 1: `2013-10-09 14:06:17`, Unix `1381327577`  
Clean row 1: `2013-10-09 13:06:17`, Unix `1381323977`  
Difference: exactly −3600 s on both fields.  
UK was in BST (UTC+1) on 9 Oct 2013; clocks went back 27 Oct 2013.  
**Action for DATA.md**: state that raw timestamps are local-time (BST or GMT depending on date) and that cleaning converted everything to UTC.

### 2.2 Part 1 vs. Part 2 (House 1)
| File | Date range | Missing sensor encoding |
|---|---|---|
| RAW_House1_Part1.csv | Oct 2013 – Oct 2014 | 0 (unavailable = zero) |
| RAW_House1_Part2.csv | Jun 2014 – end | NaN (unavailable = NaN) |
| Overlap | Jun – Oct 2014 | Both files cover this window |

The Part1 / Part2 distinction is documented in the README:  
> Part1 = sensors not available stored as 0; Part2 = NaN to distinguish from genuine zero.

**Action**: concatenate both parts; deduplicate the ~4-month overlap before resampling.

### 2.3 Duplicate Unix timestamps
- 65,326 unique Unix values appear in two consecutive rows in Part1.
- 89% of duplicate pairs share the same Aggregate value → sensor-polling collision artifact (poll window = 6–8 s, sensor may update anywhere in that window).
- 11% have differing values → genuinely ambiguous; apply a stated dedup rule (keep first or take mean).

### 2.4 Spikes and impossible values (raw Part1, first 200k rows)
| Channel | Spikes > 4 kW |
|---|---|
| Aggregate | 94 |
| Appliance1 (Fridge) | 9 |
| Appliance6 (Dishwasher) | 2 |
| Appliance8 (Television) | 5 |

Max Aggregate observed: 11,193 W (raw).  
The clean pipeline replaced IAM spikes > 4,000 W with zero.  
**Action in raw cleaning**: flag values > 4,000 W as NaN, log the count, do not silently drop.

### 2.5 Gap distribution (raw, first 500k rows)
- Median inter-row gap: 2–3 s (not 8 s — the 6–8 s figure is the polling interval, not the log interval).
- Gaps > 5 min: 69 occurrences.
- Largest gap: 25.6 h (Nov 7–8, 2013).
- Standard deviation of gaps: 22.8 s → very right-skewed; a histogram is needed for the report.

---

## 3. Washing Machine — Column Map (all 19 houses)

Verified by checking non-zero activity and peak power in each house's Appliance column.  
Peak power range: 2,100–3,072 W across all houses — consistent with European front-loaders with an internal heater.

| House | WM Column | Peak W (sample) | Notes |
|---|---|---|---|
| 1 | Appliance5 | 3,072 | Also Appliance4 = Washer-Dryer |
| 2 | Appliance2 | 2,422 | |
| 3 | Appliance6 | 2,115 | Tumble dryer present |
| 4 | Appliance4 | 2,576 | **Two WMs**: also Appliance5 |
| 5 | Appliance3 | 2,212 | Standby leak at 1–5 W (see §3.1) |
| 6 | Appliance2 | 2,220 | |
| 7 | Appliance5 | 2,248 | Tumble dryer present |
| 8 | Appliance4 | 2,603 | Also Appliance3 = Washer-Dryer |
| 9 | Appliance3 | 2,345 | Also Appliance2 = Washer-Dryer |
| 10 | Appliance5 | 2,236 | |
| 11 | Appliance3 | 2,422 | |
| 12 | — | — | **No WM** — exclude |
| 13 | Appliance3 | 2,336 | Appliance changed 25 Mar 2015 |
| 15 | Appliance3 | 2,504 | Tumble dryer present |
| 16 | Appliance5 | 2,204 | |
| 17 | Appliance4 | 2,285 | Tumble dryer present |
| 18 | Appliance5 | 2,367 | Also Appliance4 = Washer-Dryer |
| 19 | Appliance2 | 2,450 | |
| 20 | Appliance4 | 2,390 | Tumble dryer present |
| 21 | Appliance3 | 2,363 | |

### 3.1 House 5 standby anomaly
Value distribution of Appliance3 (first 200k rows):

| Bucket | Count |
|---|---|
| 0 W | 6 |
| 1–5 W | 163,520 |
| 5–20 W | 2,449 |
| 20–50 W | 1,468 |
| 50–100 W | 1,812 |
| 100–500 W | 3,375 |
| > 1 kW | 1,030 |

75% of rows sit at 1–5 W — forward-filled standby current.  
Only 3.1% of rows exceed 50 W; 2.2% exceed 100 W.  
**Action**: use **≥ 50 W** as the on-threshold (not the 20 W default) — tune per-house in EDA.

---

## 4. House Coverage by Cohort

| Cohort | Houses | Approx. start | Note |
|---|---|---|---|
| Early | 1–9 | Sep–Dec 2013 | Affected by Feb 2014 outage |
| Late | 10–21 (no 14) | Jan–Mar 2014 | Mostly post-outage |

Most late-cohort houses start after the Feb 2014 outage, so that gap won't appear in their data.

---

## 5. Model & Validation Plan

### 5.1 Standard house split (from D'Incecco et al. 2019 / transferNILM)
| Role | Houses |
|---|---|
| Train | 2, 5, 7, 9, 15, 16, 17 |
| Validate | 18 |
| **Test** | **8** |

House 8 is a deliberately hard test: it has a Washer-Dryer (Appliance3) alongside the WM (Appliance4), making it a realistic failure-analysis case.

### 5.2 Models (in order of complexity)
1. **Always-off baseline** — exposes the misleading-MAE trap (WM is off >90% of the time)
2. **Rule-based** — detect ~2 kW rises lasting 10–30 min in the aggregate
3. **LightGBM** — rolling features (5, 15, 30, 60, 120 min), hour, weekday
4. **Seq2Point CNN** (Zhang et al., AAAI 2018) — 5 conv layers + dense, window ~199 samples at 1-min resolution

### 5.3 Metrics
| Metric | Why |
|---|---|
| MAE overall and when on | Standard; split exposes always-off trap |
| SAE (Signal Aggregate Error) | Total energy accuracy — what utilities care about |
| Daily energy error (kWh/day) | Billing-style accuracy |
| F1 for on-state | Timing accuracy |
| Cycle-level F1 + per-cycle energy error | Ties to EDA and business use |

---

## 6. Practical / Business Hook

Switching from 60°C → 30°C saves ~0.65 kWh per cycle (confirmed from literature).  
At 4–5 washes/week per household: **~130–170 kWh/year** saved.  
This saving is directly measurable from the EDA: fraction of each cycle's energy spent in the ~2 kW heating phase.  
**Measurement approach for `Recommendation.md`**: difference-in-differences on heating-phase energy per cycle, nudged vs. control households, over ≥ 8 weeks.

---

## 7. Two Adjustments to the Roadmap

| Item | Roadmap says | Evidence says |
|---|---|---|
| WM on-threshold | 20 W | Start at **50 W**; tune per-house (House 5 standby at 1–5 W invalidates 20 W) |
| Raw House 1 input | One file | **Two files** (Part1 + Part2) with a ~4-month overlap; concatenate and dedup |

---

## 8. References

1. Murray, D. et al. (2017). An electrical load measurements dataset of UK households from a two-year longitudinal study. *Scientific Data*, 4, 160122. https://strathprints.strath.ac.uk/58873/
2. REFIT Cleaned Data README. University of Strathclyde. https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt
3. Zhang, C. et al. (2018). Sequence-to-point learning with neural networks for non-intrusive load monitoring. *AAAI*. https://homepages.inf.ed.ac.uk/csutton/publications/seq2pointNilm.pdf
4. D'Incecco, M. et al. (2019). Transfer learning for non-intrusive load monitoring. *IEEE Transactions on Smart Grid*. https://ar5iv.arxiv.org/html/1902.08835
5. Kelly, J. & Knottenbelt, W. (2015). Neural NILM: Deep neural networks applied to energy disaggregation. *ACM BuildSys*.
6. Selectra UK (2024). Washing 30 instead of 60 degrees — energy savings. https://selectra.co.uk/energy/news/washing-30-instead-of-60-degrees-savings
