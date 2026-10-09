# Section 2 — Washing Machine EDA: Lessons Learned

## What the Assignment Asked

Explore washing machine usage patterns across all REFIT households: cycle frequency,
duration, energy consumption, hot vs cold wash behaviour, and time-of-use patterns.
Produce a cycle-level dataset suitable for Section 3 NILM work.

---

## Step 1 — Research: Understanding the Machine and the Market

A washing machine is not a simple on/off appliance — it cycles through distinct power
phases within a single wash programme. Before writing any detection code we mapped the
physical reality.

**Power phases in a typical UK front-loader (40°C cotton, 75 min):**

```mermaid
timeline
    title Power draw across one wash cycle
    section Heating (0–20 min)
        1800–2500 W : Heating element raises drum temperature
                    : Thermostat cycles ON / OFF
    section Wash (20–40 min)
        200–400 W : Motor agitates drum
                  : No heat
    section Drain (40–43 min)
        50–80 W : Drain pump only
                : Brief but deep power drop
    section Rinse (43–60 min)
        200–400 W : Fill · agitate · drain — repeated 1–3×
    section Spin (60–75 min)
        200–350 W : High-speed drum spin
```

**UK market context (2013–2015, relevant to when REFIT was collected):**

We looked up dominant washing machine brands sold in the UK during the data collection
window: Beko, Hotpoint, Bosch, Indesit. The shortest wash programme on the market was
the **Bosch SpeedPerfect 15-minute** cycle. No domestic programme runs longer than
3 hours. These two facts gave us our minimum (15 min) and maximum (180 min) cycle bounds
— grounded in real products, not guessed from the data.

---

## Step 2 — Data Complexity We Found

### The drain-phase fragmentation problem

The most important discovery came from inspecting the *detected* cycle durations before
any tuning: the median was **19 minutes** instead of the expected 60–90 minutes.

```mermaid
sequenceDiagram
    participant WM as WM Power
    participant A1 as Fixed 100W threshold
    participant A2 as Hysteresis (our fix)

    Note over WM: Heating phase (0–20 min)
    WM->>A1: 1800–2500 W ✓ ON
    WM->>A2: 1800–2500 W ✓ ON

    Note over WM: Agitation (20–40 min)
    WM->>A1: 200–400 W ✓ ON
    WM->>A2: 200–400 W ✓ ON

    Note over WM: Drain pump (40–43 min)
    WM->>A1: 50–80 W ✗ EXITS — cycle ends at 40 min
    WM->>A2: 50–80 W ✓ STAYS — above off_thresh (25 W)

    Note over WM: Rinse + Spin (43–75 min)
    WM->>A1: 200–400 W NEW cycle starts (fragmented)
    WM->>A2: 200–400 W ✓ same cycle continues

    Note over WM: Machine off (75+ min)
    WM->>A1: 0 W — OFF
    WM->>A2: 0 W — 5-min patience → EXIT at 75 min ✓
```

A fixed 100 W ON/OFF threshold worked fine for the heating phase but treated the
2–4 minute drain pump (50–80 W) as the machine turning off. Every hot wash was
being split into a heating segment + a rinse/spin segment — neither of which is a
real cycle.

### Why House 5 was the clue

House 5 showed a median cycle duration of ~50 minutes under the broken algorithm.
It turned out to run 45% cold washes — programmes that stay at 200–400 W throughout
with no heating spike. Without a deep drain dip, the full cycle was captured. This
anomaly led directly to the diagnosis above.

### Variable machine signatures across 19 houses

Not all washing machines have the same idle power or standby draw. A 100 W ON threshold
that works for one machine may be too high for another with a weaker motor. We needed
per-house thresholds derived from each machine's own power distribution.

---

## Step 3 — Our Approach and Why

### Dynamic per-house thresholds

Instead of a single global threshold we derived ON and OFF thresholds from each
machine's active power distribution:

```mermaid
flowchart TD
    A([WM power series\nall 1-min readings]) --> B[Filter: keep readings > 5 W\nremove true-off zeros]
    B --> C[Compute percentiles\nof active distribution]
    C --> D[p05 ≈ drain pump level\ntypically 50–80 W]
    C --> E[p20 ≈ minimum agitation level\ntypically 100–200 W]
    D --> F[OFF threshold = max p05 × 0.6 · 25W\nrounded to 5 W]
    E --> G[ON threshold = max p20 · 60W\nrounded to 10 W]
    F --> H{≥ 200 active readings?}
    G --> H
    H -->|No| I[Fallback: ON=100W · OFF=50W]
    H -->|Yes| J([House-specific ON + OFF thresholds])
    I --> J

    style A fill:#ddeeff,stroke:#2b7be0,color:#0a2a5e
    style J fill:#dff5ec,stroke:#2eaa6a,color:#0d3d23
    style I fill:#fff3e0,stroke:#e05a2b,color:#7a2a00
```

The key insight: the OFF threshold is set *below* the drain pump level (p05 × 0.6),
so the machine stays in the ACTIVE state during drainage. It only exits when the
drum has fully stopped and power returns to true standby (<25 W).

### Hysteresis state machine

Two thresholds without patience logic would still fail: a single 1-minute reading
below the OFF threshold would incorrectly end the cycle. We added a 5-minute patience
window (OFF_PATIENCE):

```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> CONFIRMING : reading ≥ on_thresh
    CONFIRMING --> IDLE : streak broken before 3 min
    CONFIRMING --> ACTIVE : on_thresh sustained ≥ 3 min\ncycle START recorded

    ACTIVE --> ACTIVE : power ≥ off_thresh
    ACTIVE --> DRAINING : power < off_thresh\nlow_streak counter starts

    DRAINING --> ACTIVE : power returns ≥ off_thresh\nbefore 5-min patience
    DRAINING --> VALIDATE : below off_thresh for 5 consecutive min\ncycle END recorded

    VALIDATE --> CYCLE_OK : 15 min ≤ duration ≤ 180 min
    VALIDATE --> DISCARD : outside bounds

    CYCLE_OK --> IDLE : record cycle metadata
    DISCARD --> IDLE
```

The 3-minute entry streak prevents transient spikes (kettle, microwave) from
triggering a cycle. The 5-minute exit patience bridges the drain phase (2–4 min)
with a safety margin.

### Hot vs cold wash classification

```mermaid
flowchart LR
    A([Detected cycle]) --> B{Any 1-min bin\n≥ 1800 W for ≥ 3 min?}
    B -->|Yes| C[HOT WASH\nheating element confirmed active]
    B -->|No| D[COLD / WARM WASH\nmotor only · 30°C or less]
    C --> E[Record heat_mins · heat_energy_wh]
    D --> F[hot_wash = False]
    E --> G[Energy saving = hot_median − cold_median]
    F --> G

    style C fill:#fde8e0,stroke:#e05a2b,color:#7a2a00
    style D fill:#ddeeff,stroke:#2b7be0,color:#0a2a5e
```

1800 W is the rated minimum for a UK front-loader heating element — below this, only
the motor is running. The 3-minute minimum prevents a brief spike from misclassifying
a cold wash.

---

## Step 4 — Results

| Metric | V1 (fixed 100 W) | V2 (hysteresis) |
|---|---|---|
| Total cycles detected | 3 502 | **6 776** |
| Median duration | 19 min | **66 min** |
| Median energy (cold wash) | ~50 Wh | **~200 Wh** |
| Median energy (hot wash) | ~537 Wh | **~500 Wh** |
| Hot wash fraction | 88.8% | **87.4%** |
| Energy saving hot → cold | 481 Wh/cycle | **401 Wh/cycle** |

The V1 numbers were physically wrong: 19-minute cycles and 50 Wh cold washes are not
washing machine cycles. The V2 numbers match what appliance manufacturers and
energy auditors report for UK households.

**Notable finding — House 19:** Only 9% hot washes. Nearly all cycles at 30°C or
cold. This is a genuine behavioural outlier across the cohort, not a detection error —
the machine's power trace confirms it.

**Usage timing:** Saturday 8–10 am is the busiest slot (2.1% of all cycles). Weekday
morning peak at 7–10 am. Tuesdays and Wednesdays noticeably quieter across the cohort.

---

## Lessons Learned

**1. Physical domain knowledge diagnosed the bug before the data did.**
We knew the drain pump runs at 50–80 W before we ran a single query. That knowledge
made the 19-minute median immediately suspicious rather than accepted.

**2. One outlier house can be a diagnostic gift.**
House 5's correct duration under a broken algorithm pointed directly to the cold-wash
vs hot-wash difference as the mechanism. Treating anomalies as signals rather than
noise accelerated the fix.

**3. Thresholds should come from the data's own distribution, not from intuition.**
A 100 W ON threshold was a reasonable starting guess. The data showed that wash
machines in this cohort have p20 active readings between 60–80 W — a 100 W threshold
was already clipping the low end of normal agitation. Deriving from percentiles makes
the threshold adaptive without adding complexity.

**4. "More cycles detected" is not automatically better.**
Going from 3 502 to 6 776 cycles looks like improvement — and it is — but the
right validation is whether duration and energy now match known appliance behaviour.
We validated against UK manufacturer cycle time specs, not just against V1.

**5. The hot/cold energy gap is the most actionable finding.**
A 401 Wh median saving per cycle translates to roughly 80–120 cycles per household
per year at 30°C instead of 40–60°C. That is a concrete recommendation for Section 4.
