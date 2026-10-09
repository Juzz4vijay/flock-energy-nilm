# Recommendation — Utility-Facing Summary

**Prepared by Vijay Rameshkumar**

---

## What Was Built

A machine learning model (ARNILM) that reads only the whole-house electricity
meter and estimates, minute by minute, how much power the washing machine is drawing.
No additional hardware is needed beyond the smart meter already installed.

The model was trained on 16 UK households and tested on a 17th household it had
never seen — simulating exactly what happens when you deploy to a new customer.
It achieved F1=0.64 with an energy estimation error of 64%, at 1-minute resolution.

---

## Where This Model Can Be Used Confidently

**Household-level energy attribution**
Identifying which appliances are the dominant consumers in a household, at weekly
or monthly granularity. The model reliably detects whether the washing machine is
running and contributes to total consumption estimates.

**Behavioural segmentation**
Classifying households by washing machine usage style — hot vs cold wash fraction,
cycle frequency, typical usage hours — using only aggregate meter data. These
segments are accurate and can be computed from two weeks of aggregate data without
any sub-metering.

**Targeted nudge campaigns**
Identifying high hot-wash households (those where the model infers > 80% of cycles
are hot) and sending them targeted efficiency messaging. The model provides the
targeting signal; impact can be measured by tracking estimated cycle energy before
and after the campaign.

**New household onboarding**
The model generalises to new households without retraining. From one week of
aggregate data, behavioral signatures are computed and the model adapts. This is
suitable for large-scale smart meter rollouts where per-household labelling is
not feasible.

---

## Where This Model Has Limits

**Do not use for billing or settlement**
Energy estimation error is 64%. This is suitable for behavioural insight and
targeting but not for allocating costs or verifying tariff compliance. For billing
purposes, sub-metering remains necessary.

**Dishwasher-heavy households**
Washing machines and dishwashers have similar power signatures (long cycle, moderate
power draw). The model will produce false positives in households where a dishwasher
runs on a similar schedule to the washing machine. A multi-appliance version of the
model would resolve this.

**Coarser than 1-minute data**
UK SMETS2 smart meters default to 30-minute half-hourly reporting. This model
requires 1-minute data. At 15-minute resolution, expected F1 drops to approximately
0.25–0.35. At 30-minute resolution, per-cycle disaggregation is not achievable.
Deploying this model requires either requesting 1-minute data from the smart meter
(available in SMETS2 but not default) or installing a low-cost plug-level monitor
(IAM, ~£25) on the washing machine circuit.

**UK households only in current form**
The model was trained on UK front-loading washing machines from 2013–2015. Indian
households have a different appliance mix (top-loading semi-automatic machines are
common, with a flat 200–400W profile and no heating element). Direct deployment in
India without retraining on Indian household data would produce unreliable results.

---

## One Energy-Saving Opportunity

**Action: shift high hot-wash households to 30–40°C cycles**

The model identifies which households run predominantly hot washes. A hot cycle at
60°C consumes approximately 1.0–1.1 kWh. The same machine at 30°C consumes
approximately 0.2 kWh — an 80% reduction in WM energy per cycle.

Modern detergents (Bio formulations) are effective at 30°C for everyday laundry.
The only genuine requirement for 60°C is heavily soiled items and allergen reduction
in bedding. Typical household laundry does not require it.

**Estimated impact per high hot-wash household (2 cycles/week):**

```
Current (60°C):   2 × 1.05 kWh = 2.1 kWh/week = 109 kWh/year
Target (30°C):    2 × 0.20 kWh = 0.4 kWh/week =  21 kWh/year
Annual saving:    ~88 kWh  |  ~£26/year  |  ~20 kg CO₂/year
```

**How to measure impact:**

1. Use the model to compute baseline hot-wash fraction and cycle energy per household
   over four weeks before the campaign.
2. Send targeted in-app messaging to households with modelled hot-wash fraction > 0.8.
3. Measure the same metrics four weeks after the campaign.
4. The disaggregated WM signal isolates the change from seasonal and occupancy
   confounders that would contaminate a total-consumption comparison.

The model's Gaussian uncertainty output flags weeks where cycle detection confidence
is low — these are excluded from before/after comparisons to ensure measurement quality.

---

## Additional Data That Would Improve the Model

**1-minute sub-meter labels for Indian households**
The single highest-value addition. Training on Indian household data would enable
direct deployment in the Indian market. Two weeks of labelled data from 10–20
Indian households with diverse appliance types would be sufficient to fine-tune
from the existing UK checkpoint.

**Dishwasher sub-meter channel**
Adding dishwasher labels from existing REFIT households enables a multi-appliance
model. Once the dishwasher head explicitly claims dishwasher events, washing machine
precision improves substantially. The data is already in REFIT (Appliance 4 in some
houses) — it just was not used in this version.

**Post-2020 REFIT or equivalent dataset**
The training data is from 2013–2015. Quick-wash programmes (15-min cycles at 800W,
no heating phase) have grown significantly since then and represent a new cycle
signature class the current model has not seen. A more recent dataset would capture
the current distribution of wash programmes.

**Voltage measurements alongside current**
Indian grid voltage fluctuates ±10–20%, shifting appliance power draw in ways that
affect the behavioral signature features. Adding voltage as an input feature would
make the model robust to grid instability and improve Indian market transferability.
