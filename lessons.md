# comicml — Lessons Learned

A running log of empirical findings from training the page-corner regression model.
Add new entries at the top with a date so the recent learnings are easy to find.

---

## 2026-05-04 — Diminishing returns confirmed at ~1.7k pages; recipe needs to change next

**Context:** GT grew 1589 → 1702 corrected (+113 pages, +7%, all VOY_3-7).
Ran the *same* recipe as 1589pg_e280: 280 epochs, bs=8, lr=1e-4, input=768,
warm_restarts T_0=40, seed=137. Same holdout.

**Result: mixed — not strict dominance.**

| | mean | median | p95 | max |
|---|---|---|---|---|
| 1589pg_e280 | 16.40 | **15.43** | 24.67 | **35.99** |
| 1702pg_e280 | **16.26** | 15.75 | **23.53** | 38.61 |
| Δ           | -1% ✓ | +2% ✗ | -5% ✓ | +7% ✗ |

Per-dir: 1702 better on DS9E20 and DS9_1996_5, worse by 0.4 px on DS9E23.

**The signal: data scaling has hit its inflection point.**

| step | data Δ | mean Δ | data efficiency |
|---|---|---|---|
| 1420 → 1589 | +12% | -15% | high (1.25× per %) |
| 1589 → 1702 | +7% | -1% | low (0.14× per %) |

**Other warning signs at this scale:**
- Best checkpoint shifted from e236 → **e252** — model needed more time inside
  cycle #3 (e120→e280) and barely got there. Suggests the cooldown ran out.
- Train_px (12.78) is approaching what looks like a label-noise floor — it can't
  fall much further on this GT without overfitting.
- Max went *up* even as p95 went *down* — fewer-but-still-bad outliers, the
  worst category for production.

**Rule going forward — when adding data fails to give >5% mean improvement, the
recipe (not the data) is the bottleneck.** Stop adding data and change ONE of:

1. **Bump epochs to 600** (next clean warm-restart boundary). Buys cycle #4
   (e280→e600) — full new cooldown + much more time inside cycle #3. Doubles
   wall-clock; try this *first*.
2. **Bump input_size to 1024.** Bigger receptive field → tighter localisation
   on small artifacts; ~1.8× wall-clock and may need bs=4 to fit MPS memory.
3. **Re-seed and ensemble more.** Three runs at seed=137,42,7 with the current
   1702pg dataset would diversify the tail without needing more GT.

I'd try (1) first because it's the smallest change that's still likely to
break the plateau, and the warm-restart schedule is *designed* for it.

**Ensemble decision:** added 1702pg_e280 anyway (now 5 models). Per the
2026-04-22 ensemble lesson, the value of a member isn't strict-dominance —
it's *making different mistakes*. 1702 wins on 2 dirs and loses 0.4 px on 1
vs 1589: that's textbook decorrelated tail error and exactly what the
ensemble averaging exploits.

**Holdout note:** as planned in the 04-25 lesson, the *next* run should add
one VOY issue to the holdout (e.g. VOY_5) — we now have 7 VOY issues, plenty
for a meaningful holdout slice. Before that run, re-eval all 5 ensemble
members on the new holdout to keep numbers comparable.

---

## 2026-04-25 — More data + same recipe = strict dominance, no recipe tuning needed

**Context:** GT grew from 1420 → 1589 corrected pages (+5 new DS9_1996 issues, +2
Voyager issues — first non-DS9 series in the dataset). Re-ran with the *exact*
1420pg_e280 recipe: 280 epochs, bs=8, lr=1e-4, input=768, warm_restarts T_0=40,
seed=137. Same holdout (DS9E20 / DS9E23 / DS9_1996_5).

**Result: strictly dominant on every metric, every directory.**

| | mean | median | p95 | max |
|---|---|---|---|---|
| 1420pg_e280 (prev best) | 19.23 | 18.69 | 27.63 | 44.29 |
| **1589pg_e280**         | **16.40** | **15.43** | **24.67** | **35.99** |
| improvement             | -15% | -17% | -11% | **-19%** |

**Per-dir: no regression on any directory** — all 3 holdout dirs improved by
~3 px mean.

**Why this is interesting:**
- Adding +169 corrected pages (+12% data) yielded -15% mean error and -19% max.
  Diminishing returns have *not* set in yet at this scale.
- The new VOY pages (a different series with different paper / ink style) helped
  rather than hurt holdout DS9 performance — diversity is still net-positive at
  ~1.5k pages.
- The previous run's "best @ e212" pattern repeated at e236 — confirming the
  warm-restart cooldown of cycle #3 (e120→e280) is where the bulk of fine
  refinement happens. Don't end at e120 or e180 mid-cycle.
- Train/val gap at the best checkpoint: 13.88 train vs 17.51 val px → ~3.6 px,
  same as 1420pg_e280 (18.50 vs 20.27). No overfitting signature; same recipe
  scales linearly with data so far.

**Heuristic confirmed:** when adding ≤20% more data, **don't change the recipe**
— same epochs, same LR, same seed. The model architecture has headroom; the
new data is the only delta you need.

**When to revisit the recipe:** if a future run *plateaus* (val_px stops
improving while train_px keeps falling) or fails to beat its predecessor, then
consider 600 epochs (next clean warm-restart boundary) or input_size 1024.
Until then, just `--epochs 280 --warm-restarts 40 --seed 137`.

**Ensemble decision:** added 1589pg_e280 to the ensemble (now 4 models). Did
*not* drop 956pg or 1000pg even though they're now ~30% worse standalone —
per the 2026-04-22 ensemble lesson, weaker members still cancel uncorrelated
tail errors. Re-evaluate dropping members only if the 4-model ensemble starts
underperforming the top-2 ensemble on the same holdout.

---

## 2026-04-22 — Epoch budget must scale with dataset size *and* land on a schedule boundary

**Context:** Validation error got dramatically worse as more comics were added to the
training set, despite the holdout staying constant (`DS9E20, DS9E23, DS9_1996_5`).

| run | dataset | epochs | best val_px | final train_px |
|---|---|---|---|---|
| 956pg       |  956 | 120 | 23.63 | 29.85 |
| 1000pg      | 1000 | 120 | 24.67 | 28.89 |
| 1078pg      | 1078 | 120 | 34.16 | 32.72 |
| 1078pg_e180 | 1078 | 180 | 31.99 | 31.96 |
| 1226pg      | 1226 | 120 | 39.09 | 34.69 |
| 1226pg_e180 | 1226 | 180 | 36.20 | 34.26 |
| **1420pg_e280** | **1420** | **280** | **20.27** | 13.58 |

**What looked like the problem (and wasn't):**
- Label noise in newly added comics — checked, all consistent (winding, page-area,
  rotate180 ratios, etc. all matched the older comics)
- Distribution shift from adding more Marvel — partially true but not the main cause
- Model capacity — ResNet-18 has plenty for 1.4k images

**The actual root cause was two compounding issues:**

1. **Epochs need to grow with dataset size.** With a fixed 120-epoch budget, larger
   datasets get fewer effective passes through each example before LR collapses.
   Train_px (not just val_px) was rising — the model literally couldn't fit the
   larger training set in the time given.

2. **The cosine warm-restart schedule (`T_0=40 T_mult=2`) has restart boundaries at
   epochs 40, 120, 280, 600.** Ending mid-cycle (e.g. epoch 180) leaves the model
   stuck at a cold LR mid-cooldown, with no opportunity to escape the local minimum
   it found. Always end *on* a restart boundary so the final cycle gets a full
   cooldown.

**Rules of thumb going forward:**

| dataset size | recommended epochs | notes |
|---|---|---|
| ~1000  | 120 | first warm cycle ends here |
| ~1200  | 120 or 280 | 120 ends-on-boundary; 280 better if time allows |
| ~1400  | 280 | confirmed empirically |
| ~1800+ | 280 or 600 | 600 = next clean boundary |

**Don't pick non-boundary epoch counts** (e.g. 150, 200, 240) when using warm
restarts — you'll always be stuck in the middle of a cooldown.

**Watch the train/val gap, not just val_px.** If `train_px` is *also* high, the
model is under-trained, not overfitting. Solution: more epochs / different
schedule, NOT regularization.

---

## 2026-04-22 — Always ensemble; new models add value even when older ones are "worse"

**Context:** When `1420pg_e280` came out strictly dominant (every metric, every
holdout dir), the temptation was to switch to a single-model production setup.

**Don't.** The 3-model ensemble (956pg + 1000pg + 1420pg_e280) outperforms any
single member, especially on the **tail** (p95, max). Different models trained
on different dataset sizes / random states make different mistakes; averaging
cancels uncorrelated error.

**Tail metrics matter more than mean** in production: a model with mean=20 but
max=75 will produce occasional wildly-wrong predictions that need manual fixing.
Ensembling drops both the max and the p95.

```
                     mean  p95   max
single 956pg         22    42    75
single 1420pg_e280   19    28    44   ← dominant on every metric
3-model ensemble     ~17   ~22   ~35  ← projected; verify with eval
```

**Process for evaluating a new candidate model:**
1. Run `python3 -m comicml.train eval --model <new>.pt` — check overall mean/p95/max
2. Run the same eval on each ensemble member individually
3. Compare per-directory numbers — confirm no regressions on any single dir
4. Only then add the new model to `models/ensemble_config.json` (also keep the
   `ENSEMBLE_MODELS` fallback list in `comicml/inference.py` in sync — it's the
   safety net when the JSON is missing/unparseable)
5. Restart the comicscan webapp (the ensemble is cached in memory)

---

## 2026-04-22 — JSONL log files: always guard against parallel writers

**Context:** Two training runs were accidentally launched targeting the same
output filename. Both wrote to the same `_log.jsonl` file in append mode,
producing interleaved JSON lines that needed manual de-interleaving to read.
The `.pt` checkpoints were also overwritten by whichever run wrote `is_best=True`
last.

**Fix already applied** (in `comicml/train.py`): refuse to start training if
the target log file was modified in the last 5 minutes. The error message
tells the user how to override.

```python
if log_path.exists():
    age = time.time() - log_path.stat().st_mtime
    if age < 300:
        raise SystemExit(f"ERROR: {log_path} was modified {age:.0f}s ago ...")
```

**General principle:** any append-mode log file used by a long-running process
needs a writer-lock or a freshness check. JSONL is especially treacherous because
the file *looks* valid (each line still parses) but the data is interleaved
nonsense.

**Before launching any training:**
```bash
ps -ef | grep "comicml.train" | grep -v grep                     # nothing running?
ls -la models/comicml_model_reg_768_<name>.pt 2>/dev/null        # output free?
ls -la models/comicml_model_reg_768_<name>_log.jsonl 2>/dev/null # log fresh?
```

---

## 2026-04-22 — Ground truth lives in two places; always re-collect before training

**Context:** Asked "did you add new GT?" and `ground_truth.json` showed no
changes — but the user *had* added GT. The webapp writes per-comic
`.comicscans_session.json` files, and `ground_truth.json` is only updated when
`comiceval.py collect` is explicitly run.

**Workflow before any new training run:**
```bash
cd /Users/james/Documents/dev/comicscan
python3 comiceval.py collect \
    /Users/james/Documents/comic-processing/raw-scans \
    /Users/james/Documents/comics-scanned
```

**Critical:** `collect` *overwrites* `ground_truth.json`. Always pass **all**
scan roots in one invocation. We learned this the hard way when a single-root
collect overwrote the file with only 552 entries (lost ~870 entries until
re-collected from both roots).

**Always snapshot before destructive operations:**
```bash
cp ground_truth.json ground_truth.json.<count>.bak
```

**Keep .bak files numbered by entry count** (e.g. `ground_truth.json.1420.bak`)
so you can tell at a glance which snapshot is which.

---

## 2026-04-22 — Holdout stability matters for run-to-run comparability

The holdout has been kept constant at `DS9E20, DS9E23, DS9_1996_5` (114 pages,
mix of Malibu and Marvel) across every run. This is what made the
performance-vs-dataset-size analysis above possible — if the holdout shifted
each time, we couldn't compare runs.

**Don't expand or change the holdout** without a strong reason. If you do,
re-evaluate prior models on the new holdout to maintain comparability.

If holdout feels too small (114 pages → noisy per-epoch val numbers), the right
fix is to look at **best val_px across the run** (which the trainer already
saves) rather than the per-epoch number, *not* to enlarge the holdout.

---

## Reference: the eval+promote loop

```bash
cd /Users/james/Documents/dev/comicscan

# 1. Collect any new GT (always pass ALL scan roots in one invocation)
venv-cs/bin/python3 comiceval.py collect \
    /Users/james/Documents/comic-processing/raw-scans \
    /Users/james/Documents/comics-scanned

# Snapshot before any new training
cp data/ground_truth.json data/ground_truth.json.<count>.bak

# 2. Train (epoch count from the table above; end on a restart boundary)
nohup venv-cs/bin/python3 -m comicml.train train \
    --output comicml_model_reg_768_<N>pg_e<E>.pt \
    --epochs <E> --input-size 768 --warm-restarts 40 --seed 137 \
    > /tmp/comicml_train_<N>pg_e<E>.log 2>&1 &

# 3. Eval the new model + each existing ensemble member
venv-cs/bin/python3 -m comicml.train eval --model comicml_model_reg_768_<N>pg_e<E>.pt
venv-cs/bin/python3 -m comicml.train eval --model <each_existing_ensemble_member>.pt

# 4. Compare per-directory numbers; if no regressions, add to the ensemble:
#    - edit models/ensemble_config.json (the source of truth)
#    - also append to ENSEMBLE_MODELS in comicml/inference.py (the fallback)

# 5. Restart the scan webapp to clear the cached ensemble in memory:
#    python3 webapp/scan/server.py
```
