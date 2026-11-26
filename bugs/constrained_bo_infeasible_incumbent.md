# Constrained BO uses guessed infeasible values as the incumbent

| | |
| --- | --- |
| Status | Open. No library code has been changed. |
| Found | 2026-11-26, on `79a5f98` plus the 3D FEM examples |
| Component | `gp_omni.opt`: batch loop (tested), async loop (same code path, not tested) |
| Severity | High. Constrained campaigns can stop improving. |

## Summary

The acquisition functions take the incumbent (the "best value so far") from every training
target, including the values the surrogate **guessed** for infeasible and pending points.
A guess can be far better than anything feasible. The loop then looks for improvement over
a target that may be unreachable.

## Where it happens

1. `src/opt/parallel/hallucination.py:37-38` replaces the output of each infeasible or
   pending point with the prediction of a model fitted on the feasible points, then refits
   on all of them. Line 57 adds in-batch pending points labelled feasible.
2. The loops pass these targets on as `y_obs`.
3. The incumbent is their maximum, with no feasibility mask:
   - `src/opt/acquisition/single_objective/__init__.py:156` (`f_best` for PI and EI)
   - `src/opt/acquisition/single_objective/__init__.py:97` (floor of the MES max values)
   - `src/opt/search.py:100` (start point of the CMA-ES search)

The product `acquisition * P(feasible)` does not undo this. `P(feasible)` weights the
value at a candidate, but the threshold inside EI is already wrong.

## Minimal reproduction

No FEM packages needed. Save as `repro.py` and run `uv run --locked python repro.py`.

```python
import numpy as np
import torch

from gp_omni.opt.parallel.hallucination import fit_models

torch.manual_seed(0)
x = np.array([[0.05], [0.15], [0.6], [0.7], [0.8], [0.9]])
feasible = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
objective = np.where(feasible > 0, -x[:, 0] - 1.0, 0.0)  # x < 0.5 is infeasible
_, _, hallucinated = fit_models(x, objective.copy(), feasible, 200)
print("best feasible value     ", objective[feasible > 0].max())
print("incumbent the loop uses ", hallucinated.max())
```

Output on macOS arm64:

```text
best feasible value      -1.6
incumbent the loop uses  -1.0500191924435611
```

Expected: the incumbent is at most -1.6.

## Evidence from the FEM campaigns

Replaying three campaigns from `examples/fem` (objective is `-mass / reference mass`, so
every real value is negative):

| Case | Evaluations replayed | Best feasible value | Incumbent used | Incumbent point |
| --- | --- | --- | --- | --- |
| stiffened_panel | 12 | -1.030 | -0.680 | infeasible |
| stiffened_panel | 44 | -1.030 | -0.451 | infeasible |
| lifting_lug | 28 | -1.172 | **+0.499** | infeasible |
| lattice_beam | 12 | -1.053 | -0.675 | infeasible |

A positive value would mean negative mass. At 28 evaluations of `stiffened_panel` and
`lattice_beam`, the next proposal had expected improvement exactly 0 and was infeasible.
For `lifting_lug` it was 3e-64.

## Experiment: cap the targets passed to the proposal

A runtime patch, not committed, capped the targets handed to the acquisition search at the
best real feasible value:

```python
original = batch_bo._propose

def patched(self, batch_id, x_obs):
    _, y, f = self._load()
    saved = self.y_hallucinated
    if (f > 0).any():
        self.y_hallucinated = np.minimum(saved, y[f > 0].max())
    try:
        return original(self, batch_id, x_obs)
    finally:
        self.y_hallucinated = saved

batch_bo._propose = patched
```

Improvement over the reference design, one campaign per case, seed 0, 44 candidates,
Linux x86-64. "Sampling" is the median over 4,000 draws of 44 designs from an 80-design
Latin hypercube sweep of each case. It is an estimate, not a separate campaign, and for
stress-limited cases other than `drone_arm` the sweep stresses were rescaled to the final
stress sampling by their ratio at the reference design.

| Case | Current library | With cap | Sampling | Effect of cap |
| --- | --- | --- | --- | --- |
| stiffened_panel | 0.0% | 33.3% | 31.9% | better |
| drone_arm | 5.1% | 26.4% | 18.9% | better |
| lifting_lug | 0.0% | 17.7% | 15.6% | better |
| connecting_rod | 15.6% | 16.4% | 12.0% | about the same |
| l_bracket | 31.1% | 30.7% | 23.3% | about the same |
| flexure_hinge | 89.7% | 88.7% | 71.0% | about the same |
| cold_plate | 7.8% | 7.8% | 4.4% | same |
| lattice_beam | 0.0% | 0.0% | 8.8% | still stuck |
| pressure_nozzle | 15.2% | 8.8% | 27.7% | worse |
| heat_sink | 28.6% | 9.1% | 21.7% | much worse |

## What is and is not established

Established:

- The incumbent can come from an infeasible point's guessed value. The code, the
  reproduction and the replays agree.
- Removing that effect from the proposal step changed the three cases above from
  little or no improvement to better than the sampling estimate.

Not established:

- **The cap is not a complete fix.** Two cases got worse and `lattice_beam` did not move.
  With one seed per case, seed-to-seed variation cannot be separated from the effect.
- Untested hypothesis for the remaining gap: the cap changes only the incumbent and the
  search start. The surrogate is still trained on the guessed values, which can keep
  drawing proposals into infeasible regions. A cap also ties several guesses with the
  incumbent.
- A related design point: the objective of an infeasible design is discarded
  (`0` is written to `output.dat`), even when the simulation computed it. Using it would
  likely reduce the extrapolation.

## Suggested next steps

1. Add the reproduction as a failing test: the incumbent never exceeds the best real
   feasible value.
2. Compute the incumbent and the search start from real feasible observations only, and
   pass them explicitly instead of deriving them from the hallucinated targets. Cover
   both loops, PI, EI, MES and KG.
3. Compare before and after over at least 10 seeds on the analytical constrained
   benchmarks and the ten FEM cases before merging.
4. Handle the case with no real feasible point yet.

## Minor, separate: result files are read while being written

During tests, `jobs.py` raised `UserWarning: loadtxt: input contained no data` on a
`complete.dat`. `_run_job` in `src/opt/parallel/asyncbo/loop.py` writes the file in place
while the loop polls it. `_num` treats an empty file as the default (unfinished), so the
next poll recovers. Writing to a temporary file and renaming it would remove the race.
