# Neural-network surrogate for ammonia oxidation rates

This folder contains a small research workflow for learning reaction rates from
the Kraehnert ammonia-oxidation mechanism over platinum. The central artifact
is [`NNmultioutput_mass_conservation.ipynb`](./NNmultioutput_mass_conservation.ipynb):
it generates a solver-based dataset (or reads one already generated), trains
neural-network surrogates, compares conservation treatments and architectures,
and includes a high-temperature extrapolation experiment.

The network is a **surrogate trained on simulated rates**. It is not itself a
first-principles mechanism, and the experiments here do not establish accuracy
against independent laboratory measurements.

## Project contents

- [`Krahnert.py`](./Krahnert.py) implements the heterogeneous kinetic model as
  a differential-algebraic system (DAE), solves for steady-state surface
  coverages, and computes rates.
- [`sampling.py`](./sampling.py) samples the mechanism's input domain and
  writes `rates.csv`.
- [`NNmultioutput_mass_conservation.ipynb`](./NNmultioutput_mass_conservation.ipynb)
  contains the main training, model-comparison, visualization, and
  extrapolation workflow.
- [`NNtrain.ipynb`](./NNtrain.ipynb) is an earlier/smaller neural-network
  training experiment.
- [`mechanism_test.ipynb`](./mechanism_test.ipynb) explores the mechanistic
  model.
- [`requirements.txt`](./requirements.txt) lists the Python dependencies.
- `model_checkpoints/` contains cached neural-network checkpoints used by the
  main notebook. A checkpoint is only reusable when its recorded data/config
  fingerprint matches.

## Chemistry and mechanistic simulator

The source implementation follows the Kraehnert and Baerns model of ammonia
oxidation on platinum foil:

> Kraehnert, R. and Baerns, M. (2008), “Kinetics of ammonia oxidation over Pt
> foil studied in a micro-structured quartz-reactor,” *Chemical Engineering
> Journal*, 137, 361–375.

The mechanistic model has two surface-site families, `a` and `b`, with
coverages for vacant and adsorbed species. Their algebraic site balances are

$$
\theta_b + \theta_{b-\mathrm{NH_3}} = 1,
\qquad
\theta_a + \theta_{a-\mathrm O} + \theta_{a-\mathrm{NO}} +
\theta_{a-\mathrm N} = 1.
$$

The DAE solver finds steady-state coverages for the supplied temperature and
partial pressures. Ten elementary/non-elementary surface steps are then used
to calculate the gas-phase rates. The network learns the mapping from operating
conditions to these **simulated steady-state rates**; it does not learn the
surface coverages or solve the DAE directly.

For the four rates predicted by the notebook, the three global product
pathways can be written in terms of forward extents $\xi_1,\xi_2,\xi_3$:

$$
\begin{aligned}
4\mathrm{NH_3} + 5\mathrm{O_2} &\longrightarrow
    4\mathrm{NO} + 6\mathrm{H_2O},\\
4\mathrm{NH_3} + 3\mathrm{O_2} &\longrightarrow
    2\mathrm{N_2} + 6\mathrm{H_2O},\\
4\mathrm{NH_3} + 4\mathrm{O_2} &\longrightarrow
    2\mathrm{N_2O} + 6\mathrm{H_2O}.
\end{aligned}
$$

With the sign convention that ammonia consumption is negative and product
formation is positive,

$$
r_{\mathrm{NH_3}}=-4(\xi_1+\xi_2+\xi_3),\quad
r_{\mathrm{NO}}=4\xi_1,\quad
r_{\mathrm{N_2}}=2\xi_2,\quad
r_{\mathrm{N_2O}}=2\xi_3.
$$

Eliminating the extents gives the nitrogen-atom balance among the four
predicted rates:

$$
\boxed{
r_{\mathrm{NH_3}} + r_{\mathrm{NO}} +
2r_{\mathrm{N_2}} + 2r_{\mathrm{N_2O}} = 0.
}
$$

The coefficient vector in the code is
$\mathbf c=(1,2,1,2)^\mathsf T$, in the output order
`[rNH3, rN2, rNO, rN2O]`. This is a nitrogen balance for the tracked rates. It
is not a full elemental balance over all species: the network does not predict
oxygen, water, or the surface species.

## Dataset

Run `sampling.py` from this folder to generate `rates.csv`. It evaluates
100,000 Latin-hypercube samples with six worker processes at a fixed total
pressure of 500 kPa. The four independently sampled input quantities span:

| Input | Range | Unit |
|---|---:|---|
| Temperature, `T` | 500–1500 | K |
| Ammonia mole fraction, `xNH3` | 0.001–0.2 | — |
| Oxygen mole fraction, `xO2` | 0.001–0.2 | — |
| Nitric oxide mole fraction, `xNO` | 0.001–0.2 | — |

The input fractions are sampled independently over those bounds, as in the
mechanism driver; they are not renormalized to sum to one. `rates.csv` contains
these input columns and four outputs: `rNH3`, `rN2`, `rNO`, and `rN2O`. The
rate calculations in `Krahnert.py` use $\mathrm{mol\,m^{-2}\,s^{-1}}$.

The generated CSV is intentionally excluded from Git via `.gitignore`, so it
must be generated locally before running the notebook if it is not already
available. This avoids committing a large derived dataset; it also means that
the exact sampled rows are not distributed with the repository.

## Setup and running

From this directory, install the listed dependencies into an existing Python
environment:

```bash
python -m pip install -r requirements.txt
```

Then generate the dataset if needed:

```bash
python sampling.py
```

Finally, open `NNmultioutput_mass_conservation.ipynb` in Jupyter or VS Code and
run its cells from top to bottom. Dataset generation involves many DAE solves
and can take a while. Training/search cells are marked as optional or
opt-in in the notebook; inspect their run flags before executing them. GPU use
is optional; the notebook selects CUDA when available and otherwise uses CPU.

## Neural-network formulation

The main notebook predicts all four rates together with one shared
feed-forward network. Its transformed input vector is

$$
\mathbf x =
\left[
  \frac{1}{T},
  \ln(x_{\mathrm{NH_3}}),
  \ln(x_{\mathrm{O_2}}),
  \ln(x_{\mathrm{NO}})
\right].
$$

Each feature is standardized using the training rows' mean and standard
deviation. The target rates keep their signed, original units; they are not
log-transformed. Training-set rate standard deviations are used to put the
multi-output loss terms on comparable scales.

For target $j$, let $s_j$ be its training-set standard deviation, and let
$y_{ij}$ and $\hat y_{ij}$ be the true and predicted rate for sample $i$.
The normalized data loss is

$$
\mathcal L_{\mathrm{data}}
= \frac{1}{4N}\sum_{i=1}^N\sum_{j=1}^4
\left(\frac{\hat y_{ij}-y_{ij}}{s_j}\right)^2.
$$

Two different approaches to the nitrogen-balance condition were compared.
They are alternatives, not two names for the same technique.

### 1. Soft conservation penalty

For each predicted sample, define the balance residual

```math
g_i = \mathbf c^\mathsf T \hat{\mathbf y}_i
= \hat r_{\mathrm{NH_3},i} + 2\hat r_{\mathrm{N_2},i}
+ \hat r_{\mathrm{NO},i} + 2\hat r_{\mathrm{N_2O},i}.
```

The symbol $\odot$ denotes element-wise multiplication. For example, with
$\mathbf c=(1,2,1,2)$ and
$\mathbf s=(s_{\mathrm{NH_3}},s_{\mathrm{N_2}},s_{\mathrm{NO}},s_{\mathrm{N_2O}})$,
$\mathbf c\odot\mathbf s=(s_{\mathrm{NH_3}},2s_{\mathrm{N_2}},s_{\mathrm{NO}},2s_{\mathrm{N_2O}})$.
The notebook scales this residual by

```math
s_{\mathrm{bal}} =
\left\|\mathbf c \odot \mathbf s\right\|_2,
```

where $\|\cdot\|_2$ is the Euclidean norm, and minimizes

```math
\mathcal L_{\mathrm{soft}}
= \mathcal L_{\mathrm{data}}
+ \lambda \frac{1}{N}\sum_{i=1}^N
\left(\frac{g_i}{s_{\mathrm{bal}}}\right)^2.
```

The hyperparameter $\lambda$ controls a trade-off. A larger penalty
encourages smaller balance residuals, but does **not** guarantee zero residual
and can reduce the accuracy of individual rates. The notebook explored
different penalty strengths, including $\lambda=10$.

### 2. Exact output projection

The exact-balance models first produce an unconstrained output
$\mathbf f(\mathbf x)$, then project it onto the hyperplane
$\mathbf c^\mathsf T\mathbf y=0$. With $q_j=s_j^2$, the projection is

```math
\hat{\mathbf y}
= \mathbf f
- \frac{\mathbf c^\mathsf T\mathbf f}
{\sum_j c_j^2 q_j}\,(\mathbf q\odot\mathbf c).
```

Indeed,

```math
\mathbf c^\mathsf T\hat{\mathbf y}
= \mathbf c^\mathsf T\mathbf f
- \frac{\mathbf c^\mathsf T\mathbf f}
{\sum_j c_j^2q_j}\sum_j c_j^2q_j
=0.
```

This is the smallest correction to the raw output in the
training-standard-deviation-weighted metric: it distributes the correction
across rates according to their training variances. The constraint is imposed
on every prediction by construction, up to floating-point roundoff. Unlike
the soft penalty, exact projection does not ask gradient descent to discover
the balance relation.

The simulator's target rates can have tiny nonzero balance residuals because
of numerical solution tolerances. Therefore, an exactly balanced prediction
cannot match every tiny label residual; this may contribute an irreducible
error. Exact conservation is a structural property of the outputs, not proof
that every individual rate is accurate or that the network independently
learned the conservation law.

## Experiments and evaluation

The main notebook records experiments comparing separate and shared
multi-output networks, soft-penalty weights, depth/width, activation functions,
learning rates, weight decay, exact output projection, and per-rate
validation/error plots. Use validation results to compare candidates. The
saved original split has about 70,000 training rows, 15,000 validation rows,
and 15,000 test rows, created by random row-wise splitting with seed 2026.
Because temperature is randomly mixed across these splits, that split tests
interpolation over the sampled domain, not temperature extrapolation.

The most recent architecture search selected a four-hidden-layer,
192-units-per-layer GELU network with exact balance projection and learning
rate $3\times10^{-4}$ (112,900 trainable parameters). That model was then
used in the dedicated high-temperature experiment.

### High-temperature extrapolation experiment

The extrapolation cell sets its cutoff at two-thirds of the observed
temperature interval:

$$
T_{\mathrm{cut}} = T_{\min}+\frac{2}{3}(T_{\max}-T_{\min})
\approx 1166.7\ \mathrm{K}.
$$

It randomly divides rows at or below the cutoff into training (about 56,666
rows) and in-range validation (10,000 rows). The upper third (33,334 rows,
approximately 1166.7–1500 K) is excluded from that cell's fitting of weights,
feature scaling, projection scales, and early-stopping/model-checkpoint
selection. The model achieves upper-range $R^2$ values around 0.9994–0.9999
for the four rates in the saved run; the per-rate RMSE and absolute-error
results are printed by the notebook.

Interpret this as promising evidence that a network trained on the lower
temperature range can reproduce part of the simulator's smooth rate
structure above that range. Keep the caveats in view:

1. The upper-range rows had been part of the full dataset used in earlier
   random-split architecture/model comparisons. The architecture was selected
   using those earlier validation results. The high-temperature results are
   therefore not an untouched, end-to-end independent estimate, even though
   those rows were not used for the extrapolation cell's weight updates.
2. The labels come from the same Kraehnert simulator throughout. This tests
   extrapolation of that simulator's mapping, not accuracy against independent
   experiments or a different physical model.
3. Composition remains inside the sampled input domain. The experiment holds
   out a temperature region; it does not test extrapolation to unseen mixture
   compositions.
4. $R^2$ alone can look excellent when the rate varies over a wide range.
   Inspect the reported RMSE, MAE, signed error extrema, and scatter plots as
   well.

For a stronger generalization claim, freeze the modeling choices using only
training-range data, then evaluate on newly generated high-temperature
simulations that were never used in prior model comparisons. Repeat with
multiple random seeds, compare simple baselines, and ultimately evaluate
against independent measurements if available.

## Reproducibility and interpretation notes

- The notebook uses seed 2026 for its original row split and training setup;
  the sampler also seeds its Latin-hypercube generation. GPU/CPU math and
  numerical DAE convergence can still affect exact results.
- The original random test split was described as a one-time final evaluation,
  but later architecture investigations used validation results and the
  high-temperature cell repartitions rows from the full dataset. Treat the
  earlier test results as already observed, not as a pristine holdout for new
  decisions.
- The main notebook contains earlier experiment outputs as well as code.
  Restart the kernel and run the required setup/helper cells when changing
  experiments; inspect each cell's opt-in flag before running a sweep.
- “Exact conservation” refers to the four-output nitrogen balance above. It
  does not mean the model solves the surface DAE or enforces every physical
  law in the kinetic mechanism.
