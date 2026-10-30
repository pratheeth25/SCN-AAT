# Fuzzy logic used in this project

> **Attribution.** The IT2-ANFIS method described here comes from Khaled, Mallak, Kaymak and Genga (2026,
> [arXiv:2601.18897](https://arxiv.org/abs/2601.18897)). Their code base is
> [QusaiKhaled/XUQ](https://github.com/QusaiKhaled/XUQ). The model code used in this project was written from the
> paper's equations, because the original repository does not include its model files. The general fuzzy-logic concepts
> are credited to the authors in §10.

This document explains every fuzzy-logic and soft-computing idea behind the 7-rule, 9-rule and 9-rule + PSO models,
from fuzzy sets to interval type-2 TSK ANFIS. Each idea also says where it appears in the code.
Implementation details are in [implementation.md](implementation.md); results are in [results.md](results.md).

---

## 1. Crisp sets vs fuzzy sets

- **Crisp set:** an element belongs (1) or does not belong (0). Example: "a hot day is a day with T ≥ 30 °C", so 29.9 °C is not hot.
- **Fuzzy set (Zadeh, 1965):** membership is a **degree** μ(x) ∈ [0, 1]. A 29.9 °C day can be hot to degree 0.95.
- **Membership function (MF):** the curve x ↦ μ(x). Common shapes are triangular, trapezoidal, bell and **Gaussian**:
  `μ(x) = exp(−½ ((x − c) / σ)²)`, with centre c and spread σ.
- **Linguistic variable:** a variable whose values are words ("inflow is LOW / MEDIUM / HIGH"), each word being a fuzzy set.

## 2. Fuzzy-set operations

| Operation | Crisp meaning | Fuzzy version | Used here |
|---|---|---|---|
| AND (intersection) | both true | **t-norm**: min(a, b) or **product a·b** | product, to combine a rule's conditions |
| OR (union) | either true | s-norm: max(a, b) or a + b − ab | not used |
| NOT (complement) | opposite | 1 − a | not used |

The product t-norm is smooth (differentiable), so gradient-based training works through it. In the code
(`_IT2TSKNetwork._compute_memberships`) the product is computed as a **sum of logarithms**, which avoids numerical
underflow when many inputs are multiplied.

## 3. Fuzzy rules and fuzzy inference

A fuzzy rule has an antecedent (IF part) and a consequent (THEN part):

```
IF  x1 is A1  AND  x2 is A2  AND ... AND  xF is AF   THEN   y = ...
```

**Inference steps:**
1. **Fuzzification:** compute the membership of each input in each rule's fuzzy sets.
2. **Rule firing:** combine the memberships of one rule with the t-norm. This gives the **firing strength**, i.e. how much the rule applies.
3. **Implication and aggregation:** combine every rule's output, weighted by its firing strength.
4. **Defuzzification:** turn the result into one crisp number.

### 3.1 Mamdani vs Takagi–Sugeno–Kang (TSK)

| | Mamdani | TSK (Takagi & Sugeno, 1985) |
|---|---|---|
| THEN part | a fuzzy set ("energy is HIGH") | a function of the inputs |
| Zero-order | – | y_j = constant b_j |
| First-order | – | y_j = b_j + w_j · x (linear) |
| Output | needs centroid defuzzification | weighted average of rule outputs |
| Learning | harder | consequents are linear, so they can be solved by least squares |

**This project uses first-order TSK.** Each rule is a *local linear model* that is valid where its antecedent fires.

### 3.2 Normalised firing and the TSK output

With firing strengths f_j, the normalised weights are `f̄_j = f_j / Σ_k f_k`, which sum to 1.
The output is `ŷ = Σ_j f̄_j · y_j`, a smooth blend of the local linear models.

---

## 4. Type-1 vs type-2 fuzzy sets

- **Type-1:** the membership degree is one exact number.
- **Type-2 (Zadeh, 1975; Mendel & John, 2002):** the membership degree is itself fuzzy. This represents uncertainty *about the membership function*, such as noisy sensors, vague boundaries or disagreement between experts.
- **Interval type-2 (IT2):** the special case where every secondary membership is 1. Each x then gets an **interval** of membership degrees `[μ_L(x), μ_U(x)]`.
  - **UMF (upper MF)** and **LMF (lower MF)** are the top and bottom edges of that interval.
  - **Footprint of uncertainty (FOU):** the region between the LMF and the UMF. A wider FOU means more uncertainty.

## 5. The uncertain-mean Gaussian MF (used in every model here)

The centre of the Gaussian is known only to lie in an interval `[c1, c2]`; the spread σ is fixed.

```
UMF:  μ_U(x) = exp(−½((x − c1)/σ)²)   if x < c1
               1                         if c1 ≤ x ≤ c2
               exp(−½((x − c2)/σ)²)   if x > c2

LMF:  μ_L(x) = exp(−½((x − c2)/σ)²)   if x ≤ (c1 + c2)/2
               exp(−½((x − c1)/σ)²)   if x > (c1 + c2)/2
```

- The UMF is the envelope of all Gaussians with centre in [c1, c2]; the LMF is their lowest point. Hence **LMF ≤ UMF**.
- **Parameters per rule and per input:** c1, c2 and σ. Valid parameters require c1 ≤ c2 and σ > 0.
- **In the code:** these formulas are in `_IT2TSKNetwork._compute_memberships`. For PSO, the codec stores the centre m = (c1 + c2)/2, the half-width h ≥ 0 and log σ, so every candidate is valid by construction (`AntecedentCodec`).

## 6. IT2 rule firing, type reduction and defuzzification

1. **Firing interval:** with the product t-norm over the inputs, `f_L,j = Π_f μ_L,jf(x_f)` and `f_U,j = Π_f μ_U,jf(x_f)`.
2. **Normalisation:** `f̄_L,j = f_L,j / Σ_k f_L,k` and `f̄_U,j = f_U,j / Σ_k f_U,k`.
3. **Type reduction** turns the interval type-2 output into an interval of type-1 values:
   `y_L = Σ_j f̄_L,j · y_j` and `y_U = Σ_j f̄_U,j · y_j`.
   - The exact method is the iterative **Karnik–Mendel (KM)** algorithm.
   - The paper uses the faster **q-weighted** form (Nie–Tan style), and so does this code.
4. **Defuzzification:** `ŷ = q · y_L + (1 − q) · y_U`. The paper fixes q = 0.5. Here q = sigmoid(logit_q) starts at 0.5 and is learned.

**Note:** [y_L, y_U] measures uncertainty in the *model structure*. It is not a prediction interval. On this data it covers fewer than 10 % of test values, so calibrated intervals need e.g. split-conformal prediction (`src/utils/metrics.conformal_margin`).

### 6.1 Worked example (1 input, 2 rules, standardised x = 0.5)

| | Rule 1 | Rule 2 |
|---|---|---|
| c1, c2, σ | 0, 0.2, 1 | 1.0, 1.2, 1 |
| UMF | x > c2 → exp(−½·0.3²) = 0.956 | x < c1 → exp(−½·0.5²) = 0.882 |
| LMF | x > mid 0.1 → exp(−½·0.5²) = 0.882 | x ≤ mid 1.1 → exp(−½·0.7²) = 0.783 |
| normalised lower / upper | 0.530 / 0.520 | 0.470 / 0.480 |
| consequent y_j | 2 + 1·x = 2.5 | 4 − 1·x = 3.5 |

- y_L = 0.530·2.5 + 0.470·3.5 = 2.970
- y_U = 0.520·2.5 + 0.480·3.5 = 2.980
- With q = 0.5, **ŷ = 2.975** (standardised units; the model then converts back to kWh).

---

## 7. ANFIS: fuzzy system + neural learning

**ANFIS** ("adaptive-network-based fuzzy inference system", Jang, 1993) writes a TSK fuzzy system as a layered network, so that its parameters can be learned from data:

| Layer | Role | In IT2-ANFIS here |
|---|---|---|
| 1 | Fuzzification | LMF and UMF of every input for every rule (parameters c1, c2, σ) |
| 2 | Rule firing (t-norm) | product, giving f_L and f_U |
| 3 | Normalisation | f̄_L, f̄_U |
| 4 | Consequents | y_j = b_j + w_j · x |
| 5 | Output | type reduction to y_L, y_U, then ŷ = q·y_L + (1 − q)·y_U |

**Parameter groups:**
- **Premise (antecedent)** parameters are nonlinear: c1, c2 and σ.
- **Consequent** parameters are linear: b_j and w_j.

**Hybrid learning:** Jang's original ANFIS uses gradient descent for the premise parameters and **least squares** for the consequents. Because the output is linear in the consequents once the premises are fixed, they can be solved exactly. This project uses both ideas:

| Model | Antecedents | Consequents |
|---|---|---|
| 7 rules (M7) | Latin-hypercube placement, then frozen | **Adam** gradient descent + L1/L2 penalties, early stopping |
| 9 rules (M9) | 7 frozen + 2 placed from residual analysis, then frozen | Adam, as M7 |
| 9 rules + PSO (M9_pso) | 7 frozen + 2 tuned by **PSO** | Adam |
| 9 rules + PSO + ridge (M9_pso_ridge, M9_tscv) | 7 frozen + 2 tuned by PSO | **ridge least squares** (closed form) |

## 8. Supporting soft-computing techniques

### 8.1 Rule placement (initialisation)
- **Latin-hypercube-style placement (M7):** each input's range is cut into K equal segments, and rule r draws its centre in segment r. Rules therefore cover the whole range evenly.
- **Residual-targeted placement (the 2 new rules):** look for calendar groups (month, year, weekend) where M7's training residual has a significant mean (|t| ≥ 3). A new rule is centred on that group's average input.
- **k-means clustering:** a fallback when fewer than two groups qualify. It clusters the training rows with the largest errors.

### 8.2 Particle Swarm Optimisation (Kennedy & Eberhart, 1995)
- **What PSO is:** a population-based, gradient-free search inspired by bird flocking.
- **Particles:** each particle is one candidate set of antecedent parameters for the 2 new rules.
- **Update rule for particle i:**
  ```
  v_i ← w·v_i + c1·r1·(pbest_i − x_i) + c2·r2·(gbest − x_i)
  x_i ← x_i + v_i
  ```
  - w is the **inertia**; it falls from 0.9 (explore) to 0.4 (exploit).
  - c1 = c2 = 1.5 are the cognitive and social pulls.
  - r1 and r2 are random numbers in [0, 1].
  - Velocity is limited to 20 % of each box width, and positions are clipped to the bounds.
- **Fitness:** the validation MSE after the consequents are ridge-solved. For M9_tscv it is the mean over 3 time-ordered folds.
- **Why PSO:** this fitness is awkward to differentiate in the antecedents, because it contains a linear solve and piecewise MFs. PSO only needs function values.

### 8.3 Regularisation
- **L1 / L2 penalties** on the consequents during Adam training (0.005 and 0.01).
- **Ridge regression** `θ = (DᵀD + λI)⁻¹Dᵀy` for closed-form consequents. All terms are penalised, including biases, so a rule that hardly fires in training cannot get a huge bias.
- **Early stopping** on validation error (patience 30).

### 8.4 Fuzzy-model design choices and their effect
| Choice | Effect on this data |
|---|---|
| Number of rules (7 vs 9) | 9 rules help only on the random split, and only with PSO + ridge. See [results.md](results.md). |
| Rules on all inputs (12–16-D) | A rule fires strongly only if the input is close on *every* dimension, so firing is often near one-hot. |
| Linear (TSK) consequents | They extrapolate. A raw `year` input makes them drift on future data, which is why M9_tscv removes it. |
| IT2 vs type-1 | IT2 adds an FOU (more parameters and a type-reduction step). The paper reports a small gain over type-1. |

## 9. Glossary

| Term | Meaning |
|---|---|
| Antecedent / premise | IF part of a rule: its membership functions |
| Consequent | THEN part: here a linear function of the inputs |
| Firing strength | Degree to which a rule applies to an input (t-norm of its memberships) |
| FOU | Footprint of uncertainty: the area between UMF and LMF |
| LMF / UMF | Lower / upper membership function of an IT2 set |
| t-norm | Fuzzy AND (here: product) |
| Type reduction | IT2 output set → interval [y_L, y_U] |
| Defuzzification | Interval or fuzzy output → one number |
| TSK | Takagi–Sugeno–Kang rules with function consequents |
| ANFIS | Fuzzy system trained like a neural network |
| PSO | Particle Swarm Optimisation |
| Ridge | Least squares with an L2 penalty |

## 10. References

1. L. A. Zadeh, "Fuzzy sets," *Information and Control*, 8(3), 338–353, 1965.
2. L. A. Zadeh, "The concept of a linguistic variable and its application to approximate reasoning—I," *Information Sciences*, 8, 199–249, 1975.
3. T. Takagi and M. Sugeno, "Fuzzy identification of systems and its applications to modeling and control," *IEEE Trans. SMC*, 15(1), 116–132, 1985.
4. J.-S. R. Jang, "ANFIS: adaptive-network-based fuzzy inference system," *IEEE Trans. SMC*, 23(3), 665–685, 1993.
5. J. M. Mendel and R. I. John, "Type-2 fuzzy sets made simple," *IEEE Trans. Fuzzy Systems*, 10(2), 117–127, 2002.
6. J. Kennedy and R. Eberhart, "Particle swarm optimization," *Proc. IEEE ICNN*, 1942–1948, 1995.
7. Q. Khaled, B. Mallak, U. Kaymak and L. Genga, "Explainable Uncertainty Quantification for Wastewater Treatment Energy Prediction via Interval Type-2 Neuro-Fuzzy System," arXiv:2601.18897, 2026.
8. Q. Khaled, XUQ code repository, https://github.com/QusaiKhaled/XUQ, 2026.
