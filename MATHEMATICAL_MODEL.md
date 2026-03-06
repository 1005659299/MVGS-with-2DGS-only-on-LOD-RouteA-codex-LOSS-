# Mathematical-Physics Model: Complete Analysis

## MVGS-with-2DGS-only-on-LOD-RouteA-codex-LOSS

> Comprehensive mathematical formulation and comparison with the
> [Inverse-Depth-Probability-Sampling](https://github.com/1005659299/MVGS-with-2DGS-only-on-LOD-with-Inverse-Depth-Probability-Sampling) baseline.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Gaussian Primitive Representation (2D-GS Surfel)](#2-gaussian-primitive-representation-2d-gs-surfel)
3. [LOD Hybrid Rendering Pipeline](#3-lod-hybrid-rendering-pipeline)
4. [Deterministic Stochastic Sampling](#4-deterministic-stochastic-sampling)
5. [Loss Function Formulation](#5-loss-function-formulation)
6. [Sparse-View Adaptive Densification](#6-sparse-view-adaptive-densification)
7. [Route A: SFC-FRS++ View Selection](#7-route-a-sfc-frs-view-selection)
8. [Multi-View Regulated Training](#8-multi-view-regulated-training)
9. [Comparison Summary](#9-comparison-summary)

---

## 1. Overview

This project builds upon the **MVGS** (Multi-View Regulated Gaussian Splatting)
framework, integrating **2D Gaussian Splatting (2D-GS)** surfel primitives with a
**Level-of-Detail (LOD)** hybrid rendering pipeline.  Compared with the
Inverse-Depth-Probability-Sampling baseline, this repo introduces:

| Feature | Inverse-Depth (Baseline) | RouteA-codex-LOSS (This Repo) |
|---|---|---|
| Transition-zone sampling | Non-deterministic `torch.rand_like` | **Deterministic** camera-seeded PRNG |
| Geometric regularisation losses | Unmasked `.mean()` | **Alpha-masked** with `rend_alpha` weighting |
| Densification threshold | Fixed per-iteration | **Sparse-view adaptive** ρ^γ scaling |
| Normal-guided densification | — | Optional threshold modulation |
| View selection | None | **SFC-FRS++** greedy/uniform-pose |
| Train-view ratio tracking | — | `n_train_full` / `n_train_selected` |
| `cosine_similarity_loss` utility | — | Added for normal supervision |
| Far-cascade fallback colour | Zeros | Correct `bg_color` |

---

## 2. Gaussian Primitive Representation (2D-GS Surfel)

### 2.1 Parameter Space

Each Gaussian primitive *i* is parameterised by:

| Symbol | Dimension | Description |
|---|---|---|
| **μ**_i | ℝ³ | Centre (mean) position in world space |
| **s**_i | ℝ² | 2D scaling along the surfel's two tangent axes |
| **q**_i | ℝ⁴ | Unit quaternion encoding the surfel orientation |
| **c**_i | SH coefficients | View-dependent colour (spherical harmonics up to degree ℓ) |
| σ_i | ℝ | Pre-activation opacity (logit space) |

The 2D scaling vector **s** = (s₁, s₂) ∈ ℝ² represents the surfel as a flat
disc (zero thickness along the normal axis), a key distinction from 3D-GS which
uses **s** ∈ ℝ³.

### 2.2 Covariance / Transformation Matrix

The 2D scaling is extended to 3D by padding the third (normal) dimension with 1:

```
s_3D = [exp(s₁), exp(s₂), 1]
```

A 4×4 homogeneous transformation matrix **T** is built:

```
        ┌ R·S  │  0 ┐
  T  =  │──────┼────│
        │ μᵀ   │  1 │
        └──────┴────┘
```

where **S** = diag(s_3D) and **R** is the 3×3 rotation from quaternion **q**.
The upper-left 3×3 block is **RS** = (build_scaling_rotation(s_3D, q))ᵀ, and
the bottom row encodes the Gaussian centre **μ**. This produces the surfel
covariance in the space expected by the `diff_surfel_rasterization` CUDA kernel.

### 2.3 Activation Functions

| Parameter | Forward activation | Inverse |
|---|---|---|
| Scaling | `exp(·)` | `log(·)` |
| Opacity | `sigmoid(·)` | `logit(·)` = `log(p/(1−p))` |
| Rotation | `normalize(·)` (unit quaternion) | — |
| Colour | SH evaluation + clamp(·+0.5, min=0) | — |

---

## 3. LOD Hybrid Rendering Pipeline

### 3.1 Depth Computation

For each Gaussian centre **μ** and camera with world-to-view transform
[**R**_w2c | **t**_w2c], the view-space depth is:

```
z_i = (μ_i · R_w2c)_z + (t_w2c)_z
```

where subscript *z* denotes the depth (third) component.

### 3.2 Three-Cascade LOD

The depth axis is partitioned into three cascades controlled by two thresholds:

| Cascade | Depth range | Engine | Rendering model |
|---|---|---|---|
| 0 (Near) | z ≤ Z_near | 2D-GS (Surfel) | Ray–plane intersection (full quality) |
| 1 (Transition) | Z_near < z ≤ Z_trans | 2D-GS (Stochastic) | Probabilistic sub-sampling |
| 2 (Far) | z > Z_trans | 3D-GS (EWA) | Elliptical Weighted Average splatting |

Default values: Z_near = 2.0, Z_trans = 6.0, P_min = 0.2.

### 3.3 Perceptually-Driven Inverse-Depth Probability

For Gaussians in the transition zone, the keep-probability follows an
**inverse-depth** law motivated by the observation that closer objects subtend
larger solid angles and therefore require higher geometric fidelity:

```
P(z) = clamp(Z_near / z,  P_min,  1.0)
```

A Gaussian at depth z in the transition zone is included if a sample
`u ~ Uniform(0,1)` satisfies u < P(z).

### 3.4 Per-Pass Rendering

**Pass 1 — Far (background):** Only Gaussians with z > Z_trans are rasterised
via the 3D-GS engine. Their 2D scales (N,2) are extended to 3D by replicating
the first component: `[s₁, s₂, s₁]`.

**Pass 2 — Near + Transition (foreground):** The union of Cascade-0 and
Cascade-1 (sampled) Gaussians are rasterised via the 2D-GS surfel engine. This
produces a 7-channel **allmap**:

| Channel(s) | Content |
|---|---|
| 0 | Depth expected (Σ αᵢ·zᵢ) |
| 1 | Alpha (accumulated opacity) |
| 2–4 | Rendered normal (view space) |
| 5 | Depth median |
| 6 | Depth distortion |

### 3.5 Alpha Compositing

The final image is composited front-to-back:

```
I_final = I_near + I_far · (1 − α_near)
```

where α_near is the accumulated surfel alpha map.

### 3.6 Geometric Quantities

From the allmap the following are derived:

1. **Rendered normal** (rotated to world space):
   ```
   n_rend = (n_view · R_w2c^T)
   ```

2. **Surface depth** (expected–median blend):
   ```
   d_surf = (1 − r) · d_expected / α + r · d_median
   ```
   where r = `depth_ratio` (default 0.0).

3. **Surface normal** (from depth via finite differences):
   ```
   n_surf = normalize(∂P/∂u × ∂P/∂v)
   ```
   where ∂P/∂u, ∂P/∂v are computed by central-differencing the back-projected
   3D points.

---

## 4. Deterministic Stochastic Sampling

### 4.1 Problem (Baseline)

The baseline uses `torch.rand_like(depths)` which produces a different random
mask on every call, even for the same camera at the same iteration. This
non-reproducibility causes:
- Inconsistent gradient signals across training runs
- Flickering artefacts in evaluation

### 4.2 Solution (This Repo)

A deterministic PRNG seeds the transition-zone mask from the camera identity
and the training iteration:

```
seed(cam, t) = (FIXED_SEED + uid(cam) · PRIME + t · STRIDE)  mod  2³¹
```

Constants:
- FIXED_SEED = 0
- PRIME = 1,000,003 (large prime for decorrelation)
- STRIDE = 9,176 (co-prime to common iteration counts)

A per-call `torch.Generator` is created on the device with this seed, ensuring
bitwise identical masks for the same (camera, iteration) pair across runs while
remaining uniformly distributed across different pairs.

### 4.3 Mathematical Guarantee

For any two distinct cameras *a*, *b* and iterations *s*, *t*:

```
seed(a,s) ≡ seed(b,t)  mod 2³¹   ⟺   (uid_a − uid_b) · PRIME ≡ (t − s) · STRIDE  mod 2³¹
```

Because PRIME and STRIDE are coprime to 2³¹, collisions are astronomically
unlikely for practical uid/iteration ranges.

---

## 5. Loss Function Formulation

### 5.1 Photometric Loss (Unchanged)

```
L_photo = (1 − λ_dssim) · L₁(I, Î) + λ_dssim · (1 − SSIM(I, Î))
```

where I is the ground truth, Î is the rendered image, and λ_dssim = 0.2.

### 5.2 Alpha-Masked Normal Consistency Loss (**Improved**)

**Baseline formulation:**
```
L_normal = λ_n · mean(1 − ⟨n_rend, n_surf⟩)
```

**This repo (alpha-masked):**
```
              Σ_{p} α(p) · (1 − ⟨n_rend(p), n_surf(p)⟩)
L_normal = λ_n · ─────────────────────────────────────────
                         Σ_{p} α(p) + ε
```

where α = `rend_alpha.detach().clamp(0, 1)` and ε = 10⁻⁶.

**Physics motivation:** In regions where the surfel coverage is low (α ≈ 0),
the rendered normal and surface normal are unreliable (close to zero or
undefined). By weighting each pixel's normal error by its alpha and normalising
by the total alpha mass, the loss:
1. Suppresses noise from background/transparent pixels
2. Concentrates gradient signal on well-covered foreground surfaces
3. Prevents the loss magnitude from growing with image resolution

The `.detach()` on alpha prevents the loss from trivially driving alpha to zero,
which would reduce the loss without improving geometry.

### 5.3 Alpha-Masked Depth Distortion Loss (**Improved**)

**Baseline formulation:**
```
L_dist = λ_d · mean(D)
```

**This repo (alpha-masked):**
```
              Σ_{p} α(p) · D(p)
L_dist = λ_d · ─────────────────
               Σ_{p} α(p) + ε
```

where D is the per-pixel depth distortion from the 2D-GS rasteriser.

**Physics motivation:** Same as the normal loss — the distortion signal is
meaningful only where surfels have been rendered. Averaging over all pixels
(including empty regions where D = 0) dilutes the gradient.

### 5.4 Warm-Up Schedule (Unchanged)

```
λ_n(t) = λ_normal   if t > 7000,  else  0
λ_d(t) = λ_dist     if t > 3000,  else  0
```

Default hyperparameters: λ_normal = 0.05, λ_dist = 1000.0.

### 5.5 Total Per-View Loss

```
L_view = L_photo + L_dist + L_normal
```

Accumulated over M views per iteration:

```
L_total = Σ_{m=1}^{M}  L_view^(m)
```

### 5.6 Cosine Similarity Loss (New Utility)

An additional utility function is provided for optional external use:

```
L_cos(n_pred, n_gt, mask) = Σ_{p} mask(p) · (1 − ⟨n̂_pred(p), n̂_gt(p)⟩) / (Σ mask + ε)
```

where n̂ denotes L₂-normalised vectors. This can serve as an alternative to the
dot-product normal error when explicit normal supervision is available.

---

## 6. Sparse-View Adaptive Densification

### 6.1 Baseline Densification Threshold

The baseline uses a fixed gradient threshold τ (modulated only by camera-pair
distance):

```
τ = 0.5 · τ_base    if any camera-pair distance > 1
τ = τ_base           otherwise
```

where τ_base = `densify_grad_threshold` = 0.0002.

### 6.2 Sparse-View Ratio Compensation (**New**)

When Route A view selection reduces the training set from N_full cameras to
K = N_selected cameras, the Gaussian field receives fewer supervision signals
per iteration. To compensate, the threshold is scaled by the selection ratio:

```
ρ = K / N_full
τ' = τ · ρ^γ
```

where γ = `densify_sparse_gamma` (default 0.5, i.e. square-root scaling).

**Mathematical justification:** Under the assumption that gradient accumulation
scales linearly with the number of training views, reducing the training set by
a factor ρ reduces the expected per-Gaussian gradient by the same factor.
Scaling the threshold by ρ^γ with γ ∈ (0, 1) partially compensates, encouraging
more aggressive densification when views are sparse.

### 6.3 Normal-Guided Densification (**New, Optional**)

When `normal_guided_densify = 1`, the threshold is further modulated by the
mean alpha-weighted normal error:

```
ē_n = Σ_{p} α(p) · e_n(p) / (Σ α + ε)
f = clamp(1 − ē_n / 2,  0.5,  1.0)
τ'' = τ' · f
```

**Physics motivation:** High normal error indicates that the current Gaussian
distribution poorly represents the local surface geometry. By lowering the
densification threshold (f < 1) in these regions, the optimiser is encouraged
to add more primitives where the geometry is most inaccurate.

The clamp to [0.5, 1.0] prevents the threshold from dropping below half its
base value, maintaining training stability.

### 6.4 Cross-Ray Densification (Unchanged from MVGS)

For each pair of selected views, the highest-loss 2D bounding box is identified.
Rays from the four corners of each box are intersected to define a 3D bounding
box. All Gaussians inside any such box are marked for cloning, regardless of
their gradient magnitude. This ensures that under-represented regions in
ray-intersection volumes are densified.

---

## 7. Route A: SFC-FRS++ View Selection

### 7.1 Problem

Full training sets from COLMAP may contain hundreds of images with heavy
redundancy. Training on all views is computationally expensive and the
overlapping supervision may not improve quality.

### 7.2 Objective Function

Select a subset **S** of K views from N candidates that maximises:

```
F(S) = α · C_cov(S) + β · C_base(S) + γ · C_info(S) − δ · C_overlap(S)
```

where:

| Term | Formula | Meaning |
|---|---|---|
| **C_cov** (Coverage) | \|⋃_{k∈S} V_k\| / \|P\| | Fraction of 3D points visible from at least one selected view |
| **C_base** (Baseline) | max_{j∈S} B(k,j) | Triangulation quality: E[sin²(φ)] over shared points |
| **C_info** (Information) | I(k) / max(I) | Inverse-depth weighted information density |
| **C_overlap** (Overlap) | max_{j∈S} J(V_k, V_j) | Jaccard similarity of visibility sets (penalised) |

Default weights: α = 1.0, β = 0.5, γ = 0.2, δ = 0.1.

### 7.3 Precomputation (Layer A)

For each image k, the following are computed once:

1. **Visibility set** V_k: point indices visible in image k (from COLMAP's
   point3D_ids, filtered by positive z-depth)

2. **Information score:**
   ```
   I(k) = Σ_{p ∈ V_k}  1 / (z_p² + ε)
   ```
   Closer points contribute more heavily (inverse-square depth weighting).

3. **Overlap matrix** (N×N, symmetric):
   ```
   Overlap(k,l) = |V_k ∩ V_l| / |V_k ∪ V_l|   (Jaccard index)
   ```

4. **Baseline matrix** (N×N, symmetric):
   ```
   Baseline(k,l) = E_{p ∈ V_k ∩ V_l}[sin²(φ_p)]
   ```
   where φ_p is the angle subtended at point p between cameras k and l:
   ```
   cos(φ) = ⟨(p − C_k), (p − C_l)⟩ / (‖p − C_k‖ · ‖p − C_l‖)
   sin²(φ) = 1 − cos²(φ)
   ```

### 7.4 Greedy Selection (Layer B)

Starting from S = ∅ and covered = ∅, at each step add the view k* maximising
the marginal gain:

```
k* = argmax_{k ∉ S}  [ α · ΔC_cov(k) + β · max_{j∈S} B(k,j) + γ · I(k)/I_max − δ · max_{j∈S} J(k,j) ]
```

where ΔC_cov(k) = |V_k \ covered| / |P|.

After adding k*, update: covered ← covered ∪ V_{k*}.

### 7.5 Fallback: Uniform Pose Sampling

An alternative `uniform_pose` strategy uses farthest-point sampling in
camera-centre space: iteratively select the camera whose centre is maximally
distant from all previously selected centres.

### 7.6 Integration with Training Pipeline

The offline script produces `selected_views.json`. At training time, if
`--train_view_list` is specified:

1. `dataset_readers.py` filters `train_cam_infos` to only the selected views
2. `n_train_full` (original count) and `n_train_selected` (post-filter count)
   are stored in `nerf_normalization`
3. Normalisation uses the **full** training set's camera centres for stable
   scene scale
4. The ratio ρ = `n_train_selected / n_train_full` feeds into the adaptive
   densification threshold (§6.2)

---

## 8. Multi-View Regulated Training

### 8.1 Multi-View Constraint (from MVGS)

Each iteration samples M views (pipe.mv, default 1). The total loss accumulates
over all M views before a single backward pass:

```
L_total = Σ_{m=1}^{M} L_view^(m)
```

This multi-view supervision regularises the Gaussian attributes by preventing
overfitting to any single viewpoint.

### 8.2 Multi-View Augmented Densification

When camera-pair distances exceed a threshold (normalised distance > 1), the
densification gradient threshold is halved (τ → 0.5τ), encouraging more
aggressive point creation in regions viewed from dramatically different angles.

### 8.3 Cross-Intrinsic Guidance

The pipeline supports resolution-adaptive training through the `resolution`
parameter, enabling coarse-to-fine optimisation across different camera
intrinsics.

---

## 9. Comparison Summary

### 9.1 Rendering (gaussian_renderer/__init__.py)

| Aspect | Baseline (Inverse-Depth) | This Repo (RouteA-codex-LOSS) |
|---|---|---|
| Transition-zone RNG | `torch.rand_like(depths)` — non-deterministic, different each call | **Deterministic** per-(camera, iteration) seed via FNV-1a hash |
| render() signature | No `rng_step` parameter | Accepts `rng_step=iteration` for seed derivation |
| Far-cascade empty fallback | `torch.zeros(3, H, W)` | **`bg_color[:, None, None].expand(3, H, W)`** — correct background |

**Impact:** Deterministic sampling eliminates flickering during evaluation and
enables exact reproducibility. The correct background fallback fixes a subtle
compositing error where far regions would appear black instead of the configured
background colour (white or black).

### 9.2 Loss Functions (train.py)

| Aspect | Baseline | This Repo |
|---|---|---|
| Normal loss | `λ_n · mean(1 − ⟨n_r, n_s⟩)` | **`λ_n · Σ(α · error) / Σα`** — alpha-masked |
| Distortion loss | `λ_d · mean(D)` | **`λ_d · Σ(α · D) / Σα`** — alpha-masked |
| Alpha treatment | Not used | **Detached + clamped** α prevents trivial minimisation |
| cosine_similarity_loss | Not present | **Added** to loss_utils.py |

**Impact:** Alpha masking focuses geometric regularisation on foreground
surfaces where the signal is meaningful, improving convergence and preventing
gradient dilution in empty image regions.

### 9.3 Densification (train.py)

| Aspect | Baseline | This Repo |
|---|---|---|
| Threshold scaling | Fixed τ (with camera-distance modulation) | **ρ^γ sparse-view compensation** |
| Normal guidance | None | **Optional** threshold reduction in high-error regions |
| View count tracking | Not tracked | `n_train_full`, `n_train_selected` in Scene |
| Assertion guard | None | `assert scene.n_train_full > 0` |

**Impact:** Sparse-view compensation ensures adequate densification when the
training set is deliberately reduced by Route A view selection. Normal-guided
densification directs new primitives to geometrically challenging regions.

### 9.4 View Selection (NEW — utils/view_selection.py)

| Aspect | Baseline | This Repo |
|---|---|---|
| Module | Not present | **Full SFC-FRS++ implementation** |
| Strategies | — | `sfc_frs_greedy`, `frs_greedy`, `uniform_pose`, `random` |
| Objective | — | α·Coverage + β·Baseline + γ·Info − δ·Overlap |
| Integration | — | `--train_view_list` argument → filtered training set |

**Impact:** Enables intelligent sub-sampling of training views, dramatically
reducing training time while preserving or improving reconstruction quality
through diversity-aware selection.

### 9.5 Configuration (arguments/__init__.py)

| Parameter | Baseline | This Repo |
|---|---|---|
| `train_view_list` | — | `""` (path to selected_views.json) |
| `densify_sparse_gamma` | — | `0.5` |
| `normal_guided_densify` | — | `0` (disabled by default) |
| Default output path | `./output/` | `<repo>/output/` (relative to source tree) |

### 9.6 Scene Loading (scene/__init__.py, dataset_readers.py)

| Aspect | Baseline | This Repo |
|---|---|---|
| Colmap reader signature | `readColmapSceneInfo(path, images, eval)` | `readColmapSceneInfo(path, images, eval, *, args)` |
| View filtering | None | Route A: filter by `train_view_list` |
| Normalisation basis | Selected train views | **Full** train views (stable scale) |
| Scene metadata | `cameras_extent` only | + `n_train_full`, `n_train_selected` |

---

## Appendix A: Key Mathematical Symbols

| Symbol | Definition |
|---|---|
| **μ** ∈ ℝ³ | Gaussian centre position |
| **s** ∈ ℝ² | 2D surfel scaling (log-space) |
| **q** ∈ ℝ⁴ | Unit quaternion (orientation) |
| α ∈ [0,1] | Accumulated pixel opacity |
| z | View-space depth |
| P(z) | Inverse-depth keep probability |
| ρ | Train view selection ratio K/N |
| γ | Sparse-view exponent |
| τ | Densification gradient threshold |
| **n**_rend | Rendered normal (from alpha blending) |
| **n**_surf | Surface normal (from depth finite differences) |
| D | Depth distortion (from rasteriser) |
| F(S) | View selection objective function |
| V_k | Visibility set of camera k |

## Appendix B: Default Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| Z_near (LOD) | 2.0 | `pipe.lod_near_limit` |
| Z_trans (LOD) | 6.0 | `pipe.lod_transition_limit` |
| P_min | 0.2 | `pipe.lod_min_prob` |
| depth_ratio | 0.0 | `pipe.depth_ratio` |
| λ_dssim | 0.2 | `opt.lambda_dssim` |
| λ_normal | 0.05 | `opt.lambda_normal` |
| λ_dist | 1000.0 | `opt.lambda_dist` |
| τ_base | 0.0002 | `opt.densify_grad_threshold` |
| γ (sparse) | 0.5 | `opt.densify_sparse_gamma` |
| SFC-FRS α | 1.0 | `view_selection.select_views` |
| SFC-FRS β | 0.5 | `view_selection.select_views` |
| SFC-FRS γ | 0.2 | `view_selection.select_views` |
| SFC-FRS δ | 0.1 | `view_selection.select_views` |
