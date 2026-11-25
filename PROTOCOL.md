# Frozen analysis protocol: false-discovery-controlled HER screening (JMI)

Approved 24 November 2026, 19:17 (America/Moncton), before any model was trained.

Data
- Catalysis-Hub export of MamunHighT2019, H2gas -> Hstar, 8,856 records,
  SHA-256 2545b6912f33835e51506ec42c5362c77c5f1cd34912e0a37f95e2ba76d7ac99
- De-duplication by relaxed adsorbate structure; 2-H equation energies halved;
  records without a relaxed adsorbate structure excluded
- Unit: surface (slab composition + facet), 1,836 surfaces
- Descriptor: dG_H = min over configurations of dE_H + 0.24 eV

Target and levels
- In-window: |dG_H| <= delta; primary delta = 0.20 eV; sensitivity 0.10, 0.15, 0.30 eV
- Error level: primary alpha = 0.2; sensitivity 0.1, 0.3

Designs
- Reference: chemical systems assigned 50/25/25 to training/calibration/candidate, 100 repetitions (seeds 0-99)
- Transfer A: candidates = all L1_0 surfaces; source = L1_2 + A1; 30 % of source systems to calibration; 100 repetitions
- Transfer B: candidates = all L1_2 surfaces; source = L1_0 + A1; same rule
- Weighted conformal selection (estimated likelihood ratios, logistic domain classifier) in transfer designs only, as sensitivity

Features
- Composition-weighted elemental properties of the constituent metals, structure type one-hot, facet;
  nothing derived from slab or adsorbate calculations

Predictors (all predict dG_H; mu_hat = -|dG_hat|)
- Ridge; RBF kernel ridge; Gaussian process (RBF + white noise); gradient-boosted trees;
  quantum fidelity kernel ridge and projected quantum kernel ridge (exact state-vector simulation, PCA-compressed features)
- Classical kernels also on the same PCA-compressed features; equal tuning budgets, tuning inside the training partition only

Selection rules
- Conformal selection (Jin & Candes 2023): clipped score, randomised conformal p-values, Benjamini-Hochberg at alpha
- Top-k with k = conformal shortlist size in the same split; thresholding |dG_hat| <= delta;
  interval rule (split-conformal interval at coverage 1 - alpha inside the window)

Endpoints
- Primary: true in-window surfaces on the shortlist at delta = 0.20 eV, alpha = 0.2
- Secondary: shortlist size, FDP and its mean over repetitions, empty-shortlist frequency,
  follow-up calculations per true hit, MAE of dG_H, geometric difference, computation time
- Sensitivity: surfaces with >= 3 evaluated configurations (1,475)

## Implementation details (fixed before any model was trained)
- Elemental properties (pymatgen periodic table): group, period, Pauling electronegativity,
  atomic radius, first ionisation energy, number of valence d electrons proxy = group for groups 3-12 else 0.
  Features: composition-weighted mean and absolute A-B difference of each property, majority fraction,
  structure one-hot (A1, L1_0, L1_2). Facet is fully determined by structure type and is not a separate feature.
- Standardisation fitted on the training partition.
- Tuning: 3-fold group cross-validation (groups = chemical systems) inside the training partition, minimising MAE of dG_H.
  Ridge alpha in {0.01, 0.1, 1, 10, 100}; RBF kernel ridge alpha in {1e-3, 1e-2, 1e-1, 1} x gamma in {0.01, 0.03, 0.1, 0.3, 1};
  Gaussian process: RBF + WhiteKernel, hyperparameters by marginal likelihood;
  gradient-boosted trees (learning rate 0.05): n_estimators in {200, 500} x max_depth in {2, 3, 4};
  quantum kernels: alpha in {1e-3, 1e-2, 1e-1, 1} x bandwidth in {0.25, 0.5, 1.0, 2.0, 4.0}.
- Quantum feature map: 6 qubits, PCA to 6 components (fitted on training), components scaled to [-1, 1] then multiplied
  by the bandwidth; two layers of H on all qubits, RZ(x_q) on each qubit, RZZ(x_q x_{q+1}) on a ring.
  Fidelity kernel |<psi(x)|psi(x')>|^2; projected kernel exp(-gamma_p sum_q ||rho_q(x) - rho_q(x')||_F^2) with gamma_p = 1.
  Exact state-vector simulation (2^6 amplitudes).
- PCA-6 classical comparators: RBF kernel ridge and Gaussian process on the same 6 components.
- Conformal p-values: randomised (U_j uniform), M = 10 (larger than any attainable Y <= 0).
- Weighted conformal selection: Jin & Candes (Biometrika 2026) with deterministic pruning; weights from
  logistic regression (C = 1) on standardised features distinguishing source calibration from candidates,
  w = p/(1-p) clipped to [0.05, 20].
- Geometric difference g = sqrt(|| sqrt(K_Q) K_C^{-1} sqrt(K_Q) ||_inf) on the training partition with
  K_C regularised by 1e-3 I (Huang et al. 2021), computed for each quantum kernel against RBF on the same PCA features.
- Primary model comparison at delta = 0.20 eV, alpha = 0.2; all other combinations reported in the Supplementary Material.
