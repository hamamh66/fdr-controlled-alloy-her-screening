# False-discovery-controlled screening of bimetallic alloy surfaces for hydrogen evolution

Code and data for a statistical screening study of bimetallic alloy surfaces for the
hydrogen evolution reaction (HER). Candidate surfaces are shortlisted with conformal
selection so that the false discovery rate (FDR) of the shortlist is controlled at a
stated level, both when training and test surfaces come from the same population and
when a whole crystal-structure type is held out.

## What is in the repository

| Path | Content |
|---|---|
| `code/export_catalysishub_mamun2019_H.py` | Export of the hydrogen-adsorption records of the Catalysis-Hub publication `MamunHighT2019` (requires a personal Catalysis-Hub API key) |
| `code/audit_mamun_h.py` | Data audit: de-duplication, surface-level descriptor ΔG_H = min ΔE_H + 0.24 eV, structure-type cross-check |
| `code/screening_pipeline.py` | Screening experiments: predictors, conformal and weighted conformal selection, designs, repetitions |
| `code/summarise.py` | Aggregation of per-split results into summary tables |
| `code/*.ipynb` | The same pipeline as two notebooks (data audit; selection) |
| `data/audit_surfaces.csv` | Surface-level dataset: 1,836 surfaces (34 A1, 1,192 L1₂, 610 L1₀) |
| `data/audit_summary.json` | Audit counts |
| `data/screening/` | Summaries for every design, data subset, predictor, selection rule, window half-width and error level |
| `data/raw_runs/` | Per-split raw results (`results_*.csv`) and per-split diagnostics (`meta_*.jsonl`) |
| `PROTOCOL.md` | Analysis protocol fixed before any model was trained (SHA-256 `0d7e306fb1215d5dbc642ef42479a595e47ca4931332a59943a7b48a9f0825af`) |

## Method in brief

- **Target.** A surface is a hit when |ΔG_H| lies within a window around 0 eV.
- **Predictors.** Ridge regression, RBF kernel ridge, Gaussian process, histogram gradient
  boosting, and simulated 6-qubit fidelity and projected quantum kernels on six PCA features;
  the geometric difference between classical and quantum kernels is reported.
- **Selection.** Conformal selection with clipped scores, randomised conformal p-values and the
  Benjamini–Hochberg procedure (Jin & Candès, *JMLR* 2023); weighted conformal selection with
  deterministic pruning under covariate shift (Jin & Candès, *Biometrika* 2026).
- **Designs.** A reference design (alloy-system splits 50/25/25) and two structure-type
  transfer designs (L1₀ held out; L1₂ held out), 100 repetitions each, on the full set and on
  a sensitivity subset of 1,475 surfaces with at least three adsorption configurations.

## Data provenance

The raw Catalysis-Hub export is not redistributed here; `code/export_catalysishub_mamun2019_H.py`
recreates it. Reference export: 8,856 records, SHA-256
`2545b6912f33835e51506ec42c5362c77c5f1cd34912e0a37f95e2ba76d7ac99`.
Structure-type metadata for the cross-check come from `hydrogen-unrelaxed.db` in the public
repository [UON-comp-chem/gp-representations](https://github.com/UON-comp-chem/gp-representations)
(commit `248802301bdef08cd03d5488dd270d0ac223bea2`).

## Reproducing the results

```bash
pip install -r requirements.txt
python code/export_catalysishub_mamun2019_H.py mamun_H.csv          # prompts for a Catalysis-Hub API key
python code/audit_mamun_h.py mamun_H.csv hydrogen-unrelaxed.db data  # .db from gp-representations
for s in all min3; do python code/screening_pipeline.py data/audit_surfaces.csv runs 100 $s; done
python code/summarise.py runs data/screening
```

The full screening takes about 3 CPU-hours on two cores. The notebooks run the same steps in Google Colab.

## Citation

If you use this code or data, please cite the archived release (see `CITATION.cff`; DOI on Zenodo).

## Licence

MIT (see `LICENSE`).
