# BFG Kagome ED Analysis Pipeline

Post-processing pipeline for BFG (bond-frustrated-graphene) kagome lattice
exact diagonalization results. Computes correlations, structure factors, RDMs,
entanglement entropy, and energy spectra from symmetrized Lanczos outputs.

## Pipeline Overview

```
 Lanczos ED (external)          bfg_compute.py (SLURM array)         bfg_analyze.py (single job)
┌──────────────────┐          ┌──────────────────────────────┐     ┌──────────────────────────────┐
│ HDF5 eigenvalues │          │ Per-Jpm worker (1 per task):  │     │ Lightweight aggregation:     │
│ & eigenvectors   │───────>  │  Step 1: per-eigenstate       │──>  │  - Energy spectrum + ToS     │
│ (per Sz sector)  │          │    correlations, SF, RDM, ⟨Sz⟩│     │  - S(q)/D(q) heatmaps & cuts│
│                  │          │  Step 2: trivial GS average   │     │  - Per-eigenstate diagnostics│
└──────────────────┘          │  Step 3: 1st excited state    │     └──────────────────────────────┘
                              └──────────────────────────────┘
```

### Compute architecture (bfg_compute.py)

The compute step follows a clean per-eigenstate-first design:

1. **Per-eigenstate** (`compute_single_eigenstate`): For each GS eigenstate,
   load ψ once and compute ALL observables (spin/bond correlations, structure
   factors, nematic OP, RDM + entanglement entropy). No eigenvector is loaded
   twice.
2. **GS average** (`average_per_state_results`): Trivially average the per-state
   results. GS-averaged S(q) = mean of per-state S(q) by linearity.
3. **Excited state**: Repeat step 1-2 for the 1st excited manifold (no RDM,
   no per-state storage).

### Memory optimization

Eigenvectors live in the full 2^N Hilbert space but only have support in
the Sz-conserving sector (C(27,13) ≈ 20M states vs 2^27 = 134M). All
correlation functions operate in the reduced sector via Gosper's hack
basis generation. The SpSm correlator exploits Hermiticity (only upper
triangle computed), halving the number of searchsorted calls.

Peak memory: ~18 GB (dominated by bond-bond φ array: 54 × 20M × 16B).

## Files

| File | Purpose |
|---|---|
| `bfg_compute.py` | **Per-Jpm worker**: per-eigenstate correlations, SF, RDM, entanglement entropy. SLURM array job. |
| `bfg_analyze.py` | **Aggregation**: energy spectrum, SF plots, per-eigenstate diagnostics. Single job after compute. |
| `postprocess_unified.py` | Structure factor postprocessing library (heatmaps, BZ cuts). Imported by `bfg_analyze.py`. |
| `plot_spectrum_BFG_3x3.py` | Energy spectrum + tower-of-states plotting (3×3). Imported by `bfg_analyze.py`. |
| `plot_spectrum_BFG_2x3.py` | Energy spectrum plotting (2×3). Imported by `bfg_analyze.py`. |
| `plot_diagnostics.py` | Per-eigenstate diagnostics (⟨Sz⟩, chirality, bond energy, SF). Imported by `bfg_analyze.py`. |
| `submit_compute.sh` | SBATCH array job wrapper for `bfg_compute.py` (32 GB, 3h). |
| `submit_analyze.sh` | SBATCH wrapper for `bfg_analyze.py` (128 GB, 2h). |
| `submit_pipeline.sh` | **One-command launcher**: submits compute + analysis with dependency chaining. |

## Cluster Configurations

| Cluster | Sites | Unit cells | Sz sectors | Symmetry | Data directory |
|---|---|---|---|---|---|
| `3x3` | 27 | 9 | n_up=13 | Full (C₃ + translations) | `BFG_scan_symmetrized_pbc_3x3_nup13_negJpm/` |
| `3x3_to` | 27 | 9 | n_up=13 | Translation only | `BFG_scan_symmetrized_pbc_3x3_nup13_negJpm_translation_only/` |
| `2x3` | 18 | 6 | n_up=8,9 | Full | `BFG_scan_symmetrized_pbc_2x3_fixed_Sz/` |

## Usage

### One-command launch (recommended)

```bash
cd /scratch/zhouzb79/bfg_pipeline

# Run full pipeline (compute array + analysis) for a cluster:
./submit_pipeline.sh 3x3_to
./submit_pipeline.sh 3x3
./submit_pipeline.sh 2x3

# Compute only (no analysis):
./submit_pipeline.sh 3x3_to --compute-only

# Analysis only (compute already done):
./submit_pipeline.sh 3x3_to --analyze-only
```

### Manual submission

```bash
cd /scratch/zhouzb79/bfg_pipeline

# Submit compute array job (40 Jpm values for 3×3 translation-only)
COMP_JOB=$(sbatch --export=CLUSTER=3x3_to --array=0-39 --job-name=comp_3x3to \
  submit_compute.sh | awk '{print $4}')

# Submit analysis job (depends on compute finishing)
sbatch --dependency=afterok:${COMP_JOB} --export=CLUSTER=3x3_to \
  --job-name=ana_3x3to submit_analyze.sh
```

### Run specific analysis step only

```bash
python bfg_analyze.py --cluster 3x3_to --only spectrum
python bfg_analyze.py --cluster 3x3_to --only sf
python bfg_analyze.py --cluster 3x3_to --only diagnostics
```

### Recompute a single Jpm interactively

```bash
python bfg_compute.py --cluster 3x3_to --worker --jpm -0.05
```

### Run all Jpm sequentially (no SLURM)

```bash
python bfg_compute.py --cluster 3x3_to
```

## Output Structure

```
analysis_BFG_3x3_translation_only/
├── per_jpm/
│   └── Jpm_-0.02/
│       ├── gs/
│       │   ├── metadata.dat              # Scalars: E0, gap, n_gs, nem_abs, S_vn, ...
│       │   ├── SzSz.dat, SiSj.dat        # Correlation matrices (N×N)
│       │   ├── SpSm_real.dat, SpSm_imag.dat
│       │   ├── sz.dat, B_mean.dat, C_bb.dat
│       │   ├── Sq_disc.dat, Szz_disc.dat  # GS-averaged SF at discrete momenta
│       │   ├── Dq_disc.dat, Dq_full_disc.dat
│       │   ├── all_evals.dat              # All eigenvalues (spectrum)
│       │   ├── per_state/                 # Per-eigenstate arrays
│       │   │   ├── Sq.dat, Szz.dat        # (n_gs × n_q)
│       │   │   ├── Dq_full.dat, Dq_conn.dat
│       │   │   ├── sz_local.dat           # (n_gs × N)
│       │   │   ├── B_mean.dat             # (n_gs × N_bonds)
│       │   │   ├── SzSz_bond.dat          # (n_gs × N_bonds)
│       │   │   ├── SiSj_corr.dat          # (n_gs × N × N), 3D with shape header
│       │   │   ├── SzSz_corr.dat          # (n_gs × N × N), 3D with shape header
│       │   │   ├── sector.dat             # Sector indices
│       │   │   └── triangle_list.dat      # Triangle geometry
│       │   └── rdm/
│       │       ├── hexagon_o0/
│       │       │   ├── metadata.dat       # S_vn, S_R2, trace
│       │       │   ├── rdm_real.dat, rdm_imag.dat
│       │       │   ├── spectrum.dat
│       │       │   └── per_state/
│       │       │       ├── rdm_0_real.dat, rdm_0_imag.dat, ...
│       │       │       ├── S_vn.dat, S_R2.dat
│       │       │       └── spectrum_0.dat, ...
│       │       ├── bowtie_o0/             # 3 orientations
│       │       ├── bowtie_o1/
│       │       └── bowtie_o2/
│       └── ex/                            # 1st excited (same structure, no RDM)
├── rdm/                                   # Annotated sparse-format RDM text files
│   └── Jpm=-0.02/
│       ├── rdm_gs_averaged_hexagon_o0_ngs2.txt
│       └── rdm_eigenstate_0_hexagon_o0_ngs2.txt
├── geometry/                              # Lattice geometry
│   ├── nn_pairs.dat, bond_positions.dat
│   ├── discrete_q.dat, bz_corners.dat
│   └── discrete_q_labels.dat
├── analysis_data.csv                      # Summary CSV (all Jpm)
├── sector_fidelity/                       # GS fidelity data
├── spectrum/                              # Energy spectrum + ToS plots
├── spin_structure_factor/                 # S(q) heatmaps and cuts
├── dimer_structure_factor/                # D(q) data
└── diagnostics/                           # Per-eigenstate diagnostic plots
```

## RDM Details

RDMs are computed inside `compute_single_eigenstate()` — each eigenstate's RDM
is computed in the same pass as its correlations, with no eigenvector re-loading.
The GS-averaged + spin-flipped RDM is:

$$\rho_A = \frac{1}{n_\text{gs}} \sum_{i=1}^{n_\text{gs}}
\frac{1}{2}\left[ \text{Tr}_{\bar{A}}(|\psi_i\rangle\langle\psi_i|)
     + \text{Tr}_{\bar{A}}(|\tilde\psi_i\rangle\langle\tilde\psi_i|) \right]$$

where $|\tilde\psi_i\rangle$ is the global spin-flip partner.

**Subsystems** (3×3 clusters):
- Hexagon: 6 sites `[1,2,3,4,9,11]`, dim=64, 1 orientation
- Bowtie: 5 sites, 3 orientations, dim=32

## Dependencies

- Python 3.x with NumPy, SciPy, Matplotlib, h5py
- On Cedar/Graham: `module load StdEnv/2023 python scipy-stack hdf5`
- Virtual env: `~/pystandard`
