#!/usr/bin/env python3
"""
Unified Post-Processing for BFG Kagome ED Results
==================================================

Reads per-Jpm .dat directories (produced by bfg_compute.py)
and generates a consistent output directory structure for both
2×3 and 3×3 clusters:

  analysis_BFG_{cluster}/
    spectrum/
      spectrum_data.txt        — all eigenvalues per sector per Jpm
      global_spectrum.{png,pdf}
      ...
    spin_structure_factor/
      spin_sf_vs_Jpm.txt       — S(q) at all momenta
      szz_sf_vs_Jpm.txt        — S^zz(q) at all momenta
      structure_factor_summary.{png,pdf}
      spin_structure_factor_heatmaps.{png,pdf}
      szz_structure_factor_heatmaps.{png,pdf}
    dimer_structure_factor/
      dimer_conn_sf_vs_Jpm.txt — D_conn(q)
      dimer_full_sf_vs_Jpm.txt — D_full(q)
      dimer_structure_factor_conn_heatmaps.{png,pdf}
      dimer_structure_factor_full_heatmaps.{png,pdf}
    bond_pattern/
      bond_pattern.{png,pdf}
    fidelity/
      sector_fidelity.{png,pdf}

Usage:
  python postprocess_unified.py --cluster 3x3 [--output-dir DIR]
  python postprocess_unified.py --cluster 2x3 [--output-dir DIR]
  python postprocess_unified.py --cluster 3x3 --export-txt-only
"""

import os
import sys
import argparse
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict

# ============================================================
# Cluster configuration
# ============================================================

# --- Primitive lattice vectors (shared) ---
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3) / 2])

# Reciprocal lattice vectors
det = A1[0] * A2[1] - A1[1] * A2[0]
B1 = (2 * np.pi / det) * np.array([A2[1], -A2[0]])
B2 = (2 * np.pi / det) * np.array([-A1[1], A1[0]])

# High-symmetry points
Q_GAMMA = np.array([0.0, 0.0])
Q_K  = (2 * B1 + B2) / 3
Q_Kp = (B1 + 2 * B2) / 3
Q_M  = B1 / 2
Q_Mp = B2 / 2
Q_Mp2 = (B1 + B2) / 2

# BZ boundary (hexagonal)
BZ_CORNERS = np.array([
    (2 * B1 + B2) / 3,
    (B1 + 2 * B2) / 3,
    (-B1 + B2) / 3,
    -(2 * B1 + B2) / 3,
    -(B1 + 2 * B2) / 3,
    (B1 - B2) / 3,
])


def get_cluster_config(cluster):
    """Return configuration dict for a given cluster string."""
    if cluster == '3x3':
        return {
            'LX': 3, 'LY': 3,
            'NUM_SITES': 27, 'N_UC': 9,
            'N_UP_LIST': [13],
            'BASE_DIR': '/scratch/zhouzb79/BFG_scan_symmetrized_pbc_3x3_nup13_negJpm',
            'DEFAULT_OUTPUT': '/scratch/zhouzb79/analysis_BFG_3x3',
            'PER_JPM_SUBDIR': 'per_jpm',
            'PRECOMPUTE_PREFIX': 'precompute_ref_ham_n_up_13',
            'NN_LIST_FILENAME': 'kagome_bfg_3x3_pbc_nn_list.dat',
            'HAS_K_POINT': True,
        }
    elif cluster == '3x3_to':
        return {
            'LX': 3, 'LY': 3,
            'NUM_SITES': 27, 'N_UC': 9,
            'N_UP_LIST': [13],
            'BASE_DIR': '/scratch/zhouzb79/BFG_scan_symmetrized_pbc_3x3_nup13_negJpm_translation_only',
            'DEFAULT_OUTPUT': '/scratch/zhouzb79/analysis_BFG_3x3_translation_only',
            'PER_JPM_SUBDIR': 'per_jpm',
            'PRECOMPUTE_PREFIX': 'precompute_ref_ham_n_up_13',
            'NN_LIST_FILENAME': 'kagome_bfg_3x3_pbc_nn_list.dat',
            'HAS_K_POINT': True,
        }
    elif cluster == '2x3':
        return {
            'LX': 2, 'LY': 3,
            'NUM_SITES': 18, 'N_UC': 6,
            'N_UP_LIST': [8, 9],
            'BASE_DIR': '/scratch/zhouzb79/BFG_scan_symmetrized_pbc_2x3_fixed_Sz',
            'DEFAULT_OUTPUT': '/scratch/zhouzb79/analysis_BFG_2x3',
            'PER_JPM_SUBDIR': None,  # 2x3 uses per_jpm/ directories
            'PRECOMPUTE_PREFIX': 'precompute_ref_ham_n_up_9',
            'NN_LIST_FILENAME': 'kagome_bfg_2x3_pbc_nn_list.dat',
            'HAS_K_POINT': False,
        }
    else:
        raise ValueError(f"Unknown cluster: {cluster}. Use '2x3', '3x3', or '3x3_to'.")


# ============================================================
# Discrete momenta
# ============================================================

def _fold_into_bz(q):
    best_q = q.copy()
    best_d = np.linalg.norm(q)
    for n1 in [-2, -1, 0, 1, 2]:
        for n2 in [-2, -1, 0, 1, 2]:
            trial = q + n1 * B1 + n2 * B2
            d = np.linalg.norm(trial)
            if d < best_d - 1e-10:
                best_d = d
                best_q = trial.copy()
    return best_q


def build_discrete_momenta(LX, LY, has_K):
    """Build discrete allowed momenta for a given cluster.

    Returns (DISCRETE_Q, DISCRETE_Q_LABELS, UNIQUE_Q_INDICES).
    """
    q_list = []
    q_labels = []
    for m1 in range(LX):
        for m2 in range(LY):
            q = (m1 / LX) * B1 + (m2 / LY) * B2
            q_f = _fold_into_bz(q)
            q_list.append(q_f)
            # Label
            if np.linalg.norm(q_f) < 0.01:
                q_labels.append(r'$\Gamma$')
            elif has_K and (np.linalg.norm(q_f - Q_K) < 0.01 or
                            np.linalg.norm(q_f + Q_K) < 0.01):
                q_labels.append(r'$K$')
            elif has_K and (np.linalg.norm(q_f - Q_Kp) < 0.01 or
                            np.linalg.norm(q_f + Q_Kp) < 0.01):
                q_labels.append(r"$K'$")
            elif np.linalg.norm(q_f - Q_M) < 0.01 or np.linalg.norm(q_f + Q_M) < 0.01:
                q_labels.append(r'$M$')
            else:
                q_labels.append(rf'$\mathbf{{k}}(\frac{{{m1}}}{{{LX}}},\frac{{{m2}}}{{{LY}}})$')

    DISCRETE_Q = np.array(q_list)

    # Unique indices (q ↔ -q symmetry)
    unique_indices = []
    seen = []
    for idx, qf in enumerate(DISCRETE_Q):
        is_dup = False
        for sidx in seen:
            if (np.linalg.norm(qf - DISCRETE_Q[sidx]) < 0.01 or
                    np.linalg.norm(qf + DISCRETE_Q[sidx]) < 0.01):
                is_dup = True
                break
        if not is_dup:
            unique_indices.append(idx)
            seen.append(idx)

    return DISCRETE_Q, q_labels, unique_indices


def _generate_extended_q(discrete_q, q_extent=6.0):
    """Generate periodic copies for plotting."""
    extended_q = []
    extended_parent = []
    for n1 in range(-3, 4):
        for n2 in range(-3, 4):
            G = n1 * B1 + n2 * B2
            for i in range(len(discrete_q)):
                qt = discrete_q[i] + G
                if abs(qt[0]) <= q_extent + 0.1 and abs(qt[1]) <= q_extent + 0.1:
                    extended_q.append(qt)
                    extended_parent.append(i)
    return np.array(extended_q), np.array(extended_parent)


# ============================================================
# Data loading: per-Jpm .dat directories
# ============================================================

def _load_metadata(filepath):
    """Load scalar metadata from key = value text file."""
    meta = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            key, val = key.strip(), val.strip()
            try:
                meta[key] = int(val)
            except ValueError:
                try:
                    meta[key] = float(val)
                except ValueError:
                    meta[key] = val
    return meta


def load_per_jpm_data(output_dir):
    """Load from per_jpm/Jpm_*/ directories (all clusters)."""
    per_jpm_dir = os.path.join(output_dir, 'per_jpm')
    if not os.path.isdir(per_jpm_dir):
        print(f"  ERROR: {per_jpm_dir} not found")
        return [], []

    all_gs = []
    all_ex = []
    for dn in sorted(os.listdir(per_jpm_dir)):
        dpath = os.path.join(per_jpm_dir, dn)
        if not os.path.isdir(dpath) or not dn.startswith('Jpm_'):
            continue

        for group_name, results_list in [('gs', all_gs), ('ex', all_ex)]:
            gdir = os.path.join(dpath, group_name)
            meta_path = os.path.join(gdir, 'metadata.dat')
            if not os.path.exists(meta_path):
                continue
            meta = _load_metadata(meta_path)
            r = {
                'Jpm': float(meta.get('Jpm', 0.0)),
                'E0': float(meta.get('E0', 0.0)),
                'gap': float(meta.get('gap', 0.0)),
                'n_gs': int(meta.get('n_gs', 1)),
            }
            for ds in ['Sq_disc', 'Szz_disc', 'Dq_disc', 'Dq_full_disc',
                       'B_mean', 'all_evals']:
                fpath = os.path.join(gdir, f'{ds}.dat')
                if os.path.exists(fpath):
                    data = np.loadtxt(fpath)
                    r[ds] = data.ravel() if data.ndim > 1 and data.shape[0] == 1 else data
            r['nem_abs'] = float(meta.get('nem_abs', 0.0))
            r['nem_arg'] = float(meta.get('nem_arg', 0.0))
            r['B01'] = float(meta.get('B01', 0.0))
            r['B02'] = float(meta.get('B02', 0.0))
            r['B12'] = float(meta.get('B12', 0.0))
            results_list.append(r)

    all_gs.sort(key=lambda x: x['Jpm'])
    all_ex.sort(key=lambda x: x['Jpm'])
    return all_gs, all_ex


def load_results(cluster, cfg, output_dir):
    """High-level loader — reads per-Jpm .dat directories."""
    return load_per_jpm_data(output_dir)


# ============================================================
# Spectrum data loading (from raw eigenvalue_mapping.txt files)
# ============================================================

def load_spectrum_data(cluster, cfg):
    """Load eigenvalue mapping from raw ED results for spectrum export.

    Returns dict: {jpm_str: [{'energy': ..., 'sector': ..., ...}, ...]}
    """
    base_dir = cfg['BASE_DIR']
    spectrum = {}

    for n_up in cfg['N_UP_LIST']:
        nup_dir = os.path.join(base_dir, f'n_up={n_up}')
        if not os.path.isdir(nup_dir):
            continue
        for d in sorted(os.listdir(nup_dir)):
            if not d.startswith('Jpm='):
                continue
            jpm_str = d.split('=', 1)[1]
            mapping_file = os.path.join(nup_dir, d, 'results',
                                        'eigenvalue_mapping.txt')
            if not os.path.exists(mapping_file):
                continue

            entries = []
            with open(mapping_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        entries.append({
                            'global_idx': int(parts[0]),
                            'energy': float(parts[1]),
                            'sector': int(parts[2]),
                            'local_idx': int(parts[3]),
                            'h5_key': parts[4],
                            'n_up': n_up,
                        })

            if jpm_str not in spectrum:
                spectrum[jpm_str] = []
            spectrum[jpm_str].extend(entries)

    # Sort within each Jpm
    for jpm_str in spectrum:
        spectrum[jpm_str].sort(key=lambda x: x['energy'])

    return spectrum


def load_sector_metadata(cluster, cfg):
    """Load sector_metadata.json from the first available Jpm directory."""
    base_dir = cfg['BASE_DIR']
    for n_up in cfg['N_UP_LIST']:
        nup_dir = os.path.join(base_dir, f'n_up={n_up}')
        if not os.path.isdir(nup_dir):
            continue
        for d in sorted(os.listdir(nup_dir)):
            if not d.startswith('Jpm='):
                continue
            meta_path = os.path.join(nup_dir, d, 'automorphism_results',
                                     'sector_metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    return json.load(f)
    return None


# ============================================================
# Text export: spectrum
# ============================================================

def export_spectrum_txt(spectrum, sector_meta, cluster, cfg, output_dir):
    """Write spectrum_data.txt: all eigenvalues per sector per Jpm.

    Format:
      # BFG Kagome {cluster} Spectrum
      # Jpm  energy  sector  n_up  sector_qn  local_idx
      -0.40  -12.345678  0  13  (0,0)  0
      -0.40  -12.345670  0  13  (0,0)  1
      ...
    """
    spec_dir = os.path.join(output_dir, 'spectrum')
    os.makedirs(spec_dir, exist_ok=True)
    txt_path = os.path.join(spec_dir, 'spectrum_data.txt')

    # Build sector quantum number map
    qn_map = {}
    if sector_meta:
        for s in sector_meta.get('sectors', []):
            qn_map[s['sector_id']] = tuple(s['quantum_numbers'])

    jpm_sorted = sorted(spectrum.keys(), key=float)

    with open(txt_path, 'w') as f:
        f.write(f"# BFG Kagome {cluster} PBC: Spectrum Data\n")
        f.write(f"# Cluster: {cfg['LX']}x{cfg['LY']}, "
                f"{cfg['NUM_SITES']} sites\n")
        f.write(f"# n_up sectors: {cfg['N_UP_LIST']}\n")
        f.write(f"# Columns: Jpm  energy  sector  n_up  "
                f"sector_qn  local_idx\n")
        f.write(f"#\n")

        for jpm_str in jpm_sorted:
            entries = spectrum[jpm_str]
            for e in entries:
                sec = e['sector']
                qn = qn_map.get(sec, (sec,))
                qn_str = ','.join(str(q) for q in qn)
                n_up = e.get('n_up', cfg['N_UP_LIST'][0])
                f.write(f"{jpm_str:>8s}  {e['energy']:16.10f}  "
                        f"{sec:3d}  {n_up:3d}  ({qn_str})  "
                        f"{e['local_idx']:3d}\n")

    print(f"  Saved {txt_path}")
    print(f"    {len(jpm_sorted)} Jpm values, "
          f"{sum(len(spectrum[j]) for j in jpm_sorted)} total eigenvalues")
    return txt_path


# ============================================================
# Text export: structure factors
# ============================================================

def export_sf_txt(all_results, discrete_q, q_labels, unique_indices,
                  cluster, cfg, output_dir, prefix=''):
    """Write SF values at all momentum positions as text files.

    Creates:
      spin_structure_factor/{prefix}spin_sf_vs_Jpm.txt
      spin_structure_factor/{prefix}szz_sf_vs_Jpm.txt
      dimer_structure_factor/{prefix}dimer_conn_sf_vs_Jpm.txt
      dimer_structure_factor/{prefix}dimer_full_sf_vs_Jpm.txt
    """
    n_q = len(discrete_q)

    # Clean labels for text header
    clean_labels = []
    for i, lbl in enumerate(q_labels):
        lbl_clean = lbl.replace('$', '').replace(r'\Gamma', 'Gamma')
        lbl_clean = lbl_clean.replace(r"K'", "Kp").replace(r'\mathbf{k}', 'k')
        lbl_clean = lbl_clean.replace(r'\frac{', '').replace('}', '')
        lbl_clean = lbl_clean.replace('{', '')
        # Add q-vector coordinates
        qx, qy = discrete_q[i]
        clean_labels.append(f"{lbl_clean}({qx:.4f},{qy:.4f})")

    sf_types = [
        ('Sq_disc', 'spin_structure_factor', f'{prefix}spin_sf_vs_Jpm.txt',
         'S(q) = (1/N) sum_ij <S_i.S_j> exp(iq.dr)'),
        ('Szz_disc', 'spin_structure_factor', f'{prefix}szz_sf_vs_Jpm.txt',
         'S^zz(q) = (1/N) sum_ij <S_i^z S_j^z> exp(iq.dr)'),
        ('Dq_disc', 'dimer_structure_factor', f'{prefix}dimer_conn_sf_vs_Jpm.txt',
         'D_conn(q) = (1/N_b) sum_ab [<B_aB_b> - <B_a><B_b>] exp(iq.dR)'),
        ('Dq_full_disc', 'dimer_structure_factor', f'{prefix}dimer_full_sf_vs_Jpm.txt',
         'D_full(q) = (1/N_b) sum_ab <B_aB_b> exp(iq.dR)'),
    ]

    paths = []
    for ds_key, subdir, fname, description in sf_types:
        sf_dir = os.path.join(output_dir, subdir)
        os.makedirs(sf_dir, exist_ok=True)
        txt_path = os.path.join(sf_dir, fname)

        try:
            with open(txt_path, 'w') as f:
                f.write(f"# BFG Kagome {cluster} PBC: {description}\n")
                f.write(f"# Cluster: {cfg['LX']}x{cfg['LY']}, "
                        f"{cfg['NUM_SITES']} sites, {n_q} discrete momenta\n")
                f.write(f"#\n")

                # Header: Jpm  q0  q1  ...  qN
                header_parts = [f"{'Jpm':>8s}"]
                for i in range(n_q):
                    header_parts.append(f"{clean_labels[i]:>20s}")
                f.write("# " + "  ".join(header_parts) + "\n")

                # Data rows
                for r in all_results:
                    if ds_key not in r:
                        continue
                    vals = r[ds_key]
                    parts = [f"{r['Jpm']:8.4f}"]
                    for i in range(n_q):
                        parts.append(f"{vals[i]:20.12f}")
                    f.write("  ".join(parts) + "\n")

            print(f"  Saved {txt_path}")
            paths.append(txt_path)
        except Exception as e:
            print(f"  WARNING: I/O error writing {txt_path}: {e}")

    return paths


# ============================================================
# Plotting helpers
# ============================================================

def draw_bz(ax, color='gray', lw=1.5, ls='-', alpha=0.8):
    corners_closed = np.vstack([BZ_CORNERS, BZ_CORNERS[0]])
    ax.plot(corners_closed[:, 0], corners_closed[:, 1],
            color=color, lw=lw, ls=ls, alpha=alpha, zorder=10)


def mark_hs_points(ax, has_K=True, color='red', ms=6, fontsize=6):
    points = [('Gamma', Q_GAMMA), ('M', Q_M), ("M'", Q_Mp)]
    if has_K:
        points.extend([('K', Q_K), ("K'", Q_Kp)])
    for name, q in points:
        ax.plot(q[0], q[1], 'x', color=color, ms=ms, mew=1.0, zorder=12)
        ax.annotate(name, (q[0], q[1]), textcoords="offset points",
                    xytext=(4, 4), fontsize=fontsize, color=color,
                    fontweight='bold', zorder=12)


# ============================================================
# Plotting: Structure Factor Summary (9-panel)
# ============================================================

def plot_sf_summary(all_results, discrete_q, q_labels, unique_indices,
                    cluster, cfg, output_dir, prefix='', state_label='GS'):
    """9-panel summary: S(q), Szz(q), D(q), nematic OP, bond energy vs Jpm."""
    NUM_SITES = cfg['NUM_SITES']
    LY = cfg['LY']
    jpm_vals = [r['Jpm'] for r in all_results]
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X']

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(rf'BFG Kagome ${cfg["LX"]}\times{cfg["LY"]}$ ({state_label}): '
                 r'Structure Factors & Nematic Order vs $J_{\pm}$', fontsize=14)

    # (a) S(q)
    ax = axes[0, 0]
    for idx_i, qi in enumerate(unique_indices):
        vals = [r['Sq_disc'][qi] for r in all_results]
        ax.plot(jpm_vals, vals, f'{markers[idx_i % len(markers)]}-',
                ms=5, lw=1.2, label=q_labels[qi])
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$S(\mathbf{q})$')
    ax.set_title(r'(a) $S(\mathbf{q})$ at allowed momenta')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) Szz(q)
    ax = axes[0, 1]
    for idx_i, qi in enumerate(unique_indices):
        if 'Szz_disc' not in all_results[0]:
            break
        vals = [r['Szz_disc'][qi] for r in all_results]
        ax.plot(jpm_vals, vals, f'{markers[idx_i % len(markers)]}-',
                ms=5, lw=1.2, label=q_labels[qi])
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$S^{zz}(\mathbf{q})$')
    ax.set_title(r'(b) $S^{zz}(\mathbf{q})$')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) D_conn(q)
    ax = axes[0, 2]
    for idx_i, qi in enumerate(unique_indices):
        vals = [r['Dq_disc'][qi] for r in all_results]
        ax.plot(jpm_vals, vals, f'{markers[idx_i % len(markers)]}-',
                ms=5, lw=1.2, label=q_labels[qi])
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$D^{\rm conn}(\mathbf{q})$')
    ax.set_title(r'(c) $D^{\rm conn}(\mathbf{q})$ (connected)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) D_full(q)
    ax = axes[1, 0]
    for idx_i, qi in enumerate(unique_indices):
        vals = [r['Dq_full_disc'][qi] for r in all_results]
        ax.plot(jpm_vals, vals, f'{markers[idx_i % len(markers)]}-',
                ms=5, lw=1.2, label=q_labels[qi])
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$D^{\rm full}(\mathbf{q})$')
    ax.set_title(r'(d) $D^{\rm full}(\mathbf{q})$')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (e) Nematic OP
    ax = axes[1, 1]
    nem_abs = [r['nem_abs'] for r in all_results]
    ax.plot(jpm_vals, nem_abs, 'ko-', ms=5, lw=1.5)
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$|\Phi_{\rm nem}|$')
    ax.set_title(r'(e) Nematic OP $|\Phi_{\rm nem}|$')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(jpm_vals, [r['B01'] for r in all_results], 's-', color='tab:red',
             ms=3, lw=1, alpha=0.7, label=r'$\bar{B}_{01}$')
    ax2.plot(jpm_vals, [r['B02'] for r in all_results], '^-', color='tab:blue',
             ms=3, lw=1, alpha=0.7, label=r'$\bar{B}_{02}$')
    ax2.plot(jpm_vals, [r['B12'] for r in all_results], 'D-', color='tab:green',
             ms=3, lw=1, alpha=0.7, label=r'$\bar{B}_{12}$')
    ax2.set_ylabel(r'$\bar{B}_{st}$')
    ax2.legend(fontsize=7, loc='upper left')

    # (f) E0/site and gap
    ax = axes[1, 2]
    ax.plot(jpm_vals, [r['E0'] / NUM_SITES for r in all_results],
            'ko-', ms=4, lw=1.5, label=r'$E_0/N$')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_0/N$')
    ax.set_title(f'(f) {state_label} energy per site')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(jpm_vals, [r['gap'] for r in all_results],
             'r^-', ms=4, lw=1, alpha=0.7)
    ax2.set_ylabel(r'Gap $\Delta$', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # (g) NN bond energy
    ax = axes[2, 0]
    b_avg = [np.mean(r['B_mean']) for r in all_results]
    b_std = [np.std(r['B_mean']) for r in all_results]
    ax.plot(jpm_vals, b_avg, 'bo-', ms=4, lw=1.5)
    ax.fill_between(jpm_vals,
                    np.array(b_avg) - np.array(b_std),
                    np.array(b_avg) + np.array(b_std),
                    alpha=0.2, color='blue')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle_{\rm NN}$')
    ax.set_title(r'(g) NN bond energy (avg $\pm$ std)')
    ax.grid(True, alpha=0.3)

    # (h) Sublattice-resolved bond energy
    ax = axes[2, 1]
    ax.plot(jpm_vals, [r['B01'] for r in all_results], 'o-', ms=4, lw=1.2,
            color='tab:red', label='Sub (0,1)')
    ax.plot(jpm_vals, [r['B02'] for r in all_results], 's-', ms=4, lw=1.2,
            color='tab:blue', label='Sub (0,2)')
    ax.plot(jpm_vals, [r['B12'] for r in all_results], '^-', ms=4, lw=1.2,
            color='tab:green', label='Sub (1,2)')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$')
    ax.set_title('(h) NN bond energy by sublattice pair')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (i) Nematic OP in complex plane
    ax = axes[2, 2]
    nem_x = [r['nem_abs'] * np.cos(r['nem_arg']) for r in all_results]
    nem_y = [r['nem_abs'] * np.sin(r['nem_arg']) for r in all_results]
    sc = ax.scatter(nem_x, nem_y, c=jpm_vals, cmap='coolwarm', s=40,
                    edgecolors='black', linewidths=0.5, zorder=5)
    for r in all_results:
        phi_x = r['nem_abs'] * np.cos(r['nem_arg'])
        phi_y = r['nem_abs'] * np.sin(r['nem_arg'])
        ax.annotate(f'{r["Jpm"]:.2f}', (phi_x, phi_y),
                    fontsize=5, textcoords='offset points', xytext=(3, 3))
    ax.axhline(0, color='gray', lw=0.5, alpha=0.3)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Re $\Phi_{\rm nem}$')
    ax.set_ylabel(r'Im $\Phi_{\rm nem}$')
    ax.set_title(r'(i) Nematic OP in complex plane')
    plt.colorbar(sc, ax=ax, label=r'$J_{\pm}$', fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save: spin SF summary goes in spin_structure_factor/
    spin_dir = os.path.join(output_dir, 'spin_structure_factor')
    os.makedirs(spin_dir, exist_ok=True)
    saved = []
    for ext in ['png', 'pdf']:
        try:
            fig.savefig(os.path.join(spin_dir,
                        f'{prefix}structure_factor_summary.{ext}'),
                        dpi=200, bbox_inches='tight')
            saved.append(ext)
        except Exception as e:
            print(f"  WARNING: I/O error saving {prefix}structure_factor_summary.{ext}: {e}")
    if saved:
        print(f"  Saved {prefix}structure_factor_summary.{'/'.join(saved)} → spin_structure_factor/")
    plt.close(fig)


# ============================================================
# Plotting: SF heatmaps (scatter at discrete momenta)
# ============================================================

def plot_sf_heatmaps(all_results, discrete_q, q_labels, cluster, cfg,
                     output_dir, selected_jpms=None, prefix='',
                     state_label='GS'):
    """Structure factor scatter plots at discrete momenta."""
    has_K = cfg['HAS_K_POINT']

    if selected_jpms is None:
        # Choose a representative set of Jpm values
        jpm_vals = sorted(set(r['Jpm'] for r in all_results))
        if len(jpm_vals) > 10:
            step = max(1, len(jpm_vals) // 10)
            selected = [jpm_vals[i] for i in range(0, len(jpm_vals), step)]
            if jpm_vals[-1] not in selected:
                selected.append(jpm_vals[-1])
        else:
            selected = jpm_vals
        selected_jpms = [f"{j:.4g}" for j in selected]

    sel = []
    for j_str in selected_jpms:
        for r in all_results:
            if abs(r['Jpm'] - float(j_str)) < 1e-8:
                sel.append((j_str, r))
                break
    if not sel:
        print("  No matching Jpm values for heatmaps")
        return

    q_extent = 6.0
    ext_q, ext_parent = _generate_extended_q(discrete_q, q_extent)
    n_sel = len(sel)
    n_cols = min(5, n_sel)
    n_rows = (n_sel + n_cols - 1) // n_cols
    marker_size = 180

    def _make_scatter_fig(title, val_key, cmap_name, symmetric, subdir, fname):
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.5 * n_cols, 4.0 * n_rows))
        axes = np.atleast_2d(axes)
        fig.suptitle(title, fontsize=14, y=1.01)

        all_vals = []
        for _, r in sel:
            if val_key not in r:
                return
            disc_vals = r[val_key]
            all_vals.extend(disc_vals[ext_parent])
        all_vals = np.array(all_vals)

        if symmetric:
            vabs = max(np.max(np.abs(all_vals)), 1e-6)
            vmin, vmax = -vabs, vabs
        else:
            vmin = min(np.min(all_vals), 0)
            vmax = max(np.max(all_vals), 1e-6)

        for idx, (jpm_str, r) in enumerate(sel):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            disc_vals = r[val_key]
            ext_vals = disc_vals[ext_parent]
            sc = ax.scatter(ext_q[:, 0], ext_q[:, 1], c=ext_vals,
                            cmap=cmap_name, s=marker_size, edgecolors='black',
                            linewidths=0.5, vmin=vmin, vmax=vmax, zorder=5)
            draw_bz(ax)
            mark_hs_points(ax, has_K=has_K)
            ax.set_aspect('equal')
            ax.set_xlim(-q_extent, q_extent)
            ax.set_ylim(-q_extent, q_extent)
            ax.set_facecolor('#f0f0f0')
            ax.set_title(rf'$J_{{\pm}} = {jpm_str}$', fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel(r'$q_x$', fontsize=8)
            if col == 0:
                ax.set_ylabel(r'$q_y$', fontsize=8)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        for idx in range(n_sel, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

        plt.tight_layout()
        out_subdir = os.path.join(output_dir, subdir)
        os.makedirs(out_subdir, exist_ok=True)
        saved = []
        for ext in ['png', 'pdf']:
            try:
                fig.savefig(os.path.join(out_subdir, f'{fname}.{ext}'),
                            dpi=200, bbox_inches='tight')
                saved.append(ext)
            except Exception as e:
                print(f"  WARNING: I/O error saving {fname}.{ext}: {e}")
        if saved:
            print(f"  Saved {fname}.{'/'.join(saved)} → {subdir}/")
        plt.close(fig)

    # Spin SF → spin_structure_factor/
    _make_scatter_fig(
        rf'Spin $S(\mathbf{{q}})$ ({state_label}) [{cfg["LX"]}$\times${cfg["LY"]}]',
        'Sq_disc', 'hot', False,
        'spin_structure_factor', f'{prefix}spin_structure_factor_heatmaps')

    _make_scatter_fig(
        rf'$S^{{zz}}(\mathbf{{q}})$ ({state_label}) [{cfg["LX"]}$\times${cfg["LY"]}]',
        'Szz_disc', 'hot', False,
        'spin_structure_factor', f'{prefix}szz_structure_factor_heatmaps')

    # Dimer SF → dimer_structure_factor/
    _make_scatter_fig(
        rf'Dimer $D^{{\rm conn}}(\mathbf{{q}})$ ({state_label}) [{cfg["LX"]}$\times${cfg["LY"]}]',
        'Dq_disc', 'RdBu_r', True,
        'dimer_structure_factor', f'{prefix}dimer_conn_heatmaps')

    _make_scatter_fig(
        rf'Dimer $D^{{\rm full}}(\mathbf{{q}})$ ({state_label}) [{cfg["LX"]}$\times${cfg["LY"]}]',
        'Dq_full_disc', 'hot', False,
        'dimer_structure_factor', f'{prefix}dimer_full_heatmaps')


# ============================================================
# Plotting: BZ + discrete momenta map
# ============================================================

def plot_bz_momenta(discrete_q, q_labels, cluster, cfg, output_dir):
    """Plot the hexagonal BZ with discrete allowed momenta."""
    has_K = cfg['HAS_K_POINT']

    def _point_in_bz(q, tol=0.05):
        n = len(BZ_CORNERS)
        for i in range(n):
            edge = BZ_CORNERS[(i + 1) % n] - BZ_CORNERS[i]
            to_point = q - BZ_CORNERS[i]
            cross = edge[0] * to_point[1] - edge[1] * to_point[0]
            if cross < -tol:
                return False
        return True

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # BZ boundary
    bz_poly = MplPolygon(BZ_CORNERS, fill=True, facecolor='lightyellow',
                         edgecolor='black', lw=2, alpha=0.4)
    ax.add_patch(bz_poly)
    corners_closed = np.vstack([BZ_CORNERS, BZ_CORNERS[0]])
    ax.plot(corners_closed[:, 0], corners_closed[:, 1], 'k-', lw=2)

    # High-symmetry points
    hs_data = [('Gamma', Q_GAMMA), ('M', Q_M), ("M'", Q_Mp)]
    if has_K:
        hs_data.extend([('K', Q_K), ("K'", Q_Kp)])
    for name, q in hs_data:
        ax.plot(q[0], q[1], 'x', color='red', ms=10, mew=2, zorder=8)
        ax.annotate(name, (q[0], q[1]), textcoords="offset points",
                    xytext=(8, 8), fontsize=11, color='red', fontweight='bold')

    # Discrete momenta in/on BZ boundary
    ext_q_bz, ext_parent_bz = _generate_extended_q(discrete_q, 5.0)
    seen_pos = set()
    bz_q = []
    for k in range(len(ext_q_bz)):
        q = ext_q_bz[k]
        if not _point_in_bz(q):
            continue
        key = (round(q[0], 2), round(q[1], 2))
        if key in seen_pos:
            continue
        seen_pos.add(key)
        bz_q.append((q, q_labels[ext_parent_bz[k]]))

    for q, lbl in bz_q:
        ax.plot(q[0], q[1], 'o', color='blue', ms=12, mew=1.5,
                markeredgecolor='black', zorder=10)
        ax.annotate(lbl, (q[0], q[1]), textcoords="offset points",
                    xytext=(10, 10), fontsize=11, color='blue',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.7, edgecolor='blue'))

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$q_x$', fontsize=13)
    ax.set_ylabel(r'$q_y$', fontsize=13)
    ax.set_title(rf'Hexagonal BZ with allowed momenta '
                 rf'(${cfg["LX"]}\times{cfg["LY"]}$ cluster)', fontsize=14)
    ax.grid(True, alpha=0.2)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=10, markeredgecolor='black', label='Allowed momenta'),
        Line2D([0], [0], marker='x', color='red', ms=10, mew=2,
               linestyle='None', label='High-symmetry points'),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='lower right')

    # Save to spin_structure_factor/ (it's a geometry reference)
    spin_dir = os.path.join(output_dir, 'spin_structure_factor')
    os.makedirs(spin_dir, exist_ok=True)
    saved = []
    for ext in ['png', 'pdf']:
        try:
            fig.savefig(os.path.join(spin_dir, f'brillouin_zone_momenta.{ext}'),
                        dpi=200, bbox_inches='tight')
            saved.append(ext)
        except Exception as e:
            print(f"  WARNING: I/O error saving brillouin_zone_momenta.{ext}: {e}")
    if saved:
        print(f"  Saved brillouin_zone_momenta.{'/'.join(saved)} → spin_structure_factor/")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified post-processing for BFG kagome ED results')
    parser.add_argument('--cluster', type=str, required=True,
                        choices=['2x3', '3x3', '3x3_to'],
                        help='Cluster type: 2x3, 3x3, or 3x3_to')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: analysis_BFG_{cluster})')
    parser.add_argument('--export-txt-only', action='store_true',
                        help='Only export txt files, skip plot generation')
    parser.add_argument('--skip-spectrum', action='store_true',
                        help='Skip spectrum txt export')
    parser.add_argument('--skip-sf', action='store_true',
                        help='Skip structure factor txt export and plots')
    args = parser.parse_args()

    cluster = args.cluster
    cfg = get_cluster_config(cluster)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = cfg['DEFAULT_OUTPUT']

    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Unified Post-Processing: BFG Kagome {cluster} PBC")
    print(f"  Sites: {cfg['NUM_SITES']}, Unit cells: {cfg['N_UC']}")
    print(f"  n_up sectors: {cfg['N_UP_LIST']}")
    print(f"  Base data: {cfg['BASE_DIR']}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # Build discrete momenta
    DISCRETE_Q, Q_LABELS, UNIQUE_Q_IDX = build_discrete_momenta(
        cfg['LX'], cfg['LY'], cfg['HAS_K_POINT'])

    print(f"\n  Discrete momenta ({len(DISCRETE_Q)} total, "
          f"{len(UNIQUE_Q_IDX)} unique):")
    for i in range(len(DISCRETE_Q)):
        q = DISCRETE_Q[i]
        uniq = '*' if i in UNIQUE_Q_IDX else ' '
        lbl = Q_LABELS[i].replace('$', '')
        print(f"    {uniq} q=({q[0]:+.4f},{q[1]:+.4f})  "
              f"|q|={np.linalg.norm(q):.4f}  {lbl}")

    # ----- Spectrum txt export -----
    if not args.skip_spectrum:
        print(f"\n--- Spectrum Data ---")
        spectrum = load_spectrum_data(cluster, cfg)
        sector_meta = load_sector_metadata(cluster, cfg)
        if spectrum:
            export_spectrum_txt(spectrum, sector_meta, cluster, cfg, output_dir)
        else:
            print("  No spectrum data found")
    else:
        sector_meta = load_sector_metadata(cluster, cfg)

    # ----- Structure factor data -----
    if not args.skip_sf:
        print(f"\n--- Structure Factor Data ---")
        all_gs, all_ex = load_results(cluster, cfg, output_dir)
        print(f"  Loaded {len(all_gs)} GS results, {len(all_ex)} ES results")

        if all_gs:
            # Export txt
            print(f"\n  Exporting SF txt (GS)...")
            export_sf_txt(all_gs, DISCRETE_Q, Q_LABELS, UNIQUE_Q_IDX,
                          cluster, cfg, output_dir)

            if not args.export_txt_only:
                # Plots
                print(f"\n  Generating SF plots (GS)...")
                plot_bz_momenta(DISCRETE_Q, Q_LABELS, cluster, cfg, output_dir)
                plot_sf_summary(all_gs, DISCRETE_Q, Q_LABELS, UNIQUE_Q_IDX,
                                cluster, cfg, output_dir,
                                prefix='', state_label='GS')
                plot_sf_heatmaps(all_gs, DISCRETE_Q, Q_LABELS, cluster, cfg,
                                 output_dir, prefix='', state_label='GS')

        if all_ex:
            print(f"\n  Exporting SF txt (1st Excited)...")
            export_sf_txt(all_ex, DISCRETE_Q, Q_LABELS, UNIQUE_Q_IDX,
                          cluster, cfg, output_dir, prefix='ex_')

            if not args.export_txt_only:
                print(f"\n  Generating SF plots (1st Excited)...")
                plot_sf_summary(all_ex, DISCRETE_Q, Q_LABELS, UNIQUE_Q_IDX,
                                cluster, cfg, output_dir,
                                prefix='ex_', state_label='1st Excited')
                plot_sf_heatmaps(all_ex, DISCRETE_Q, Q_LABELS, cluster, cfg,
                                 output_dir, prefix='ex_',
                                 state_label='1st Excited')

    print(f"\n{'='*60}")
    print(f"Output directory structure:")

    for subdir in ['spectrum', 'spin_structure_factor',
                    'dimer_structure_factor']:
        full = os.path.join(output_dir, subdir)
        if os.path.isdir(full):
            files = sorted(os.listdir(full))
            print(f"  {subdir}/")
            for fn in files:
                print(f"    {fn}")
    print(f"{'='*60}")
    print("Done.")


if __name__ == '__main__':
    main()
