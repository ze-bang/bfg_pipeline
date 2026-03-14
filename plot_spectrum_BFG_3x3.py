#!/usr/bin/env python3
"""
Plot low-energy spectrum for BFG kagome 3×3 PBC cluster.

Generates:
  1) Global spectrum: all eigenvalues vs Jpm
  2) Per-symmetry-sector spectrum: color-coded by sector index (q1,q2)
  3) Sector GS energy overview

Output directory: analysis_BFG_3x3/spectrum/
"""

import os
import sys
import argparse
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

# ============================================================
# Constants
# ============================================================
BASE_DIR = '/scratch/zhouzb79/BFG_scan_symmetrized_pbc_3x3_nup13_negJpm'
N_UP = 13
NUM_SITES = 27
SZ_VAL = N_UP - NUM_SITES / 2   # = -0.5
TRANSLATION_ONLY = False  # Set True for 3x3_to (pure translation symmetry)


# ============================================================
# Data loading
# ============================================================
def discover_jpm_values():
    """Find all Jpm directories with valid data.

    Skips Jpm=0 (Heisenberg point) where the enhanced symmetry gives a
    different sector decomposition, polluting sector-resolved plots.
    """
    base = os.path.join(BASE_DIR, f'n_up={N_UP}')
    jpms = []
    if not os.path.isdir(base):
        return jpms
    for d in os.listdir(base):
        if not d.startswith('Jpm='):
            continue
        jpm_str = d.split('=', 1)[1]
        # Skip Jpm=0 (Heisenberg point)
        if float(jpm_str) == 0.0:
            print(f"  Skipping Jpm={jpm_str} (Heisenberg point)")
            continue
        mapping = os.path.join(base, f'Jpm={jpm_str}', 'results',
                               'eigenvalue_mapping.txt')
        if os.path.exists(mapping):
            jpms.append(jpm_str)
    return sorted(jpms, key=float)


def read_eigenvalue_mapping(jpm_str):
    """Read eigenvalue_mapping.txt for one Jpm."""
    mapping_file = os.path.join(BASE_DIR, f'n_up={N_UP}', f'Jpm={jpm_str}',
                                'results', 'eigenvalue_mapping.txt')
    entries = []
    if not os.path.exists(mapping_file):
        return entries
    with open(mapping_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                entries.append({
                    'global_idx': int(parts[0]),
                    'energy':     float(parts[1]),
                    'sector':     int(parts[2]),
                    'local_idx':  int(parts[3]),
                    'h5_key':     parts[4],
                })
    entries.sort(key=lambda x: x['energy'])
    return entries


def load_sector_metadata(jpm_str):
    """Load sector_metadata.json to get quantum number labels."""
    meta_path = os.path.join(BASE_DIR, f'n_up={N_UP}', f'Jpm={jpm_str}',
                             'automorphism_results', 'sector_metadata.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def collect_all_spectra(jpm_list):
    """Collect eigenvalues for all Jpm.

    Returns
    -------
    data : dict  { jpm_str: { 'jpm': float, 'entries': [...] } }
    sector_meta : dict or None
    """
    data = {}
    sector_meta = None

    for jpm_str in jpm_list:
        entries = read_eigenvalue_mapping(jpm_str)
        data[jpm_str] = {
            'jpm': float(jpm_str),
            'entries': entries,
        }
        if sector_meta is None and entries:
            sector_meta = load_sector_metadata(jpm_str)

    return data, sector_meta


def _sector_momentum_label_to(q1, q2):
    r"""Map (q1, q2) to BZ label for 3×3 translation-only.

    Generators (from automorphism analysis):
      G0 = T_{a2}   with eigenvalue omega^{q1}
      G1 = T_{-a1}  with eigenvalue omega^{q2}   (omega = e^{2pi i/3})
    so k·a2 = 2π q1/3  and  -k·a1 = 2π q2/3,
    giving k = (-q2/3) b1 + (q1/3) b2.

    High-symmetry points accessible on 3×3:
      Γ  = (0,0)
      K  = (2/3)b1 + (1/3)b2  → (q1,q2) = (1,1)   [C3-fixed]
      K' = (1/3)b1 + (2/3)b2  → (q1,q2) = (2,2)   [C3-fixed]
      M is NOT accessible (requires half-integer).

    C3 orbits (verified geometrically):
      Γ:      {(0,0)}              — C3-fixed
      K:      {(1,1)}              — C3-fixed (BZ corner)
      K':     {(2,2)}              — C3-fixed (BZ corner)
      Orb A:  {(0,1),(2,0),(1,2)}  — interior C3 triplet
      Orb B:  {(0,2),(1,0),(2,1)}  — interior C3 triplet (TR partner of A)
    """
    if (q1, q2) == (0, 0):
        return r'$\Gamma$'
    elif (q1, q2) == (1, 1):
        return r'$K$'
    elif (q1, q2) == (2, 2):
        return r"$K'$"
    return rf'$({q1},{q2})$'


def _sector_group_to(q1, q2):
    """Classify sector into C3 orbit group (TO mode).

    k = (-q2/3)*b1 + (q1/3)*b2.
    C3 action on (q1,q2) [verified geometrically]:
      (0,0)->(0,0), (0,1)->(2,0)->(1,2), (0,2)->(1,0)->(2,1),
      (1,1)->(1,1), (2,2)->(2,2).
    """
    if (q1, q2) == (0, 0):
        return 'Gamma'
    if (q1, q2) == (1, 1):  # K — C3-fixed BZ corner
        return 'K'
    if (q1, q2) == (2, 2):  # K' — C3-fixed BZ corner
        return 'Kp'
    if (q1, q2) in ((0, 1), (2, 0), (1, 2)):  # Interior orbit A
        return 'orbA'
    # (0,2), (1,0), (2,1): Interior orbit B (TR partner of A)
    return 'orbB'


def build_sector_momentum_map(sector_meta):
    r"""Determine the correct momentum label for each sector.

    Handles two cases:
    - Full symmetry (TRANSLATION_ONLY=False):  G0=T, G1=C3^{-1}∘T
      → (q1,q2) are NOT lattice momenta
      → K=(0,1), K'=(0,2)
    - Translation-only (TRANSLATION_ONLY=True): G0=T_{a2}, G1=T_{-a1}
      → k = (-q2/3)b1 + (q1/3)b2 is proper lattice momentum
      → K=(1,1), K'=(2,2)
    """
    if sector_meta is None:
        return {}
    mapping = {}
    for s in sector_meta.get('sectors', []):
        sid = s['sector_id']
        q1, q2 = s['quantum_numbers']
        if TRANSLATION_ONLY:
            mapping[sid] = _sector_momentum_label_to(q1, q2)
        else:
            mapping[sid] = _sector_momentum_label_3x3(q1, q2)
    return mapping


def get_sector_qn_label(sector_meta, sector_idx):
    """Get a human-readable quantum number label for a sector."""
    if sector_meta is None:
        return str(sector_idx)
    for s in sector_meta.get('sectors', []):
        if s['sector_id'] == sector_idx:
            qn = s['quantum_numbers']
            return ','.join(str(q) for q in qn)
    return str(sector_idx)


def get_sector_momentum_label(sector_meta, sector_idx, momentum_map=None):
    """Get the physical momentum label for a sector."""
    if momentum_map and sector_idx in momentum_map:
        return momentum_map[sector_idx]
    return get_sector_qn_label(sector_meta, sector_idx)


# ============================================================
# Plot 1: Global spectrum
# ============================================================
def plot_global_spectrum(data, jpm_sorted, output_dir, n_show=40):
    """All eigenvalues vs Jpm."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(r'BFG Kagome $3\times3$ PBC: Low-Energy Spectrum '
                 rf'($n_{{\uparrow}}={N_UP}$, $S^z={SZ_VAL:.1f}$)', fontsize=14)

    jpm_vals = np.array([float(j) for j in jpm_sorted])

    # --- (a) E/N for lowest n_show levels ---
    ax = axes[0]
    for k in range(n_show):
        evals_k = []
        for j in jpm_sorted:
            entries = data[j]['entries']
            if k < len(entries):
                evals_k.append(entries[k]['energy'] / NUM_SITES)
            else:
                evals_k.append(np.nan)
        color = 'red' if k == 0 else ('blue' if k < 10 else 'gray')
        alpha = 1.0 if k == 0 else (0.7 if k < 10 else 0.3)
        lw = 1.5 if k == 0 else (0.8 if k < 10 else 0.5)
        ax.plot(jpm_vals, evals_k, '-', lw=lw, alpha=alpha, color=color)

    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_n / N$')
    ax.set_title(f'(a) Lowest {n_show} levels')
    ax.grid(True, alpha=0.3)

    # --- (b) Energy relative to GS ---
    ax = axes[1]
    for k in range(1, n_show):
        gaps_k = []
        for j in jpm_sorted:
            entries = data[j]['entries']
            e0 = entries[0]['energy']
            if k < len(entries):
                gaps_k.append(entries[k]['energy'] - e0)
            else:
                gaps_k.append(np.nan)
        color = 'blue' if k < 10 else 'gray'
        alpha = 0.8 if k < 10 else 0.3
        lw = 1.0 if k < 10 else 0.5
        ax.plot(jpm_vals, gaps_k, '-', lw=lw, alpha=alpha, color=color)

    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_n - E_0$')
    ax.set_title(f'(b) Excitation energies (lowest {n_show})')
    ax.set_ylim(bottom=-0.01)
    ax.grid(True, alpha=0.3)

    # --- (c) GS energy per site + gap ---
    ax = axes[2]
    e0_arr = [data[j]['entries'][0]['energy'] / NUM_SITES for j in jpm_sorted]
    ax.plot(jpm_vals, e0_arr, 'ko-', ms=4, lw=1.5, label=r'$E_0/N$')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_0/N$')
    ax.set_title('(c) GS energy per site & gap')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    ax2 = ax.twinx()
    gaps = []
    deg_tol_per_site = 2e-5
    abs_tol = deg_tol_per_site * NUM_SITES
    for j in jpm_sorted:
        entries = data[j]['entries']
        e0 = entries[0]['energy']
        e1 = None
        for e in entries[1:]:
            if abs(e['energy'] - e0) > abs_tol:
                e1 = e['energy']
                break
        gaps.append(e1 - e0 if e1 is not None else 0.0)
    ax2.plot(jpm_vals, gaps, 'r^-', ms=4, lw=1, alpha=0.7)
    ax2.set_ylabel(r'Gap $\Delta$', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'global_spectrum.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved global_spectrum.png/pdf")


# ============================================================
# Plot 2: Per-symmetry-sector spectrum
# ============================================================
def _sector_momentum_label_3x3(q1, q2):
    r"""Map (q1, q2) sector quantum numbers to a proper BZ label for 3×3 full symmetry.

    Full-symmetry generators: G0 = T_{a1+a2}, G1 = C3^{-1} ∘ T.
    (q1,q2) are NOT lattice momenta.  Physical mapping (verified by S(q)):
      K  → (q1,q2) = (0,1)
      K' → (q1,q2) = (0,2)
      Γ  → (q1,q2) = (0,0)
    """
    if (q1, q2) == (0, 0):
        return r'$\Gamma$'
    elif (q1, q2) == (0, 1):
        return r'$K$'
    elif (q1, q2) == (0, 2):
        return r"$K'$"
    return rf'$({q1},{q2})$'


def _sector_momentum_label(q1, q2):
    """Dispatch to the correct labeling function based on TRANSLATION_ONLY."""
    if TRANSLATION_ONLY:
        return _sector_momentum_label_to(q1, q2)
    return _sector_momentum_label_3x3(q1, q2)


def _sector_group_full(q1, q2):
    """Classify sector into BZ-point group for full-symmetry mode."""
    if (q1, q2) == (0, 0):
        return 'Gamma'
    if (q1, q2) in ((0, 1), (0, 2)):
        return 'K'
    return 'gen'


# Palette and markers for orbit groups
_ORBIT_PALETTE_TO = {
    'Gamma': '#e74c3c',   # red
    'K':     '#27ae60',   # green  — K at (1,1)
    'Kp':    '#2980b9',   # blue   — K' at (2,2)
    'orbA':  '#f39c12',   # orange — interior orbit {(0,1),(2,0),(1,2)}
    'orbB':  '#9b59b6',   # purple — interior orbit {(0,2),(1,0),(2,1)}
}
_ORBIT_MARKER_TO = {
    'Gamma': 'o',
    'K':     'h',    # hexagon for BZ corner
    'Kp':    'H',    # hexagon for BZ corner
    'orbA':  's',
    'orbB':  'D',
}
_ORBIT_LABEL_TO = {
    'Gamma': r'$\Gamma$: $(0,0)$',
    'K':     r'$K$: $(1,1)$',
    'Kp':    r"$K'$: $(2,2)$",
    'orbA':  r'$C_3$ orbit: $\{(0,1),(2,0),(1,2)\}$',
    'orbB':  r'$C_3$ orbit: $\{(0,2),(1,0),(2,1)\}$',
}
_ORBIT_ORDER_TO = ['Gamma', 'K', 'Kp', 'orbA', 'orbB']

_ORBIT_PALETTE_FULL = {
    'Gamma': '#e74c3c',
    'K':     '#27ae60',
    'gen':   '#f1c40f',
}
_ORBIT_MARKER_FULL = {
    'Gamma': 'o',
    'K':     's',
    'gen':   '^',
}
_ORBIT_LABEL_FULL = {
    'Gamma': r'$\Gamma$',
    'K':     r'$K / K^\prime$',
    'gen':   r'Generic $C_3$ orbit',
}
_ORBIT_ORDER_FULL = ['Gamma', 'K', 'gen']


def _get_orbit_config():
    """Return (palette, markers, labels, order, group_fn) for current mode."""
    if TRANSLATION_ONLY:
        return (_ORBIT_PALETTE_TO, _ORBIT_MARKER_TO, _ORBIT_LABEL_TO,
                _ORBIT_ORDER_TO, _sector_group_to)
    return (_ORBIT_PALETTE_FULL, _ORBIT_MARKER_FULL, _ORBIT_LABEL_FULL,
            _ORBIT_ORDER_FULL, _sector_group_full)


def _sector_orbit_color(qn_map, sec):
    """Get (color, marker, group_name) for a sector."""
    palette, markers, _, _, group_fn = _get_orbit_config()
    q1, q2 = qn_map.get(sec, (sec, 0))
    grp = group_fn(q1, q2)
    return palette[grp], markers[grp], grp


def _orbit_legend_handles():
    """Build legend handles for orbit groups."""
    palette, markers, labels, order, _ = _get_orbit_config()
    handles = []
    for g in order:
        handles.append(Line2D([], [], color=palette[g], marker=markers[g],
                              ms=7, lw=0, markeredgecolor='k',
                              markeredgewidth=0.5, label=labels[g]))
    return handles


def plot_symmetry_sector_spectrum(data, jpm_sorted, sector_meta, output_dir,
                                 n_show_per_sector=5, momentum_map=None):
    """Excitation gap per sector: 3×3 grid of subfigures, one per sector."""
    jpm_vals = np.array([float(j) for j in jpm_sorted])

    # Identify all sectors
    all_sectors = set()
    for j in jpm_sorted:
        for e in data[j]['entries']:
            all_sectors.add(e['sector'])
    all_sectors = sorted(all_sectors)

    # Build sector quantum number map
    sec_qn = {}
    if sector_meta:
        for s in sector_meta.get('sectors', []):
            sec_qn[s['sector_id']] = tuple(s['quantum_numbers'])

    # Build curves: sector_curves[sec][k] = list of (jpm_val, energy)
    sector_curves = defaultdict(lambda: defaultdict(list))
    for j in jpm_sorted:
        jv = float(j)
        by_sec = defaultdict(list)
        for e in data[j]['entries']:
            by_sec[e['sector']].append(e['energy'])
        for sec in all_sectors:
            sec_evals = sorted(by_sec.get(sec, []))
            for k, ev in enumerate(sec_evals[:n_show_per_sector]):
                sector_curves[sec][k].append((jv, ev))

    # Build qn_map for orbit coloring
    qn_map = {}
    if sector_meta:
        for s in sector_meta.get('sectors', []):
            qn_map[s['sector_id']] = tuple(s['quantum_numbers'])

    # 3×3 grid: one subplot per sector
    n_sec = len(all_sectors)
    ncols = 3
    nrows = (n_sec + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(r'BFG Kagome $3\times3$: Excitation Gap per Sector '
                 rf'($n_{{\uparrow}}={N_UP}$, $S^z={SZ_VAL:.1f}$)',
                 fontsize=14, y=0.98)

    panel_labels = [f'({chr(ord("a") + i)})' for i in range(n_sec)]

    for idx, sec in enumerate(all_sectors[:n_sec]):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # Orbit color for this sector
        orbit_color, orbit_marker, orbit_grp = _sector_orbit_color(qn_map, sec)

        # Momentum label
        qn = sec_qn.get(sec, (sec,))
        if len(qn) == 2:
            mom_label = _sector_momentum_label(qn[0], qn[1])
            qn_str = f'$q_1={qn[0]},\\,q_2={qn[1]}$'
        else:
            mom_label = momentum_map.get(sec, f'sec {sec}') if momentum_map else f'sec {sec}'
            qn_str = ''

        # Compute excitation gap: E_k - E_0 within this sector
        if sec in sector_curves and 0 in sector_curves[sec]:
            e0_pts = {p[0]: p[1] for p in sector_curves[sec][0]}
            for k in sorted(sector_curves[sec].keys()):
                if k == 0:
                    continue
                pts = sector_curves[sec][k]
                xs = [p[0] for p in pts if p[0] in e0_pts]
                ys = [p[1] - e0_pts[p[0]] for p in pts if p[0] in e0_pts]
                alpha = 0.8 if k == 1 else 0.4
                lw = 1.2 if k == 1 else 0.6
                ax.plot(xs, ys, '-', lw=lw, alpha=alpha, color=orbit_color)

            # Highlight the gap (first excitation)
            if 1 in sector_curves[sec]:
                pts = sector_curves[sec][1]
                xs = [p[0] for p in pts if p[0] in e0_pts]
                ys = [p[1] - e0_pts[p[0]] for p in pts if p[0] in e0_pts]
                ax.plot(xs, ys, 'o-', ms=3, lw=1.5, color=orbit_color,
                        label=r'$\Delta_1$', zorder=5)

        title_str = f'{panel_labels[idx]} {mom_label}'
        if qn_str:
            title_str += f'  ({qn_str})'
        ax.set_title(title_str, fontsize=11, color=orbit_color,
                     fontweight='bold')
        # Colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(orbit_color)
            spine.set_linewidth(2.5)
        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel(r'$E_n - E_0^{\rm sector}$')
        ax.set_ylim(bottom=-0.01)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    # Orbit legend in last visible panel
    legend_ax = axes[0, ncols - 1]
    orbit_handles = _orbit_legend_handles()
    legend_ax.legend(handles=orbit_handles, fontsize=8, loc='upper right',
                     title='$C_3$ orbit', title_fontsize=8)

    # Turn off unused subplots
    for si in range(n_sec, nrows * ncols):
        row, col = divmod(si, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'symmetry_sector_spectrum.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved symmetry_sector_spectrum.png/pdf")


# ============================================================
# Plot 3: Sector GS energy overview + degeneracy
# ============================================================
def plot_sector_gs_energies(data, jpm_sorted, sector_meta, output_dir,
                            momentum_map=None):
    """Sector GS energies, GS-relative, and degeneracy information."""
    jpm_vals = np.array([float(j) for j in jpm_sorted])

    # Identify all sectors
    all_sectors = set()
    for j in jpm_sorted:
        for e in data[j]['entries']:
            all_sectors.add(e['sector'])
    all_sectors = sorted(all_sectors)
    n_sec = len(all_sectors)

    # Orbit-based coloring
    qn_map_gs = {}
    if sector_meta:
        for s in sector_meta.get('sectors', []):
            qn_map_gs[s['sector_id']] = tuple(s['quantum_numbers'])
    colors = {s: _sector_orbit_color(qn_map_gs, s)[0] for s in all_sectors}
    sec_markers = {s: _sector_orbit_color(qn_map_gs, s)[1] for s in all_sectors}

    # Build sector_E0[sec][jpm_idx]
    sector_E0 = {s: np.full(len(jpm_sorted), np.nan) for s in all_sectors}
    for ji, j in enumerate(jpm_sorted):
        by_sec = defaultdict(list)
        for e in data[j]['entries']:
            by_sec[e['sector']].append(e['energy'])
        for sec in all_sectors:
            if sec in by_sec:
                sector_E0[sec][ji] = min(by_sec[sec])

    if TRANSLATION_ONLY:
        subtitle = (r'Sectors labeled by lattice momentum '
                    r'$\mathbf{k} = (q_1/3)\,\mathbf{b}_1 + (q_2/3)\,\mathbf{b}_2$')
    else:
        subtitle = (r'NOTE: Sectors labeled by $(q_1,q_2)$ eigenvalues of '
                    r'$T_{\mathbf{a}_1+\mathbf{a}_2}$ and $C_3^{-1}\circ T$, '
                    r'not lattice momentum')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(r'BFG Kagome $3\times3$ PBC: Sector GS Energy Overview '
                 rf'($n_{{\uparrow}}={N_UP}$, $S^z={SZ_VAL:.1f}$)'
                 '\n' + subtitle,
                 fontsize=13)

    # Helper to get label
    def _sec_label(sec):
        mom = get_sector_momentum_label(sector_meta, sec, momentum_map)
        return f'{mom} (sec {sec})'

    # (a) E_0^sector / N
    ax = axes[0, 0]
    for sec in all_sectors:
        valid = ~np.isnan(sector_E0[sec])
        if np.any(valid):
            label = _sec_label(sec)
            ax.plot(jpm_vals[valid], sector_E0[sec][valid] / NUM_SITES,
                    marker=sec_markers[sec], ls='-',
                    color=colors[sec], ms=3, lw=1.2,
                    label=label)
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_0^{\rm sector} / N$')
    ax.set_title('(a) Sector GS energies')
    ax.legend(fontsize=8, ncol=1, loc='best')
    ax.grid(True, alpha=0.3)

    # (b) E_0^sector - E_0^global
    ax = axes[0, 1]
    e0_global = np.array([data[j]['entries'][0]['energy'] for j in jpm_sorted])
    for sec in all_sectors:
        valid = ~np.isnan(sector_E0[sec])
        if np.any(valid):
            delta = sector_E0[sec][valid] - e0_global[valid]
            label = _sec_label(sec)
            ax.plot(jpm_vals[valid], delta,
                    marker=sec_markers[sec], ls='-',
                    color=colors[sec], ms=3, lw=1.2,
                    label=label)
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_0^{\rm sector} - E_0^{\rm global}$')
    ax.set_title('(b) Sector GS relative to global GS')
    ax.legend(fontsize=8, ncol=1, loc='best')
    ax.set_ylim(bottom=-0.01)
    ax.grid(True, alpha=0.3)

    # (c) GS degeneracy at multiple per-site tolerances
    ax = axes[1, 0]
    deg_tols_per_site = [2e-5, 5e-5, 1e-4]
    styles = [('ko-', 4, 1.5), ('rs-', 3, 1.0), ('b^-', 3, 1.0)]
    for tol_ps, (style, ms, lw) in zip(deg_tols_per_site, styles):
        abs_tol = tol_ps * NUM_SITES
        gs_n_degenerate = []
        for ji, j in enumerate(jpm_sorted):
            entries = data[j]['entries']
            e0 = entries[0]['energy']
            n_deg = sum(1 for e in entries if abs(e['energy'] - e0) < abs_tol)
            gs_n_degenerate.append(n_deg)
        ax.plot(jpm_vals, gs_n_degenerate, style, ms=ms, lw=lw,
                label=rf'$\delta E/N = {tol_ps:.0e}$')

    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel('Count')
    ax.set_title('(c) GS degeneracy (per-site tolerance)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Sector gap: gap within each sector (E1_sector - E0_sector)
    ax = axes[1, 1]
    for sec in all_sectors:
        sector_gaps = []
        for ji, j in enumerate(jpm_sorted):
            by_sec = defaultdict(list)
            for e in data[j]['entries']:
                by_sec[e['sector']].append(e['energy'])
            if sec in by_sec and len(by_sec[sec]) >= 2:
                sec_evals = sorted(by_sec[sec])
                sector_gaps.append(sec_evals[1] - sec_evals[0])
            else:
                sector_gaps.append(np.nan)
        valid = ~np.isnan(np.array(sector_gaps))
        if np.any(valid):
            label = _sec_label(sec)
            ax.plot(jpm_vals[valid], np.array(sector_gaps)[valid],
                    marker=sec_markers[sec], ls='-',
                    color=colors[sec], ms=3, lw=1.0,
                    label=label)
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$\Delta^{\rm sector}$')
    ax.set_title('(d) Intra-sector gap')
    ax.legend(fontsize=8, ncol=1, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'sector_gs_energies.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved sector_gs_energies.png/pdf")


# ============================================================
# Plot 4: Detailed per-sector panels (one panel per sector)
# ============================================================
def plot_individual_sector_panels(data, jpm_sorted, sector_meta, output_dir,
                                  n_show_per_sector=10, momentum_map=None):
    """One subplot per sector showing eigenvalues."""
    jpm_vals = np.array([float(j) for j in jpm_sorted])

    all_sectors = set()
    for j in jpm_sorted:
        for e in data[j]['entries']:
            all_sectors.add(e['sector'])
    all_sectors = sorted(all_sectors)
    n_sec = len(all_sectors)

    # Build qn_map for orbit coloring
    qn_map = {}
    if sector_meta:
        for s in sector_meta.get('sectors', []):
            qn_map[s['sector_id']] = tuple(s['quantum_numbers'])

    # Grid layout
    ncols = 3
    nrows = (n_sec + ncols - 1) // ncols
    fig, all_axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                                 sharex=True)
    if nrows == 1:
        all_axes = all_axes[np.newaxis, :]
    fig.suptitle(r'BFG Kagome $3\times3$: Per-Sector Spectrum '
                 rf'($n_{{\uparrow}}={N_UP}$, $S^z={SZ_VAL:.1f}$)', fontsize=15)

    e0_global = {float(j): data[j]['entries'][0]['energy'] for j in jpm_sorted}

    for si, sec in enumerate(all_sectors):
        row, col = divmod(si, ncols)
        ax = all_axes[row, col]

        # Orbit color for this sector
        orbit_color, orbit_marker, orbit_grp = _sector_orbit_color(qn_map, sec)

        # Collect sector eigenvalues at each jpm
        for j in jpm_sorted:
            jv = float(j)
            sec_evals = sorted([e['energy'] for e in data[j]['entries']
                                if e['sector'] == sec])
            for k, ev in enumerate(sec_evals[:n_show_per_sector]):
                c = orbit_color if k == 0 else ('steelblue' if k < 5 else 'gray')
                alpha = 1.0 if k == 0 else (0.6 if k < 5 else 0.3)
                ms = 4 if k == 0 else 2
                ax.plot(jv, (ev - e0_global[jv]), 'o', color=c,
                        ms=ms, alpha=alpha)

        qn_label = get_sector_qn_label(sector_meta, sec)
        mom_label = get_sector_momentum_label(sector_meta, sec, momentum_map)
        ax.set_title(f'Sector {sec}: {mom_label} ($q$=({qn_label}))',
                     fontsize=11, color=orbit_color, fontweight='bold')
        # Colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(orbit_color)
            spine.set_linewidth(2.5)
        ax.set_ylabel(r'$E_n - E_0^{\rm global}$')
        ax.set_ylim(bottom=-0.01)
        ax.grid(True, alpha=0.3)

    # Orbit legend in first panel
    orbit_handles = _orbit_legend_handles()
    all_axes[0, 0].legend(handles=orbit_handles, fontsize=7, loc='upper right',
                          title='$C_3$ orbit', title_fontsize=7)

    # Turn off unused subplots
    for si in range(n_sec, nrows * ncols):
        row, col = divmod(si, ncols)
        all_axes[row, col].set_visible(False)

    # Set x-label on bottom row
    for col in range(ncols):
        if nrows - 1 < all_axes.shape[0]:
            all_axes[nrows - 1, col].set_xlabel(r'$J_{\pm}$')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'per_sector_panels.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved per_sector_panels.png/pdf")


# ============================================================
# Plot 5: Tower of States
# ============================================================
def plot_tower_of_states(data, jpm_sorted, sector_meta, output_dir,
                         n_tower=20, momentum_map=None):
    """Tower-of-states plot: low-energy levels vs Jpm, raw and normalized.

    Panels:
      (a) Raw excitation spectrum  (E_n - E_0) for lowest n_tower levels,
          color-coded by momentum sector (Γ=red, K/K'=green, generic=yellow).
      (b) Normalized tower: (E_n - E_0) / Δ  where Δ is the gap above
          the GS manifold.  First non-GS level maps to 1.
    """
    jpm_vals = np.array([float(j) for j in jpm_sorted])
    n_jpm = len(jpm_vals)
    deg_tol_per_site = 2e-5
    abs_tol = deg_tol_per_site * NUM_SITES

    # Identify all sectors and build color map
    all_sectors = set()
    for j in jpm_sorted:
        for e in data[j]['entries']:
            all_sectors.add(e['sector'])
    all_sectors = sorted(all_sectors)

    qn_map = {}
    if sector_meta:
        for s in sector_meta.get('sectors', []):
            qn_map[s['sector_id']] = tuple(s['quantum_numbers'])

    # Use shared orbit color scheme
    palette, marker_map, _, _, _group_fn = _get_orbit_config()

    def _sector_group(sec):
        q1, q2 = qn_map.get(sec, (sec, 0))
        return _group_fn(q1, q2)

    colors = {s: palette[_sector_group(s)] for s in all_sectors}
    markers = {s: marker_map[_sector_group(s)] for s in all_sectors}

    # Build tower arrays: for each Jpm, get lowest n_tower levels with sectors
    # tower_E[ji, k] = E_k - E_0,  tower_sec[ji, k] = sector index
    tower_E = np.full((n_jpm, n_tower), np.nan)
    tower_sec = np.full((n_jpm, n_tower), -1, dtype=int)
    tower_Enorm = np.full((n_jpm, n_tower), np.nan)
    n_gs_arr = np.zeros(n_jpm, dtype=int)
    gap_arr = np.full(n_jpm, np.nan)

    for ji, j in enumerate(jpm_sorted):
        entries = data[j]['entries']
        e0 = entries[0]['energy'] if entries else 0.0

        # GS degeneracy
        n_gs = 1
        for k in range(1, len(entries)):
            if abs(entries[k]['energy'] - e0) <= abs_tol:
                n_gs += 1
            else:
                break
        n_gs_arr[ji] = n_gs

        # Gap: first level above GS manifold
        delta = np.nan
        for k in range(n_gs, len(entries)):
            if abs(entries[k]['energy'] - e0) > abs_tol:
                delta = entries[k]['energy'] - e0
                break
        gap_arr[ji] = delta

        # Fill tower arrays
        for k in range(min(n_tower, len(entries))):
            ek = entries[k]['energy'] - e0
            tower_E[ji, k] = ek
            tower_sec[ji, k] = entries[k]['sector']
            if not np.isnan(delta) and delta > 1e-14:
                tower_Enorm[ji, k] = ek / delta

    # ---- Figure: 2 panels only ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(r'BFG Kagome $3\times3$ PBC: Tower of States '
                 rf'($N={NUM_SITES}$, $S^z={SZ_VAL:.1f}$)', fontsize=14)

    legend_handles = _orbit_legend_handles()

    # --- (a) Raw excitation energies ---
    ax = axes[0]
    for k in range(n_tower):
        # Group by sector for this level index to avoid plotting 40 series
        # Instead, connect points with the same sector across Jpm
        for sec in all_sectors:
            mask = tower_sec[:, k] == sec
            if not np.any(mask):
                continue
            xs = jpm_vals[mask]
            ys = tower_E[mask, k]
            g = _sector_group(sec)
            ax.scatter(xs, ys, c=palette[g], marker=marker_map[g],
                       s=12, alpha=0.6, edgecolors='none', zorder=3)

    # Thin line connecting each level across Jpm
    for k in range(n_tower):
        valid = ~np.isnan(tower_E[:, k])
        lw = 1.2 if k < 6 else 0.4
        alpha = 0.5 if k < 6 else 0.15
        ax.plot(jpm_vals[valid], tower_E[valid, k], '-', color='gray',
                lw=lw, alpha=alpha, zorder=1)

    ax.legend(handles=legend_handles, fontsize=8, loc='upper left')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_n - E_0$')
    ax.set_title(f'(a) Excitation spectrum (lowest {n_tower})')
    ax.grid(True, alpha=0.3)

    # --- (b) Normalized tower: (E_n - E_0) / Δ ---
    ax = axes[1]
    for k in range(n_tower):
        for sec in all_sectors:
            mask = (tower_sec[:, k] == sec) & ~np.isnan(tower_Enorm[:, k])
            if not np.any(mask):
                continue
            g = _sector_group(sec)
            ax.scatter(jpm_vals[mask], tower_Enorm[mask, k],
                       c=palette[g], marker=marker_map[g],
                       s=12, alpha=0.6, edgecolors='none', zorder=3)

    for k in range(n_tower):
        valid = ~np.isnan(tower_Enorm[:, k])
        lw = 1.2 if k < 6 else 0.4
        alpha = 0.5 if k < 6 else 0.15
        ax.plot(jpm_vals[valid], tower_Enorm[valid, k], '-', color='gray',
                lw=lw, alpha=alpha, zorder=1)

    # Reference line at 1 (= gap)
    ax.axhline(1.0, color='black', ls='--', lw=0.8, alpha=0.5)
    ax.legend(handles=legend_handles, fontsize=8, loc='upper left')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$(E_n - E_0)\, /\, \Delta$')
    ax.set_title(r'(b) Normalized tower ($\Delta$ = gap above GS manifold)')
    ax.set_ylim(-0.05, 5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'tower_of_states.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved tower_of_states.png/pdf")


# ============================================================
# Plot 6: Brillouin zone with accessible momenta (translation-only)
# ============================================================
def plot_bz_momenta(sector_meta, output_dir):
    """Plot hexagonal BZ with the 9 accessible k-points labeled.

    Only meaningful for translation-only mode where sectors are true momenta.
    """
    if not TRANSLATION_ONLY:
        return
    if sector_meta is None:
        return

    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2])
    A = np.array([a1, a2]).T
    B = 2 * np.pi * np.linalg.inv(A)   # rows = reciprocal lattice vectors
    b1, b2 = B[0], B[1]

    # Hexagonal BZ boundary vertices
    # K = (2b1+b2)/3 sits at angle 0°; corners alternate K/K' every 60°
    K_pt = (2 * b1 + b2) / 3
    norm_K = np.linalg.norm(K_pt)
    bz_corners = []
    for n in range(6):
        theta = n * np.pi / 3   # first corner at 0° (= K)
        bz_corners.append(norm_K * np.array([np.cos(theta), np.sin(theta)]))
    bz_corners.append(bz_corners[0])
    bz_x = [c[0] for c in bz_corners]
    bz_y = [c[1] for c in bz_corners]

    # Compute all 9 momenta and fold into first BZ
    sectors = sector_meta.get('sectors', [])
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Draw BZ boundary
    ax.plot(bz_x, bz_y, 'k-', lw=1.5, zorder=1)
    ax.fill(bz_x, bz_y, color='#f0f0f0', alpha=0.3, zorder=0)

    # Use shared orbit palette
    group_colors = _ORBIT_PALETTE_TO
    group_markers = _ORBIT_MARKER_TO

    for s in sectors:
        q1, q2 = s['quantum_numbers']
        k = (-q2 / 3) * b1 + (q1 / 3) * b2

        # Fold into first BZ
        best_k = k.copy()
        best_norm = np.linalg.norm(k)
        for n1 in range(-2, 3):
            for n2 in range(-2, 3):
                kp = k - n1 * b1 - n2 * b2
                if np.linalg.norm(kp) < best_norm - 1e-10:
                    best_k = kp
                    best_norm = np.linalg.norm(kp)

        grp = _sector_group_to(q1, q2)
        color = group_colors[grp]
        marker = group_markers[grp]
        label = _sector_momentum_label_to(q1, q2)
        # Strip $ for annotation text
        ann_text = label.replace('$', '').replace(r'\Gamma', 'Γ')
        ann_text = ann_text.replace(r"\mathbf{k}=", "")

        ax.scatter(best_k[0], best_k[1], c=color, s=150, marker=marker,
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.annotate(ann_text, best_k,
                    fontsize=9, ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points',
                    fontweight='bold', color=color)

    # Mark high-symmetry points that are NOT accessible
    M_pt = (b1 + b2) / 2
    # Fold M
    best_M = M_pt.copy()
    best_norm = np.linalg.norm(M_pt)
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            mp = M_pt - n1 * b1 - n2 * b2
            if np.linalg.norm(mp) < best_norm - 1e-10:
                best_M = mp
                best_norm = np.linalg.norm(mp)
    ax.scatter(best_M[0], best_M[1], c='gray', s=80,
               marker='x', zorder=4)
    ax.annotate(r'$M$ (N/A)', best_M, fontsize=8, ha='center', va='top',
                xytext=(0, -8), textcoords='offset points', color='gray')

    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_title(r'Accessible momenta on $3\times3$ kagome (translation-only)'
                 '\n'
                 r'$\mathbf{k} = (-q_2/3)\,\mathbf{b}_1 + (q_1/3)\,\mathbf{b}_2$, '
                 r'$q_1,q_2 \in \{0,1,2\}$',
                 fontsize=12)
    ax.grid(True, alpha=0.3)

    # Legend with distinct markers — use shared orbit dicts
    handles = _orbit_legend_handles()
    ax.legend(handles=handles, fontsize=9, loc='lower right')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'bz_momenta.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved bz_momenta.png/pdf")


# ============================================================
# Save data to .dat text files
# ============================================================
def save_spectrum_data(data, jpm_sorted, sector_meta, output_dir):
    """Save all spectrum data to .dat text files."""
    os.makedirs(output_dir, exist_ok=True)

    jpm_vals = np.array([float(j) for j in jpm_sorted])
    np.savetxt(os.path.join(output_dir, 'spectrum_jpm_values.dat'), jpm_vals)

    # Metadata
    with open(os.path.join(output_dir, 'spectrum_metadata.dat'), 'w') as f:
        f.write(f"num_sites = {NUM_SITES}\n")
        f.write(f"n_up = {N_UP}\n")
        f.write(f"sz = {SZ_VAL}\n")

    all_evals = []
    all_sectors_arr = []
    for j in jpm_sorted:
        entries = data[j]['entries']
        evals = np.array([e['energy'] for e in entries])
        secs = np.array([e['sector'] for e in entries], dtype=np.int32)
        all_evals.append(evals)
        all_sectors_arr.append(secs)

    max_len = max(len(a) for a in all_evals) if all_evals else 0
    evals_arr = np.full((len(jpm_sorted), max_len), np.nan)
    secs_arr = np.full((len(jpm_sorted), max_len), -1, dtype=np.int32)
    for i, (ev, sc) in enumerate(zip(all_evals, all_sectors_arr)):
        evals_arr[i, :len(ev)] = ev
        secs_arr[i, :len(sc)] = sc

    np.savetxt(os.path.join(output_dir, 'spectrum_eigenvalues.dat'), evals_arr)
    np.savetxt(os.path.join(output_dir, 'spectrum_sector_indices.dat'),
               secs_arr, fmt='%d')

    if sector_meta:
        gen_orders = sector_meta.get('generator_orders', [])
        if gen_orders:
            np.savetxt(os.path.join(output_dir, 'sector_generator_orders.dat'),
                       np.array(gen_orders, dtype=np.int32).reshape(1, -1),
                       fmt='%d')
        sectors = sector_meta.get('sectors', [])
        if sectors:
            rows = []
            for s in sectors:
                rows.append([s['sector_id']] + list(s['quantum_numbers']))
            np.savetxt(os.path.join(output_dir, 'sector_quantum_numbers.dat'),
                       np.array(rows, dtype=np.int32), fmt='%d',
                       header='sector_id quantum_numbers...')

    print(f"  Saved spectrum .dat files")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Plot low-energy spectrum for BFG 3×3 kagome cluster')
    parser.add_argument('--output-dir', type=str,
                        default='/scratch/zhouzb79/analysis_BFG_3x3/spectrum',
                        help='Output directory')
    parser.add_argument('--n-show', type=int, default=40,
                        help='Number of levels to show in global plot')
    parser.add_argument('--n-show-per-sector', type=int, default=5,
                        help='Number of levels per sector in sector plots')
    parser.add_argument('--n-tower', type=int, default=20,
                        help='Number of levels in tower-of-states plot')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Base data: {BASE_DIR}")

    print("\nDiscovering Jpm values...")
    jpm_list = discover_jpm_values()
    print(f"  Found {len(jpm_list)} Jpm values: {jpm_list[0]} to {jpm_list[-1]}")

    print("\nCollecting spectra...")
    data, sector_meta = collect_all_spectra(jpm_list)

    n_total = sum(len(data[j]['entries']) for j in jpm_list)
    print(f"  Collected {n_total} eigenvalues across {len(jpm_list)} Jpm values")

    if sector_meta:
        gen_orders = sector_meta.get('generator_orders', [])
        n_sec = len(sector_meta.get('sectors', []))
        print(f"  Symmetry: {len(gen_orders)} generators, orders {gen_orders}, "
              f"{n_sec} sectors")

    # Build momentum map from sector metadata
    momentum_map = build_sector_momentum_map(sector_meta)
    if momentum_map:
        print("  Sector → momentum mapping:")
        for sid in sorted(momentum_map.keys()):
            qn_label = get_sector_qn_label(sector_meta, sid)
            print(f"    sec {sid} (q={qn_label}) → {momentum_map[sid]}")
        if TRANSLATION_ONLY:
            print("  MODE: Translation-only — (q1,q2) are true lattice momenta.")
            print("        k = (q1/3) b1 + (q2/3) b2")
        else:
            print("  NOTE: G0 = T_{a1+a2} (translation), G1 = C3^{-1}∘T (rotation+translation)")
            print("        Sectors q1≠0 are C3 linear combinations, not pure momenta.")

    print("\nPlotting...")
    plot_global_spectrum(data, jpm_list, args.output_dir, n_show=args.n_show)
    plot_symmetry_sector_spectrum(data, jpm_list, sector_meta, args.output_dir,
                                 n_show_per_sector=args.n_show_per_sector,
                                 momentum_map=momentum_map)
    plot_sector_gs_energies(data, jpm_list, sector_meta, args.output_dir,
                            momentum_map=momentum_map)
    plot_individual_sector_panels(data, jpm_list, sector_meta, args.output_dir,
                                  n_show_per_sector=args.n_show_per_sector,
                                  momentum_map=momentum_map)
    plot_tower_of_states(data, jpm_list, sector_meta, args.output_dir,
                         n_tower=args.n_tower, momentum_map=momentum_map)
    plot_bz_momenta(sector_meta, args.output_dir)

    print("\nSaving data...")
    save_spectrum_data(data, jpm_list, sector_meta, args.output_dir)

    print("\nDone! All plots saved to:", args.output_dir)


if __name__ == '__main__':
    main()
