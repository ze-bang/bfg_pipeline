#!/usr/bin/env python3
"""
Plot low-energy spectrum for BFG kagome 3×3 PBC cluster — multi-Sz mode.

Extends the single-Sz 3×3 plotter to support multiple n_up (Sz) sectors.
Ground state properties use the overall GS; spectrum analysis combines
eigenvalues from all Sz sectors with labels.

Generates:
  1) Per-n_up spectrum: same plots as single-Sz for each n_up
  2) Combined global spectrum: all Sz sectors merged
  3) Per-Sz sector panels: separate panels per Sz value
  4) Combined tower of states: levels colored by Sz and momentum sector
  5) Combined symmetry-sector spectrum

Output directory: analysis_BFG_3x3_fixed_Sz/spectrum/
"""

import os
import sys
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

# ============================================================
# Constants (overridden by bfg_analyze.py at import time)
# ============================================================
BASE_DIR = '/scratch/zhouzb79/BFG_scan_symmetrized_pbc_3x3_fixed_Sz_translation_only'
N_UP_LIST = [14, 15]
NUM_SITES = 27
TRANSLATION_ONLY = True

# Sz = n_up - N/2
def _sz_val(n_up):
    return n_up - NUM_SITES / 2

def _sz_label(n_up):
    sz = _sz_val(n_up)
    if sz == int(sz):
        return rf'$S^z={int(sz)}$'
    # half-integer
    num = int(2 * sz)
    return rf'$S^z={num}/2$'

# Colors for Sz sectors
_SZ_COLORS = {
    14: '#2980b9',   # blue  — Sz=1/2
    15: '#e74c3c',   # red   — Sz=3/2
    13: '#27ae60',   # green — Sz=-1/2 (if ever added)
    12: '#f39c12',   # orange
}

_SZ_MARKERS = {
    14: 'o',
    15: 's',
    13: '^',
    12: 'D',
}


# ============================================================
# Data loading
# ============================================================
def discover_jpm_values():
    """Find all Jpm directories with valid data across all n_up sectors."""
    jpms = set()
    for n_up in N_UP_LIST:
        base = os.path.join(BASE_DIR, f'n_up={n_up}')
        if not os.path.isdir(base):
            continue
        for d in os.listdir(base):
            if not d.startswith('Jpm='):
                continue
            jpm_str = d.split('=', 1)[1]
            if float(jpm_str) == 0.0:
                print(f"  Skipping Jpm={jpm_str} (Heisenberg point)")
                continue
            mapping = os.path.join(base, f'Jpm={jpm_str}', 'results',
                                   'eigenvalue_mapping.txt')
            if os.path.exists(mapping):
                jpms.add(jpm_str)
    return sorted(jpms, key=float)


def read_eigenvalue_mapping(n_up, jpm_str):
    """Read eigenvalue_mapping.txt for one (n_up, Jpm)."""
    mapping_file = os.path.join(BASE_DIR, f'n_up={n_up}', f'Jpm={jpm_str}',
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
                    'n_up':       n_up,
                })
    entries.sort(key=lambda x: x['energy'])
    return entries


def load_sector_metadata(n_up, jpm_str):
    """Load sector_metadata.json to get quantum number labels."""
    meta_path = os.path.join(BASE_DIR, f'n_up={n_up}', f'Jpm={jpm_str}',
                             'automorphism_results', 'sector_metadata.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def collect_all_spectra(jpm_list):
    """Collect eigenvalues for all Jpm, all Sz sectors.

    Returns
    -------
    data : dict
        data[jpm_str] = {
            'jpm': float,
            'all_entries': [...],           # merged, sorted by energy
            'by_nup': {n_up: [...], ...},   # per Sz sector
        }
    sector_meta : dict or None
    """
    data = {}
    sector_meta = None

    for jpm_str in jpm_list:
        all_entries = []
        by_nup = {}
        for n_up in N_UP_LIST:
            entries = read_eigenvalue_mapping(n_up, jpm_str)
            by_nup[n_up] = entries
            all_entries.extend(entries)
            if sector_meta is None and entries:
                sector_meta = load_sector_metadata(n_up, jpm_str)

        all_entries.sort(key=lambda x: x['energy'])
        data[jpm_str] = {
            'jpm': float(jpm_str),
            'all_entries': all_entries,
            'by_nup': by_nup,
        }

    return data, sector_meta


# ============================================================
# Sector momentum labeling (translation-only)
# ============================================================
def _sector_momentum_label_to(q1, q2):
    if (q1, q2) == (0, 0):
        return r'$\Gamma$'
    elif (q1, q2) == (1, 1):
        return r'$K$'
    elif (q1, q2) == (2, 2):
        return r"$K'$"
    return rf'$({q1},{q2})$'


def _sector_group_to(q1, q2):
    if (q1, q2) == (0, 0):
        return 'Gamma'
    if (q1, q2) == (1, 1):
        return 'K'
    if (q1, q2) == (2, 2):
        return 'Kp'
    if (q1, q2) in ((0, 1), (2, 0), (1, 2)):
        return 'orbA'
    return 'orbB'


def build_sector_momentum_map(sector_meta):
    if sector_meta is None:
        return {}
    mapping = {}
    for s in sector_meta.get('sectors', []):
        sid = s['sector_id']
        q1, q2 = s['quantum_numbers']
        mapping[sid] = _sector_momentum_label_to(q1, q2)
    return mapping


def get_sector_qn_label(sector_meta, sector_idx):
    if sector_meta is None:
        return str(sector_idx)
    for s in sector_meta.get('sectors', []):
        if s['sector_id'] == sector_idx:
            qn = s['quantum_numbers']
            return ','.join(str(q) for q in qn)
    return str(sector_idx)


def get_sector_momentum_label(sector_meta, sector_idx, momentum_map=None):
    if momentum_map and sector_idx in momentum_map:
        return momentum_map[sector_idx]
    return get_sector_qn_label(sector_meta, sector_idx)


# ============================================================
# Orbit coloring
# ============================================================
_ORBIT_PALETTE = {
    'Gamma': '#e74c3c',
    'K':     '#27ae60',
    'Kp':    '#2980b9',
    'orbA':  '#f39c12',
    'orbB':  '#9b59b6',
}
_ORBIT_MARKER = {
    'Gamma': 'o', 'K': 'h', 'Kp': 'H', 'orbA': 's', 'orbB': 'D',
}
_ORBIT_LABEL = {
    'Gamma': r'$\Gamma$: $(0,0)$',
    'K':     r'$K$: $(1,1)$',
    'Kp':    r"$K'$: $(2,2)$",
    'orbA':  r'$C_3$ orbit: $\{(0,1),(2,0),(1,2)\}$',
    'orbB':  r'$C_3$ orbit: $\{(0,2),(1,0),(2,1)\}$',
}
_ORBIT_ORDER = ['Gamma', 'K', 'Kp', 'orbA', 'orbB']


def _sector_orbit_color(qn_map, sec):
    q1, q2 = qn_map.get(sec, (sec, 0))
    grp = _sector_group_to(q1, q2)
    return _ORBIT_PALETTE[grp], _ORBIT_MARKER[grp], grp


def _orbit_legend_handles():
    handles = []
    for g in _ORBIT_ORDER:
        handles.append(Line2D([], [], color=_ORBIT_PALETTE[g],
                              marker=_ORBIT_MARKER[g],
                              ms=7, lw=0, markeredgecolor='k',
                              markeredgewidth=0.5, label=_ORBIT_LABEL[g]))
    return handles


def _sz_legend_handles():
    handles = []
    for n_up in N_UP_LIST:
        c = _SZ_COLORS.get(n_up, 'gray')
        m = _SZ_MARKERS.get(n_up, 'o')
        handles.append(Line2D([], [], color=c, marker=m, ms=7, lw=0,
                              markeredgecolor='k', markeredgewidth=0.5,
                              label=_sz_label(n_up)))
    return handles


# ============================================================
# Plot 1: Global spectrum (all Sz merged)
# ============================================================
def plot_global_spectrum(data, jpm_sorted, output_dir, n_show=40):
    """All eigenvalues vs Jpm, merged from all Sz sectors."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sz_str = ', '.join(_sz_label(n).strip('$') for n in N_UP_LIST)
    fig.suptitle(r'BFG Kagome $3\times3$ PBC: Low-Energy Spectrum '
                 rf'({sz_str} merged)', fontsize=14)

    jpm_vals = np.array([float(j) for j in jpm_sorted])

    # (a) E/N for lowest n_show levels
    ax = axes[0]
    for k in range(n_show):
        evals_k = []
        for j in jpm_sorted:
            entries = data[j]['all_entries']
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

    # (b) Excitation energies
    ax = axes[1]
    for k in range(1, n_show):
        gaps_k = []
        for j in jpm_sorted:
            entries = data[j]['all_entries']
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

    # (c) GS energy per site + gap
    ax = axes[2]
    e0_arr = [data[j]['all_entries'][0]['energy'] / NUM_SITES
              for j in jpm_sorted]
    ax.plot(jpm_vals, e0_arr, 'ko-', ms=4, lw=1.5, label=r'$E_0/N$')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_0/N$')
    ax.set_title('(c) GS energy per site & gap')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    ax2 = ax.twinx()
    deg_tol = 2e-5 * NUM_SITES
    gaps = []
    for j in jpm_sorted:
        entries = data[j]['all_entries']
        e0 = entries[0]['energy']
        e1 = None
        for e in entries[1:]:
            if abs(e['energy'] - e0) > deg_tol:
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
# Plot 2: Per-Sz sector spectrum
# ============================================================
def plot_sz_sector_spectrum(data, jpm_sorted, output_dir, n_show=30):
    """Separate panels for each Sz sector."""
    jpm_vals = np.array([float(j) for j in jpm_sorted])
    n_nup = len(N_UP_LIST)

    fig, axes = plt.subplots(n_nup, 3, figsize=(20, 6 * n_nup))
    if n_nup == 1:
        axes = axes[np.newaxis, :]
    sz_str = ', '.join(_sz_label(n).strip('$') for n in N_UP_LIST)
    fig.suptitle(r'BFG Kagome $3\times3$ PBC: Per-$S^z$ Sector Spectrum',
                 fontsize=14)

    for ri, n_up in enumerate(N_UP_LIST):
        sz = _sz_val(n_up)
        color_main = _SZ_COLORS.get(n_up, 'blue')

        # (a) E/N
        ax = axes[ri, 0]
        for k in range(n_show):
            evals_k = []
            for j in jpm_sorted:
                entries = data[j]['by_nup'].get(n_up, [])
                if k < len(entries):
                    evals_k.append(entries[k]['energy'] / NUM_SITES)
                else:
                    evals_k.append(np.nan)
            c = color_main if k == 0 else ('steelblue' if k < 10 else 'gray')
            alpha = 1.0 if k == 0 else (0.7 if k < 10 else 0.3)
            lw = 1.5 if k == 0 else (0.8 if k < 10 else 0.5)
            ax.plot(jpm_vals, evals_k, '-', lw=lw, alpha=alpha, color=c)
        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel(r'$E_n / N$')
        ax.set_title(f'({chr(ord("a")+ri*3)}) {_sz_label(n_up)} '
                     f'($n_{{\\uparrow}}={n_up}$): E/N')
        ax.grid(True, alpha=0.3)

        # (b) E_n - E_0 within this Sz sector
        ax = axes[ri, 1]
        for k in range(1, n_show):
            gaps_k = []
            for j in jpm_sorted:
                entries = data[j]['by_nup'].get(n_up, [])
                if not entries:
                    gaps_k.append(np.nan)
                    continue
                e0 = entries[0]['energy']
                if k < len(entries):
                    gaps_k.append(entries[k]['energy'] - e0)
                else:
                    gaps_k.append(np.nan)
            c = color_main if k < 5 else 'gray'
            alpha = 0.8 if k < 10 else 0.3
            lw = 1.0 if k < 10 else 0.5
            ax.plot(jpm_vals, gaps_k, '-', lw=lw, alpha=alpha, color=c)
        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel(r'$E_n - E_0^{S^z}$')
        ax.set_title(f'({chr(ord("a")+ri*3+1)}) {_sz_label(n_up)}: '
                     f'Excitations')
        ax.set_ylim(bottom=-0.01)
        ax.grid(True, alpha=0.3)

        # (c) Sector GS energy
        ax = axes[ri, 2]
        e0_sec = []
        for j in jpm_sorted:
            entries = data[j]['by_nup'].get(n_up, [])
            if entries:
                e0_sec.append(entries[0]['energy'] / NUM_SITES)
            else:
                e0_sec.append(np.nan)
        ax.plot(jpm_vals, e0_sec, 'o-', color=color_main, ms=4, lw=1.5,
                label=f'{_sz_label(n_up)}')
        # Also show global GS for comparison
        e0_global = [data[j]['all_entries'][0]['energy'] / NUM_SITES
                     for j in jpm_sorted]
        ax.plot(jpm_vals, e0_global, 'k--', lw=1, alpha=0.5,
                label='Global GS')
        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel(r'$E_0/N$')
        ax.set_title(f'({chr(ord("a")+ri*3+2)}) {_sz_label(n_up)}: '
                     f'GS energy')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'sz_sector_spectrum.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved sz_sector_spectrum.png/pdf")


# ============================================================
# Plot 3: Combined Sz+symmetry sector spectrum
# ============================================================
def plot_combined_spectrum(data, jpm_sorted, sector_meta, output_dir,
                           n_show=30, momentum_map=None):
    """Combined spectrum: levels colored by Sz, markers by momentum sector."""
    jpm_vals = np.array([float(j) for j in jpm_sorted])
    deg_tol = 2e-5 * NUM_SITES

    qn_map = {}
    if sector_meta:
        for s in sector_meta.get('sectors', []):
            qn_map[s['sector_id']] = tuple(s['quantum_numbers'])

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    sz_str = ', '.join(_sz_label(n).strip('$') for n in N_UP_LIST)
    fig.suptitle(r'BFG Kagome $3\times3$ PBC: Combined Multi-$S^z$ Spectrum '
                 rf'({sz_str})', fontsize=14)

    # (a) All levels colored by Sz sector
    ax = axes[0]
    for n_up in N_UP_LIST:
        c = _SZ_COLORS.get(n_up, 'gray')
        m = _SZ_MARKERS.get(n_up, 'o')
        for j in jpm_sorted:
            jv = float(j)
            entries = data[j]['by_nup'].get(n_up, [])
            for k, e in enumerate(entries[:n_show]):
                alpha = 0.8 if k < 10 else 0.3
                ms = 4 if k == 0 else 2
                ax.plot(jv, e['energy'] / NUM_SITES, marker=m, color=c,
                        ms=ms, alpha=alpha, linestyle='none')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_n / N$')
    ax.set_title(f'(a) All levels by $S^z$')
    ax.legend(handles=_sz_legend_handles(), fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # (b) Excitation energy relative to global GS, colored by Sz
    ax = axes[1]
    for n_up in N_UP_LIST:
        c = _SZ_COLORS.get(n_up, 'gray')
        m = _SZ_MARKERS.get(n_up, 'o')
        for j in jpm_sorted:
            jv = float(j)
            e0_global = data[j]['all_entries'][0]['energy']
            entries = data[j]['by_nup'].get(n_up, [])
            for k, e in enumerate(entries[:n_show]):
                dE = e['energy'] - e0_global
                alpha = 0.8 if k < 10 else 0.3
                ms = 4 if k == 0 else 2
                ax.plot(jv, dE, marker=m, color=c, ms=ms, alpha=alpha,
                        linestyle='none')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_n - E_0^{\rm global}$')
    ax.set_title(r'(b) Excitations vs global GS')
    ax.legend(handles=_sz_legend_handles(), fontsize=9, loc='best')
    ax.set_ylim(bottom=-0.02)
    ax.grid(True, alpha=0.3)

    # (c) GS energy per sector (per n_up)
    ax = axes[2]
    for n_up in N_UP_LIST:
        c = _SZ_COLORS.get(n_up, 'gray')
        m = _SZ_MARKERS.get(n_up, 'o')
        e0_nup = []
        for j in jpm_sorted:
            entries = data[j]['by_nup'].get(n_up, [])
            if entries:
                e0_nup.append(entries[0]['energy'] / NUM_SITES)
            else:
                e0_nup.append(np.nan)
        ax.plot(jpm_vals, e0_nup, marker=m, color=c, ms=4, lw=1.5,
                label=_sz_label(n_up))
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_0^{S^z} / N$')
    ax.set_title('(c) GS energy per $S^z$ sector')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'combined_spectrum.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved combined_spectrum.png/pdf")


# ============================================================
# Plot 4: Per-symmetry-sector spectrum (per n_up)
# ============================================================
def plot_symmetry_sector_spectrum(data, jpm_sorted, sector_meta, output_dir,
                                 n_show_per_sector=5, momentum_map=None):
    """Per-symmetry-sector excitation gaps, one figure per n_up."""
    jpm_vals = np.array([float(j) for j in jpm_sorted])

    for n_up in N_UP_LIST:
        sz_lab = _sz_label(n_up)

        # Identify all sectors for this n_up
        all_sectors = set()
        for j in jpm_sorted:
            for e in data[j]['by_nup'].get(n_up, []):
                all_sectors.add(e['sector'])
        all_sectors = sorted(all_sectors)
        if not all_sectors:
            continue

        qn_map = {}
        if sector_meta:
            for s in sector_meta.get('sectors', []):
                qn_map[s['sector_id']] = tuple(s['quantum_numbers'])

        # Build per-sector curves
        sector_curves = defaultdict(lambda: defaultdict(list))
        for j in jpm_sorted:
            jv = float(j)
            by_sec = defaultdict(list)
            for e in data[j]['by_nup'].get(n_up, []):
                by_sec[e['sector']].append(e['energy'])
            for sec in all_sectors:
                sec_evals = sorted(by_sec.get(sec, []))
                for k, ev in enumerate(sec_evals[:n_show_per_sector]):
                    sector_curves[sec][k].append((jv, ev))

        n_sec = len(all_sectors)
        ncols = 3
        nrows = (n_sec + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
        if nrows == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(r'BFG Kagome $3\times3$: Sector Excitation Gap — '
                     rf'{sz_lab} ($n_{{\uparrow}}={n_up}$)',
                     fontsize=14, y=0.98)

        for idx, sec in enumerate(all_sectors[:n_sec]):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]
            orbit_color, orbit_marker, orbit_grp = _sector_orbit_color(qn_map, sec)

            qn = qn_map.get(sec, (sec,))
            if len(qn) == 2:
                mom_label = _sector_momentum_label_to(qn[0], qn[1])
                qn_str = f'$q_1={qn[0]},\\,q_2={qn[1]}$'
            else:
                mom_label = momentum_map.get(sec, f'sec {sec}') if momentum_map else f'sec {sec}'
                qn_str = ''

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
                if 1 in sector_curves[sec]:
                    pts = sector_curves[sec][1]
                    xs = [p[0] for p in pts if p[0] in e0_pts]
                    ys = [p[1] - e0_pts[p[0]] for p in pts if p[0] in e0_pts]
                    ax.plot(xs, ys, 'o-', ms=3, lw=1.5, color=orbit_color,
                            label=r'$\Delta_1$', zorder=5)

            title_str = f'{mom_label}'
            if qn_str:
                title_str += f'  ({qn_str})'
            ax.set_title(title_str, fontsize=11, color=orbit_color,
                         fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor(orbit_color)
                spine.set_linewidth(2.5)
            ax.set_xlabel(r'$J_{\pm}$')
            ax.set_ylabel(r'$E_n - E_0^{\rm sector}$')
            ax.set_ylim(bottom=-0.01)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')

        orbit_handles = _orbit_legend_handles()
        axes[0, ncols - 1].legend(handles=orbit_handles, fontsize=8,
                                  loc='upper right', title='$C_3$ orbit',
                                  title_fontsize=8)

        for si in range(n_sec, nrows * ncols):
            row, col = divmod(si, ncols)
            axes[row, col].set_visible(False)

        plt.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(os.path.join(output_dir,
                        f'symmetry_sector_spectrum_nup{n_up}.{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved symmetry_sector_spectrum_nup{n_up}.png/pdf")


# ============================================================
# Plot 5: Sector GS energy overview (per n_up + combined)
# ============================================================
def plot_sector_gs_energies(data, jpm_sorted, sector_meta, output_dir,
                            momentum_map=None):
    """Sector GS energies — one figure per n_up + combined plot."""
    jpm_vals = np.array([float(j) for j in jpm_sorted])

    qn_map = {}
    if sector_meta:
        for s in sector_meta.get('sectors', []):
            qn_map[s['sector_id']] = tuple(s['quantum_numbers'])

    def _sec_color_marker(sec):
        return _sector_orbit_color(qn_map, sec)

    def _sec_label(sec):
        mom = get_sector_momentum_label(sector_meta, sec, momentum_map)
        return f'{mom} (sec {sec})'

    # Per-n_up figures
    for n_up in N_UP_LIST:
        sz_lab = _sz_label(n_up)
        all_sectors = set()
        for j in jpm_sorted:
            for e in data[j]['by_nup'].get(n_up, []):
                all_sectors.add(e['sector'])
        all_sectors = sorted(all_sectors)
        if not all_sectors:
            continue

        sector_E0 = {s: np.full(len(jpm_sorted), np.nan) for s in all_sectors}
        for ji, j in enumerate(jpm_sorted):
            by_sec = defaultdict(list)
            for e in data[j]['by_nup'].get(n_up, []):
                by_sec[e['sector']].append(e['energy'])
            for sec in all_sectors:
                if sec in by_sec:
                    sector_E0[sec][ji] = min(by_sec[sec])

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(r'BFG Kagome $3\times3$: Sector GS Overview — '
                     rf'{sz_lab} ($n_{{\uparrow}}={n_up}$)', fontsize=13)

        colors = {s: _sec_color_marker(s)[0] for s in all_sectors}
        markers = {s: _sec_color_marker(s)[1] for s in all_sectors}

        # (a) E/N
        ax = axes[0, 0]
        for sec in all_sectors:
            valid = ~np.isnan(sector_E0[sec])
            if np.any(valid):
                ax.plot(jpm_vals[valid], sector_E0[sec][valid] / NUM_SITES,
                        marker=markers[sec], ls='-', color=colors[sec],
                        ms=3, lw=1.2, label=_sec_label(sec))
        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel(r'$E_0^{\rm sector}/N$')
        ax.set_title('(a) Sector GS energies')
        ax.legend(fontsize=8, ncol=1, loc='best')
        ax.grid(True, alpha=0.3)

        # (b) relative to global GS
        ax = axes[0, 1]
        e0_global = np.array([data[j]['all_entries'][0]['energy']
                              for j in jpm_sorted])
        for sec in all_sectors:
            valid = ~np.isnan(sector_E0[sec])
            if np.any(valid):
                delta = sector_E0[sec][valid] - e0_global[valid]
                ax.plot(jpm_vals[valid], delta, marker=markers[sec], ls='-',
                        color=colors[sec], ms=3, lw=1.2,
                        label=_sec_label(sec))
        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel(r'$E_0^{\rm sector} - E_0^{\rm global}$')
        ax.set_title('(b) Sector GS vs global GS')
        ax.legend(fontsize=8, loc='best')
        ax.set_ylim(bottom=-0.01)
        ax.grid(True, alpha=0.3)

        # (c) Degeneracy
        ax = axes[1, 0]
        deg_tols = [2e-5, 5e-5, 1e-4]
        styles = [('ko-', 4, 1.5), ('rs-', 3, 1.0), ('b^-', 3, 1.0)]
        for tol, (sty, ms, lw) in zip(deg_tols, styles):
            abs_tol = tol * NUM_SITES
            degs = []
            for j in jpm_sorted:
                entries = data[j]['by_nup'].get(n_up, [])
                if not entries:
                    degs.append(0)
                    continue
                e0 = entries[0]['energy']
                degs.append(sum(1 for e in entries
                                if abs(e['energy'] - e0) < abs_tol))
            ax.plot(jpm_vals, degs, sty, ms=ms, lw=lw,
                    label=rf'$\delta E/N = {tol:.0e}$')
        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel('Count')
        ax.set_title(f'(c) {sz_lab} GS degeneracy')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # (d) Intra-sector gap
        ax = axes[1, 1]
        for sec in all_sectors:
            s_gaps = []
            for j in jpm_sorted:
                by_sec = defaultdict(list)
                for e in data[j]['by_nup'].get(n_up, []):
                    by_sec[e['sector']].append(e['energy'])
                if sec in by_sec and len(by_sec[sec]) >= 2:
                    evals = sorted(by_sec[sec])
                    s_gaps.append(evals[1] - evals[0])
                else:
                    s_gaps.append(np.nan)
            valid = ~np.isnan(np.array(s_gaps))
            if np.any(valid):
                ax.plot(jpm_vals[valid], np.array(s_gaps)[valid],
                        marker=markers[sec], ls='-', color=colors[sec],
                        ms=3, lw=1.0, label=_sec_label(sec))
        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel(r'$\Delta^{\rm sector}$')
        ax.set_title('(d) Intra-sector gap')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(os.path.join(output_dir,
                        f'sector_gs_energies_nup{n_up}.{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved sector_gs_energies_nup{n_up}.png/pdf")


# ============================================================
# Plot 6: Per-sector panels (per n_up)
# ============================================================
def plot_individual_sector_panels(data, jpm_sorted, sector_meta, output_dir,
                                  n_show_per_sector=10, momentum_map=None):
    """One subplot per sector showing eigenvalues — per n_up."""
    jpm_vals = np.array([float(j) for j in jpm_sorted])

    qn_map = {}
    if sector_meta:
        for s in sector_meta.get('sectors', []):
            qn_map[s['sector_id']] = tuple(s['quantum_numbers'])

    for n_up in N_UP_LIST:
        sz_lab = _sz_label(n_up)
        all_sectors = set()
        for j in jpm_sorted:
            for e in data[j]['by_nup'].get(n_up, []):
                all_sectors.add(e['sector'])
        all_sectors = sorted(all_sectors)
        if not all_sectors:
            continue

        n_sec = len(all_sectors)
        ncols = 3
        nrows = (n_sec + ncols - 1) // ncols
        fig, all_axes = plt.subplots(nrows, ncols,
                                     figsize=(7 * ncols, 5 * nrows),
                                     sharex=True)
        if nrows == 1:
            all_axes = all_axes[np.newaxis, :]
        fig.suptitle(r'BFG Kagome $3\times3$: Per-Sector Spectrum — '
                     rf'{sz_lab} ($n_{{\uparrow}}={n_up}$)', fontsize=15)

        e0_global = {float(j): data[j]['all_entries'][0]['energy']
                     for j in jpm_sorted}

        for si, sec in enumerate(all_sectors):
            row, col = divmod(si, ncols)
            ax = all_axes[row, col]
            orbit_color, orbit_marker, _ = _sector_orbit_color(qn_map, sec)

            for j in jpm_sorted:
                jv = float(j)
                sec_evals = sorted([e['energy'] for e in
                                    data[j]['by_nup'].get(n_up, [])
                                    if e['sector'] == sec])
                for k, ev in enumerate(sec_evals[:n_show_per_sector]):
                    c = orbit_color if k == 0 else (
                        'steelblue' if k < 5 else 'gray')
                    alpha = 1.0 if k == 0 else (0.6 if k < 5 else 0.3)
                    ms = 4 if k == 0 else 2
                    ax.plot(jv, ev - e0_global[jv], 'o', color=c,
                            ms=ms, alpha=alpha)

            qn_label = get_sector_qn_label(sector_meta, sec)
            mom_label = get_sector_momentum_label(sector_meta, sec,
                                                  momentum_map)
            ax.set_title(f'Sector {sec}: {mom_label} ($q$=({qn_label}))',
                         fontsize=11, color=orbit_color, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor(orbit_color)
                spine.set_linewidth(2.5)
            ax.set_ylabel(r'$E_n - E_0^{\rm global}$')
            ax.set_ylim(bottom=-0.01)
            ax.grid(True, alpha=0.3)

        orbit_handles = _orbit_legend_handles()
        all_axes[0, 0].legend(handles=orbit_handles, fontsize=7,
                              loc='upper right', title='$C_3$ orbit',
                              title_fontsize=7)

        for si in range(n_sec, nrows * ncols):
            row, col = divmod(si, ncols)
            all_axes[row, col].set_visible(False)
        for col in range(ncols):
            if nrows - 1 < all_axes.shape[0]:
                all_axes[nrows - 1, col].set_xlabel(r'$J_{\pm}$')

        plt.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(os.path.join(output_dir,
                        f'per_sector_panels_nup{n_up}.{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved per_sector_panels_nup{n_up}.png/pdf")


# ============================================================
# Plot 7: Tower of States (combined multi-Sz)
# ============================================================
def plot_tower_of_states(data, jpm_sorted, sector_meta, output_dir,
                         n_tower=20, momentum_map=None):
    """Tower-of-states: low-energy levels vs Jpm, combined from all Sz sectors.

    Panels:
      (a) Raw excitation spectrum colored by Sz sector
      (b) Raw excitation spectrum colored by momentum sector
      (c) Normalized tower colored by Sz
      (d) Normalized tower colored by momentum
    """
    jpm_vals = np.array([float(j) for j in jpm_sorted])
    n_jpm = len(jpm_vals)
    deg_tol = 2e-5 * NUM_SITES

    qn_map = {}
    if sector_meta:
        for s in sector_meta.get('sectors', []):
            qn_map[s['sector_id']] = tuple(s['quantum_numbers'])

    # Build tower arrays from merged entries
    tower_E = np.full((n_jpm, n_tower), np.nan)
    tower_sec = np.full((n_jpm, n_tower), -1, dtype=int)
    tower_nup = np.full((n_jpm, n_tower), -1, dtype=int)
    tower_Enorm = np.full((n_jpm, n_tower), np.nan)
    gap_arr = np.full(n_jpm, np.nan)

    for ji, j in enumerate(jpm_sorted):
        entries = data[j]['all_entries']
        if not entries:
            continue
        e0 = entries[0]['energy']

        # Find gap above GS manifold
        delta = np.nan
        for k in range(1, len(entries)):
            if abs(entries[k]['energy'] - e0) > deg_tol:
                delta = entries[k]['energy'] - e0
                break
        gap_arr[ji] = delta

        for k in range(min(n_tower, len(entries))):
            ek = entries[k]['energy'] - e0
            tower_E[ji, k] = ek
            tower_sec[ji, k] = entries[k]['sector']
            tower_nup[ji, k] = entries[k]['n_up']
            if not np.isnan(delta) and delta > 1e-14:
                tower_Enorm[ji, k] = ek / delta

    # ---- Figure: 2×2 panels ----
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    sz_str = ', '.join(_sz_label(n).strip('$') for n in N_UP_LIST)
    fig.suptitle(r'BFG Kagome $3\times3$ PBC: Tower of States — '
                 rf'Multi-$S^z$ ({sz_str})', fontsize=14)

    sz_handles = _sz_legend_handles()
    orbit_handles = _orbit_legend_handles()

    # --- (a) Raw excitation energy, colored by Sz ---
    ax = axes[0, 0]
    for k in range(n_tower):
        for n_up in N_UP_LIST:
            mask = tower_nup[:, k] == n_up
            if not np.any(mask):
                continue
            c = _SZ_COLORS.get(n_up, 'gray')
            m = _SZ_MARKERS.get(n_up, 'o')
            ax.scatter(jpm_vals[mask], tower_E[mask, k], c=c, marker=m,
                       s=12, alpha=0.6, edgecolors='none', zorder=3)
        # Connecting lines
        valid = ~np.isnan(tower_E[:, k])
        lw = 1.2 if k < 6 else 0.4
        alpha = 0.5 if k < 6 else 0.15
        ax.plot(jpm_vals[valid], tower_E[valid, k], '-', color='gray',
                lw=lw, alpha=alpha, zorder=1)
    ax.legend(handles=sz_handles, fontsize=8, loc='upper left')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_n - E_0$')
    ax.set_title(f'(a) Excitations by $S^z$ (lowest {n_tower})')
    ax.grid(True, alpha=0.3)

    # --- (b) Raw, colored by momentum sector ---
    ax = axes[0, 1]
    all_sectors_set = set()
    for k in range(n_tower):
        for ji in range(n_jpm):
            if tower_sec[ji, k] >= 0:
                all_sectors_set.add(tower_sec[ji, k])
    all_sectors = sorted(all_sectors_set)
    for k in range(n_tower):
        for sec in all_sectors:
            mask = tower_sec[:, k] == sec
            if not np.any(mask):
                continue
            q1, q2 = qn_map.get(sec, (sec, 0))
            grp = _sector_group_to(q1, q2)
            ax.scatter(jpm_vals[mask], tower_E[mask, k],
                       c=_ORBIT_PALETTE[grp], marker=_ORBIT_MARKER[grp],
                       s=12, alpha=0.6, edgecolors='none', zorder=3)
        valid = ~np.isnan(tower_E[:, k])
        lw = 1.2 if k < 6 else 0.4
        alpha = 0.5 if k < 6 else 0.15
        ax.plot(jpm_vals[valid], tower_E[valid, k], '-', color='gray',
                lw=lw, alpha=alpha, zorder=1)
    ax.legend(handles=orbit_handles, fontsize=8, loc='upper left')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_n - E_0$')
    ax.set_title(f'(b) Excitations by momentum (lowest {n_tower})')
    ax.grid(True, alpha=0.3)

    # --- (c) Normalized tower, colored by Sz ---
    ax = axes[1, 0]
    for k in range(n_tower):
        for n_up in N_UP_LIST:
            mask = (tower_nup[:, k] == n_up) & ~np.isnan(tower_Enorm[:, k])
            if not np.any(mask):
                continue
            c = _SZ_COLORS.get(n_up, 'gray')
            m = _SZ_MARKERS.get(n_up, 'o')
            ax.scatter(jpm_vals[mask], tower_Enorm[mask, k], c=c, marker=m,
                       s=12, alpha=0.6, edgecolors='none', zorder=3)
        valid = ~np.isnan(tower_Enorm[:, k])
        lw = 1.2 if k < 6 else 0.4
        alpha = 0.5 if k < 6 else 0.15
        ax.plot(jpm_vals[valid], tower_Enorm[valid, k], '-', color='gray',
                lw=lw, alpha=alpha, zorder=1)
    ax.axhline(1.0, color='black', ls='--', lw=0.8, alpha=0.5)
    ax.legend(handles=sz_handles, fontsize=8, loc='upper left')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$(E_n - E_0)\, /\, \Delta$')
    ax.set_title(r'(c) Normalized tower by $S^z$')
    ax.set_ylim(-0.05, 5)
    ax.grid(True, alpha=0.3)

    # --- (d) Normalized tower, colored by momentum ---
    ax = axes[1, 1]
    for k in range(n_tower):
        for sec in all_sectors:
            mask = (tower_sec[:, k] == sec) & ~np.isnan(tower_Enorm[:, k])
            if not np.any(mask):
                continue
            q1, q2 = qn_map.get(sec, (sec, 0))
            grp = _sector_group_to(q1, q2)
            ax.scatter(jpm_vals[mask], tower_Enorm[mask, k],
                       c=_ORBIT_PALETTE[grp], marker=_ORBIT_MARKER[grp],
                       s=12, alpha=0.6, edgecolors='none', zorder=3)
        valid = ~np.isnan(tower_Enorm[:, k])
        lw = 1.2 if k < 6 else 0.4
        alpha = 0.5 if k < 6 else 0.15
        ax.plot(jpm_vals[valid], tower_Enorm[valid, k], '-', color='gray',
                lw=lw, alpha=alpha, zorder=1)
    ax.axhline(1.0, color='black', ls='--', lw=0.8, alpha=0.5)
    ax.legend(handles=orbit_handles, fontsize=8, loc='upper left')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$(E_n - E_0)\, /\, \Delta$')
    ax.set_title(r'(d) Normalized tower by momentum')
    ax.set_ylim(-0.05, 5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'tower_of_states.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved tower_of_states.png/pdf")


# ============================================================
# Plot 8: Tower of states — dual-marker mode (Sz + momentum)
# ============================================================
def plot_tower_of_states_dual(data, jpm_sorted, sector_meta, output_dir,
                               n_tower=30, momentum_map=None):
    """Tower of states with both Sz and momentum encoded.

    Each point has: fill color = Sz sector, edge marker shape = momentum orbit.
    """
    jpm_vals = np.array([float(j) for j in jpm_sorted])
    n_jpm = len(jpm_vals)
    deg_tol = 2e-5 * NUM_SITES

    qn_map = {}
    if sector_meta:
        for s in sector_meta.get('sectors', []):
            qn_map[s['sector_id']] = tuple(s['quantum_numbers'])

    tower_E = np.full((n_jpm, n_tower), np.nan)
    tower_sec = np.full((n_jpm, n_tower), -1, dtype=int)
    tower_nup = np.full((n_jpm, n_tower), -1, dtype=int)
    tower_Enorm = np.full((n_jpm, n_tower), np.nan)

    for ji, j in enumerate(jpm_sorted):
        entries = data[j]['all_entries']
        if not entries:
            continue
        e0 = entries[0]['energy']
        delta = np.nan
        for k in range(1, len(entries)):
            if abs(entries[k]['energy'] - e0) > deg_tol:
                delta = entries[k]['energy'] - e0
                break
        for k in range(min(n_tower, len(entries))):
            ek = entries[k]['energy'] - e0
            tower_E[ji, k] = ek
            tower_sec[ji, k] = entries[k]['sector']
            tower_nup[ji, k] = entries[k]['n_up']
            if not np.isnan(delta) and delta > 1e-14:
                tower_Enorm[ji, k] = ek / delta

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sz_str = ', '.join(_sz_label(n).strip('$') for n in N_UP_LIST)
    fig.suptitle(r'BFG Kagome $3\times3$: Tower of States — '
                 rf'{sz_str} (color=$S^z$, shape=momentum)', fontsize=14)

    for pi, (y_arr, ylabel, title_sfx, ylim) in enumerate([
        (tower_E, r'$E_n - E_0$', 'raw', None),
        (tower_Enorm, r'$(E_n - E_0)/\Delta$', 'normalized', (-0.05, 5)),
    ]):
        ax = axes[pi]
        for k in range(n_tower):
            valid = ~np.isnan(y_arr[:, k])
            lw = 1.2 if k < 6 else 0.4
            alpha_l = 0.5 if k < 6 else 0.15
            ax.plot(jpm_vals[valid], y_arr[valid, k], '-', color='gray',
                    lw=lw, alpha=alpha_l, zorder=1)
            for n_up in N_UP_LIST:
                for sec in sorted(qn_map.keys()):
                    mask = (tower_nup[:, k] == n_up) & \
                           (tower_sec[:, k] == sec) & valid
                    if not np.any(mask):
                        continue
                    q = qn_map.get(sec, (sec, 0))
                    grp = _sector_group_to(q[0], q[1])
                    ax.scatter(jpm_vals[mask], y_arr[mask, k],
                               c=_SZ_COLORS.get(n_up, 'gray'),
                               marker=_ORBIT_MARKER[grp],
                               s=18, alpha=0.7, edgecolors='k',
                               linewidths=0.3, zorder=3)

        if pi == 1:
            ax.axhline(1.0, color='black', ls='--', lw=0.8, alpha=0.5)

        # Combined legend
        handles = []
        handles.append(Line2D([], [], color='none', marker='none',
                              label=r'$\bf{S^z\ sector:}$'))
        handles.extend(_sz_legend_handles())
        handles.append(Line2D([], [], color='none', marker='none',
                              label=r'$\bf{Momentum:}$'))
        handles.extend(_orbit_legend_handles())
        ax.legend(handles=handles, fontsize=7, loc='upper left', ncol=1)

        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel(ylabel)
        ax.set_title(f'({chr(ord("a")+pi)}) {title_sfx.capitalize()}')
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir,
                    f'tower_of_states_dual.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved tower_of_states_dual.png/pdf")


# ============================================================
# Plot 9: BZ momenta
# ============================================================
def plot_bz_momenta(sector_meta, output_dir):
    """Plot hexagonal BZ with accessible k-points."""
    if sector_meta is None:
        return

    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2])
    A = np.array([a1, a2]).T
    B = 2 * np.pi * np.linalg.inv(A)
    b1, b2 = B[0], B[1]

    K_pt = (2 * b1 + b2) / 3
    norm_K = np.linalg.norm(K_pt)
    bz_corners = []
    for n in range(6):
        theta = n * np.pi / 3
        bz_corners.append(norm_K * np.array([np.cos(theta), np.sin(theta)]))
    bz_corners.append(bz_corners[0])
    bz_x = [c[0] for c in bz_corners]
    bz_y = [c[1] for c in bz_corners]

    sectors = sector_meta.get('sectors', [])
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(bz_x, bz_y, 'k-', lw=1.5, zorder=1)
    ax.fill(bz_x, bz_y, color='#f0f0f0', alpha=0.3, zorder=0)

    for s in sectors:
        q1, q2 = s['quantum_numbers']
        k = (-q2 / 3) * b1 + (q1 / 3) * b2
        best_k, best_norm = k.copy(), np.linalg.norm(k)
        for n1 in range(-2, 3):
            for n2 in range(-2, 3):
                kp = k - n1 * b1 - n2 * b2
                if np.linalg.norm(kp) < best_norm - 1e-10:
                    best_k, best_norm = kp, np.linalg.norm(kp)

        grp = _sector_group_to(q1, q2)
        color = _ORBIT_PALETTE[grp]
        marker = _ORBIT_MARKER[grp]
        label = _sector_momentum_label_to(q1, q2)
        ann_text = label.replace('$', '').replace(r'\Gamma', 'Gamma')

        ax.scatter(best_k[0], best_k[1], c=color, s=150, marker=marker,
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.annotate(ann_text, best_k, fontsize=9, ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points',
                    fontweight='bold', color=color)

    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_title(r'Accessible momenta on $3\times3$ kagome (translation-only)',
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    handles = _orbit_legend_handles()
    ax.legend(handles=handles, fontsize=9, loc='lower right')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'bz_momenta.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved bz_momenta.png/pdf")


# ============================================================
# Save spectrum data
# ============================================================
def save_spectrum_data(data, jpm_sorted, sector_meta, output_dir):
    """Save all spectrum data to .dat text files."""
    os.makedirs(output_dir, exist_ok=True)
    jpm_vals = np.array([float(j) for j in jpm_sorted])
    np.savetxt(os.path.join(output_dir, 'spectrum_jpm_values.dat'), jpm_vals)

    with open(os.path.join(output_dir, 'spectrum_metadata.dat'), 'w') as f:
        f.write(f"num_sites = {NUM_SITES}\n")
        f.write(f"n_up_list = {N_UP_LIST}\n")
        for n_up in N_UP_LIST:
            f.write(f"sz_nup{n_up} = {_sz_val(n_up)}\n")

    # Per-n_up spectrum data
    for n_up in N_UP_LIST:
        all_evals = []
        all_secs = []
        for j in jpm_sorted:
            entries = data[j]['by_nup'].get(n_up, [])
            all_evals.append(np.array([e['energy'] for e in entries]))
            all_secs.append(np.array([e['sector'] for e in entries],
                                      dtype=np.int32))
        max_len = max(len(a) for a in all_evals) if all_evals else 0
        ev_arr = np.full((len(jpm_sorted), max_len), np.nan)
        sc_arr = np.full((len(jpm_sorted), max_len), -1, dtype=np.int32)
        for i, (ev, sc) in enumerate(zip(all_evals, all_secs)):
            ev_arr[i, :len(ev)] = ev
            sc_arr[i, :len(sc)] = sc
        np.savetxt(os.path.join(output_dir,
                   f'spectrum_eigenvalues_nup{n_up}.dat'), ev_arr)
        np.savetxt(os.path.join(output_dir,
                   f'spectrum_sector_indices_nup{n_up}.dat'), sc_arr, fmt='%d')

    # Combined spectrum
    all_evals_comb = []
    all_secs_comb = []
    all_nup_comb = []
    for j in jpm_sorted:
        entries = data[j]['all_entries']
        all_evals_comb.append(np.array([e['energy'] for e in entries]))
        all_secs_comb.append(np.array([e['sector'] for e in entries],
                                       dtype=np.int32))
        all_nup_comb.append(np.array([e['n_up'] for e in entries],
                                      dtype=np.int32))
    max_len = max(len(a) for a in all_evals_comb) if all_evals_comb else 0
    ev_arr = np.full((len(jpm_sorted), max_len), np.nan)
    sc_arr = np.full((len(jpm_sorted), max_len), -1, dtype=np.int32)
    nu_arr = np.full((len(jpm_sorted), max_len), -1, dtype=np.int32)
    for i, (ev, sc, nu) in enumerate(zip(all_evals_comb, all_secs_comb,
                                          all_nup_comb)):
        ev_arr[i, :len(ev)] = ev
        sc_arr[i, :len(sc)] = sc
        nu_arr[i, :len(nu)] = nu
    np.savetxt(os.path.join(output_dir, 'spectrum_eigenvalues_combined.dat'),
               ev_arr)
    np.savetxt(os.path.join(output_dir, 'spectrum_sector_indices_combined.dat'),
               sc_arr, fmt='%d')
    np.savetxt(os.path.join(output_dir, 'spectrum_nup_combined.dat'),
               nu_arr, fmt='%d')

    # Sector quantum numbers
    if sector_meta:
        gen_orders = sector_meta.get('generator_orders', [])
        if gen_orders:
            np.savetxt(os.path.join(output_dir,
                       'sector_generator_orders.dat'),
                       np.array(gen_orders, dtype=np.int32).reshape(1, -1),
                       fmt='%d')
        sectors = sector_meta.get('sectors', [])
        if sectors:
            rows = [[s['sector_id']] + list(s['quantum_numbers'])
                    for s in sectors]
            np.savetxt(os.path.join(output_dir,
                       'sector_quantum_numbers.dat'),
                       np.array(rows, dtype=np.int32), fmt='%d',
                       header='sector_id quantum_numbers...')

    print(f"  Saved spectrum .dat files")


# ============================================================
# Main (standalone entry point)
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot multi-Sz spectrum for BFG 3x3 kagome cluster')
    parser.add_argument('--output-dir', type=str,
                        default='/scratch/zhouzb79/analysis_BFG_3x3_fixed_Sz/spectrum')
    parser.add_argument('--n-show', type=int, default=40)
    parser.add_argument('--n-tower', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Base data: {BASE_DIR}")
    print(f"Sz sectors: {[_sz_label(n) for n in N_UP_LIST]}")

    jpm_list = discover_jpm_values()
    print(f"Found {len(jpm_list)} Jpm values: {jpm_list}")

    data, sector_meta = collect_all_spectra(jpm_list)
    momentum_map = build_sector_momentum_map(sector_meta)

    print("\nSaving data...")
    save_spectrum_data(data, jpm_list, sector_meta, args.output_dir)

    print("\nPlotting...")
    plot_global_spectrum(data, jpm_list, args.output_dir, n_show=args.n_show)
    plot_sz_sector_spectrum(data, jpm_list, args.output_dir)
    plot_combined_spectrum(data, jpm_list, sector_meta, args.output_dir,
                           momentum_map=momentum_map)
    plot_symmetry_sector_spectrum(data, jpm_list, sector_meta,
                                 args.output_dir, momentum_map=momentum_map)
    plot_sector_gs_energies(data, jpm_list, sector_meta, args.output_dir,
                            momentum_map=momentum_map)
    plot_individual_sector_panels(data, jpm_list, sector_meta,
                                 args.output_dir, momentum_map=momentum_map)
    plot_tower_of_states(data, jpm_list, sector_meta, args.output_dir,
                         n_tower=args.n_tower, momentum_map=momentum_map)
    plot_tower_of_states_dual(data, jpm_list, sector_meta, args.output_dir,
                               n_tower=args.n_tower, momentum_map=momentum_map)
    plot_bz_momenta(sector_meta, args.output_dir)

    print(f"\nDone! All plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
