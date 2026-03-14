#!/usr/bin/env python3
"""
Plot low-energy spectrum for BFG kagome 2×3 PBC cluster.

Generates:
  1) Global spectrum: all eigenvalues vs Jpm (both Sz sectors merged)
  2) Per-Sz-sector spectrum: separate panels for n_up=8 (Sz=-1) and n_up=9 (Sz=0)
  3) Per-symmetry-sector spectrum: color-coded by sector index
  4) Combined Sz + symmetry sector spectrum

Output directory: analysis_BFG_2x3/spectrum/
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, 'BFG_scan_symmetrized_pbc_2x3_fixed_Sz')
N_UP_LIST = [8, 9]
NUM_SITES = 18
SZ_LABELS = {8: r'$S^z = -1$', 9: r'$S^z = 0$'}

# Skip Jpm=0.0 (Heisenberg point): enhanced symmetry gives 64 sectors
# instead of the usual 6, which pollutes sector-resolved plots.
JPM_STRS = ["0.2", "0.15", "0.12", "0.10", "0.09", "0.08", "0.07", "0.06",
            "0.05", "0.04", "0.03", "0.02", "0.01", "-0.025", "-0.05",
            "-0.075", "-0.1", "-0.125", "-0.15", "-0.175", "-0.2", "-0.225",
            "-0.25", "-0.275", "-0.3", "-0.325", "-0.35", "-0.375", "-0.4"]


# ============================================================
# Data loading
# ============================================================
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


def collect_all_spectra():
    """Collect eigenvalues for all Jpm, both Sz sectors.

    Returns
    -------
    data : dict
        data[jpm_str] = {
            'jpm': float,
            'all_entries': [...],           # merged list
            'by_nup': {n_up: [...], ...},   # per Sz sector
        }
    jpm_sorted : list of str, sorted by float(jpm)
    sector_meta : dict or None  (sector metadata from first available)
    """
    data = {}
    sector_meta = None

    for jpm_str in JPM_STRS:
        jpm_val = float(jpm_str)
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
            'jpm': jpm_val,
            'all_entries': all_entries,
            'by_nup': by_nup,
        }

    jpm_sorted = sorted(data.keys(), key=lambda s: float(s))
    return data, jpm_sorted, sector_meta


def get_sector_qn_label(sector_meta, sector_idx):
    """Get a human-readable quantum number label for a sector."""
    if sector_meta is None:
        return str(sector_idx)
    for s in sector_meta.get('sectors', []):
        if s['sector_id'] == sector_idx:
            qn = s['quantum_numbers']
            return ','.join(str(q) for q in qn)
    return str(sector_idx)


# ============================================================
# Plot 1: Global spectrum
# ============================================================
def plot_global_spectrum(data, jpm_sorted, output_dir, n_show=40):
    """All eigenvalues vs Jpm, merged from both Sz sectors."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(r'BFG Kagome $2\times3$ PBC: Low-Energy Spectrum '
                 r'($n_\uparrow = 8, 9$ merged)', fontsize=14)

    jpm_vals = np.array([float(j) for j in jpm_sorted])

    # --- (a) E/N for lowest n_show levels ---
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

    # --- (b) Energy relative to GS: E_n - E_0 ---
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

    # --- (c) GS energy per site + gap ---
    ax = axes[2]
    e0_arr = [data[j]['all_entries'][0]['energy'] / NUM_SITES for j in jpm_sorted]
    ax.plot(jpm_vals, e0_arr, 'ko-', ms=4, lw=1.5, label=r'$E_0/N$')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_0/N$')
    ax.set_title('(c) GS energy per site & gap')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    ax2 = ax.twinx()
    gaps = []
    for j in jpm_sorted:
        entries = data[j]['all_entries']
        e0 = entries[0]['energy']
        # find first entry NOT degenerate with GS
        e1 = None
        for e in entries[1:]:
            if abs(e['energy'] - e0) > 1e-6:
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
# Plot 2: Per-Sz-sector spectrum
# ============================================================
def plot_sz_sector_spectrum(data, jpm_sorted, output_dir, n_show=30):
    """Separate panels for each Sz sector (n_up=8, 9)."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(r'BFG Kagome $2\times3$ PBC: Spectrum by $S^z$ Sector', fontsize=14)

    jpm_vals = np.array([float(j) for j in jpm_sorted])

    for row, n_up in enumerate(N_UP_LIST):
        sz_val = n_up - NUM_SITES / 2
        label = SZ_LABELS[n_up]

        # --- (a) Absolute energies ---
        ax = axes[row, 0]
        for k in range(n_show):
            evals_k = []
            for j in jpm_sorted:
                entries = data[j]['by_nup'].get(n_up, [])
                if k < len(entries):
                    evals_k.append(entries[k]['energy'] / NUM_SITES)
                else:
                    evals_k.append(np.nan)
            color = 'red' if k == 0 else ('blue' if k < 10 else 'gray')
            alpha = 1.0 if k == 0 else (0.7 if k < 10 else 0.3)
            ax.plot(jpm_vals, evals_k, '-', lw=0.8, alpha=alpha, color=color)
        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel(r'$E_n / N$')
        ax.set_title(f'{label}: $E_n/N$ (lowest {n_show})')
        ax.grid(True, alpha=0.3)

        # --- (b) Gaps relative to sector GS ---
        ax = axes[row, 1]
        for k in range(1, n_show):
            gaps_k = []
            for j in jpm_sorted:
                entries = data[j]['by_nup'].get(n_up, [])
                if len(entries) == 0:
                    gaps_k.append(np.nan)
                    continue
                e0_sec = entries[0]['energy']
                if k < len(entries):
                    gaps_k.append(entries[k]['energy'] - e0_sec)
                else:
                    gaps_k.append(np.nan)
            color = 'blue' if k < 10 else 'gray'
            alpha = 0.8 if k < 10 else 0.3
            ax.plot(jpm_vals, gaps_k, '-', lw=0.8, alpha=alpha, color=color)
        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel(r'$E_n - E_0^{\rm sector}$')
        ax.set_title(f'{label}: excitation gaps')
        ax.set_ylim(bottom=-0.01)
        ax.grid(True, alpha=0.3)

        # --- (c) Sector GS energy + gap ---
        ax = axes[row, 2]
        e0_sec = []
        gaps_sec = []
        for j in jpm_sorted:
            entries = data[j]['by_nup'].get(n_up, [])
            if entries:
                e0_sec.append(entries[0]['energy'] / NUM_SITES)
                e1 = None
                for e in entries[1:]:
                    if abs(e['energy'] - entries[0]['energy']) > 1e-6:
                        e1 = e['energy']
                        break
                gaps_sec.append(e1 - entries[0]['energy'] if e1 is not None else 0.0)
            else:
                e0_sec.append(np.nan)
                gaps_sec.append(np.nan)

        ax.plot(jpm_vals, e0_sec, 'ko-', ms=4, lw=1.5, label=r'$E_0^{\rm sector}/N$')
        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel(r'$E_0^{\rm sector}/N$')
        ax.set_title(f'{label}: sector GS & gap')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

        ax2 = ax.twinx()
        ax2.plot(jpm_vals, gaps_sec, 'r^-', ms=3, lw=1, alpha=0.7)
        ax2.set_ylabel(r'$\Delta_{\rm sector}$', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'sz_sector_spectrum.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved sz_sector_spectrum.png/pdf")


# ============================================================
# Plot 3: Per-symmetry-sector spectrum (within each Sz sector)
# ============================================================
def _momentum_label(q0, q1):
    r"""Map (q0, q1) quantum numbers to a BZ momentum label.

    Allowed k = (q0/2)*b1 + (q1/3)*b2  with q0 in {0,1}, q1 in {0,1,2}.
    Only (0,0) = Gamma and (1,0) = M hit true high-symmetry points.
    (0,1)<->(0,2) and (1,1)<->(1,2) are time-reversal partners.
    """
    if q0 == 0 and q1 == 0:
        return r'$\Gamma$'
    elif q0 == 1 and q1 == 0:
        return r'$M$'
    # Generic points: label by k-vector in units of (b1, b2)
    return rf'$\mathbf{{k}}=(\frac{{{q0}}}{{2}},\frac{{{q1}}}{{3}})$'


def plot_symmetry_sector_spectrum(data, jpm_sorted, sector_meta, output_dir,
                                 n_show_per_sector=5):
    """Excitation gap per momentum sector: 2×3 grid of subfigures."""
    jpm_vals = np.array([float(j) for j in jpm_sorted])

    # Identify all sectors across all data
    all_sectors = set()
    for j in jpm_sorted:
        for n_up in N_UP_LIST:
            for e in data[j]['by_nup'].get(n_up, []):
                all_sectors.add(e['sector'])
    all_sectors = sorted(all_sectors)

    # Build sector quantum number map from metadata
    sec_qn = {}
    if sector_meta:
        for s in sector_meta.get('sectors', []):
            sec_qn[s['sector_id']] = tuple(s['quantum_numbers'])

    for n_up in N_UP_LIST:
        sz_label = SZ_LABELS[n_up]

        # Collect per-sector eigenvalue curves
        sector_curves = defaultdict(lambda: defaultdict(list))
        for j in jpm_sorted:
            entries = data[j]['by_nup'].get(n_up, [])
            jv = float(j)
            by_sec = defaultdict(list)
            for e in entries:
                by_sec[e['sector']].append(e['energy'])
            for sec in all_sectors:
                sec_evals = sorted(by_sec.get(sec, []))
                for k, ev in enumerate(sec_evals[:n_show_per_sector]):
                    sector_curves[sec][k].append((jv, ev))

        # 2×3 grid: one subplot per sector
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(rf'BFG Kagome $2\times3$: Excitation Gap per Momentum Sector '
                     rf'({sz_label})', fontsize=14, y=0.98)

        panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

        for idx, sec in enumerate(all_sectors[:6]):
            row, col = divmod(idx, 3)
            ax = axes[row, col]

            # Momentum label
            qn = sec_qn.get(sec, (sec,))
            if len(qn) == 2:
                mom_label = _momentum_label(qn[0], qn[1])
                qn_str = f'$q_0={qn[0]},\\,q_1={qn[1]}$'
            else:
                mom_label = f'sec {sec}'
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
                    ax.plot(xs, ys, '-', lw=lw, alpha=alpha, color='C0')

                # Highlight the gap (first excitation)
                if 1 in sector_curves[sec]:
                    pts = sector_curves[sec][1]
                    xs = [p[0] for p in pts if p[0] in e0_pts]
                    ys = [p[1] - e0_pts[p[0]] for p in pts if p[0] in e0_pts]
                    ax.plot(xs, ys, 'o-', ms=3, lw=1.5, color='C3',
                            label=r'$\Delta_1$', zorder=5)

            title_str = f'{panel_labels[idx]} {mom_label}'
            if qn_str:
                title_str += f'  ({qn_str})'
            ax.set_title(title_str, fontsize=11)
            ax.set_xlabel(r'$J_{\pm}$')
            ax.set_ylabel(r'$E_n - E_0^{\rm sector}$')
            ax.set_ylim(bottom=-0.01)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')

        plt.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(os.path.join(output_dir,
                        f'symmetry_sector_spectrum_nup{n_up}.{ext}'),
                        dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved symmetry_sector_spectrum_nup{n_up}.png/pdf")


# ============================================================
# Plot 4: Combined Sz + symmetry sector overview
# ============================================================
def plot_combined_spectrum(data, jpm_sorted, sector_meta, output_dir,
                          n_show_per_sector=3):
    """All eigenvalues merged, colored by (n_up, sector) pair."""
    jpm_vals = np.array([float(j) for j in jpm_sorted])

    # Build curves for each (n_up, sector) pair
    all_pairs = set()
    pair_curves = defaultdict(lambda: defaultdict(list))

    for j in jpm_sorted:
        jv = float(j)
        for n_up in N_UP_LIST:
            entries = data[j]['by_nup'].get(n_up, [])
            by_sec = defaultdict(list)
            for e in entries:
                by_sec[e['sector']].append(e['energy'])
            for sec, evals in by_sec.items():
                pair = (n_up, sec)
                all_pairs.add(pair)
                for k, ev in enumerate(sorted(evals)[:n_show_per_sector]):
                    pair_curves[pair][k].append((jv, ev))

    all_pairs = sorted(all_pairs)

    # Two-tone coloring: blues for n_up=8, reds for n_up=9
    n_sec_8 = len([p for p in all_pairs if p[0] == 8])
    n_sec_9 = len([p for p in all_pairs if p[0] == 9])
    colors_8 = plt.cm.Blues(np.linspace(0.3, 1.0, max(n_sec_8, 1)))
    colors_9 = plt.cm.Reds(np.linspace(0.3, 1.0, max(n_sec_9, 1)))

    pair_colors = {}
    i8, i9 = 0, 0
    for p in all_pairs:
        if p[0] == 8:
            pair_colors[p] = colors_8[i8]
            i8 += 1
        else:
            pair_colors[p] = colors_9[i9]
            i9 += 1

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(r'BFG Kagome $2\times3$ PBC: Spectrum by ($S^z$, Sector)', fontsize=14)

    # --- (a) Absolute spectrum ---
    ax = axes[0]
    for pair in all_pairs:
        for k in sorted(pair_curves[pair].keys()):
            pts = pair_curves[pair][k]
            xs = [p[0] for p in pts]
            ys = [p[1] / NUM_SITES for p in pts]
            alpha = 0.7 if k == 0 else 0.3
            lw = 1.0 if k == 0 else 0.4
            ax.plot(xs, ys, '-', color=pair_colors[pair], alpha=alpha, lw=lw)

    # Legend handles
    h8 = Line2D([], [], color='blue', lw=2, label=r'$n_\uparrow=8$ ($S^z=-1$)')
    h9 = Line2D([], [], color='red', lw=2, label=r'$n_\uparrow=9$ ($S^z=0$)')
    ax.legend(handles=[h8, h9], fontsize=10)
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_n / N$')
    ax.set_title(f'(a) All levels by ($S^z$, sector)')
    ax.grid(True, alpha=0.3)

    # --- (b) E_n - E_0 spectrum ---
    ax = axes[1]
    e0_global = {float(j): data[j]['all_entries'][0]['energy']
                 for j in jpm_sorted}
    for pair in all_pairs:
        for k in sorted(pair_curves[pair].keys()):
            pts = pair_curves[pair][k]
            xs = [p[0] for p in pts]
            ys = [p[1] - e0_global[p[0]] for p in pts]
            alpha = 0.7 if k == 0 else 0.3
            lw = 1.0 if k == 0 else 0.4
            ax.plot(xs, ys, '-', color=pair_colors[pair], alpha=alpha, lw=lw)

    ax.legend(handles=[h8, h9], fontsize=10)
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_n - E_0$')
    ax.set_title('(b) Excitation energies')
    ax.set_ylim(bottom=-0.01)
    ax.grid(True, alpha=0.3)

    # --- (c) Sector GS energies (both Sz) ---
    ax = axes[2]
    for n_up in N_UP_LIST:
        sz_lbl = SZ_LABELS[n_up]
        base_color = 'tab:blue' if n_up == 8 else 'tab:red'
        # Get the overall lowest energy per Sz sector at each Jpm
        e0_sz = []
        for j in jpm_sorted:
            entries = data[j]['by_nup'].get(n_up, [])
            if entries:
                e0_sz.append(entries[0]['energy'] / NUM_SITES)
            else:
                e0_sz.append(np.nan)
        ax.plot(jpm_vals, e0_sz, 'o-', color=base_color, ms=4, lw=1.5,
                label=f'{sz_lbl} GS')

    # Also plot global GS
    e0_all = [data[j]['all_entries'][0]['energy'] / NUM_SITES for j in jpm_sorted]
    ax.plot(jpm_vals, e0_all, 'k*-', ms=6, lw=1, alpha=0.8, label='Global GS')

    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_0 / N$')
    ax.set_title(r'(c) GS energy: per $S^z$ sector vs global')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'combined_sz_sector_spectrum.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved combined_sz_sector_spectrum.png/pdf")


# ============================================================
# Plot 5: Sector GS energy overview (per symmetry sector, both Sz merged)
# ============================================================
def plot_sector_gs_energies(data, jpm_sorted, sector_meta, output_dir):
    """Plot the lowest energy in each symmetry sector across both Sz sectors."""
    jpm_vals = np.array([float(j) for j in jpm_sorted])

    # Collect: for each sector, the lowest energy across both n_up at each Jpm
    all_sectors = set()
    for j in jpm_sorted:
        for n_up in N_UP_LIST:
            for e in data[j]['by_nup'].get(n_up, []):
                all_sectors.add(e['sector'])
    all_sectors = sorted(all_sectors)
    n_sec = len(all_sectors)

    # Build sector_E0[sec][jpm_idx] = min energy in that sector (across both n_up)
    sector_E0 = {s: np.full(len(jpm_sorted), np.nan) for s in all_sectors}
    for ji, j in enumerate(jpm_sorted):
        for n_up in N_UP_LIST:
            for e in data[j]['by_nup'].get(n_up, []):
                s = e['sector']
                if np.isnan(sector_E0[s][ji]) or e['energy'] < sector_E0[s][ji]:
                    sector_E0[s][ji] = e['energy']

    # Use a good colormap
    if n_sec <= 10:
        cmap = plt.cm.tab10
    elif n_sec <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar
    colors = {s: cmap(i / max(n_sec - 1, 1)) for i, s in enumerate(all_sectors)}

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(r'BFG Kagome $2\times3$ PBC: Sector GS Energies', fontsize=14)

    # (a) E_0^sector / N
    ax = axes[0]
    for sec in all_sectors:
        valid = ~np.isnan(sector_E0[sec])
        if np.any(valid):
            qn_label = get_sector_qn_label(sector_meta, sec)
            ax.plot(jpm_vals[valid], sector_E0[sec][valid] / NUM_SITES,
                    '-', color=colors[sec], lw=0.8, alpha=0.6)
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_0^{\rm sector} / N$')
    ax.set_title('(a) Sector GS energies')
    ax.grid(True, alpha=0.3)

    # (b) E_0^sector - E_0^global
    ax = axes[1]
    e0_global = np.array([data[j]['all_entries'][0]['energy'] for j in jpm_sorted])
    for sec in all_sectors:
        valid = ~np.isnan(sector_E0[sec])
        if np.any(valid):
            delta = sector_E0[sec][valid] - e0_global[valid]
            ax.plot(jpm_vals[valid], delta, '-', color=colors[sec], lw=0.8, alpha=0.6)
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel(r'$E_0^{\rm sector} - E_0^{\rm global}$')
    ax.set_title('(b) Sector GS relative to global GS')
    ax.set_ylim(bottom=-0.01)
    ax.grid(True, alpha=0.3)

    # (c) GS sector ID: which sector(s) host the global GS
    ax = axes[2]
    gs_sectors_per_jpm = []
    gs_n_degenerate = []
    for ji, j in enumerate(jpm_sorted):
        e0 = data[j]['all_entries'][0]['energy']
        gs_secs = set()
        n_deg = 0
        for e in data[j]['all_entries']:
            if abs(e['energy'] - e0) < 1e-6:
                gs_secs.add(e['sector'])
                n_deg += 1
        gs_sectors_per_jpm.append(len(gs_secs))
        gs_n_degenerate.append(n_deg)

    ax.plot(jpm_vals, gs_n_degenerate, 'ko-', ms=4, lw=1.5,
            label='GS degeneracy (both $S^z$)')
    ax.plot(jpm_vals, gs_sectors_per_jpm, 'rs-', ms=3, lw=1, alpha=0.7,
            label='# sectors hosting GS')
    ax.set_xlabel(r'$J_{\pm}$')
    ax.set_ylabel('Count')
    ax.set_title('(c) GS degeneracy & sector count')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'sector_gs_energies.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved sector_gs_energies.png/pdf")


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

    for n_up in N_UP_LIST:
        all_evals = []
        all_sectors = []
        for j in jpm_sorted:
            entries = data[j]['by_nup'].get(n_up, [])
            evals = np.array([e['energy'] for e in entries])
            secs = np.array([e['sector'] for e in entries], dtype=np.int32)
            all_evals.append(evals)
            all_sectors.append(secs)

        # Pad to same length
        max_len = max(len(a) for a in all_evals) if all_evals else 0
        evals_arr = np.full((len(jpm_sorted), max_len), np.nan)
        secs_arr = np.full((len(jpm_sorted), max_len), -1, dtype=np.int32)
        for i, (ev, sc) in enumerate(zip(all_evals, all_sectors)):
            evals_arr[i, :len(ev)] = ev
            secs_arr[i, :len(sc)] = sc

        np.savetxt(os.path.join(output_dir,
                   f'spectrum_n_up_{n_up}_eigenvalues.dat'), evals_arr)
        np.savetxt(os.path.join(output_dir,
                   f'spectrum_n_up_{n_up}_sector_indices.dat'),
                   secs_arr, fmt='%d')

    print(f"  Saved spectrum .dat files")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Plot low-energy spectrum for BFG 2×3 kagome cluster')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(SCRIPT_DIR, 'analysis_BFG_2x3', 'spectrum'),
                        help='Output directory')
    parser.add_argument('--n-show', type=int, default=40,
                        help='Number of levels to show in global plot')
    parser.add_argument('--n-show-per-sector', type=int, default=5,
                        help='Number of levels per sector in sector plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Base data: {BASE_DIR}")
    print(f"Jpm values: {len(JPM_STRS)}")

    print("\nCollecting spectra...")
    data, jpm_sorted, sector_meta = collect_all_spectra()

    n_total = sum(len(data[j]['all_entries']) for j in jpm_sorted)
    print(f"  Collected {n_total} eigenvalues across {len(jpm_sorted)} Jpm values")

    if sector_meta:
        gen_orders = sector_meta.get('generator_orders', [])
        n_sec = len(sector_meta.get('sectors', []))
        print(f"  Symmetry: {len(gen_orders)} generators, orders {gen_orders}, "
              f"{n_sec} sectors")

    print("\nPlotting...")
    plot_global_spectrum(data, jpm_sorted, args.output_dir, n_show=args.n_show)
    plot_sz_sector_spectrum(data, jpm_sorted, args.output_dir, n_show=min(args.n_show, 30))
    plot_symmetry_sector_spectrum(data, jpm_sorted, sector_meta, args.output_dir,
                                 n_show_per_sector=args.n_show_per_sector)
    plot_combined_spectrum(data, jpm_sorted, sector_meta, args.output_dir,
                           n_show_per_sector=args.n_show_per_sector)
    plot_sector_gs_energies(data, jpm_sorted, sector_meta, args.output_dir)

    print("\nSaving data...")
    save_spectrum_data(data, jpm_sorted, sector_meta, args.output_dir)

    print("\nDone! All plots saved to:", args.output_dir)


if __name__ == '__main__':
    main()
