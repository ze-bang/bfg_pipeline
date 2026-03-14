#!/usr/bin/env python3
"""
Visualize per-eigenstate diagnostics:
  1. sz_local[si, j]   — sublattice magnetization per GS eigenstate
  2. chirality[si, t]  — scalar chirality per GS eigenstate per triangle

Reads from analysis_BFG_3x3/per_jpm/Jpm_*/gs/per_state/*.dat
"""
import os
import sys
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon as MplPolygon

OUTPUT_DIR = '/scratch/zhouzb79/analysis_BFG_3x3'
PER_JPM_DIR = os.path.join(OUTPUT_DIR, 'per_jpm')
NUM_SITES = 27


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


def load_all_diagnostics():
    """Load sz_local and chirality from all per-Jpm .dat directories."""
    results = []
    jpm_dirs = sorted(glob.glob(os.path.join(PER_JPM_DIR, 'Jpm_*')))
    for jpm_dir in jpm_dirs:
        if not os.path.isdir(jpm_dir):
            continue
        basename = os.path.basename(jpm_dir)
        jpm_str = basename.replace('Jpm_', '')
        jpm_val = float(jpm_str)

        ps_dir = os.path.join(jpm_dir, 'gs', 'per_state')
        if not os.path.isdir(ps_dir):
            continue

        sz_path = os.path.join(ps_dir, 'sz_local.dat')
        if not os.path.exists(sz_path):
            continue

        sz_local = np.loadtxt(sz_path)
        if sz_local.ndim == 1:
            sz_local = sz_local.reshape(1, -1)
        n_gs = sz_local.shape[0]

        # Chirality (optional)
        chi_path = os.path.join(ps_dir, 'chirality.dat')
        chirality = np.loadtxt(chi_path) if os.path.exists(chi_path) else None
        if chirality is not None and chirality.ndim == 1:
            chirality = chirality.reshape(1, -1)

        # Sectors
        sec_path = os.path.join(ps_dir, 'sector.dat')
        sectors = np.loadtxt(sec_path, dtype=int).ravel() \
            if os.path.exists(sec_path) else np.zeros(n_gs, dtype=int)

        # Triangle list
        tri_path = os.path.join(ps_dir, 'triangle_list.dat')
        tri_list = np.loadtxt(tri_path, dtype=int) \
            if os.path.exists(tri_path) else None

        # Bond energies
        bm_path = os.path.join(ps_dir, 'B_mean.dat')
        B_mean = np.loadtxt(bm_path) if os.path.exists(bm_path) else None
        if B_mean is not None and B_mean.ndim == 1:
            B_mean = B_mean.reshape(1, -1)

        szsz_path = os.path.join(ps_dir, 'SzSz_bond.dat')
        SzSz_bond = np.loadtxt(szsz_path) if os.path.exists(szsz_path) else None
        if SzSz_bond is not None and SzSz_bond.ndim == 1:
            SzSz_bond = SzSz_bond.reshape(1, -1)

        # Per-eigenstate structure factors
        sq_path = os.path.join(ps_dir, 'Sq.dat')
        ps_Sq = np.loadtxt(sq_path) if os.path.exists(sq_path) else None
        if ps_Sq is not None and ps_Sq.ndim == 1:
            ps_Sq = ps_Sq.reshape(1, -1)

        szz_path = os.path.join(ps_dir, 'Szz.dat')
        ps_Szz = np.loadtxt(szz_path) if os.path.exists(szz_path) else None
        if ps_Szz is not None and ps_Szz.ndim == 1:
            ps_Szz = ps_Szz.reshape(1, -1)

        # Per-eigenstate dimer structure factors
        dq_full_path = os.path.join(ps_dir, 'Dq_full.dat')
        ps_Dq_full = np.loadtxt(dq_full_path) if os.path.exists(dq_full_path) else None
        if ps_Dq_full is not None and ps_Dq_full.ndim == 1:
            ps_Dq_full = ps_Dq_full.reshape(1, -1)

        dq_conn_path = os.path.join(ps_dir, 'Dq_conn.dat')
        ps_Dq_conn = np.loadtxt(dq_conn_path) if os.path.exists(dq_conn_path) else None
        if ps_Dq_conn is not None and ps_Dq_conn.ndim == 1:
            ps_Dq_conn = ps_Dq_conn.reshape(1, -1)

        results.append({
            'Jpm': jpm_val,
            'jpm_str': jpm_str,
            'n_gs': n_gs,
            'sz_local': sz_local,
            'chirality': chirality,
            'sectors': sectors,
            'triangle_list': tri_list,
            'B_mean': B_mean,
            'SzSz_bond': SzSz_bond,
            'ps_Sq': ps_Sq,
            'ps_Szz': ps_Szz,
            'ps_Dq_full': ps_Dq_full,
            'ps_Dq_conn': ps_Dq_conn,
        })

    results.sort(key=lambda r: r['Jpm'])
    return results


def plot_sz_local_summary(results, output_dir):
    """
    Multi-panel figure — one dot per GS eigenstate (no averaging):
      (a) sz range per GS eigenstate vs Jpm
      (b) sublattice-averaged sz per GS eigenstate vs Jpm
      (c) spatial stddev of sz per GS eigenstate vs Jpm
      (d) site-resolved sz per GS eigenstate
    """
    sub_colors = ['tab:red', 'tab:blue', 'tab:green']
    sub_labels = ['Sublattice A (0)', 'Sublattice B (1)', 'Sublattice C (2)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(r'Per-GS-eigenstate $\langle\psi_i|S_j^z|\psi_i\rangle$',
                 fontsize=14)

    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for r_idx, r in enumerate(results):
        sz = r['sz_local']  # (n_gs, 27)
        jpm = r['Jpm']
        n_gs = r['n_gs']

        for si in range(n_gs):
            sz_i = sz[si]  # (27,)

            # (a) range
            rng = sz_i.max() - sz_i.min()
            ax_a.scatter(jpm, rng, c='steelblue', s=15, alpha=0.6,
                         edgecolors='none', zorder=3)

            # (b) sublattice averages
            for s in range(3):
                sub_sites = list(range(s, 27, 3))
                sub_avg = sz_i[sub_sites].mean()
                ax_b.scatter(jpm, sub_avg, c=sub_colors[s], s=12, alpha=0.5,
                             edgecolors='none', zorder=3,
                             label=sub_labels[s] if (r_idx == 0 and si == 0) else None)

            # (c) spatial std
            ax_c.scatter(jpm, np.std(sz_i), c='steelblue', s=15, alpha=0.6,
                         edgecolors='none', zorder=3)

            # (d) site-resolved scatter
            for s in range(3):
                sub_sites = list(range(s, 27, 3))
                ax_d.scatter([jpm] * len(sub_sites), sz_i[sub_sites],
                             c=sub_colors[s], s=6, alpha=0.25,
                             edgecolors='none',
                             label=sub_labels[s] if (r_idx == 0 and si == 0) else None)

    ax_a.set_xlabel(r'$J_{\pm}$')
    ax_a.set_ylabel(r'$\max_j \langle S_j^z \rangle - \min_j \langle S_j^z \rangle$')
    ax_a.set_title(r'(a) $\langle S^z \rangle$ range per GS eigenstate')
    ax_a.grid(True, alpha=0.3)

    ax_b.set_xlabel(r'$J_{\pm}$')
    ax_b.set_ylabel(r'$\langle S^z \rangle_{\rm sublat}$')
    ax_b.set_title(r'(b) Sublattice-averaged $\langle S^z \rangle$ per GS eigenstate')
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3)

    ax_c.set_xlabel(r'$J_{\pm}$')
    ax_c.set_ylabel(r'$\sigma_{j}[\langle S_j^z \rangle]$')
    ax_c.set_title(r'(c) Spatial std of $\langle S^z \rangle$ per GS eigenstate')
    ax_c.grid(True, alpha=0.3)

    ax_d.set_xlabel(r'$J_{\pm}$')
    ax_d.set_ylabel(r'$\langle S_j^z \rangle$')
    ax_d.set_title(r'(d) Site-resolved $\langle S_j^z \rangle$ (all GS eigenstates)')
    ax_d.legend(fontsize=7, markerscale=3)
    ax_d.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'diagnostic_sz_local.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved diagnostic_sz_local")


def plot_chirality_summary(results, output_dir):
    """
    Multi-panel figure for scalar chirality:
      (a) |κ|_max vs Jpm
      (b) κ by triangle type (up vs down) vs Jpm
      (c) per-triangle chirality for selected Jpm
      (d) chirality distribution across eigenstates
    """
    results_chi = [r for r in results if r['chirality'] is not None]
    if not results_chi:
        print("  No chirality data, skipping")
        return

    jpm_vals = [r['Jpm'] for r in results_chi]

    # Triangle classification: up = same UC (0,4,7,9,11,13,15,16,17), down = rest
    # Based on triangle enumeration
    all_tris = results_chi[0]['triangle_list']
    n_tri = len(all_tris)

    # Classify based on sites: up-triangle = sites all in same UC
    up_idx = []
    dn_idx = []
    for t_idx, (si, sj, sk) in enumerate(all_tris):
        uci, ucj, uck = si // 3, sj // 3, sk // 3
        if uci == ucj == uck:
            up_idx.append(t_idx)
        else:
            dn_idx.append(t_idx)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(r'Per-GS-eigenstate scalar chirality $\kappa_\triangle = \langle \mathbf{S}_i \cdot (\mathbf{S}_j \times \mathbf{S}_k) \rangle$',
                 fontsize=13)

    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Pick a few Jpm values for the per-triangle bar chart (c)
    selected = [r for r in results_chi if r['Jpm'] in
                [-0.40, -0.30, -0.20, -0.10, -0.05]]
    if not selected:
        step = max(1, len(results_chi) // 5)
        selected = results_chi[::step][:5]

    for r in results_chi:
        chi = r['chirality']  # (n_gs, 18)
        jpm = r['Jpm']
        n_gs = r['n_gs']

        for si in range(n_gs):
            chi_i = chi[si]  # (18,)

            # (a) |κ|_max per GS eigenstate
            ax_a.scatter(jpm, np.max(np.abs(chi_i)), c='steelblue',
                         s=15, alpha=0.6, edgecolors='none', zorder=3)

            # (b) mean κ on up vs down triangles per GS eigenstate
            ax_b.scatter(jpm, chi_i[up_idx].mean(), c='tab:red',
                         s=12, alpha=0.5, edgecolors='none', zorder=3,
                         marker='s',
                         label=(r'$\langle\kappa\rangle_{\triangle\rm up}$'
                                if (r is results_chi[0] and si == 0) else None))
            ax_b.scatter(jpm, chi_i[dn_idx].mean(), c='tab:blue',
                         s=12, alpha=0.5, edgecolors='none', zorder=3,
                         marker='^',
                         label=(r'$\langle\kappa\rangle_{\triangledown\rm dn}$'
                                if (r is results_chi[0] and si == 0) else None))

            # (d) total chirality Σ_△ κ per GS eigenstate
            ax_d.scatter(jpm, chi_i.sum(), c='steelblue',
                         s=15, alpha=0.6, edgecolors='none', zorder=3)

    # (c) Per-triangle chirality bar chart — every GS eigenstate
    x = np.arange(n_tri)
    # Count total bars
    total_bars = sum(r['n_gs'] for r in selected)
    width = 0.8 / max(total_bars, 1)
    bar_idx = 0
    gs_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for r in selected:
        chi = r['chirality']  # (n_gs, 18)
        for si in range(r['n_gs']):
            c = gs_colors[bar_idx % 10]
            ax_c.bar(x + bar_idx * width, chi[si], width,
                     label=f"Jpm={r['Jpm']:.2f} GS{si}", alpha=0.8, color=c)
            bar_idx += 1

    ax_a.set_xlabel(r'$J_{\pm}$')
    ax_a.set_ylabel(r'$|\kappa|_{\rm max}$')
    ax_a.set_title(r'(a) Max $|\kappa|$ per GS eigenstate')
    ax_a.grid(True, alpha=0.3)

    ax_b.set_xlabel(r'$J_{\pm}$')
    ax_b.set_ylabel(r'$\langle \kappa \rangle$')
    ax_b.set_title(r'(b) Mean $\kappa$: up $\triangle$ vs down $\triangledown$ per GS eigenstate')
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3)
    ax_b.axhline(0, color='gray', lw=0.5, ls='--')

    ax_c.set_xlabel('Triangle index')
    ax_c.set_ylabel(r'$\kappa_\triangle$')
    ax_c.set_title(r'(c) Per-triangle $\kappa$ (all GS eigenstates)')
    ax_c.legend(fontsize=5, ncol=2)
    ax_c.grid(True, alpha=0.3, axis='y')
    ax_c.axhline(0, color='gray', lw=0.5, ls='--')
    for t_idx in up_idx:
        ax_c.axvspan(t_idx - 0.4, t_idx + 0.4 + width * total_bars,
                     alpha=0.05, color='red')

    ax_d.set_xlabel(r'$J_{\pm}$')
    ax_d.set_ylabel(r'$\sum_\triangle \kappa_\triangle$')
    ax_d.set_title(r'(d) Total chirality per GS eigenstate')
    ax_d.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'diagnostic_chirality.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved diagnostic_chirality")


def plot_sz_lattice_patterns(results, output_dir):
    """
    Lattice plots of ⟨Sz⟩ pattern for first eigenstate at selected Jpm values.
    Shows √3×√3 magnetic structure.
    """
    # Kagome positions
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2])
    offsets = np.array([[0, 0], [0.5, 0], [0.25, np.sqrt(3) / 4]])

    positions = []
    for i in range(3):
        for j in range(3):
            for s in range(3):
                positions.append(i * a1 + j * a2 + offsets[s])
    positions = np.array(positions)

    selected = [r for r in results if r['Jpm'] in
                [-0.40, -0.30, -0.20, -0.15, -0.10, -0.05]]
    if not selected:
        step = max(1, len(results) // 6)
        selected = results[::step][:6]

    # Build list of all (Jpm, GS_index, sz_pattern) panels
    panels = []
    for r in selected:
        for si in range(r['n_gs']):
            panels.append((r['Jpm'], si, r['sz_local'][si], r['n_gs']))

    n_panels = len(panels)
    if n_panels == 0:
        return

    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    if n_panels == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    sz_all = np.concatenate([p[2] for p in panels])
    vabs = max(abs(sz_all.min()), abs(sz_all.max()))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    sc = None
    for idx, (jpm, si, sz, n_gs) in enumerate(panels):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        sc = ax.scatter(positions[:, 0], positions[:, 1],
                        c=sz, cmap='RdBu_r', norm=norm,
                        s=120, edgecolors='black', linewidths=0.5, zorder=5)

        for site_idx in range(27):
            ax.annotate(f'{sz[site_idx]:+.3f}',
                        positions[site_idx],
                        fontsize=3.5, ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points')

        ax.set_aspect('equal')
        ax.set_title(f'Jpm={jpm:.2f}, GS[{si}]/{n_gs}', fontsize=9)
        ax.set_xlim(-0.3, 3.5)
        ax.set_ylim(-0.3, 3.0)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(r'$\langle S_j^z \rangle$ pattern on kagome lattice (each GS eigenstate)',
                 fontsize=13)
    if sc is not None:
        fig.colorbar(sc, ax=axes, shrink=0.6, label=r'$\langle S_j^z \rangle$')
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'diagnostic_sz_lattice.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved diagnostic_sz_lattice")


def plot_bond_energy(results, output_dir):
    """
    Per-GS-eigenstate bond energy decomposition:
      (a) Avg bond energy <S·S> per GS eigenstate vs Jpm
      (b) Ising part <SzSz> per GS eigenstate vs Jpm
      (c) XY part (S+S- + S-S+)/2 per GS eigenstate vs Jpm
      (d) Bond energy std (spatial variation across bonds) per eigenstate
    """
    results_bond = [r for r in results if r['B_mean'] is not None]
    if not results_bond:
        print("  No B_mean data, skipping bond energy plot")
        return

    has_decomp = any(r['SzSz_bond'] is not None for r in results_bond)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(r'Per-GS-eigenstate NN bond energy $\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$',
                 fontsize=14)

    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for r in results_bond:
        bm = r['B_mean']       # (n_gs, 54)
        szsz = r['SzSz_bond']  # (n_gs, 54) or None
        jpm = r['Jpm']

        for si in range(r['n_gs']):
            # (a) Average bond energy <S·S>
            avg_full = bm[si].mean()
            ax_a.scatter(jpm, avg_full, c='steelblue', s=15, alpha=0.6,
                         edgecolors='none', zorder=3)

            if szsz is not None:
                # (b) Ising part
                avg_zz = szsz[si].mean()
                ax_b.scatter(jpm, avg_zz, c='tab:red', s=15, alpha=0.6,
                             edgecolors='none', zorder=3)

                # (c) XY part = full - SzSz
                xy = bm[si] - szsz[si]
                avg_xy = xy.mean()
                ax_c.scatter(jpm, avg_xy, c='tab:green', s=15, alpha=0.6,
                             edgecolors='none', zorder=3)

            # (d) Spatial variation (std across bonds)
            ax_d.scatter(jpm, np.std(bm[si]), c='steelblue', s=15, alpha=0.6,
                         edgecolors='none', zorder=3,
                         label=(r'$\sigma_b[\langle S\cdot S \rangle]$'
                                if (r is results_bond[0] and si == 0) else None))
            if szsz is not None:
                ax_d.scatter(jpm, np.std(szsz[si]), c='tab:red', s=10, alpha=0.4,
                             edgecolors='none', zorder=3, marker='s',
                             label=(r'$\sigma_b[\langle S^z S^z \rangle]$'
                                    if (r is results_bond[0] and si == 0) else None))
                ax_d.scatter(jpm, np.std(bm[si] - szsz[si]), c='tab:green',
                             s=10, alpha=0.4, edgecolors='none', zorder=3, marker='^',
                             label=(r'$\sigma_b[\mathrm{XY}]$'
                                    if (r is results_bond[0] and si == 0) else None))

    ax_a.set_xlabel(r'$J_{\pm}$')
    ax_a.set_ylabel(r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle_{\rm avg}$')
    ax_a.set_title(r'(a) Avg NN bond energy (full $\mathbf{S}\cdot\mathbf{S}$) per GS eigenstate')
    ax_a.grid(True, alpha=0.3)

    ax_b.set_xlabel(r'$J_{\pm}$')
    ax_b.set_ylabel(r'$\langle S_i^z S_j^z \rangle_{\rm avg}$')
    ax_b.set_title(r'(b) Ising part $\langle S^z S^z \rangle$ per GS eigenstate')
    ax_b.grid(True, alpha=0.3)
    if not has_decomp:
        ax_b.text(0.5, 0.5, 'SzSz\_bond data\nnot yet computed',
                  transform=ax_b.transAxes, ha='center', va='center', fontsize=12)

    ax_c.set_xlabel(r'$J_{\pm}$')
    ax_c.set_ylabel(r'$\langle S^+ S^- + S^- S^+ \rangle_{\rm avg} / 2$')
    ax_c.set_title(r'(c) XY part per GS eigenstate')
    ax_c.grid(True, alpha=0.3)
    if not has_decomp:
        ax_c.text(0.5, 0.5, 'SzSz\_bond data\nnot yet computed',
                  transform=ax_c.transAxes, ha='center', va='center', fontsize=12)

    ax_d.set_xlabel(r'$J_{\pm}$')
    ax_d.set_ylabel(r'$\sigma_b$ (std across bonds)')
    ax_d.set_title(r'(d) Spatial variation of bond energy per GS eigenstate')
    ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'diagnostic_bond_energy.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved diagnostic_bond_energy")


def plot_transverse_moments(results, output_dir):
    """
    Per-GS-eigenstate transverse vs longitudinal structure factor decomposition:
      (a) S^zz(q) at each q per eigenstate vs Jpm
      (b) S^perp(q) = S(q) - S^zz(q) per eigenstate vs Jpm
      (c) Ratio S^perp / S^total per eigenstate vs Jpm
      (d) Anisotropy: S^perp/2 vs S^zz (isotropy check)
    """
    results_sf = [r for r in results if r['ps_Sq'] is not None
                  and r['ps_Szz'] is not None]
    if not results_sf:
        print("  No per-eigenstate SF data, skipping transverse moments")
        return

    n_q = results_sf[0]['ps_Sq'].shape[1]
    # Use K-point (index 4 for 3x3) or M-point (index 1)
    # Pick the q with the largest GS-averaged S(q) as the representative
    avg_sq = np.mean(np.vstack([r['ps_Sq'] for r in results_sf]), axis=0)
    q_rep = int(np.argmax(avg_sq))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        r'Per-GS-eigenstate transverse moments '
        rf'($q$ index {q_rep}, largest $S(q)$)',
        fontsize=14)

    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for r in results_sf:
        jpm = r['Jpm']
        sq = r['ps_Sq']     # (n_gs, n_q)
        szz = r['ps_Szz']   # (n_gs, n_q)

        for si in range(r['n_gs']):
            # (a) S^zz at each q
            for qi in range(n_q):
                ax_a.scatter(jpm, szz[si, qi], s=10, alpha=0.4,
                             edgecolors='none', zorder=3)

            # (b) S^perp at representative q
            sperp = sq[si, q_rep] - szz[si, q_rep]
            ax_b.scatter(jpm, sperp, c='tab:green', s=15, alpha=0.6,
                         edgecolors='none', zorder=3)

            # (c) Ratio S^perp / S^total at representative q
            s_total = sq[si, q_rep]
            ratio = sperp / s_total if abs(s_total) > 1e-12 else 0.0
            ax_c.scatter(jpm, ratio, c='tab:purple', s=15, alpha=0.6,
                         edgecolors='none', zorder=3)

            # (d) S^perp/2 vs S^zz (isotropy check: should be 1 if SU(2))
            szz_val = szz[si, q_rep]
            sperp_half = sperp / 2.0
            ax_d.scatter(szz_val, sperp_half, c='steelblue', s=15, alpha=0.6,
                         edgecolors='none', zorder=3)

    ax_a.set_xlabel(r'$J_{\pm}$')
    ax_a.set_ylabel(r'$S^{zz}(\mathbf{q})$')
    ax_a.set_title(r'(a) $S^{zz}(\mathbf{q})$ at all $q$ per GS eigenstate')
    ax_a.grid(True, alpha=0.3)

    ax_b.set_xlabel(r'$J_{\pm}$')
    ax_b.set_ylabel(r'$S^{\perp}(\mathbf{q})$')
    ax_b.set_title(rf'(b) Transverse SF $S^{{\perp}} = S - S^{{zz}}$ (q idx {q_rep})')
    ax_b.grid(True, alpha=0.3)

    ax_c.set_xlabel(r'$J_{\pm}$')
    ax_c.set_ylabel(r'$S^{\perp} / S^{\mathrm{total}}$')
    ax_c.set_title(rf'(c) Transverse fraction (q idx {q_rep})')
    ax_c.axhline(2/3, color='gray', ls='--', lw=1,
                 label=r'$2/3$ (SU(2) isotropic)')
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3)

    ax_d.set_xlabel(r'$S^{zz}(\mathbf{q})$')
    ax_d.set_ylabel(r'$S^{\perp}(\mathbf{q})/2$')
    ax_d.set_title(rf'(d) Isotropy check: $S^{{\perp}}/2$ vs $S^{{zz}}$ (q idx {q_rep})')
    lims = ax_d.get_xlim()
    ax_d.plot(lims, lims, 'k--', lw=0.8, label='isotropic')
    ax_d.legend(fontsize=8)
    ax_d.set_aspect('equal')
    ax_d.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'diagnostic_transverse_sf.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved diagnostic_transverse_sf")


# ============================================================
# Momentum helpers (for per-eigenstate SF labels)
# ============================================================
_A1 = np.array([1.0, 0.0])
_A2 = np.array([0.5, np.sqrt(3) / 2])
_det = _A1[0] * _A2[1] - _A1[1] * _A2[0]
_B1 = (2 * np.pi / _det) * np.array([_A2[1], -_A2[0]])
_B2 = (2 * np.pi / _det) * np.array([-_A1[1], _A1[0]])
_Q_K = (2 * _B1 + _B2) / 3
_Q_Kp = (_B1 + 2 * _B2) / 3
_Q_M = _B1 / 2


def _fold_into_bz(q):
    best_q, best_d = q.copy(), np.linalg.norm(q)
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            trial = q + n1 * _B1 + n2 * _B2
            d = np.linalg.norm(trial)
            if d < best_d - 1e-10:
                best_d, best_q = d, trial.copy()
    return best_q


def _build_q_info(LX, LY):
    """Build momentum labels and unique indices for discrete momenta."""
    q_list, q_labels = [], []
    for m1 in range(LX):
        for m2 in range(LY):
            q = (m1 / LX) * _B1 + (m2 / LY) * _B2
            q_f = _fold_into_bz(q)
            q_list.append(q_f)
            if np.linalg.norm(q_f) < 0.01:
                q_labels.append(r'$\Gamma$')
            elif (np.linalg.norm(q_f - _Q_K) < 0.01
                  or np.linalg.norm(q_f + _Q_K) < 0.01):
                q_labels.append(r'$K$')
            elif (np.linalg.norm(q_f - _Q_Kp) < 0.01
                  or np.linalg.norm(q_f + _Q_Kp) < 0.01):
                q_labels.append(r"$K'$")
            elif (np.linalg.norm(q_f - _Q_M) < 0.01
                  or np.linalg.norm(q_f + _Q_M) < 0.01):
                q_labels.append(r'$M$')
            else:
                q_labels.append(
                    rf'$\mathbf{{k}}(\frac{{{m1}}}{{{LX}}},'
                    rf'\frac{{{m2}}}{{{LY}}})$')
    discrete_q = np.array(q_list)
    # Unique indices (fold q ↔ -q)
    unique_idx, seen = [], []
    for idx, qf in enumerate(discrete_q):
        is_dup = False
        for sidx in seen:
            if (np.linalg.norm(qf - discrete_q[sidx]) < 0.01
                    or np.linalg.norm(qf + discrete_q[sidx]) < 0.01):
                is_dup = True
                break
        if not is_dup:
            unique_idx.append(idx)
            seen.append(idx)
    return q_labels, unique_idx


# ============================================================
# Per-eigenstate structure factors (spin + dimer)
# ============================================================

def plot_per_eigenstate_structure_factors(results, output_dir):
    """
    Per-eigenstate structure factors at discrete momenta vs Jpm.
    2×2 panels: (a) S(q), (b) S^zz(q), (c) D^full(q), (d) D^conn(q).
    Each unique q gets its own marker/color.
    """
    results_sf = [r for r in results if r['ps_Sq'] is not None]
    if not results_sf:
        print("  No per-eigenstate SF data, skipping")
        return

    n_q = results_sf[0]['ps_Sq'].shape[1]
    LX = LY = int(round(n_q ** 0.5))
    q_labels, unique_idx = _build_q_info(LX, LY)

    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    has_szz = any(r['ps_Szz'] is not None for r in results_sf)
    has_dimer_full = any(r.get('ps_Dq_full') is not None for r in results_sf)
    has_dimer_conn = any(r.get('ps_Dq_conn') is not None for r in results_sf)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        r'Per-GS-eigenstate structure factors at discrete momenta',
        fontsize=14)
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for qi_idx, qi in enumerate(unique_idx):
        c = colors[qi_idx % len(colors)]
        m = markers[qi_idx % len(markers)]
        lbl = q_labels[qi]

        for r_idx, r in enumerate(results_sf):
            jpm = r['Jpm']
            first = (r_idx == 0)
            for si in range(r['n_gs']):
                ax_a.scatter(jpm, r['ps_Sq'][si, qi],
                             c=[c], marker=m, s=20, alpha=0.6,
                             edgecolors='none', zorder=3,
                             label=lbl if (first and si == 0) else None)
                if has_szz and r['ps_Szz'] is not None:
                    ax_b.scatter(jpm, r['ps_Szz'][si, qi],
                                 c=[c], marker=m, s=20, alpha=0.6,
                                 edgecolors='none', zorder=3,
                                 label=lbl if (first and si == 0) else None)
                if has_dimer_full and r.get('ps_Dq_full') is not None:
                    ax_c.scatter(jpm, r['ps_Dq_full'][si, qi],
                                 c=[c], marker=m, s=20, alpha=0.6,
                                 edgecolors='none', zorder=3,
                                 label=lbl if (first and si == 0) else None)
                if has_dimer_conn and r.get('ps_Dq_conn') is not None:
                    ax_d.scatter(jpm, r['ps_Dq_conn'][si, qi],
                                 c=[c], marker=m, s=20, alpha=0.6,
                                 edgecolors='none', zorder=3,
                                 label=lbl if (first and si == 0) else None)

    ax_a.set_xlabel(r'$J_{\pm}$')
    ax_a.set_ylabel(r'$S(\mathbf{q})$')
    ax_a.set_title(r'(a) $S(\mathbf{q})$ per GS eigenstate')
    ax_a.legend(fontsize=7)
    ax_a.grid(True, alpha=0.3)

    ax_b.set_xlabel(r'$J_{\pm}$')
    ax_b.set_ylabel(r'$S^{zz}(\mathbf{q})$')
    ax_b.set_title(r'(b) $S^{zz}(\mathbf{q})$ per GS eigenstate')
    ax_b.legend(fontsize=7)
    ax_b.grid(True, alpha=0.3)
    if not has_szz:
        ax_b.text(0.5, 0.5, 'No $S^{zz}$ data',
                  transform=ax_b.transAxes, ha='center', va='center')

    ax_c.set_xlabel(r'$J_{\pm}$')
    ax_c.set_ylabel(r'$D^{\rm full}(\mathbf{q})$')
    ax_c.set_title(r'(c) $D^{\rm full}(\mathbf{q})$ per GS eigenstate')
    ax_c.legend(fontsize=7)
    ax_c.grid(True, alpha=0.3)
    if not has_dimer_full:
        ax_c.text(0.5, 0.5, 'No dimer SF data',
                  transform=ax_c.transAxes, ha='center', va='center')

    ax_d.set_xlabel(r'$J_{\pm}$')
    ax_d.set_ylabel(r'$D^{\rm conn}(\mathbf{q})$')
    ax_d.set_title(r'(d) $D^{\rm conn}(\mathbf{q})$ per GS eigenstate')
    ax_d.legend(fontsize=7)
    ax_d.grid(True, alpha=0.3)
    if not has_dimer_conn:
        ax_d.text(0.5, 0.5, 'No dimer SF data',
                  transform=ax_d.transAxes, ha='center', va='center')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir,
                    f'diagnostic_per_eigenstate_sf.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved diagnostic_per_eigenstate_sf")


# ============================================================
# RDM subsystem geometry
# ============================================================

# Subsystem definitions (mirrors CLUSTER_CONFIG in bfg_compute.py)
_SUBSYSTEMS_3x3 = {
    'hexagon': {
        'sites_list': [[1, 2, 3, 4, 9, 11]],
    },
    'bowtie': {
        'sites_list': [
            [9, 10, 11, 4, 12],
            [3, 5, 4, 11, 12],
            [4, 11, 12, 14, 13],
        ],
    },
}


def plot_rdm_subsystem_geometry(output_dir, num_sites=27):
    """Plot kagome lattice with RDM subsystem sites highlighted.

    One panel per (subsystem name, orientation).
    """
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2])
    offsets = np.array([[0, 0], [0.5, 0], [0.25, np.sqrt(3) / 4]])

    LX = LY = 3
    positions = []
    for i in range(LX):
        for j in range(LY):
            for s in range(3):
                positions.append(i * a1 + j * a2 + offsets[s])
    positions = np.array(positions)

    # Build panel list
    panels = []
    for name, sub_cfg in _SUBSYSTEMS_3x3.items():
        for oi, sites in enumerate(sub_cfg['sites_list']):
            panels.append((name, oi, sites))

    n_panels = len(panels)
    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.0 * nrows))
    if n_panels == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    sub_colors = {'hexagon': '#e74c3c', 'bowtie': '#2980b9'}

    for idx, (name, oi, sites) in enumerate(panels):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # All sites (gray background)
        ax.scatter(positions[:, 0], positions[:, 1],
                   c='lightgray', s=80, edgecolors='gray',
                   linewidths=0.5, zorder=4)
        # Highlight subsystem sites
        c = sub_colors.get(name, '#f39c12')
        ax.scatter(positions[sites, 0], positions[sites, 1],
                   c=c, s=160, edgecolors='black',
                   linewidths=1.2, zorder=5)
        # Label all sites
        for si in range(num_sites):
            in_sub = si in sites
            ax.annotate(str(si), positions[si],
                        fontsize=6 if in_sub else 5,
                        ha='center', va='center',
                        color='white' if in_sub else 'gray',
                        fontweight='bold' if in_sub else 'normal',
                        zorder=6)

        ax.set_aspect('equal')
        ax.set_title(f'{name} orient {oi}\nsites = {sites}', fontsize=9)
        ax.set_xlim(-0.3, 3.5)
        ax.set_ylim(-0.3, 3.0)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(r'RDM Subsystem Geometries ($3\times3$ PBC kagome)',
                 fontsize=14)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir,
                    f'rdm_subsystem_geometry.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved rdm_subsystem_geometry")


def main():
    print("Loading diagnostics data...")
    results = load_all_diagnostics()
    print(f"  {len(results)} Jpm values with diagnostics data")

    if not results:
        print("No data found!")
        sys.exit(1)

    for r in results:
        chi_info = ""
        if r['chirality'] is not None:
            chi_info = f"|κ|_max={np.max(np.abs(r['chirality'])):.2e}"
        sz_range = r['sz_local'][0].max() - r['sz_local'][0].min()
        print(f"  Jpm={r['Jpm']:+.2f}: n_gs={r['n_gs']:d}, "
              f"sz_range={sz_range:.4f}, {chi_info}")

    output_dir = os.path.join(OUTPUT_DIR, 'diagnostics')
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating plots...")
    plot_sz_local_summary(results, output_dir)
    plot_chirality_summary(results, output_dir)
    plot_sz_lattice_patterns(results, output_dir)
    plot_bond_energy(results, output_dir)
    plot_transverse_moments(results, output_dir)
    plot_per_eigenstate_structure_factors(results, output_dir)
    plot_rdm_subsystem_geometry(output_dir)
    print("\nDone. Output in:", output_dir)


if __name__ == '__main__':
    main()
