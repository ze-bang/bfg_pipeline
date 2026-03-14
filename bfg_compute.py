#!/usr/bin/env python3
"""
Unified BFG Kagome ED Post-Processing: Compute Phase
=====================================================
Replaces: analyze_BFG_3x3.py, compute_structure_factors_BFG_2x3.py,
          patch_per_state_szsz_bond.py

Usage:
  # SLURM worker mode (one Jpm at a time):
  python bfg_compute.py --cluster 3x3 --worker --index $SLURM_ARRAY_TASK_ID
  python bfg_compute.py --cluster 2x3 --worker --index 5

  # Sequential mode (all Jpm values):
  python bfg_compute.py --cluster 2x3
  python bfg_compute.py --cluster 3x3 --skip-fidelity

  # Specific Jpm:
  python bfg_compute.py --cluster 3x3 --jpm -0.10
"""
# Suppress multithreading (SLURM single-core workers)
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import time
import gc
import argparse
import csv
import numpy as np
import h5py  # only for reading input ed_results.h5 from Lanczos solver

# ============================================================
# Cluster configuration
# ============================================================
SCRATCH = '/scratch/zhouzb79'

CLUSTER_CONFIG = {
    '3x3': {
        'LX': 3, 'LY': 3,
        'NUM_SITES': 27, 'N_UC': 9,
        'N_UP_LIST': [13],
        'BASE_DIR': os.path.join(SCRATCH,
                                 'BFG_scan_symmetrized_pbc_3x3_nup13_negJpm'),
        'OUTPUT_DIR': os.path.join(SCRATCH, 'analysis_BFG_3x3'),
        'PRECOMPUTE_NUP': 13,       # n_up used for geometry files
        'DEG_TOL': 2e-5,            # Per-site degeneracy tolerance
        'SUBSYSTEMS': {
            'hexagon': {
                'sites_list': [[1, 2, 3, 4, 9, 11]],
                'dim': 64,
            },
            'bowtie': {
                'sites_list': [
                    [9, 10, 11, 4, 12],
                    [3, 5, 4, 11, 12],
                    [4, 11, 12, 14, 13],
                ],
                'dim': 32,
            },
        },
    },
    '3x3_to': {
        'LX': 3, 'LY': 3,
        'NUM_SITES': 27, 'N_UC': 9,
        'N_UP_LIST': [13],
        'BASE_DIR': os.path.join(SCRATCH,
                                 'BFG_scan_symmetrized_pbc_3x3_nup13_negJpm_translation_only'),
        'OUTPUT_DIR': os.path.join(SCRATCH, 'analysis_BFG_3x3_translation_only'),
        'PRECOMPUTE_NUP': 13,
        'DEG_TOL': 2e-5,
        'SUBSYSTEMS': {
            'hexagon': {
                'sites_list': [[1, 2, 3, 4, 9, 11]],
                'dim': 64,
            },
            'bowtie': {
                'sites_list': [
                    [9, 10, 11, 4, 12],
                    [3, 5, 4, 11, 12],
                    [4, 11, 12, 14, 13],
                ],
                'dim': 32,
            },
        },
    },
    '2x3': {
        'LX': 2, 'LY': 3,
        'NUM_SITES': 18, 'N_UC': 6,
        'N_UP_LIST': [8, 9],
        'BASE_DIR': os.path.join(SCRATCH,
                                 'BFG_scan_symmetrized_pbc_2x3_fixed_Sz'),
        'OUTPUT_DIR': os.path.join(SCRATCH, 'analysis_BFG_2x3'),
        'PRECOMPUTE_NUP': 9,
        'DEG_TOL': 1e-6,
        'SUBSYSTEMS': {},           # No RDM for 2×3
    },
}


def get_config(cluster):
    if cluster not in CLUSTER_CONFIG:
        print(f"ERROR: unknown cluster '{cluster}'. Use: {list(CLUSTER_CONFIG)}")
        sys.exit(1)
    cfg = CLUSTER_CONFIG[cluster].copy()
    cfg['cluster'] = cluster
    cfg['FULL_DIM'] = 1 << cfg['NUM_SITES']
    return cfg


# ============================================================
# Kagome lattice geometry (shared)
# ============================================================
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3) / 2])

# Reciprocal lattice vectors
_det = A1[0] * A2[1] - A1[1] * A2[0]
B1 = (2 * np.pi / _det) * np.array([A2[1], -A2[0]])
B2 = (2 * np.pi / _det) * np.array([-A1[1], A1[0]])

# High-symmetry points
Q_GAMMA = np.array([0.0, 0.0])
Q_K  = (2 * B1 + B2) / 3
Q_Kp = (B1 + 2 * B2) / 3
Q_M  = B1 / 2
Q_Mp = B2 / 2
Q_Mp2 = (B1 + B2) / 2

BZ_CORNERS = np.array([
    (2 * B1 + B2) / 3,
    (B1 + 2 * B2) / 3,
    (-B1 + B2) / 3,
    -(2 * B1 + B2) / 3,
    -(B1 + 2 * B2) / 3,
    (B1 - B2) / 3,
])


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


def build_discrete_momenta(LX, LY):
    """Build discrete momenta for an LX×LY supercell.

    Returns (DISCRETE_Q, DISCRETE_Q_LABELS, UNIQUE_Q_INDICES)
    """
    discrete_q = []
    labels = []
    for m1 in range(LX):
        for m2 in range(LY):
            q = (m1 / LX) * B1 + (m2 / LY) * B2
            q_f = _fold_into_bz(q)
            discrete_q.append(q_f)
            if np.linalg.norm(q_f) < 0.01:
                labels.append(r'$\Gamma$')
            elif np.linalg.norm(q_f - Q_K) < 0.01 or np.linalg.norm(q_f + Q_K) < 0.01:
                labels.append(r'$K$')
            elif np.linalg.norm(q_f - Q_Kp) < 0.01 or np.linalg.norm(q_f + Q_Kp) < 0.01:
                labels.append(r"$K'$")
            elif np.linalg.norm(q_f - Q_M) < 0.01 or np.linalg.norm(q_f + Q_M) < 0.01:
                labels.append(r'$M$')
            else:
                labels.append(
                    rf'$\mathbf{{k}}(\frac{{{m1}}}{{{LX}}},\frac{{{m2}}}{{{LY}}})$')
    discrete_q = np.array(discrete_q)

    # Unique momenta (modulo q ↔ -q)
    unique_indices = []
    seen = []
    for idx, qf in enumerate(discrete_q):
        is_dup = False
        for sidx in seen:
            if (np.linalg.norm(qf - discrete_q[sidx]) < 0.01 or
                    np.linalg.norm(qf + discrete_q[sidx]) < 0.01):
                is_dup = True
                break
        if not is_dup:
            unique_indices.append(idx)
            seen.append(idx)

    return discrete_q, labels, unique_indices


# ============================================================
# Geometry parsing (shared)
# ============================================================
def parse_positions(filepath):
    pos = []
    sublattices = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            sublattices.append(int(parts[2]))
            pos.append([float(parts[3]), float(parts[4])])
    return np.array(pos), np.array(sublattices)


def parse_nn_list(filepath):
    nn_dict = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = list(map(int, line.split()))
            site_id = parts[0]
            nn_dict[site_id] = parts[2:]
    pairs = set()
    for i, nbrs in nn_dict.items():
        for j in nbrs:
            pairs.add((min(i, j), max(i, j)))
    return nn_dict, sorted(pairs)


def minimum_image_displacement(dr, L1_vec, L2_vec):
    best_dr = dr.copy()
    best_d = np.linalg.norm(dr)
    for n1 in [-1, 0, 1]:
        for n2 in [-1, 0, 1]:
            trial = dr + n1 * L1_vec + n2 * L2_vec
            d = np.linalg.norm(trial)
            if d < best_d:
                best_d = d
                best_dr = trial
    return best_dr


def compute_min_image_dr_matrix(positions, L1_vec, L2_vec):
    N = len(positions)
    dr_raw = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dr_min = np.zeros_like(dr_raw)
    for i in range(N):
        for j in range(N):
            dr_min[i, j] = minimum_image_displacement(dr_raw[i, j], L1_vec, L2_vec)
    return dr_min


def compute_bond_midpoints(positions, nn_pairs, L1_vec, L2_vec):
    bond_positions = []
    bond_displacements = []
    for (i, j) in nn_pairs:
        dr = positions[j] - positions[i]
        dr_min = minimum_image_displacement(dr, L1_vec, L2_vec)
        mid = positions[i] + dr_min / 2
        bond_positions.append(mid)
        bond_displacements.append(dr_min)
    return np.array(bond_positions), np.array(bond_displacements)


def compute_min_image_bond_dr_matrix(bond_positions, L1_vec, L2_vec):
    N_b = len(bond_positions)
    dR_raw = bond_positions[:, np.newaxis, :] - bond_positions[np.newaxis, :, :]
    dR = np.zeros_like(dR_raw)
    for a in range(N_b):
        for b in range(N_b):
            dR[a, b] = minimum_image_displacement(dR_raw[a, b], L1_vec, L2_vec)
    return dR


def load_geometry(cfg):
    """Load positions, NN list, and compute displacement matrices."""
    LX, LY = cfg['LX'], cfg['LY']
    L1_vec = LX * A1
    L2_vec = LY * A2

    precompute_dir = os.path.join(
        cfg['BASE_DIR'],
        f'precompute_ref_ham_n_up_{cfg["PRECOMPUTE_NUP"]}')
    pos_path = os.path.join(precompute_dir, 'positions.dat')
    nn_path = os.path.join(precompute_dir,
                           f'kagome_bfg_{LX}x{LY}_pbc_nn_list.dat')

    positions, sublattices = parse_positions(pos_path)
    nn_dict, nn_pairs = parse_nn_list(nn_path)
    bond_pos, bond_disp = compute_bond_midpoints(positions, nn_pairs,
                                                  L1_vec, L2_vec)
    dr_min = compute_min_image_dr_matrix(positions, L1_vec, L2_vec)
    dR_min = compute_min_image_bond_dr_matrix(bond_pos, L1_vec, L2_vec)

    return {
        'positions': positions,
        'sublattices': sublattices,
        'nn_dict': nn_dict,
        'nn_pairs': nn_pairs,
        'bond_positions': bond_pos,
        'bond_displacements': bond_disp,
        'dr_min': dr_min,
        'dR_min': dR_min,
        'L1': L1_vec,
        'L2': L2_vec,
    }


# ============================================================
# Data loading
# ============================================================
def discover_jpm_values(cfg):
    """Find all Jpm values with complete data."""
    jpms = set()
    for n_up in cfg['N_UP_LIST']:
        base = os.path.join(cfg['BASE_DIR'], f'n_up={n_up}')
        if not os.path.isdir(base):
            continue
        for d in os.listdir(base):
            if not d.startswith('Jpm='):
                continue
            jpm_str = d.split('=', 1)[1]
            mapping = os.path.join(base, f'Jpm={jpm_str}', 'results',
                                   'eigenvalue_mapping.txt')
            h5 = os.path.join(base, f'Jpm={jpm_str}', 'results',
                              'ed_results.h5')
            if os.path.exists(mapping) and os.path.exists(h5):
                jpms.add(jpm_str)
    return sorted(jpms, key=float)


def read_eigenvalue_mapping(cfg, n_up, jpm_str):
    """Parse eigenvalue_mapping.txt for a given (n_up, Jpm)."""
    mapping_file = os.path.join(cfg['BASE_DIR'], f'n_up={n_up}',
                                f'Jpm={jpm_str}', 'results',
                                'eigenvalue_mapping.txt')
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
                    'energy': float(parts[1]),
                    'sector': int(parts[2]),
                    'local_idx': int(parts[3]),
                    'h5_key': parts[4],
                    'n_up': n_up,
                })
    entries.sort(key=lambda x: x['energy'])
    return entries


def find_state_manifolds(cfg, jpm_str):
    """Find GS and 1st excited state manifolds across all n_up sectors.

    Returns (gs_info, ex_info) as dicts or (None, None).
    """
    all_entries = []
    for n_up in cfg['N_UP_LIST']:
        all_entries.extend(read_eigenvalue_mapping(cfg, n_up, jpm_str))

    if not all_entries:
        return None, None

    all_entries.sort(key=lambda x: x['energy'])
    E0 = all_entries[0]['energy']

    abs_tol = cfg['DEG_TOL'] * cfg['NUM_SITES']
    gs_entries = [e for e in all_entries if abs(e['energy'] - E0) < abs_tol]

    remaining = [e for e in all_entries if e['energy'] > E0 + abs_tol]
    if remaining:
        E1 = remaining[0]['energy']
        ex_entries = [e for e in remaining if abs(e['energy'] - E1) < abs_tol]
    else:
        E1 = None
        ex_entries = []

    all_evals = np.array([e['energy'] for e in all_entries])
    gap = float(E1 - E0) if E1 is not None else 0.0

    gs_info = {'E0': E0, 'entries': gs_entries, 'n_gs': len(gs_entries),
               'all_evals': all_evals}
    ex_info = {'E1': E1, 'entries': ex_entries, 'n_ex': len(ex_entries),
               'gap': gap}
    return gs_info, ex_info


def load_eigenvector(cfg, n_up, jpm_str, h5_key):
    """Load eigenvector from ed_results.h5 (full Hilbert space)."""
    h5_path = os.path.join(cfg['BASE_DIR'], f'n_up={n_up}',
                           f'Jpm={jpm_str}', 'results', 'ed_results.h5')
    with h5py.File(h5_path, 'r') as f:
        ed = f['eigendata']
        if h5_key in ed:
            data = ed[h5_key][:]
        else:
            num = int(h5_key.split('_')[1])
            if num >= 1000000:
                alt = f'eigenvector_{num // 100}'
            else:
                alt = f'eigenvector_{num * 100}'
            data = ed[alt][:]

    if data.dtype.names and 'real' in data.dtype.names:
        vec = data['real'] + 1j * data['imag']
    else:
        vec = data.astype(np.complex128)
    del data

    nrm = np.linalg.norm(vec)
    if nrm > 0:
        vec /= nrm
    return vec


# ============================================================
# Correlation functions — reduced Sz-sector
# ============================================================
def _sz_sector_basis(N, n_up):
    """Generate all bit patterns of N bits with exactly n_up set bits (sorted)."""
    if n_up == 0:
        return np.array([0], dtype=np.int64)
    basis = []
    x = (1 << n_up) - 1
    limit = 1 << N
    while x < limit:
        basis.append(x)
        c = x & -x
        r = x + c
        x = (((r ^ x) >> 2) // c) | r
    return np.array(basis, dtype=np.int64)


def compute_spin_correlations(basis, psi_sector, N):
    """Compute spin correlations in Sz-sector representation.

    basis: sorted int64 array of basis states (bit patterns)
    psi_sector: complex128 coefficients at those states

    Exploits SpSm Hermiticity: SpSm[j,i] = conj(SpSm[i,j]),
    so only upper triangle is computed (halves searchsorted calls).
    """
    prob = np.abs(psi_sector) ** 2

    # Precompute bit arrays (avoids redundant extraction in loops)
    bits = np.empty((N, len(basis)), dtype=np.int8)
    for i in range(N):
        bits[i] = (basis >> i) & 1

    b_exp = np.zeros(N)
    for i in range(N):
        b_exp[i] = np.dot(prob, bits[i])
    sz = b_exp - 0.5

    SzSz = np.zeros((N, N))
    for i in range(N):
        SzSz[i, i] = 0.25
        for j in range(i + 1, N):
            val = np.dot(prob, bits[i] * bits[j]) \
                  - b_exp[i] / 2 - b_exp[j] / 2 + 0.25
            SzSz[i, j] = SzSz[j, i] = val

    # SpSm is Hermitian: (S+_i S-_j)† = S+_j S-_i, so only upper triangle
    SpSm = np.zeros((N, N), dtype=complex)
    for i in range(N):
        SpSm[i, i] = b_exp[i]
    for i in range(N):
        for j in range(i + 1, N):
            flip_mask = np.int64((1 << i) | (1 << j))
            valid = (bits[i] == 0) & (bits[j] == 1)
            src = np.where(valid)[0]
            tgt_states = basis[src] ^ flip_mask
            tgt = np.searchsorted(basis, tgt_states)
            SpSm[i, j] = np.dot(np.conj(psi_sector[tgt]),
                                psi_sector[src])
            SpSm[j, i] = np.conj(SpSm[i, j])

    del bits

    SmSp = np.conj(SpSm.T)
    SiSj = SzSz + 0.5 * np.real(SpSm + SmSp)
    return SzSz, SpSm, SiSj, sz


# ============================================================
# Bond operator
# ============================================================
def _apply_bond(basis, psi_sector, i, j):
    """Apply S_i · S_j in Sz-sector representation."""
    bi = (basis >> i) & 1
    bj = (basis >> j) & 1
    phi = ((bi - 0.5) * (bj - 0.5)).astype(np.complex128) * psi_sector
    flip_mask = np.int64((1 << i) | (1 << j))
    valid_pm = (bi == 0) & (bj == 1)
    src_pm = np.where(valid_pm)[0]
    tgt_pm = np.searchsorted(basis, basis[src_pm] ^ flip_mask)
    phi[tgt_pm] += 0.5 * psi_sector[src_pm]
    valid_mp = (bi == 1) & (bj == 0)
    src_mp = np.where(valid_mp)[0]
    tgt_mp = np.searchsorted(basis, basis[src_mp] ^ flip_mask)
    phi[tgt_mp] += 0.5 * psi_sector[src_mp]
    return phi


# ============================================================
# Bond-bond correlations
# ============================================================
def compute_bond_bond_correlations(basis, psi_sector, nn_pairs, N):
    """Compute C_{ab} = <B_a B_b> and B_mean = <B_a>."""
    N_b = len(nn_pairs)
    dim_sector = len(psi_sector)

    phi_array = np.empty((N_b, dim_sector), dtype=np.complex128)
    B_mean = np.zeros(N_b)
    for a, (i, j) in enumerate(nn_pairs):
        phi_array[a] = _apply_bond(basis, psi_sector, i, j)
        B_mean[a] = np.dot(np.conj(psi_sector), phi_array[a]).real

    C = (phi_array @ np.conj(phi_array).T).real

    del phi_array
    gc.collect()
    return C, B_mean


# ============================================================
# Structure factors (shared)
# ============================================================
def compute_sq_at_points(corr_matrix, dr_min, q_points):
    N = corr_matrix.shape[0]
    result = []
    for q in q_points:
        phase = np.exp(1j * (q[0] * dr_min[:, :, 0] + q[1] * dr_min[:, :, 1]))
        result.append(np.sum(corr_matrix * phase).real / N)
    return np.array(result)


def compute_dq_at_points(C_bb, B_mean, dR_min, q_points):
    N_b = len(B_mean)
    D_conn_mat = C_bb - np.outer(B_mean, B_mean)
    conn_list = []
    full_list = []
    for q in q_points:
        phase = np.exp(1j * (q[0] * dR_min[:, :, 0] + q[1] * dR_min[:, :, 1]))
        conn_list.append(np.sum(D_conn_mat * phase).real / N_b)
        full_list.append(np.sum(C_bb * phase).real / N_b)
    return np.array(conn_list), np.array(full_list)


def compute_nematic_op(B_mean, nn_pairs, sublattices):
    omega = np.exp(2j * np.pi / 3)
    bond_types = {(0, 1): [], (0, 2): [], (1, 2): []}
    for b_idx, (i, j) in enumerate(nn_pairs):
        si, sj = min(sublattices[i], sublattices[j]), max(sublattices[i], sublattices[j])
        bond_types[(si, sj)].append(B_mean[b_idx])
    B01 = np.mean(bond_types[(0, 1)])
    B02 = np.mean(bond_types[(0, 2)])
    B12 = np.mean(bond_types[(1, 2)])
    Phi = B01 + B02 * omega + B12 * omega**2
    return np.abs(Phi), np.angle(Phi), B01, B02, B12


# ============================================================
# RDM computation (shared)
# ============================================================
def compute_rdm(nz_idx, psi_nz, subsystem_sites, n_total):
    """Compute RDM from sparse representation."""
    from scipy.sparse import csr_matrix

    n_sub = len(subsystem_sites)
    dim_sub = 2 ** n_sub
    env_sites = sorted(set(range(n_total)) - set(subsystem_sites))
    n_nz = len(nz_idx)

    sub_config = np.zeros(n_nz, dtype=np.int64)
    for j, site in enumerate(subsystem_sites):
        sub_config |= (((nz_idx >> site) & 1).astype(np.int64)) << j

    env_config = np.zeros(n_nz, dtype=np.int64)
    for k, site in enumerate(env_sites):
        env_config |= (((nz_idx >> site) & 1).astype(np.int64)) << k

    unique_env, env_local = np.unique(env_config, return_inverse=True)
    M = csr_matrix((psi_nz, (sub_config, env_local)),
                   shape=(dim_sub, len(unique_env)))
    return (M @ M.conj().T).toarray()


def entanglement_entropy(rdm, tol=1e-14):
    eigenvalues = np.linalg.eigvalsh(rdm)
    eigenvalues = eigenvalues[eigenvalues > tol]
    S_vn = -np.sum(eigenvalues * np.log(eigenvalues))
    S_R2 = -np.log(np.sum(eigenvalues ** 2))
    return S_vn, S_R2, eigenvalues


# ============================================================
# Per-eigenstate SzSz bond energy
# ============================================================
def compute_per_state_szsz_bond(basis, psi_sector, nn_pairs, N):
    """Compute diagonal <S_i^z S_j^z> for each NN bond."""
    N_b = len(nn_pairs)
    szsz_bond = np.zeros(N_b)
    prob = np.abs(psi_sector) ** 2
    for b_idx, (i, j) in enumerate(nn_pairs):
        bi = ((basis >> i) & 1).astype(np.float64) - 0.5
        bj = ((basis >> j) & 1).astype(np.float64) - 0.5
        szsz_bond[b_idx] = np.dot(prob, bi * bj)
    return szsz_bond


# ============================================================
# Per-eigenstate scalar chirality
# ============================================================
def compute_triangles(nn_pairs, N):
    """Find all triangles from NN pairs."""
    from collections import defaultdict
    adj = defaultdict(set)
    for i, j in nn_pairs:
        adj[i].add(j)
        adj[j].add(i)
    triangles = []
    for i in range(N):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k <= j or k not in adj[i]:
                    continue
                triangles.append((i, j, k))
    return triangles


def compute_scalar_chirality(psi, triangles, N):
    """Compute <S_i · (S_j × S_k)> for each triangle."""
    # Placeholder: full chirality requires 3-point correlators
    chirality = np.zeros(len(triangles))
    return chirality


# ============================================================
# Core per-eigenstate computation
# ============================================================
def compute_single_eigenstate(cfg, entry, jpm_str, geo, discrete_q,
                              basis, subsystems=None):
    """Compute ALL observables for a single eigenstate.

    Returns a dict with every per-state quantity:
      correlations, structure factors, bond patterns, ⟨Sz⟩,
      nematic OP, RDM (if subsystems provided).
    """
    N = cfg['NUM_SITES']
    nn_pairs = geo['nn_pairs']
    sublattices = geo['sublattices']

    # --- Load eigenvector, extract Sz sector ---
    psi_full = load_eigenvector(cfg, entry['n_up'], jpm_str,
                                entry['h5_key'])
    psi_sector = psi_full[basis].copy()
    del psi_full  # free ~2 GB immediately

    # --- Spin correlations ---
    SzSz, SpSm, SiSj, sz = compute_spin_correlations(basis, psi_sector, N)

    # --- Bond correlations ---
    C_bb, B_mean = compute_bond_bond_correlations(
        basis, psi_sector, nn_pairs, N)
    szsz_bond = compute_per_state_szsz_bond(basis, psi_sector, nn_pairs, N)

    # --- Structure factors (spin + dimer) ---
    Sq = compute_sq_at_points(SiSj, geo['dr_min'], discrete_q)
    Szz = compute_sq_at_points(SzSz, geo['dr_min'], discrete_q)
    Dq_conn, Dq_full = compute_dq_at_points(
        C_bb, B_mean, geo['dR_min'], discrete_q)

    # --- Nematic OP ---
    nem_abs, nem_arg, B01, B02, B12 = compute_nematic_op(
        B_mean, nn_pairs, sublattices)

    result = {
        'sector': entry['sector'],
        # Matrices
        'SzSz': SzSz, 'SpSm': SpSm, 'SiSj': SiSj,
        'C_bb': C_bb,
        # Vectors
        'sz': sz, 'B_mean': B_mean, 'SzSz_bond': szsz_bond,
        # Structure factors
        'Sq': Sq, 'Szz': Szz, 'Dq_full': Dq_full, 'Dq_conn': Dq_conn,
        # Nematic
        'nem_abs': nem_abs, 'nem_arg': nem_arg,
        'B01': B01, 'B02': B02, 'B12': B12,
    }

    # --- RDM (if subsystems specified) ---
    # Per-eigenstate: store raw ρ(ψ) WITHOUT spin-flip averaging.
    # Spin-flip averaging is done only for the final GS-averaged RDM
    # in average_per_state_results().
    if subsystems:
        nz = basis                  # nonzero indices in full space
        psi_nz = psi_sector         # nonzero coefficients

        rdm_per_state = {}
        for name, sub_cfg in subsystems.items():
            for oi, sites in enumerate(sub_cfg['sites_list']):
                try:
                    rdm = compute_rdm(nz, psi_nz, sites, N)
                    S_vn, S_R2, spectrum = entanglement_entropy(rdm)
                    rdm_per_state[(name, oi)] = {
                        'rdm': rdm, 'spectrum': spectrum,
                        'S_vn': S_vn, 'S_R2': S_R2,
                        'trace': np.trace(rdm).real,
                        'sites': sites,
                    }
                except Exception as e:
                    print(f"      ERROR RDM {name}_o{oi}: {e}")
                    continue
        result['rdm'] = rdm_per_state

    del psi_sector
    gc.collect()
    return result


def average_per_state_results(per_state_list, subsystems=None):
    """Average per-eigenstate results to get GS manifold quantities.

    Averaging is linear — the GS-averaged S(q) IS the mean of
    per-eigenstate S(q), and similarly for all other quantities.
    """
    n = len(per_state_list)

    # Average arrays by key
    array_keys = ['SzSz', 'SiSj', 'C_bb', 'sz', 'B_mean',
                  'Sq', 'Szz', 'Dq_full', 'Dq_conn']
    result = {}
    for key in array_keys:
        result[key] = np.mean([s[key] for s in per_state_list], axis=0)

    # SpSm is complex
    result['SpSm'] = np.mean([s['SpSm'] for s in per_state_list], axis=0)

    # Nematic OP from averaged bond energies
    result['nem_abs'] = np.mean([s['nem_abs'] for s in per_state_list])
    result['nem_arg'] = np.mean([s['nem_arg'] for s in per_state_list])
    result['B01'] = np.mean([s['B01'] for s in per_state_list])
    result['B02'] = np.mean([s['B02'] for s in per_state_list])
    result['B12'] = np.mean([s['B12'] for s in per_state_list])

    # SF are already in result as Sq, Szz, etc. — just the averaged versions
    # Rename to match the expected keys in save/plot code
    result['Sq_disc'] = result.pop('Sq')
    result['Szz_disc'] = result.pop('Szz')
    result['Dq_full_disc'] = result.pop('Dq_full')
    result['Dq_disc'] = result.pop('Dq_conn')

    # Per-state arrays (stacked)
    result['per_state_Sq'] = np.array([s['Sq'] for s in per_state_list])
    result['per_state_Szz'] = np.array([s['Szz'] for s in per_state_list])
    result['per_state_Dq_full'] = np.array(
        [s['Dq_full'] for s in per_state_list])
    result['per_state_Dq_conn'] = np.array(
        [s['Dq_conn'] for s in per_state_list])
    result['per_state_B_mean'] = np.array(
        [s['B_mean'] for s in per_state_list])
    result['per_state_sz_local'] = np.array(
        [s['sz'] for s in per_state_list])
    result['per_state_SzSz_bond'] = np.array(
        [s['SzSz_bond'] for s in per_state_list])
    result['per_state_sector'] = np.array(
        [s['sector'] for s in per_state_list], dtype=int)
    result['per_state_SiSj'] = np.array(
        [s['SiSj'] for s in per_state_list])
    result['per_state_SzSz'] = np.array(
        [s['SzSz'] for s in per_state_list])

    # RDM: average per-eigenstate raw RDMs + spin-flip for final result.
    # Per-eigenstate RDMs are raw ρ(ψ) (no spin-flip).
    # The GS-averaged RDM is (1/n) Σ_k (ρ(ψ_k) + ρ(ψ̃_k)) / 2,
    # where ψ̃ is the global spin-flip of ψ.
    if subsystems and all('rdm' in s for s in per_state_list):
        rdm_results = {}
        # Collect all (name, oi) keys that appear in all states
        all_keys = set(per_state_list[0].get('rdm', {}).keys())
        for s in per_state_list[1:]:
            all_keys &= set(s.get('rdm', {}).keys())

        # We need flip_mask and basis for spin-flip RDMs.
        # Reconstruct from cfg passed through subsystems.
        # Actually, we stored raw per-state RDMs; to get spin-flip
        # copies we need to recompute them.  Instead, store per-state
        # raw ρ(ψ) as-is and build the averaged+flipped RDM from
        # the per-eigenstate wavefunctions stored earlier.
        # However wavefunctions are not stored here — so we build
        # the spin-flipped RDMs from the raw ones using the property
        # that ρ_A(ψ̃) = P ρ_A(ψ) P, where P is the subsystem
        # spin-flip operator (bit-flip in Fock basis = self-inverse
        # permutation matrix).
        n_total_rdm = 2 * n  # each state contributes ψ + ψ̃
        for key in all_keys:
            per_rdm = [s['rdm'][key] for s in per_state_list]
            sites = per_rdm[0]['sites']
            n_sub = len(sites)
            dim_sub = 1 << n_sub
            # P = antidiag permutation that flips all subsystem spins
            P = np.zeros((dim_sub, dim_sub))
            for idx in range(dim_sub):
                flipped = ((1 << n_sub) - 1) ^ idx
                P[flipped, idx] = 1.0
            # GS-averaged RDM with spin-flip
            rdm_avg = np.zeros_like(per_rdm[0]['rdm'])
            for r_i in per_rdm:
                rdm_raw = r_i['rdm']
                rdm_flip = P @ rdm_raw @ P  # ρ_A(ψ̃) = P ρ_A(ψ) P
                rdm_avg += (rdm_raw + rdm_flip) / 2.0
            rdm_avg /= n
            S_vn, S_R2, spectrum = entanglement_entropy(rdm_avg)
            tr = np.trace(rdm_avg).real
            if abs(tr - 1.0) > 1e-4:
                name, oi = key
                print(f"      WARNING: {name}_o{oi} Tr={tr:.8f} "
                      f"deviates from 1!")
            rdm_results[key] = {
                'rdm': rdm_avg, 'spectrum': spectrum,
                'S_vn': S_vn, 'S_R2': S_R2,
                'trace': tr,
                'sites': sites,
                'per_state_rdm': [r['rdm'] for r in per_rdm],
                'per_state_S_vn': np.array([r['S_vn'] for r in per_rdm]),
                'per_state_S_R2': np.array([r['S_R2'] for r in per_rdm]),
                'per_state_spectra': [r['spectrum'] for r in per_rdm],
            }
            name, oi = key
            result[f'{name}_o{oi}_S_vn'] = S_vn
            result[f'{name}_o{oi}_S_R2'] = S_R2
            result[f'{name}_o{oi}_trace'] = tr
        result['rdm_results'] = rdm_results
        result['n_total_rdm'] = n_total_rdm

    return result


def process_one_jpm(cfg, jpm_str, geo, discrete_q, q_labels, unique_q_idx):
    """Process one Jpm value: GS + 1st excited state.

    Clean logic:
      1. Per-eigenstate: compute ALL quantities for each GS eigenstate
      2. Average: trivially mean the per-state results
      3. Repeat (without per-state storage) for 1st excited state

    Returns (gs_result, ex_result) or (None, None).
    """
    N = cfg['NUM_SITES']
    t_total = time.time()

    gs_info, ex_info = find_state_manifolds(cfg, jpm_str)
    if gs_info is None:
        print(f"  WARNING: no data for Jpm={jpm_str}")
        return None, None

    E0 = gs_info['E0']
    n_gs = gs_info['n_gs']
    gap = ex_info['gap']
    all_evals = gs_info['all_evals']
    subsystems = cfg.get('SUBSYSTEMS', {})

    print(f"  Jpm={jpm_str}: E0={E0:.8f} ({E0/N:.8f}/site), "
          f"gap={gap:.8f}, n_gs={n_gs}")

    # Pre-compute Sz-sector bases
    n_up_values = sorted(set(e['n_up'] for e in gs_info['entries']))
    if ex_info['entries']:
        n_up_values = sorted(set(n_up_values) |
                             set(e['n_up'] for e in ex_info['entries']))
    bases = {n_up: _sz_sector_basis(N, n_up) for n_up in n_up_values}
    for n_up, b in bases.items():
        print(f"      Sz-sector basis n_up={n_up}: "
              f"{len(b)} states (vs 2^{N}={1<<N})")

    # ---- Step 1: Per-eigenstate GS computation ----
    print(f"    Computing per-eigenstate observables ({n_gs} GS states)...")
    per_state_list = []
    for si, entry in enumerate(gs_info['entries']):
        t0 = time.time()
        basis = bases[entry['n_up']]
        r = compute_single_eigenstate(
            cfg, entry, jpm_str, geo, discrete_q, basis,
            subsystems=subsystems or None)
        per_state_list.append(r)
        dt = time.time() - t0
        rdm_info = ''
        if 'rdm' in r:
            rdm_info = ', '.join(
                f'{k[0]}_o{k[1]}: S_vN={v["S_vn"]:.4f}'
                for k, v in r['rdm'].items())
            rdm_info = f'  RDM: {rdm_info}'
        print(f"      [{si+1}/{n_gs}] {entry['h5_key']} "
              f"(n_up={entry['n_up']}): {dt:.1f}s{rdm_info}")

    # ---- Step 2: Average over GS manifold (trivial) ----
    print(f"    Averaging over {n_gs} GS eigenstates...")
    gs_avg = average_per_state_results(per_state_list, subsystems)

    # Triangles (geometry, same for all states)
    triangles = compute_triangles(geo['nn_pairs'], N)
    if triangles:
        gs_avg['triangle_list'] = np.array(triangles, dtype=int)

    gs_result = {
        'Jpm': float(jpm_str), 'E0': E0, 'gap': gap,
        'n_gs': n_gs, 'n_use': n_gs,
        'all_evals': all_evals,
        'S_rho_GS': np.log(n_gs) if n_gs > 0 else 0.0,
        **gs_avg,
    }

    disc_info = ', '.join(
        f'S({q_labels[i].strip("$")})={gs_avg["Sq_disc"][i]:.4f}'
        for i in unique_q_idx)
    print(f"    GS: |Φ_nem|={gs_avg['nem_abs']:.4f}, {disc_info}")

    # ---- 1st excited state (no per-state storage, no RDM) ----
    ex_result = None
    if ex_info['E1'] is not None and ex_info['n_ex'] > 0:
        E1 = ex_info['E1']
        n_ex = ex_info['n_ex']
        print(f"    1st excited: E1={E1:.8f}, n_ex={n_ex}")
        print(f"    Computing per-eigenstate observables "
              f"({n_ex} ES states)...")
        ex_per_state = []
        for si, entry in enumerate(ex_info['entries']):
            t0 = time.time()
            basis = bases[entry['n_up']]
            r = compute_single_eigenstate(
                cfg, entry, jpm_str, geo, discrete_q, basis,
                subsystems=None)  # no RDM for excited states
            ex_per_state.append(r)
            dt = time.time() - t0
            print(f"      [{si+1}/{n_ex}] {entry['h5_key']} "
                  f"(n_up={entry['n_up']}): {dt:.1f}s")

        print(f"    Averaging over {n_ex} ES eigenstates...")
        ex_avg = average_per_state_results(ex_per_state)

        ex_result = {
            'Jpm': float(jpm_str), 'E0': E1, 'gap': gap,
            'n_gs': n_ex, 'n_use': n_ex,
            'all_evals': all_evals,
            **ex_avg,
        }
        # Drop per-state arrays for ES (not needed)
        for key in list(ex_result.keys()):
            if key.startswith('per_state_'):
                del ex_result[key]

    dt_total = time.time() - t_total
    print(f"    Total: {dt_total:.0f}s ({dt_total/60:.1f}min)")

    return gs_result, ex_result


# ============================================================
# Save/load helpers for .dat text format
# ============================================================
def _save_metadata(filepath, meta_dict, header_lines=None):
    """Save scalar metadata as key = value text file."""
    with open(filepath, 'w') as f:
        if header_lines:
            for line in header_lines:
                f.write(f"# {line}\n")
            f.write("#\n")
        for key, val in meta_dict.items():
            if isinstance(val, float):
                f.write(f"{key} = {val:.15e}\n")
            else:
                f.write(f"{key} = {val}\n")


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


# ============================================================
# Save results — unified per-Jpm .dat files
# ============================================================
def save_per_jpm_result(gs_result, ex_result, per_jpm_dir, jpm_str,
                        cfg, geo, discrete_q, q_labels):
    """Save per-Jpm intermediate results as .dat text files.

    Directory structure:
      per_jpm/Jpm_{jpm_str}/
        gs/  (or ex/)
          metadata.dat         — scalar parameters
          SzSz.dat, SiSj.dat   — array datasets
          per_state/           — per-eigenstate arrays
          rdm/                 — RDM data per subsystem
    """
    N = cfg['NUM_SITES']
    nn_pairs = geo['nn_pairs']
    N_b = len(nn_pairs)
    n_q = len(discrete_q)
    # Cleaned q labels for headers (strip $ and LaTeX)
    ql = [l.replace('$', '').replace('\\', '') for l in q_labels]
    q_col_header = '  '.join(f'q{i}={ql[i]}' for i in range(n_q))

    jpm_dir = os.path.join(per_jpm_dir, f'Jpm_{jpm_str}')

    for group_name, r in [('gs', gs_result), ('ex', ex_result)]:
        if r is None:
            continue
        gdir = os.path.join(jpm_dir, group_name)
        os.makedirs(gdir, exist_ok=True)

        state_label = 'Ground state' if group_name == 'gs' else '1st excited'
        n_eig = r.get('n_gs', 1)  # always an int

        # Scalar metadata
        meta = {}
        for attr in ['Jpm', 'E0', 'gap', 'n_gs', 'n_use',
                     'nem_abs', 'nem_arg', 'B01', 'B02', 'B12']:
            if attr in r:
                meta[attr] = r[attr]
        if 'n_total_rdm' in r:
            meta['n_total_rdm'] = r['n_total_rdm']
        for key in r:
            if key.endswith('_S_vn') or key.endswith('_S_R2') \
                    or key.endswith('_trace'):
                meta[key] = r[key]
        meta_header = [
            f"BFG Kagome ED — {state_label} manifold metadata",
            f"Cluster: {cfg['cluster']} ({N} sites, {cfg['N_UC']} unit cells)",
            f"Jpm = {jpm_str}",
            "",
            "Key definitions:",
            "  Jpm        — nearest-neighbor transverse coupling J±",
            "  E0         — absolute ground state energy",
            "  gap        — energy gap to 1st excited manifold",
            "  n_gs       — number of degenerate states in manifold",
            "  n_use      — number of states used for averaging",
            "  nem_abs    — |Φ_nematic| (nematic order parameter magnitude)",
            "  nem_arg    — arg(Φ_nematic) in radians",
            "  B01,B02,B12 — mean bond energy by sublattice pair (0-1, 0-2, 1-2)",
            "  *_S_vn     — von Neumann entanglement entropy (from GS-averaged+spin-flipped RDM)",
            "  *_S_R2     — Rényi-2 entanglement entropy (from GS-averaged+spin-flipped RDM)",
            "  *_trace    — Tr(ρ_A) for RDM subsystem (should be 1.0)",
            "  n_total_rdm — total density matrices that enter the average (= 2 × n_gs)",
        ]
        _save_metadata(os.path.join(gdir, 'metadata.dat'), meta,
                       header_lines=meta_header)

        # --- Array datasets ---
        # Site-site correlation matrices
        site_hdr = (f"Jpm={jpm_str}, {state_label}, "
                    f"averaged over {n_eig} eigenstates\n"
                    f"N={N} sites, matrix shape ({N} x {N})\n"
                    f"Row i, Column j → correlator(site_i, site_j)")

        for ds, desc in [
            ('SzSz', f'<S^z_i S^z_j> spin-spin correlation matrix.\n{site_hdr}'),
            ('SiSj', f'<S_i · S_j> Heisenberg correlation matrix.\n{site_hdr}'),
            ('C_bb', f'<B_a B_b> bond-bond correlation matrix.\n'
                     f'Jpm={jpm_str}, {state_label}, '
                     f'averaged over {n_eig} eigenstates\n'
                     f'{N_b} NN bonds, matrix shape ({N_b} x {N_b})\n'
                     f'Row a, Column b → <(S_i·S_j)(S_k·S_l)>'),
        ]:
            if ds in r:
                np.savetxt(os.path.join(gdir, f'{ds}.dat'),
                           np.atleast_2d(r[ds]), header=desc)

        # Vectors
        vec_descs = {
            'sz': (f'<S^z_i> local magnetization per site.\n'
                   f'Jpm={jpm_str}, {state_label}, '
                   f'averaged over {n_eig} eigenstates\n'
                   f'{N} values, one per site (site index 0..{N-1})'),
            'B_mean': (f'<S_i · S_j> bond expectation per NN bond.\n'
                       f'Jpm={jpm_str}, {state_label}, '
                       f'averaged over {n_eig} eigenstates\n'
                       f'{N_b} values, one per NN bond\n'
                       f'Bond index → (site_i, site_j): ' +
                       ', '.join(f'{a}→({i},{j})'
                                 for a, (i, j) in enumerate(nn_pairs))),
        }
        for ds, desc in vec_descs.items():
            if ds in r:
                np.savetxt(os.path.join(gdir, f'{ds}.dat'),
                           np.atleast_2d(r[ds]), header=desc)

        # Structure factors at discrete momenta
        sf_q_hdr = (f'Jpm={jpm_str}, {state_label}, '
                    f'averaged over {n_eig} eigenstates\n'
                    f'{n_q} discrete momenta: {q_col_header}')
        for ds, desc in [
            ('Sq_disc', f'S(q) spin structure factor (full <S_i·S_j>).\n'
                        f'{sf_q_hdr}'),
            ('Szz_disc', f'S^zz(q) longitudinal spin structure factor '
                         f'(<S^z_i S^z_j>).\n{sf_q_hdr}'),
            ('Dq_disc', f'D_conn(q) connected dimer structure factor.\n'
                        f'{sf_q_hdr}'),
            ('Dq_full_disc', f'D_full(q) full dimer structure factor.\n'
                             f'{sf_q_hdr}'),
        ]:
            if ds in r:
                np.savetxt(os.path.join(gdir, f'{ds}.dat'),
                           np.atleast_2d(r[ds]), header=desc)

        # Eigenvalues
        if 'all_evals' in r:
            np.savetxt(os.path.join(gdir, 'all_evals.dat'),
                       np.atleast_2d(r['all_evals']),
                       header=(f'All eigenvalues for Jpm={jpm_str}\n'
                               f'Sorted in ascending order. '
                               f'{len(r["all_evals"])} total.'))

        # SpSm (complex)
        if 'SpSm' in r:
            spsm_hdr = (f'<S^+_i S^-_j> off-diagonal spin correlator.\n'
                        f'{site_hdr}')
            np.savetxt(os.path.join(gdir, 'SpSm_real.dat'),
                       r['SpSm'].real,
                       header=f'Real part of {spsm_hdr}')
            np.savetxt(os.path.join(gdir, 'SpSm_imag.dat'),
                       r['SpSm'].imag,
                       header=f'Imaginary part of {spsm_hdr}')

        # --- Per-eigenstate data ---
        if 'per_state_Sq' in r:
            psdir = os.path.join(gdir, 'per_state')
            os.makedirs(psdir, exist_ok=True)

            ps_hdr = (f'Jpm={jpm_str}, {state_label}, '
                      f'{n_eig} eigenstates\n'
                      f'Each row = one eigenstate (row 0..{n_eig-1})')

            ps_descs = {
                'per_state_Sq': (
                    'Sq.dat',
                    f'S(q) per eigenstate. {ps_hdr}\n'
                    f'{n_q} columns = discrete momenta: {q_col_header}'),
                'per_state_Szz': (
                    'Szz.dat',
                    f'S^zz(q) per eigenstate. {ps_hdr}\n'
                    f'{n_q} columns = discrete momenta: {q_col_header}'),
                'per_state_Dq_full': (
                    'Dq_full.dat',
                    f'D_full(q) per eigenstate. {ps_hdr}\n'
                    f'{n_q} columns = discrete momenta: {q_col_header}'),
                'per_state_Dq_conn': (
                    'Dq_conn.dat',
                    f'D_conn(q) per eigenstate. {ps_hdr}\n'
                    f'{n_q} columns = discrete momenta: {q_col_header}'),
                'per_state_B_mean': (
                    'B_mean.dat',
                    f'<S_i·S_j> per NN bond, per eigenstate. {ps_hdr}\n'
                    f'{N_b} columns = NN bonds\n'
                    f'Bond index → (site_i, site_j): ' +
                    ', '.join(f'{a}→({i},{j})'
                              for a, (i, j) in enumerate(nn_pairs))),
                'per_state_sz_local': (
                    'sz_local.dat',
                    f'<S^z_i> per site, per eigenstate. {ps_hdr}\n'
                    f'{N} columns = sites (index 0..{N-1})'),
                'per_state_SzSz_bond': (
                    'SzSz_bond.dat',
                    f'<S^z_i S^z_j> per NN bond, per eigenstate. {ps_hdr}\n'
                    f'{N_b} columns = NN bonds\n'
                    f'Bond index → (site_i, site_j): ' +
                    ', '.join(f'{a}→({i},{j})'
                              for a, (i, j) in enumerate(nn_pairs))),
            }
            for key, (fname, desc) in ps_descs.items():
                if key in r:
                    np.savetxt(os.path.join(psdir, fname), r[key],
                               header=desc)

            if 'per_state_sector' in r:
                np.savetxt(os.path.join(psdir, 'sector.dat'),
                           r['per_state_sector'].reshape(1, -1), fmt='%d',
                           header=(f'Symmetry sector index per eigenstate.\n'
                                   f'Jpm={jpm_str}, {n_eig} eigenstates\n'
                                   f'One integer per eigenstate'))
            if 'triangle_list' in r:
                np.savetxt(os.path.join(psdir, 'triangle_list.dat'),
                           r['triangle_list'], fmt='%d',
                           header=(f'Triangle list for scalar chirality.\n'
                                   f'Each row: site_i site_j site_k '
                                   f'forming a triangle\n'
                                   f'{len(r["triangle_list"])} triangles '
                                   f'from {N_b} NN bonds'))
            # 3D arrays: reshape to 2D with shape header
            for key, fname, desc in [
                ('per_state_SiSj', 'SiSj_corr.dat',
                 f'<S_i·S_j> full correlation matrix per eigenstate.\n'
                 f'Jpm={jpm_str}, {n_eig} eigenstates\n'
                 f'3D array ({n_eig} x {N} x {N}) reshaped to '
                 f'({n_eig*N} x {N})\n'
                 f'Rows [{N}*k : {N}*(k+1)] = eigenstate k'),
                ('per_state_SzSz', 'SzSz_corr.dat',
                 f'<S^z_i S^z_j> correlation matrix per eigenstate.\n'
                 f'Jpm={jpm_str}, {n_eig} eigenstates\n'
                 f'3D array ({n_eig} x {N} x {N}) reshaped to '
                 f'({n_eig*N} x {N})\n'
                 f'Rows [{N}*k : {N}*(k+1)] = eigenstate k'),
            ]:
                if key in r:
                    arr = r[key]  # (n_gs, N, N)
                    s = arr.shape
                    np.savetxt(os.path.join(psdir, fname),
                               arr.reshape(s[0] * s[1], s[2]),
                               header=f'shape: {s[0]} {s[1]} {s[2]}\n{desc}')

        # --- RDM sub-results ---
        if 'rdm_results' in r:
            for (name, oi), rdm_data in r['rdm_results'].items():
                rdir = os.path.join(gdir, 'rdm', f'{name}_o{oi}')
                os.makedirs(rdir, exist_ok=True)
                sites = rdm_data['sites']
                dim_sub = rdm_data['rdm'].shape[0]
                rdm_meta = {
                    'S_vn': rdm_data['S_vn'],
                    'S_R2': rdm_data['S_R2'],
                    'trace': rdm_data['trace'],
                }
                rdm_meta_hdr = [
                    f"RDM metadata: {name} orientation {oi}",
                    f"Jpm={jpm_str}, {state_label}",
                    f"GS-averaged over {n_eig} eigenstates, with spin-flip",
                    f"  ρ_avg = (1/{n_eig}) Σ_k [ρ(ψ_k) + ρ(ψ̃_k)] / 2",
                    f"  where ψ̃ = global spin-flip of ψ",
                    f"Subsystem sites: {sites} ({len(sites)} sites, "
                    f"Hilbert dim = {dim_sub})",
                    "",
                    "S_vn  = von Neumann entropy  -Tr(ρ ln ρ)  of the averaged RDM",
                    "S_R2  = Rényi-2 entropy  -ln(Tr(ρ²))  of the averaged RDM",
                    "trace = Tr(ρ_A), should be 1.0",
                ]
                _save_metadata(os.path.join(rdir, 'metadata.dat'),
                               rdm_meta, header_lines=rdm_meta_hdr)

                rdm_hdr = (f'Reduced density matrix ({name} orient {oi}).\n'
                           f'Jpm={jpm_str}, {state_label}\n'
                           f'GS-averaged over {n_eig} eigenstates WITH spin-flip:\n'
                           f'  ρ_avg = (1/{n_eig}) Σ_k [ρ(ψ_k) + ρ(ψ̃_k)] / 2\n'
                           f'Subsystem sites: {sites}\n'
                           f'Matrix shape: ({dim_sub} x {dim_sub})\n'
                           f'Rows/cols = subsystem Fock states 0..{dim_sub-1} '
                           f'(binary encoding of {len(sites)} subsystem spins)')
                np.savetxt(os.path.join(rdir, 'sites.dat'),
                           np.array(sites, dtype=int).reshape(1, -1),
                           fmt='%d',
                           header=(f'Subsystem site indices for {name} '
                                   f'orient {oi}\n'
                                   f'{len(sites)} sites out of {N} total'))
                np.savetxt(os.path.join(rdir, 'rdm_real.dat'),
                           rdm_data['rdm'].real,
                           header=f'Real part of {rdm_hdr}')
                np.savetxt(os.path.join(rdir, 'rdm_imag.dat'),
                           rdm_data['rdm'].imag,
                           header=f'Imaginary part of {rdm_hdr}')
                np.savetxt(os.path.join(rdir, 'spectrum.dat'),
                           rdm_data['spectrum'],
                           header=(f'Entanglement spectrum (eigenvalues '
                                   f'of ρ_avg) for {name} orient {oi}.\n'
                                   f'Jpm={jpm_str}, {state_label}, '
                                   f'GS-averaged + spin-flip\n'
                                   f'{len(rdm_data["spectrum"])} eigenvalues, '
                                   f'sorted in ascending order'))

                # Per-eigenstate RDMs
                if 'per_state_rdm' in rdm_data:
                    ps_rdir = os.path.join(rdir, 'per_state')
                    os.makedirs(ps_rdir, exist_ok=True)
                    ps_rdm_hdr = (f'{name} orient {oi}, '
                                  f'sites={sites}, dim={dim_sub}')
                    for si, rdm_i in enumerate(rdm_data['per_state_rdm']):
                        hdr_i = (f'Per-eigenstate RDM — raw ρ(ψ_{si}), '
                                 f'NO spin-flip averaging.\n'
                                 f'Eigenstate {si} of {n_eig} in the '
                                 f'{state_label} manifold.\n'
                                 f'Jpm={jpm_str}, {ps_rdm_hdr}\n'
                                 f'To obtain the spin-flip copy: '
                                 f'ρ(ψ̃) = P ρ(ψ) P with P = subsystem '
                                 f'spin-flip permutation.')
                        np.savetxt(os.path.join(ps_rdir,
                                   f'rdm_{si}_real.dat'), rdm_i.real,
                                   header=f'Real part. {hdr_i}')
                        np.savetxt(os.path.join(ps_rdir,
                                   f'rdm_{si}_imag.dat'), rdm_i.imag,
                                   header=f'Imaginary part. {hdr_i}')
                    np.savetxt(os.path.join(ps_rdir, 'S_vn.dat'),
                               rdm_data['per_state_S_vn'],
                               header=(f'Von Neumann entropy S_vN = -Tr(ρ ln ρ) '
                                       f'per eigenstate.\n'
                                       f'Computed from raw ρ(ψ) — '
                                       f'NO spin-flip averaging.\n'
                                       f'Jpm={jpm_str}, {n_eig} values, '
                                       f'{ps_rdm_hdr}'))
                    np.savetxt(os.path.join(ps_rdir, 'S_R2.dat'),
                               rdm_data['per_state_S_R2'],
                               header=(f'Rényi-2 entropy S_R2 = -ln(Tr(ρ²)) '
                                       f'per eigenstate.\n'
                                       f'Computed from raw ρ(ψ) — '
                                       f'NO spin-flip averaging.\n'
                                       f'Jpm={jpm_str}, {n_eig} values, '
                                       f'{ps_rdm_hdr}'))
                    for si, spec_i in enumerate(
                            rdm_data['per_state_spectra']):
                        np.savetxt(os.path.join(ps_rdir,
                                   f'spectrum_{si}.dat'), spec_i,
                                   header=(f'Entanglement spectrum '
                                           f'(eigenvalues of ρ(ψ_{si})), '
                                           f'NO spin-flip.\n'
                                           f'Eigenstate {si} of {n_eig}, '
                                           f'Jpm={jpm_str}, {ps_rdm_hdr}'))

    print(f"  Saved per-Jpm results: {jpm_dir}")


def save_rdm_txt_files(gs_result, cfg, jpm_str, output_dir):
    """Export GS-averaged RDM matrices as text files.

    Output: {output_dir}/rdm/Jpm={jpm_str}/rdm_gs_averaged_{name}_o{oi}_ngs{n}.txt
    """
    if gs_result is None or 'rdm_results' not in gs_result:
        return

    rdm_dir = os.path.join(output_dir, 'rdm', f'Jpm={jpm_str}')
    os.makedirs(rdm_dir, exist_ok=True)

    E0 = gs_result['E0']
    n_gs = gs_result['n_gs']
    n_total_rdm = gs_result.get('n_total_rdm', 2 * n_gs)
    N = cfg['NUM_SITES']
    deg_tol = cfg.get('DEG_TOL', 2e-5)

    for (name, oi), rdm_data in gs_result['rdm_results'].items():
        rdm = rdm_data['rdm']
        sites = rdm_data['sites']
        dim_sub = rdm.shape[0]
        trace = rdm_data['trace']
        S_vn = rdm_data['S_vn']
        S_R2 = rdm_data['S_R2']

        fname = f"rdm_gs_averaged_{name}_o{oi}_ngs{n_gs}.txt"
        path = os.path.join(rdm_dir, fname)

        with open(path, 'w') as f:
            f.write("# GS-Averaged Reduced Density Matrix (with spin-flip)\n")
            f.write("#\n")
            f.write(f"# ρ_avg = (1/{n_gs}) Σ_k [ρ(ψ_k) + ρ(ψ̃_k)] / 2\n")
            f.write("# where ψ̃ = global spin-flip of ψ (all spins inverted).\n")
            f.write("# This restores Sz-symmetry broken by working in a single Sz sector.\n")
            f.write("#\n")
            f.write(f"# Jpm: {jpm_str}\n")
            f.write(f"# E0: {E0:.15e}\n")
            f.write(f"# n_gs (degenerate eigenstates): {n_gs}\n")
            f.write(f"# n_total_rdm (states × 2 for spin-flip): {n_total_rdm}\n")
            f.write(f"# deg_tol: {deg_tol:.1e}\n")
            f.write(f"# Geometry: {name} orient {oi}\n")
            f.write(f"# Total sites in cluster: {N}\n")
            f.write(f"# Subsystem sites: {sites}\n")
            f.write(f"# Subsystem Hilbert dimension: {dim_sub}\n")
            f.write(f"# Trace(ρ_avg): {trace:.15e}  (should be 1.0)\n")
            f.write(f"# S_vN: {S_vn:.15e}  (von Neumann entropy of ρ_avg)\n")
            f.write(f"# S_R2: {S_R2:.15e}  (Rényi-2 entropy of ρ_avg)\n")
            f.write("#\n")
            f.write("# MATRIX FORMAT: row  col  Re(ρ)  Im(ρ)\n")
            f.write("# Only nonzero entries (|ρ_ij| > 1e-15) are listed.\n")
            f.write("#\n")
            for i in range(dim_sub):
                for j in range(dim_sub):
                    val = rdm[i, j]
                    if abs(val) > 1e-15:
                        f.write(f"{i:3d}  {j:3d}  "
                                f"{val.real:+.15e}  {val.imag:+.15e}\n")

        print(f"  Saved RDM: {fname}")

        # Per-eigenstate RDMs
        if 'per_state_rdm' in rdm_data:
            for si, rdm_i in enumerate(rdm_data['per_state_rdm']):
                tr_i = np.trace(rdm_i).real
                s_vn_i = rdm_data['per_state_S_vn'][si]
                s_r2_i = rdm_data['per_state_S_R2'][si]
                fname_i = (f"rdm_eigenstate_{si}_{name}_o{oi}_"
                           f"ngs{n_gs}.txt")
                path_i = os.path.join(rdm_dir, fname_i)
                dim_sub = rdm_i.shape[0]
                with open(path_i, 'w') as f:
                    f.write(f"# Per-Eigenstate RDM — raw ρ(ψ_{si}), NO spin-flip averaging\n")
                    f.write("#\n")
                    f.write("# This is the raw reduced density matrix for a single eigenstate.\n")
                    f.write("# No spin-flip copy is included. To construct the spin-flip partner:\n")
                    f.write("#   ρ(ψ̃) = P · ρ(ψ) · P,  where P is the subsystem spin-flip\n")
                    f.write("#   permutation matrix (self-inverse perm that flips all sub-bits).\n")
                    f.write("#\n")
                    f.write(f"# Jpm: {jpm_str}\n")
                    f.write(f"# E0: {E0:.15e}\n")
                    f.write(f"# Eigenstate index: {si} (of {n_gs} degenerate states)\n")
                    f.write(f"# Geometry: {name} orient {oi}\n")
                    f.write(f"# Total sites in cluster: {N}\n")
                    f.write(f"# Subsystem sites: {sites}\n")
                    f.write(f"# Subsystem Hilbert dimension: {dim_sub}\n")
                    f.write(f"# Trace(ρ): {tr_i:.15e}  (should be 1.0)\n")
                    f.write(f"# S_vN: {s_vn_i:.15e}  (von Neumann, raw ρ)\n")
                    f.write(f"# S_R2: {s_r2_i:.15e}  (Rényi-2, raw ρ)\n")
                    f.write("#\n")
                    f.write("# MATRIX FORMAT: row  col  Re(ρ)  Im(ρ)\n")
                    f.write("# Only nonzero entries (|ρ_ij| > 1e-15) are listed.\n")
                    f.write("#\n")
                    for i in range(dim_sub):
                        for j in range(dim_sub):
                            val = rdm_i[i, j]
                            if abs(val) > 1e-15:
                                f.write(f"{i:3d}  {j:3d}  "
                                        f"{val.real:+.15e}  "
                                        f"{val.imag:+.15e}\n")
                print(f"  Saved RDM: {fname_i}")


def save_monolithic(all_results, nn_pairs, bond_positions, discrete_q,
                    q_labels, cfg, output_dir, prefix=''):
    """Save summary CSV and geometry .dat files."""
    N = cfg['NUM_SITES']
    LX, LY = cfg['LX'], cfg['LY']

    # CSV
    disc_fields = []
    for i in range(len(discrete_q)):
        m1, m2 = i // LY, i % LY
        disc_fields.extend([f'S_q{m1}{m2}', f'Szz_q{m1}{m2}',
                           f'Dconn_q{m1}{m2}', f'Dfull_q{m1}{m2}'])
    fields = (['Jpm', 'E0_per_site', 'gap', 'n_gs',
               'B_mean_avg', 'B_mean_std',
               'nem_abs', 'nem_arg', 'B01', 'B02', 'B12'] + disc_fields)

    csv_path = os.path.join(output_dir, f'{prefix}analysis_data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for r in all_results:
            row = {
                'Jpm': r['Jpm'], 'E0_per_site': r['E0'] / N,
                'gap': r['gap'], 'n_gs': r['n_gs'],
                'B_mean_avg': np.mean(r['B_mean']),
                'B_mean_std': np.std(r['B_mean']),
                'nem_abs': r['nem_abs'], 'nem_arg': r['nem_arg'],
                'B01': r['B01'], 'B02': r['B02'], 'B12': r['B12'],
            }
            for i in range(len(discrete_q)):
                m1, m2 = i // LY, i % LY
                row[f'S_q{m1}{m2}'] = r['Sq_disc'][i]
                row[f'Szz_q{m1}{m2}'] = r['Szz_disc'][i]
                row[f'Dconn_q{m1}{m2}'] = r['Dq_disc'][i]
                row[f'Dfull_q{m1}{m2}'] = r['Dq_full_disc'][i]
            writer.writerow(row)
    print(f"  Saved {csv_path}")

    # Geometry .dat files
    geo_dir = os.path.join(output_dir, f'{prefix}geometry')
    os.makedirs(geo_dir, exist_ok=True)
    np.savetxt(os.path.join(geo_dir, 'nn_pairs.dat'),
               np.array(nn_pairs), fmt='%d',
               header=(f'Nearest-neighbor bond list for {cfg["cluster"]} '
                       f'kagome cluster ({N} sites).\n'
                       f'{len(nn_pairs)} bonds total.\n'
                       f'Each row: site_i  site_j  (0-indexed)'))
    np.savetxt(os.path.join(geo_dir, 'bond_positions.dat'), bond_positions,
               header=(f'Bond midpoint positions in real space.\n'
                       f'{len(bond_positions)} bonds, one row per bond.\n'
                       f'Columns: x  y  (Cartesian coordinates)'))
    np.savetxt(os.path.join(geo_dir, 'discrete_q.dat'), discrete_q,
               header=(f'Discrete momenta for structure factor evaluation.\n'
                       f'{len(discrete_q)} q-points on the {LX}x{LY} '
                       f'reciprocal mesh.\n'
                       f'Columns: qx  qy  (reciprocal space coordinates)'))
    np.savetxt(os.path.join(geo_dir, 'bz_corners.dat'), BZ_CORNERS,
               header=('Corners of the first Brillouin zone (hexagonal).\n'
                       'Each row: qx  qy  (reciprocal space coordinates)'))
    with open(os.path.join(geo_dir, 'discrete_q_labels.dat'), 'w') as f:
        f.write('# Labels for each discrete q-point (one per line).\n')
        f.write(f'# {len(q_labels)} labels matching rows of '
                f'discrete_q.dat.\n')
        for l in q_labels:
            f.write(l.replace('$', '') + '\n')
    print(f"  Saved {geo_dir}")


# ============================================================
# Fidelity computation (shared)
# ============================================================
def compute_sector_fidelity(cfg, jpm_list, output_dir):
    """Compute GS fidelity between consecutive Jpm values, per sector."""
    n_jpm = len(jpm_list)

    all_mappings = {}
    for n_up in cfg['N_UP_LIST']:
        for jpm_str in jpm_list:
            all_mappings[(n_up, jpm_str)] = read_eigenvalue_mapping(
                cfg, n_up, jpm_str)

    nup_sectors = sorted(set(
        (e['n_up'], e['sector'])
        for entries in all_mappings.values() for e in entries))

    n_sec = len(nup_sectors)
    print(f"\n  Fidelity: {n_sec} (n_up, sector) pairs, {n_jpm} Jpm values")

    jpm_vals = np.array([float(j) for j in jpm_list])
    fidelity = np.full((n_sec, n_jpm - 1), np.nan)

    for si, (n_up, sector) in enumerate(nup_sectors):
        psi_prev = None
        for ji, jpm_str in enumerate(jpm_list):
            entries = all_mappings[(n_up, jpm_str)]
            sec_entries = [e for e in entries if e['sector'] == sector]
            if not sec_entries:
                psi_prev = None
                continue
            best = min(sec_entries, key=lambda e: e['energy'])
            psi = load_eigenvector(cfg, n_up, jpm_str, best['h5_key'])
            if psi_prev is not None:
                f_val = abs(np.dot(np.conj(psi_prev), psi))**2
                fidelity[si, ji - 1] = f_val
                if f_val < 0.99:
                    print(f"    (n_up={n_up},s={sector}) "
                          f"{jpm_list[ji-1]}->{jpm_str}: F={f_val:.8f}")
            psi_prev = psi
        del psi_prev
        gc.collect()

    fid_dir = os.path.join(output_dir, 'sector_fidelity')
    os.makedirs(fid_dir, exist_ok=True)
    np.savetxt(os.path.join(fid_dir, 'jpm_values.dat'), jpm_vals,
               header=(f'Jpm values used in fidelity computation.\n'
                       f'{len(jpm_vals)} values, sorted.'))
    np.savetxt(os.path.join(fid_dir, 'nup_sectors.dat'),
               np.array(nup_sectors, dtype=int), fmt='%d',
               header=(f'(n_up, sector) pairs indexing fidelity matrix rows.\n'
                       f'{n_sec} pairs total.\n'
                       f'Columns: n_up  sector_index'))
    np.savetxt(os.path.join(fid_dir, 'fidelity.dat'), fidelity,
               header=(f'Ground-state fidelity |<ψ(Jpm_i)|ψ(Jpm_{{i+1}})>|² '
                       f'between consecutive Jpm values.\n'
                       f'Matrix shape: ({n_sec} sectors x {n_jpm - 1} '
                       f'transitions).\n'
                       f'Row k = (n_up, sector) pair from nup_sectors.dat.\n'
                       f'Column j = transition from Jpm_j to Jpm_{{j+1}}.\n'
                       f'NaN entries = sector not present at that Jpm.'))
    print(f"  Saved fidelity: {fid_dir}")
    return fidelity


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Unified BFG Kagome ED computation (2×3 or 3×3)')
    parser.add_argument('--cluster', required=True, choices=['2x3', '3x3', '3x3_to'],
                        help='Cluster size')
    parser.add_argument('--jpm', type=str, default='all',
                        help='Specific Jpm value or "all"')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: cluster-specific)')
    parser.add_argument('--worker', action='store_true',
                        help='Worker mode: process ONE Jpm and save per-Jpm')
    parser.add_argument('--index', type=int, default=None,
                        help='0-based index into sorted Jpm list (SLURM)')
    parser.add_argument('--skip-fidelity', action='store_true',
                        help='Skip fidelity computation')
    parser.add_argument('--deg-tol', type=float, default=None,
                        help='Override degeneracy tolerance (per site)')
    args = parser.parse_args()

    cfg = get_config(args.cluster)
    if args.deg_tol is not None:
        cfg['DEG_TOL'] = args.deg_tol
    output_dir = args.output_dir or cfg['OUTPUT_DIR']
    os.makedirs(output_dir, exist_ok=True)

    N = cfg['NUM_SITES']
    LX, LY = cfg['LX'], cfg['LY']

    print(f"{'='*60}")
    print(f"BFG Kagome {args.cluster} — {N} sites, "
          f"n_up={cfg['N_UP_LIST']}")
    print(f"  Base data: {cfg['BASE_DIR']}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # --- Lattice geometry ---
    print("\nLoading lattice geometry...")
    geo = load_geometry(cfg)
    print(f"  {N} sites, {len(geo['nn_pairs'])} NN bonds, {cfg['N_UC']} UCs")

    # --- Discrete momenta ---
    discrete_q, q_labels, unique_q_idx = build_discrete_momenta(LX, LY)
    print(f"\n  Discrete momenta ({len(discrete_q)} total, "
          f"{len(unique_q_idx)} unique):")
    for i in range(len(discrete_q)):
        q = discrete_q[i]
        uniq = '*' if i in unique_q_idx else ' '
        print(f"    {uniq} q=({q[0]:+.4f},{q[1]:+.4f})  "
              f"|q|={np.linalg.norm(q):.4f}  {q_labels[i]}")

    # --- Discover Jpm values ---
    all_jpm = discover_jpm_values(cfg)
    print(f"\n  Found {len(all_jpm)} Jpm values with complete data")

    # ========== Worker mode ==========
    if args.worker:
        if args.index is not None:
            if args.index < 0 or args.index >= len(all_jpm):
                print(f"ERROR: --index={args.index} out of range "
                      f"(0..{len(all_jpm)-1})")
                sys.exit(1)
            jpm_str = all_jpm[args.index]
        elif args.jpm.lower() != 'all':
            jpm_str = args.jpm
        else:
            print("ERROR: --worker requires --index=N or --jpm=VALUE")
            sys.exit(1)

        per_jpm_dir = os.path.join(output_dir, 'per_jpm')
        print(f"\n{'='*60}")
        print(f"WORKER MODE — Jpm = {jpm_str}")
        print(f"{'='*60}")

        t0 = time.time()
        gs_r, ex_r = process_one_jpm(
            cfg, jpm_str, geo, discrete_q, q_labels, unique_q_idx)
        dt = time.time() - t0
        print(f"\n  Worker time: {dt:.0f}s ({dt/60:.1f}min)")
        save_per_jpm_result(gs_r, ex_r, per_jpm_dir, jpm_str,
                            cfg, geo, discrete_q, q_labels)
        save_rdm_txt_files(gs_r, cfg, jpm_str, output_dir)
        print("Worker done.")
        return

    # ========== Sequential mode ==========
    if args.jpm.lower() == 'all':
        jpm_list = all_jpm
    else:
        jpm_list = [args.jpm]

    print(f"\nProcessing {len(jpm_list)} Jpm values...")
    all_gs, all_ex = [], []
    t_global = time.time()

    for ji, jpm_str in enumerate(jpm_list):
        print(f"\n*** [{ji+1}/{len(jpm_list)}] Jpm = {jpm_str} ***")
        t0 = time.time()
        try:
            gs_r, ex_r = process_one_jpm(
                cfg, jpm_str, geo, discrete_q, q_labels, unique_q_idx)
            if gs_r is not None:
                all_gs.append(gs_r)
            if ex_r is not None:
                all_ex.append(ex_r)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Also save per-Jpm (for consistency)
        per_jpm_dir = os.path.join(output_dir, 'per_jpm')
        save_per_jpm_result(gs_r, ex_r, per_jpm_dir, jpm_str,
                            cfg, geo, discrete_q, q_labels)
        save_rdm_txt_files(gs_r, cfg, jpm_str, output_dir)

        dt = time.time() - t0
        elapsed = time.time() - t_global
        remaining = (len(jpm_list) - ji - 1) * (elapsed / (ji + 1))
        print(f"  Time: {dt:.0f}s. Elapsed: {elapsed/60:.1f}min. "
              f"Est remaining: {remaining/60:.1f}min")

    if not all_gs:
        print("No results to save!")
        return

    print(f"\n{'='*60}")
    print(f"Processed {len(all_gs)} GS + {len(all_ex)} ES "
          f"in {(time.time()-t_global)/60:.1f} min")
    print(f"{'='*60}")

    # Save monolithic
    print("\nSaving monolithic results...")
    save_monolithic(all_gs, geo['nn_pairs'], geo['bond_positions'],
                    discrete_q, q_labels, cfg, output_dir, prefix='')
    if all_ex:
        save_monolithic(all_ex, geo['nn_pairs'], geo['bond_positions'],
                        discrete_q, q_labels, cfg, output_dir, prefix='ex_')

    # Fidelity
    if not args.skip_fidelity and len(jpm_list) > 1:
        print("\nComputing sector-resolved fidelity...")
        compute_sector_fidelity(cfg, jpm_list, output_dir)

    # Summary table
    print(f"\n{'='*100}")
    header = f"  {'Jpm':>6s}  {'E0/N':>10s}  {'gap':>10s}  {'n_gs':>4s}"
    for i in unique_q_idx:
        lbl = q_labels[i].strip('$').replace('\\', '')
        header += f"  {'S('+lbl+')':>10s}"
    header += f"  {'|Φ_nem|':>8s}"
    print(header)
    print(f"{'='*100}")
    for r in sorted(all_gs, key=lambda x: x['Jpm']):
        line = (f"  {r['Jpm']:6.2f}  {r['E0']/N:10.6f}  "
                f"{r['gap']:10.6f}  {r['n_gs']:4d}")
        for i in unique_q_idx:
            line += f"  {r['Sq_disc'][i]:10.4f}"
        line += f"  {r['nem_abs']:8.4f}"
        print(line)

    print(f"\n=== Done. Output in {output_dir} ===")


if __name__ == '__main__':
    main()
