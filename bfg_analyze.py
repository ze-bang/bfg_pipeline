#!/usr/bin/env python3
"""
Unified BFG Kagome ED Analysis Runner
======================================
Thin wrapper that orchestrates all existing analysis scripts
with a single --cluster interface.

Usage:
  python bfg_analyze.py --cluster 3x3              # everything
  python bfg_analyze.py --cluster 2x3              # everything
  python bfg_analyze.py --cluster 3x3 --only spectrum
  python bfg_analyze.py --cluster 3x3 --only sf
  python bfg_analyze.py --cluster 3x3 --only diagnostics
  python bfg_analyze.py --cluster 3x3 --export-txt-only
"""
import os
import sys
import argparse
import importlib.util
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRATCH = '/scratch/zhouzb79'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CLUSTER_CONFIG = {
    '3x3': {
        'output_dir': os.path.join(SCRATCH, 'analysis_BFG_3x3'),
        'num_sites': 27,
    },
    '3x3_to': {
        'output_dir': os.path.join(SCRATCH, 'analysis_BFG_3x3_translation_only'),
        'num_sites': 27,
    },
    '3x3_to_fsz': {
        'output_dir': os.path.join(SCRATCH, 'analysis_BFG_3x3_fixed_Sz'),
        'num_sites': 27,
    },
    '2x3': {
        'output_dir': os.path.join(SCRATCH, 'analysis_BFG_2x3'),
        'num_sites': 18,
    },
}


def _import_module(script_name):
    """Import a script as a module by file path."""
    path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(path):
        print(f"  WARNING: {script_name} not found, skipping")
        return None
    spec = importlib.util.spec_from_file_location(
        script_name.replace('.py', ''), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_spectrum(cluster, output_dir, txt_only=False):
    """Run spectrum plotting (tower of states, sector panels, etc.)."""
    print(f"\n{'='*60}")
    print(f"  SPECTRUM ANALYSIS — {cluster}")
    print(f"{'='*60}")

    if cluster in ('3x3', '3x3_to'):
        mod = _import_module('plot_spectrum_BFG_3x3.py')
        if mod is None:
            return
        # Override BASE_DIR for translation_only variant
        if cluster == '3x3_to':
            mod.BASE_DIR = os.path.join(
                SCRATCH, 'BFG_scan_symmetrized_pbc_3x3_nup13_negJpm_translation_only')
            mod.TRANSLATION_ONLY = True
        else:
            mod.TRANSLATION_ONLY = False
        spec_dir = os.path.join(output_dir, 'spectrum')
        os.makedirs(spec_dir, exist_ok=True)

        print("  Discovering Jpm values...")
        jpm_list = mod.discover_jpm_values()
        if not jpm_list:
            print("  No Jpm data found!")
            return
        print(f"  Found {len(jpm_list)} Jpm values")

        print("  Collecting spectra...")
        data, sector_meta = mod.collect_all_spectra(jpm_list)
        momentum_map = mod.build_sector_momentum_map(sector_meta)

        print("  Saving spectrum data...")
        mod.save_spectrum_data(data, jpm_list, sector_meta, spec_dir)

        if not txt_only:
            print("  Plotting...")
            mod.plot_global_spectrum(data, jpm_list, spec_dir)
            mod.plot_symmetry_sector_spectrum(
                data, jpm_list, sector_meta, spec_dir,
                momentum_map=momentum_map)
            mod.plot_sector_gs_energies(
                data, jpm_list, sector_meta, spec_dir,
                momentum_map=momentum_map)
            mod.plot_individual_sector_panels(
                data, jpm_list, sector_meta, spec_dir,
                momentum_map=momentum_map)
            mod.plot_tower_of_states(
                data, jpm_list, sector_meta, spec_dir,
                momentum_map=momentum_map)
            mod.plot_bz_momenta(sector_meta, spec_dir)

    elif cluster == '3x3_to_fsz':
        mod = _import_module('plot_spectrum_BFG_3x3_multi_sz.py')
        if mod is None:
            return
        mod.BASE_DIR = os.path.join(
            SCRATCH, 'BFG_scan_symmetrized_pbc_3x3_fixed_Sz_translation_only')
        mod.N_UP_LIST = [14, 15]
        mod.NUM_SITES = 27
        mod.TRANSLATION_ONLY = True
        spec_dir = os.path.join(output_dir, 'spectrum')
        os.makedirs(spec_dir, exist_ok=True)

        print("  Discovering Jpm values...")
        jpm_list = mod.discover_jpm_values()
        if not jpm_list:
            print("  No Jpm data found!")
            return
        print(f"  Found {len(jpm_list)} Jpm values")

        print("  Collecting spectra...")
        data, sector_meta = mod.collect_all_spectra(jpm_list)
        momentum_map = mod.build_sector_momentum_map(sector_meta)

        print("  Saving spectrum data...")
        mod.save_spectrum_data(data, jpm_list, sector_meta, spec_dir)

        if not txt_only:
            print("  Plotting...")
            mod.plot_global_spectrum(data, jpm_list, spec_dir)
            mod.plot_sz_sector_spectrum(data, jpm_list, spec_dir)
            mod.plot_combined_spectrum(data, jpm_list, sector_meta,
                                       spec_dir, momentum_map=momentum_map)
            mod.plot_symmetry_sector_spectrum(
                data, jpm_list, sector_meta, spec_dir,
                momentum_map=momentum_map)
            mod.plot_sector_gs_energies(
                data, jpm_list, sector_meta, spec_dir,
                momentum_map=momentum_map)
            mod.plot_individual_sector_panels(
                data, jpm_list, sector_meta, spec_dir,
                momentum_map=momentum_map)
            mod.plot_tower_of_states(
                data, jpm_list, sector_meta, spec_dir,
                momentum_map=momentum_map)
            mod.plot_tower_of_states_dual(
                data, jpm_list, sector_meta, spec_dir,
                momentum_map=momentum_map)
            mod.plot_bz_momenta(sector_meta, spec_dir)

    elif cluster == '2x3':
        mod = _import_module('plot_spectrum_BFG_2x3.py')
        if mod is None:
            return
        spec_dir = os.path.join(output_dir, 'spectrum')
        os.makedirs(spec_dir, exist_ok=True)

        print("  Collecting spectra (both Sz sectors)...")
        data, jpm_sorted, sector_meta = mod.collect_all_spectra()
        if not data:
            print("  No data found!")
            return

        print("  Saving spectrum data...")
        mod.save_spectrum_data(data, jpm_sorted, sector_meta, spec_dir)

        if not txt_only:
            print("  Plotting...")
            mod.plot_global_spectrum(data, jpm_sorted, spec_dir)
            mod.plot_sz_sector_spectrum(data, jpm_sorted, spec_dir)
            mod.plot_symmetry_sector_spectrum(
                data, jpm_sorted, sector_meta, spec_dir)
            mod.plot_combined_spectrum(
                data, jpm_sorted, sector_meta, spec_dir)
            mod.plot_sector_gs_energies(
                data, jpm_sorted, sector_meta, spec_dir)

    print(f"  Done: {spec_dir}")


def run_structure_factors(cluster, output_dir, txt_only=False):
    """Run SF export + plotting via postprocess_unified.py."""
    print(f"\n{'='*60}")
    print(f"  STRUCTURE FACTORS — {cluster}")
    print(f"{'='*60}")

    mod = _import_module('postprocess_unified.py')
    if mod is None:
        return

    cfg = mod.get_cluster_config(cluster)
    DISCRETE_Q, Q_LABELS, UNIQUE_Q_IDX = mod.build_discrete_momenta(
        cfg['LX'], cfg['LY'], cfg['HAS_K_POINT'])

    # Spectrum txt export
    print("  Exporting spectrum txt...")
    spectrum = mod.load_spectrum_data(cluster, cfg)
    sector_meta = mod.load_sector_metadata(cluster, cfg)
    if spectrum:
        mod.export_spectrum_txt(spectrum, sector_meta, cluster, cfg, output_dir)

    # SF data
    print("  Loading SF data...")
    all_gs, all_ex = mod.load_results(cluster, cfg, output_dir)
    print(f"  Loaded {len(all_gs)} GS, {len(all_ex)} ES results")

    if all_gs:
        print("  Exporting SF txt...")
        mod.export_sf_txt(all_gs, DISCRETE_Q, Q_LABELS, UNIQUE_Q_IDX,
                          cluster, cfg, output_dir)
        if not txt_only:
            print("  Plotting SF...")
            mod.plot_bz_momenta(DISCRETE_Q, Q_LABELS, cluster, cfg, output_dir)
            mod.plot_sf_summary(all_gs, DISCRETE_Q, Q_LABELS, UNIQUE_Q_IDX,
                                cluster, cfg, output_dir)
            mod.plot_sf_heatmaps(all_gs, DISCRETE_Q, Q_LABELS, cluster, cfg,
                                 output_dir)

    if all_ex:
        print("  Exporting ES SF txt...")
        mod.export_sf_txt(all_ex, DISCRETE_Q, Q_LABELS, UNIQUE_Q_IDX,
                          cluster, cfg, output_dir, prefix='ex_')
        if not txt_only:
            mod.plot_sf_summary(all_ex, DISCRETE_Q, Q_LABELS, UNIQUE_Q_IDX,
                                cluster, cfg, output_dir,
                                prefix='ex_', state_label='1st Excited')
            mod.plot_sf_heatmaps(all_ex, DISCRETE_Q, Q_LABELS, cluster, cfg,
                                 output_dir, prefix='ex_',
                                 state_label='1st Excited')

    print(f"  Done: {output_dir}")


def run_diagnostics(cluster, output_dir):
    """Run per-eigenstate diagnostics (3x3 only — requires per_state data)."""
    print(f"\n{'='*60}")
    print(f"  PER-EIGENSTATE DIAGNOSTICS — {cluster}")
    print(f"{'='*60}")

    if cluster not in ('3x3', '3x3_to', '3x3_to_fsz'):
        print("  Diagnostics only available for 3x3 (requires per_state data)")
        return

    mod = _import_module('plot_diagnostics.py')
    if mod is None:
        return

    # Override the module's hardcoded paths
    mod.OUTPUT_DIR = output_dir
    mod.PER_JPM_DIR = os.path.join(output_dir, 'per_jpm')
    mod.NUM_SITES = CLUSTER_CONFIG[cluster]['num_sites']

    print("  Loading diagnostics data...")
    results = mod.load_all_diagnostics()
    print(f"  {len(results)} Jpm values with diagnostics data")

    if not results:
        print("  No diagnostics data found!")
        return

    diag_dir = os.path.join(output_dir, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)

    print("  Plotting...")
    mod.plot_sz_local_summary(results, diag_dir)
    mod.plot_chirality_summary(results, diag_dir)
    mod.plot_sz_lattice_patterns(results, diag_dir)
    mod.plot_realspace_bond_energy(results, diag_dir)
    mod.plot_per_eigenstate_structure_factors(results, diag_dir)
    mod.plot_spin_structure_factor(results, diag_dir)
    mod.plot_rdm_subsystem_geometry(diag_dir)

    print(f"  Done: {diag_dir}")



def main():
    parser = argparse.ArgumentParser(
        description='Unified BFG Kagome ED analysis runner')
    parser.add_argument('--cluster', required=True,
                        choices=['2x3', '3x3', '3x3_to', '3x3_to_fsz'],
                        help='Cluster: 2x3, 3x3, 3x3_to, or 3x3_to_fsz (multi-Sz)')
    parser.add_argument('--output-dir', default=None,
                        help='Override output directory')
    parser.add_argument('--only', choices=['spectrum', 'sf', 'diagnostics'],
                        help='Run only one analysis step')
    parser.add_argument('--export-txt-only', action='store_true',
                        help='Skip plot generation, export data only')
    args = parser.parse_args()

    cluster = args.cluster
    output_dir = args.output_dir or CLUSTER_CONFIG[cluster]['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    print(f"BFG Kagome ED Analysis — {cluster} cluster")
    print(f"Output: {output_dir}")

    # Default steps exclude 'rdm' — RDM is computed per-Jpm by bfg_compute.py
    steps = [args.only] if args.only else ['spectrum', 'sf', 'diagnostics']

    if 'spectrum' in steps:
        run_spectrum(cluster, output_dir, txt_only=args.export_txt_only)

    if 'sf' in steps:
        run_structure_factors(cluster, output_dir,
                              txt_only=args.export_txt_only)

    if 'diagnostics' in steps:
        run_diagnostics(cluster, output_dir)

    dt = time.time() - t0
    print(f"\n{'='*60}")
    print(f"All done in {dt:.1f}s. Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
