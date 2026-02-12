"""
Generate All Figures

Master script to reproduce all figures from the paper.
Run this script from the analysis/ directory.

Usage:
    python generate_all_figures.py
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name, description):
    """Run a Python script and report status."""
    print(f"\n{'='*60}")
    print(f"Generating: {description}")
    print(f"Script: {script_name}")
    print('='*60)

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False
    )

    if result.returncode == 0:
        print(f"[OK] {description}")
    else:
        print(f"[FAILED] {description}")

    return result.returncode == 0


def main():
    print("="*60)
    print("GENERATING ALL FIGURES")
    print("="*60)

    # Ensure output directories exist
    figures_dir = Path('../results/figures')
    for i in range(2, 8):
        (figures_dir / f'figure{i}').mkdir(parents=True, exist_ok=True)

    results = {}

    # Figure 2: Model Comparison
    results['Figure 2 - Model Comparison'] = run_script(
        'compare_all_models_figure_2.py',
        'Figure 2: Model Performance Comparison'
    )

    results['Figure 2 - Panel H (Beta Analysis)'] = run_script(
        'extract_beta_analysis_panel_h_figure_2.py',
        'Figure 2 Panel H: Physics vs Neural Contributions'
    )

    # Figure 3: City Analysis
    results['Figure 3 - City Analysis'] = run_script(
        'create_city_analysis_figure_3.py',
        'Figure 3: City-Level Spatial Dynamics'
    )

    results['Figure 3 - Panel F (Interaction Matrix)'] = run_script(
        'panel_f_interaction_matrix__figure_3.py',
        'Figure 3 Panel F: Ethnic Interaction Matrix'
    )

    # Figure 4: Universal Dynamics
    results['Figure 4 - Universal Dynamics'] = run_script(
        'create_universal_dynamics_figure_4.py',
        'Figure 4: Universal Spatial Dynamics'
    )

    # Figure 5: Turing Patterns
    results['Figure 5 - Panel A (Toronto Overview)'] = run_script(
        'panel_a_toronto_overview_figure_5.py',
        'Figure 5 Panel A: Toronto Overview'
    )

    results['Figure 5 - Panel B (Turing Zoom)'] = run_script(
        'panel_b_turing_zoom_figure_5.py',
        'Figure 5 Panel B: Turing Pattern Zoom'
    )

    results['Figure 5 - Panel D (Dynamics)'] = run_script(
        'panel_d_dynamics_separated_figure_5.py',
        'Figure 5 Panel D: Pattern Formation Dynamics'
    )

    results['Figure 5 - Panel E (Pattern Characteristics)'] = run_script(
        'panel_e_pattern_characteristics_separated_figure_5.py',
        'Figure 5 Panel E: Multi-scale Pattern Analysis'
    )

    results['Figure 5 - Panel I (Diffusion)'] = run_script(
        'create_panel_i_figure_5.py',
        'Figure 5 Panel I: Diffusion Coefficients'
    )

    # Figure 6: Crystallization
    results['Figure 6 - Crystallization'] = run_script(
        'create_graphpde_crystallization_figure_6.py',
        'Figure 6: Settlement Crystallization'
    )

    # Figure 7: Energy Landscape
    results['Figure 7 - Energy Landscape'] = run_script(
        'create_energy_landscape_toronto_chinese_figure_7.py',
        'Figure 7: Energy Landscape Analysis'
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    success_count = sum(results.values())
    total_count = len(results)

    for name, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"  {status} {name}")

    print(f"\nCompleted: {success_count}/{total_count}")

    print("\nOutput directories:")
    print("  ./figures/figure2/ - Model comparison")
    print("  ./figures/figure3/ - City analysis")
    print("  ./figures/figure4/ - Universal dynamics")
    print("  ./figures/figure5/ - Turing patterns")
    print("  ./figures/figure6/ - Crystallization")
    print("  ./figures/figure7/ - Energy landscape")


if __name__ == "__main__":
    main()
