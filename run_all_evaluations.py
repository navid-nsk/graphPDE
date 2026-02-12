"""
run_all_evaluations.py

Master script for peer review - runs all evaluation and figure generation scripts.

This script executes the following in sequence:
1. evaluate_test_graphpde.py          - Main model evaluation on test set
2. evaluate_multistep_graphpde.py     - Multi-step horizon evaluation
3. generalizability/traffic/evaluate.py - Traffic domain generalizability
4. analysis/generate_all_figures.py   - Generate all analysis figures

All outputs are saved to the results/ folder.

Usage:
    python run_all_evaluations.py
    python run_all_evaluations.py --skip_figures    # Skip figure generation
    python run_all_evaluations.py --only_figures    # Only generate figures
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

# Get the base directory (where this script is located)
BASE_DIR = Path(__file__).parent.resolve()

# Scripts to run in sequence
EVALUATION_SCRIPTS = [
    {
        'name': 'Main Model Evaluation',
        'script': BASE_DIR / 'evaluate_test_graphpde.py',
        'description': 'Evaluates GraphPDE on test set (single-step prediction)',
    },
    {
        'name': 'Multi-step Horizon Evaluation',
        'script': BASE_DIR / 'experiments' / 'multistep' / 'evaluate_multistep_graphpde.py',
        'description': 'Evaluates multi-step predictions (5, 10, 15, 20-year horizons)',
    },
    {
        'name': 'Traffic Generalizability',
        'script': BASE_DIR / 'generalizability' / 'traffic' / 'evaluate.py',
        'description': 'Evaluates model generalizability on traffic prediction domain',
    },
]

FIGURE_SCRIPT = {
    'name': 'Figure Generation',
    'script': BASE_DIR / 'analysis' / 'generate_all_figures.py',
    'description': 'Generates all analysis figures for the paper',
}


def print_header(text, char='=', width=80):
    """Print a formatted header."""
    print('\n' + char * width)
    print(text.center(width))
    print(char * width)


def print_subheader(text, char='-', width=80):
    """Print a formatted subheader."""
    print('\n' + char * width)
    print(text)
    print(char * width)


def run_script(script_info, script_num=None, total=None):
    """
    Run a single Python script and return success status.

    Args:
        script_info: Dict with 'name', 'script', 'description'
        script_num: Current script number (for display)
        total: Total number of scripts (for display)

    Returns:
        Tuple of (success: bool, elapsed_time: float)
    """
    script_path = script_info['script']

    # Header
    if script_num and total:
        header = f"[{script_num}/{total}] {script_info['name']}"
    else:
        header = script_info['name']

    print_subheader(header)
    print(f"Script: {script_path}")
    print(f"Description: {script_info['description']}")
    print()

    # Check if script exists
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False, 0.0

    # Run the script
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            capture_output=False,  # Show output in real-time
            text=True
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n[SUCCESS] {script_info['name']} completed in {elapsed:.1f}s")
            return True, elapsed
        else:
            print(f"\n[FAILED] {script_info['name']} failed with return code {result.returncode}")
            return False, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] {script_info['name']} raised exception: {e}")
        return False, elapsed


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run all evaluation and figure generation scripts'
    )
    parser.add_argument('--skip_figures', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--only_figures', action='store_true',
                        help='Only run figure generation')
    parser.add_argument('--skip_traffic', action='store_true',
                        help='Skip traffic generalizability evaluation')

    args = parser.parse_args()

    # Start time
    total_start = time.time()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Header
    print_header('GRAPHPDE EVALUATION SUITE')
    print(f"\nStarted at: {timestamp}")
    print(f"Base directory: {BASE_DIR}")
    print(f"Results will be saved to: {BASE_DIR / 'results'}")

    # Create results directory
    results_dir = BASE_DIR / 'results'
    results_dir.mkdir(exist_ok=True)

    # Track results
    results = []

    if args.only_figures:
        # Only run figure generation
        scripts_to_run = [FIGURE_SCRIPT]
    else:
        # Build list of scripts to run
        scripts_to_run = []
        for script in EVALUATION_SCRIPTS:
            if args.skip_traffic and 'traffic' in str(script['script']).lower():
                continue
            scripts_to_run.append(script)

        # Add figure generation unless skipped
        if not args.skip_figures:
            scripts_to_run.append(FIGURE_SCRIPT)

    total_scripts = len(scripts_to_run)

    print(f"\nScripts to run: {total_scripts}")
    for i, script in enumerate(scripts_to_run, 1):
        print(f"  {i}. {script['name']}")

    # Run each script
    print_header('RUNNING EVALUATIONS')

    for i, script_info in enumerate(scripts_to_run, 1):
        success, elapsed = run_script(script_info, i, total_scripts)
        results.append({
            'name': script_info['name'],
            'success': success,
            'elapsed': elapsed
        })

    # Summary
    total_elapsed = time.time() - total_start

    print_header('EVALUATION SUMMARY')

    print(f"\n{'Script':<40} {'Status':<10} {'Time':>10}")
    print('-' * 62)

    n_success = 0
    n_failed = 0

    for r in results:
        status = 'SUCCESS' if r['success'] else 'FAILED'
        status_symbol = '[OK]' if r['success'] else '[X]'
        time_str = f"{r['elapsed']:.1f}s"
        print(f"{r['name']:<40} {status_symbol:<10} {time_str:>10}")

        if r['success']:
            n_success += 1
        else:
            n_failed += 1

    print('-' * 62)
    print(f"{'Total':<40} {'':<10} {total_elapsed:.1f}s")

    print(f"\nResults: {n_success} succeeded, {n_failed} failed")
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")

    # Final status
    if n_failed == 0:
        print_header('ALL EVALUATIONS COMPLETED SUCCESSFULLY', char='*')
        print(f"\nResults saved to: {results_dir}")
        return 0
    else:
        print_header('SOME EVALUATIONS FAILED', char='!')
        print(f"\n{n_failed} script(s) failed. Please check the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
