"""
Model Accuracy Analysis with Physics-Informed Validation
=========================================================
Tests the full ML + Physics pipeline against labeled audio data.

Samples files from each category, runs the analyzer, and reports:
  - ML classification accuracy
  - Physics validation agreement rates
  - Per-category breakdown
"""

import os
import sys
import json
import time
import random
import traceback

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "utils"))

from analyze import MachineHealthAnalyzer

# ── Configuration ──
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # GRASP root
SAMPLES_PER_CATEGORY = 5  # files to test per category (keep it fast)

# Define test categories with ground-truth labels
TEST_CATEGORIES = [
    {
        "name": "Normal Engine Idle (Healthy)",
        "folder": os.path.join(BASE_DIR, "healthy", "car_diagnostics_dataset_idle_state_normal_engine_idle"),
        "pattern": "normal_engine_idle_{}_std.wav",
        "expected_status": "normal",
        "expected_fault": None,
        "max_index": 264,
    },
    {
        "name": "Normal Brakes (Healthy)",
        "folder": os.path.join(BASE_DIR, "healthy", "car_diagnostics_dataset_braking_state_normal_brakes"),
        "pattern": "normal_brakes_{}_std.wav",
        "expected_status": "normal",
        "expected_fault": None,
        "max_index": 77,
    },
    {
        "name": "Bad Ignition (Faulty)",
        "folder": os.path.join(BASE_DIR, "anomalous", "car_diagnostics_dataset_startup_state_bad_ignition"),
        "pattern": "bad_ignition_{}_std.wav",
        "expected_status": "faulty",
        "expected_fault": "bad_ignition",
        "max_index": 62,
    },
    {
        "name": "Dead Battery (Faulty)",
        "folder": os.path.join(BASE_DIR, "anomalous", "car_diagnostics_dataset_startup_state_dead_battery"),
        "pattern": "dead_battery_{}_std.wav",
        "expected_status": "faulty",
        "expected_fault": "dead_battery",
        "max_index": 57,
    },
    {
        "name": "Worn Out Brakes (Faulty)",
        "folder": os.path.join(BASE_DIR, "anomalous", "car_diagnostics_dataset_braking_state_worn_out_brakes"),
        "pattern": "worn_out_brakes_{}_std.wav",
        "expected_status": "faulty",
        "expected_fault": "worn_brakes",
        "max_index": 76,
    },
    {
        "name": "Bearing Fault (.mat)",
        "folder": os.path.join(SCRIPT_DIR, "data", "test", "faulty"),
        "files": ["IR007_0_1797.mat"],  # Explicit file list
        "expected_status": "faulty",
        "expected_fault": "bearing_fault",
    },
]


def pick_sample_files(category, n=SAMPLES_PER_CATEGORY):
    """Pick n random files from a category."""
    if "files" in category:
        return [os.path.join(category["folder"], f) for f in category["files"]]

    indices = random.sample(range(1, category["max_index"] + 1), min(n, category["max_index"]))
    paths = []
    for idx in indices:
        path = os.path.join(category["folder"], category["pattern"].format(idx))
        if os.path.exists(path):
            paths.append(path)
    return paths


def run_analysis():
    print("=" * 70)
    print("  MODEL ACCURACY ANALYSIS WITH PHYSICS VALIDATION")
    print("=" * 70)
    print(f"  Samples per category: {SAMPLES_PER_CATEGORY}")
    print(f"  Categories: {len(TEST_CATEGORIES)}")
    print()

    # Load analyzer
    print("[*] Loading ML models...")
    t0 = time.time()
    analyzer = MachineHealthAnalyzer()
    print(f"    Models loaded in {time.time() - t0:.1f}s\n")

    # Tracking
    total_files = 0
    total_correct_fault = 0
    total_fault_applicable = 0
    physics_agree = 0
    physics_disagree = 0
    physics_na = 0
    category_results = []

    # Warning-aware metrics (global accumulators)
    global_strict_correct = 0     # exact match only
    global_relaxed_correct = 0    # warning counts as acceptable for faulty
    global_partial_score = 0.0    # faulty=1.0, warning=0.5, miss=0.0

    for cat in TEST_CATEGORIES:
        print("-" * 70)
        print(f"  Category: {cat['name']}")
        print(f"  Expected: status={cat['expected_status']}, fault={cat['expected_fault']}")
        print("-" * 70)

        files = pick_sample_files(cat)
        if not files:
            print("  [!] No files found, skipping.\n")
            continue

        cat_strict = 0
        cat_relaxed = 0
        cat_partial = 0.0
        cat_correct_fault = 0
        cat_fault_applicable = 0
        cat_physics = {"agree": 0, "disagree": 0, "na": 0}
        cat_modulations = []  # track physics modulation actions

        for fpath in files:
            fname = os.path.basename(fpath)
            total_files += 1

            try:
                t1 = time.time()
                result = analyzer.analyze(fpath)
                elapsed = time.time() - t1

                status = result.get("status", "unknown")
                fault = result.get("failure_type")
                confidence = result.get("confidence", 0) or 0
                pv = result.get("physics_validation", {})
                pv_consistent = pv.get("consistent")
                pv_reason = pv.get("reason", "")
                pv_modulation = pv.get("modulation_applied")
                if pv_modulation:
                    cat_modulations.append(pv_modulation)

                expected = cat["expected_status"]

                # -- Warning-aware scoring --
                if expected == "normal":
                    # normal -> only exact match is correct
                    strict_ok = (status == "normal")
                    relaxed_ok = (status == "normal")
                    partial = 1.0 if status == "normal" else 0.0
                else:
                    # expected == "faulty"
                    strict_ok = (status == "faulty")
                    relaxed_ok = (status in ("faulty", "warning"))  # warning = acceptable detection
                    if status == "faulty":
                        partial = 1.0
                    elif status == "warning":
                        partial = 0.5
                    else:
                        partial = 0.0

                cat_strict += int(strict_ok)
                cat_relaxed += int(relaxed_ok)
                cat_partial += partial
                global_strict_correct += int(strict_ok)
                global_relaxed_correct += int(relaxed_ok)
                global_partial_score += partial

                # -- Fault-type accuracy (only for faulty expected) --
                if cat["expected_fault"] is not None:
                    cat_fault_applicable += 1
                    total_fault_applicable += 1
                    fault_correct = (fault == cat["expected_fault"])
                    if fault_correct:
                        cat_correct_fault += 1
                        total_correct_fault += 1
                else:
                    fault_correct = None

                # -- Physics validation tracking --
                if pv_consistent is True:
                    cat_physics["agree"] += 1
                    physics_agree += 1
                elif pv_consistent is False:
                    cat_physics["disagree"] += 1
                    physics_disagree += 1
                else:
                    cat_physics["na"] += 1
                    physics_na += 1

                # -- Per-file output --
                if strict_ok:
                    mark = "OK"
                elif relaxed_ok:
                    mark = "PART"  # partial credit (warning for expected faulty)
                else:
                    mark = "MISS"

                fault_str = f", fault={fault}" if fault else ""
                physics_str = {True: "AGREE", False: "DISAGREE", None: "N/A"}.get(pv_consistent, "N/A")
                mod_str = f"  mod={pv_modulation}" if pv_modulation else ""

                print(
                    f"  [{mark:4s}] {fname:40s} "
                    f"status={status:7s}{fault_str:25s} "
                    f"conf={confidence:.2f}  physics={physics_str:8s}  "
                    f"({elapsed:.1f}s){mod_str}"
                )
                if pv_reason:
                    print(f"        -> {pv_reason[:100]}")

            except Exception as e:
                print(f"  [ERR] {fname}: {e}")
                traceback.print_exc()

        # Category summary
        n = len(files)
        print(f"\n  Category Summary:")
        print(f"    Strict accuracy:  {cat_strict}/{n} ({cat_strict/n*100:.0f}%)")
        print(f"    Relaxed accuracy: {cat_relaxed}/{n} ({cat_relaxed/n*100:.0f}%)")
        print(f"    Partial score:    {cat_partial:.1f}/{n} ({cat_partial/n*100:.0f}%)")
        if cat_fault_applicable > 0:
            print(f"    Fault accuracy:   {cat_correct_fault}/{cat_fault_applicable} ({cat_correct_fault/cat_fault_applicable*100:.0f}%)")
        print(f"    Physics:  agree={cat_physics['agree']}  disagree={cat_physics['disagree']}  n/a={cat_physics['na']}")
        if cat_modulations:
            from collections import Counter as _Counter
            mod_counts = _Counter(cat_modulations)
            parts = [f"{k}={v}" for k, v in mod_counts.items()]
            print(f"    Physics modulations: {', '.join(parts)}")
        print()

        category_results.append({
            "name": cat["name"],
            "files_tested": n,
            "strict_correct": cat_strict,
            "relaxed_correct": cat_relaxed,
            "partial_score": cat_partial,
            "fault_correct": cat_correct_fault,
            "fault_applicable": cat_fault_applicable,
            "physics": cat_physics,
        })

    # -- Overall Summary --
    print("=" * 70)
    print("  OVERALL RESULTS")
    print("=" * 70)
    print(f"  Total files analyzed:    {total_files}")
    print()

    if total_files > 0:
        strict_pct = global_strict_correct / total_files * 100
        relaxed_pct = global_relaxed_correct / total_files * 100
        avg_partial = global_partial_score / total_files
        print(f"  Strict accuracy:         {global_strict_correct}/{total_files} ({strict_pct:.1f}%)")
        print(f"    (only exact status match counts)")
        print(f"  Relaxed accuracy:        {global_relaxed_correct}/{total_files} ({relaxed_pct:.1f}%)")
        print(f"    (warning counts as acceptable for expected-faulty)")
        print(f"  Avg partial score:       {avg_partial:.2f} / 1.00 ({avg_partial*100:.1f}%)")
        print(f"    (faulty=1.0, warning=0.5, miss=0.0)")
    if total_fault_applicable > 0:
        print(f"  Fault-type accuracy:     {total_correct_fault}/{total_fault_applicable} ({total_correct_fault/total_fault_applicable*100:.1f}%)")
    print()
    print("  Physics Validation Summary:")
    total_physics = physics_agree + physics_disagree + physics_na
    if total_physics:
        print(f"    Agreed with ML:        {physics_agree}/{total_physics} ({physics_agree/total_physics*100:.1f}%)")
        print(f"    Contradicted ML:       {physics_disagree}/{total_physics} ({physics_disagree/total_physics*100:.1f}%)")
        print(f"    Not applicable:        {physics_na}/{total_physics} ({physics_na/total_physics*100:.1f}%)")
    print()

    # Per-category table
    print("  Per-Category Breakdown:")
    header = (
        f"  {'Category':<35s} {'Strict':>7s} {'Relaxed':>8s} "
        f"{'Partial':>8s} {'Fault':>7s} {'Ph-OK':>6s} {'Ph-X':>5s} {'Ph-NA':>6s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cr in category_results:
        n = cr["files_tested"]
        s_str = f"{cr['strict_correct']}/{n}"
        r_str = f"{cr['relaxed_correct']}/{n}"
        p_str = f"{cr['partial_score']:.1f}/{n}"
        f_str = f"{cr['fault_correct']}/{cr['fault_applicable']}" if cr["fault_applicable"] > 0 else "n/a"
        print(
            f"  {cr['name']:<35s} {s_str:>7s} {r_str:>8s} "
            f"{p_str:>8s} {f_str:>7s} "
            f"{cr['physics']['agree']:>6d} {cr['physics']['disagree']:>5d} {cr['physics']['na']:>6d}"
        )
    print("=" * 70)


if __name__ == "__main__":
    random.seed(42)  # Reproducible sampling
    run_analysis()
