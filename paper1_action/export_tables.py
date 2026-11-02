"""Export machine-readable paper result tables from validated aggregate JSON."""
from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TABLES = HERE / "tables"
KINDS = ("ann", "leaky_ann", "snn", "gru")


def write(name, fields, rows):
    TABLES.mkdir(exist_ok=True)
    with (TABLES / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    summaries = json.loads((RESULTS / "action_decision_summary.json").read_text())
    primary = []
    for path, group in summaries.items():
        for kind in KINDS:
            row = group[kind]
            primary.append({"path": path, "model": kind, "seeds": 5,
                            "n_parameters": row["parameters"],
                            "accuracy_mean": row["accuracy"]["mean"],
                            "accuracy_sample_sd": row["accuracy"]["sample_sd"],
                            "macro_f1_mean": row["macro_f1"]["mean"],
                            "macro_f1_sample_sd": row["macro_f1"]["sample_sd"],
                            "coverage_mean": row["selective_coverage"]["mean"],
                            "accepted_error_mean": row["selective_error"]["mean"]})
    fields = list(primary[0])
    write("recurrent_accuracy.csv", fields, primary)

    baseline = []
    for stem in ("slurp_asr_bow", "slurp_oracle_bow",
                 "speech_massive_fr_asr_bow"):
        row = json.loads((RESULTS / f"{stem}.json").read_text())
        baseline.append({"path": stem, "model": "tfidf_bigram_logistic",
                         "accuracy": row["test"]["accuracy"],
                         "macro_f1": row["test"]["macro_f1"],
                         "coverage": row["test"]["selective_coverage"],
                         "accepted_error": row["test"]["selective_error"]})
    write("text_baselines.csv", list(baseline[0]), baseline)

    calibration = json.loads((RESULTS / "calibration_risk_summary.json").read_text())
    cal_rows = [{"path": path, "model": model, **metrics}
                for path, models in calibration.items()
                for model, metrics in models.items()]
    write("calibration_and_selective_risk.csv",
          ["path", "model", "ece_before_mean", "ece_after_mean",
           "aurc_mean", "aurc_sample_sd"], cal_rows)

    costs = json.loads((RESULTS / "decision_cost.json").read_text())
    cost_rows = []
    for path, levels in costs["five_seed_means"].items():
        for defer_cost, models in levels.items():
            for model, metrics in models.items():
                cost_rows.append({"path": path, "model": model,
                                  "defer_cost": defer_cost, "fits": 5, **metrics})
    for defer_cost, item in costs["bow_asr_single_fit"].items():
        cost_rows.append({"path": "asr", "model": "tfidf_bigram_logistic",
                          "defer_cost": defer_cost, "fits": 1, **item["test"]})
    write("action_deferral_cost.csv", list(cost_rows[0]), cost_rows)

    intervals = []
    for path, group in summaries.items():
        for comparator in ("ann", "leaky_ann", "gru"):
            row = group[f"paired_accuracy_difference_snn_minus_{comparator}"]
            intervals.append({"path": path, "comparison": f"snn_minus_{comparator}",
                              "mean": row["mean"],
                              "hierarchical_95ci_low": row["hierarchical_95ci"][0],
                              "hierarchical_95ci_high": row["hierarchical_95ci"][1],
                              "paired_utterance_95ci_low": row["bootstrap_95ci"][0],
                              "paired_utterance_95ci_high": row["bootstrap_95ci"][1]})
    write("paired_accuracy_intervals.csv", list(intervals[0]), intervals)
    print(f"Wrote {len(list(TABLES.glob('*.csv')))} tables to {TABLES}")


if __name__ == "__main__":
    main()
