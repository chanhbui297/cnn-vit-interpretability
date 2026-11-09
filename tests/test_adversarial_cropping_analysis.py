"""Unit tests for adversarial-cropping result analysis."""

import unittest

import pandas as pd

from experiments.adversarial_cropping.analyze_results import (
    build_transfer_metrics,
    select_worst_conditions,
)


class AdversarialCroppingAnalysisTest(unittest.TestCase):
    def test_worst_condition_uses_documented_tie_order(self):
        rows = []
        for model_name in ("resnet50", "vit_b_16"):
            for condition, severity, position in (
                ("area_065__center", 0.65, "center"),
                ("area_050__top_left", 0.50, "top_left"),
                ("area_050__center", 0.50, "center"),
            ):
                rows.append({
                    "model_name": model_name,
                    "condition": condition,
                    "severity": severity,
                    "position": position,
                    "absolute_top1_drop_pp": 5.0,
                })

        selected = select_worst_conditions(pd.DataFrame(rows))

        self.assertEqual(
            selected.set_index("model_name")["condition"].to_dict(),
            {"resnet50": "area_050__center", "vit_b_16": "area_050__center"},
        )

    def test_transfer_uses_source_condition_unchanged(self):
        metrics = pd.DataFrame([
            {"model_name": "resnet50", "condition": "crop_a", "severity": 0.5,
             "position": "center", "absolute_top1_drop_pp": 6.0},
            {"model_name": "vit_b_16", "condition": "crop_a", "severity": 0.5,
             "position": "center", "absolute_top1_drop_pp": 4.0},
            {"model_name": "resnet50", "condition": "crop_b", "severity": 0.5,
             "position": "top_left", "absolute_top1_drop_pp": 3.0},
            {"model_name": "vit_b_16", "condition": "crop_b", "severity": 0.5,
             "position": "top_left", "absolute_top1_drop_pp": 7.0},
        ])
        worst = pd.DataFrame([
            {"model_name": "resnet50", "condition": "crop_a"},
            {"model_name": "vit_b_16", "condition": "crop_b"},
        ])

        transfer = build_transfer_metrics(metrics, worst)

        self.assertEqual(len(transfer), 4)
        self.assertEqual(
            set(zip(transfer["source_model"], transfer["selected_condition"])),
            {("resnet50", "crop_a"), ("vit_b_16", "crop_b")},
        )


if __name__ == "__main__":
    unittest.main()
