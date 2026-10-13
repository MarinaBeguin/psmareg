"""Smoke checks of the command-line entry points and shipped recipes."""

from __future__ import annotations

import importlib
from pathlib import Path
import unittest

import train_fine_tuning
import train_synthetic_registration_pretraining

CONFIG_DIR = Path(__file__).resolve().parents[1] / "psmareg" / "config"
ENTRY_POINTS = (
    "train_synthetic_registration_pretraining",
    "train_fine_tuning",
    "scripts.visualize_registration",
    "scripts.visualize_val_registration_grid",
)


class EntryPointTests(unittest.TestCase):
    """Catch moved names and recipes that no longer load."""

    def test_entry_points_import(self) -> None:
        for name in ENTRY_POINTS:
            with self.subTest(entry_point=name):
                self.assertTrue(hasattr(importlib.import_module(name), "main"))

    def test_final_recipe_loads_with_command_line_overrides(self) -> None:
        config = train_fine_tuning.parse_fine_tuning_config([
            "--config",
            str(CONFIG_DIR / "fine_tuning_final.py"), "--epochs", "2", "--set",
            "num_workers=0", "--set", "pretrained_checkpoint=none"
        ])
        self.assertEqual(config.optimization.epochs, 2)
        self.assertEqual(config.runtime.num_workers, 0)
        self.assertIsNone(config.initialization.pretrained_checkpoint)

    def test_stage_recipes_load_and_chain(self) -> None:
        models = {}
        for stage in ("a", "b", "c"):
            recipe = CONFIG_DIR / f"synthetic_pretraining_stage_{stage}.py"
            parse = (train_synthetic_registration_pretraining.
                     parse_synthetic_pretraining_config)
            config = parse(["--config", str(recipe), "--set", "lr=1e-5"])
            self.assertEqual(config.optimization.lr, 1e-5)
            models[stage] = config.architecture.model
        self.assertEqual(models["a"], models["b"])
        self.assertEqual(models["c"], "affine_transmorph_v2_cascade")

    def test_unknown_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown"):
            train_fine_tuning.parse_fine_tuning_config([
                "--config",
                str(CONFIG_DIR / "fine_tuning_final.py"), "--set",
                "not_a_parameter=1"
            ])


if __name__ == "__main__":
    unittest.main()
