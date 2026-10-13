"""Protect the documented manifest, cache, and checkpoint path workflow."""

import unittest
from pathlib import Path

from psmareg.config.fine_tuning_config import load_fine_tuning_config
from psmareg.config.synthetic_pretraining_config import (
    load_synthetic_pretraining_config,)


class ConfigPathTests(unittest.TestCase):
    """Ensure the default stages connect without a release directory."""

    def test_stage_outputs_feed_fine_tuning(self) -> None:
        stages = {
            name: load_synthetic_pretraining_config(
                Path(f"psmareg/config/synthetic_pretraining_stage_{name}.py"))
            for name in "abc"
        }
        final = load_fine_tuning_config(
            Path("psmareg/config/fine_tuning_final.py"))

        self.assertEqual(
            stages["b"].initialization.start_from,
            stages["a"].data.required("output_dir") / "best_val.pt")
        self.assertEqual(
            final.initialization.pretrained_checkpoint,
            stages["c"].data.required("output_dir") / "best_val.pt")
        self.assertEqual(
            final.initialization.compatible_pretrained_checkpoint,
            stages["b"].data.required("output_dir") / "best_val.pt")

    def test_all_stages_use_generated_manifests(self) -> None:
        final = load_fine_tuning_config(
            Path("psmareg/config/fine_tuning_final.py"))
        self.assertEqual(final.data.train_manifest,
                         Path("manifests/paired/train_pairs.csv"))
        self.assertEqual(final.data.val_manifest,
                         Path("manifests/paired/val_pairs.csv"))
        for name in "abc":
            stage = load_synthetic_pretraining_config(
                Path(f"psmareg/config/synthetic_pretraining_stage_{name}.py"))
            self.assertEqual(stage.data.exclude_subjects_manifest,
                             final.data.val_manifest)


if __name__ == "__main__":
    unittest.main()
