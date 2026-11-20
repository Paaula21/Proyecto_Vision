"""El holdout por especie no puede aprobar una validación con frutas repetidas."""

import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_split


class SpeciesSplitTests(unittest.TestCase):
    def test_leaky_holdout_is_rejected_unless_explicitly_diagnostic(self):
        data = PROJECT_ROOT / "fruits_part1" / "train"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "species_holdout"
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "splits" / "make_species_split.py"),
                 "--data", str(data), "--out", str(output)],
                check=True, capture_output=True, text=True,
            )
            checks = json.loads((output / "checks.json").read_text(encoding="utf-8"))
            self.assertFalse(checks["passed"])
            self.assertFalse(checks["no_fruit_leak"])
            self.assertTrue(checks["diagnostic_only"])
            self.assertGreater(checks["n_shared_fruit_ids"], 0)

            with self.assertRaisesRegex(ValueError, "no ha superado"):
                load_split(data, output, 0)
            with warnings.catch_warnings(record=True) as emitted:
                train_ids, val_ids, fold, _ = load_split(data, output, 0, allow_leaky=True)
            self.assertEqual((len(train_ids), len(val_ids)), (160, 40))
            self.assertEqual(fold["n_shared_fruit_ids"], checks["n_shared_fruit_ids"])
            self.assertTrue(emitted)

            checks["passed"] = True
            (output / "checks.json").write_text(json.dumps(checks), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no puede aprobarse"):
                load_split(data, output, 0)


if __name__ == "__main__":
    unittest.main()
