"""CPU regression: relocation preserves calibration and config bytes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'zero-task/migration_render.py'
spec = importlib.util.spec_from_file_location('migration', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RemapTests(unittest.TestCase):
    def test_calibration_and_bytes_survive_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for side in ('left', 'right'):
                name = f'curobo_{side}.yml'
                (root / name).write_text('planner: {frame_bias: [1, 2, 3]}')
                original = f'/robot/aloha-agilex/{name}'
                fixed = module.remap_robot_config(original, root)
                self.assertIn('aloha-agilex', fixed)
                self.assertEqual(Path(fixed).read_bytes(), (root / name).read_bytes())
                self.assertEqual(module.remap_robot_config(original, root), fixed)
                self.assertNotIn('aloha-agilex', module.remap_robot_config(f'/robot/other/{name}', root))

    def test_incompatible_existing_alias_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'curobo_left.yml').write_text('new')
            (root / 'aloha-agilex').mkdir()
            (root / 'aloha-agilex/curobo_left.yml').write_text('stale')
            with self.assertRaises(ValueError):
                module.remap_robot_config('/robot/aloha-agilex/curobo_left.yml', root)

    def test_cannot_enable_aloha_for_other_robot(self):
        with self.assertRaises(ValueError):
            module.remap_robot_config('/other/curobo_left.yml', '/migration/aloha-agilex')

    def test_unrelated_config_unchanged(self):
        self.assertEqual(module.remap_robot_config('/other/custom.yml', '/migration'), '/other/custom.yml')


if __name__ == '__main__':
    unittest.main()
