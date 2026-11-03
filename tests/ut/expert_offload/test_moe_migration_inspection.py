# SPDX-License-Identifier: Apache-2.0
"""跨宿主机/容器读取的CPU回归；不调用真实docker、ldd或设备。"""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class MigrationInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "script/inspect_moe_migration.py"
        spec = importlib.util.spec_from_file_location("migration_inspection", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_elf_architecture_and_bounded_library_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, machine, expected in (("liburma.so.0", 183, "aarch64"), ("other", 62, "x86_64")):
                header = bytearray(20)
                header[:6] = b"\x7fELF\x02\x01"
                header[18:20] = machine.to_bytes(2, "little")
                (root / name).write_bytes(header)
                self.assertEqual(self.module.elf_machine(root / name), expected)
            hidden = root / "docker"
            hidden.mkdir()
            (hidden / "liburma.so.0").write_bytes(b"not a library")
            found, _ = self.module.find_urma([root])
            self.assertEqual(found, [str(root / "liburma.so.0")])

    def test_inspect_filters_environment_and_unrelated_docker_settings(self):
        record = dict(
            Config={"Image": "example", "Env": ["TOKEN=secret"]}, HostConfig={"Privileged": False, "Binds": ["unused"]}
        )
        result = self.module.docker_summary(record)
        self.assertEqual(result["image"], "example")
        self.assertNotIn("secret", json.dumps(result))
        self.assertNotIn("Binds", result["settings"])

    def test_host_runs_container_read_only_via_stdin(self):
        record = dict(Name="/new_moe", Config={"Image": "base"}, State={"Running": True})
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            patch.object(self.module.sys, "platform", "linux"),
            patch.object(self.module, "local_platform"),
            patch.object(self.module.shutil, "which", return_value="docker"),
            patch.object(
                self.module, "capture", side_effect=[(0, json.dumps([record]), ""), (0, "container ok", "")]
            ) as run,
        ):
            self.assertEqual(self.module.main(["--container", "new_moe"]), 0)
        self.assertEqual(run.call_args_list[0].args[0], ["docker", "inspect", "new_moe"])
        args = run.call_args_list[1].args[0]
        self.assertEqual(args[:7], ["docker", "exec", "-i", "new_moe", "python3", "-B", "-"])
        self.assertIn("def main", run.call_args_list[1].kwargs["input_text"])
        self.assertIn("container ok", output.getvalue())

    def test_stopped_container_is_not_started(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(self.module.sys, "platform", "linux"),
            patch.object(self.module, "local_platform"),
            patch.object(self.module.shutil, "which", return_value="docker"),
            patch.object(self.module, "capture", return_value=(0, '[{"State":{"Running":false}}]', "")) as run,
        ):
            self.assertEqual(self.module.main(["--container", "stopped"]), 1)
            self.assertEqual(run.call_count, 1)

    def test_model_config_prints_only_structural_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "DeepSeek-V4.1-Flash"
            model.mkdir()
            config = dict(model_type="deepseek_v4", quantization_config={"quant_method": "fp8"}, private_token="secret")
            path = model / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            before = path.read_bytes()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.module.source_and_models(root / "absent", root)
            self.assertIn("quant_method", output.getvalue())
            self.assertNotIn("secret", output.getvalue())
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
