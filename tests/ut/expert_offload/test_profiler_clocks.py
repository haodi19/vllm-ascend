# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
import inspect_moe_profiler_clocks as clocks


class ClockMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "run_ascend_pt"
        self.root.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_discovery_accepts_trace_and_parent_and_deduplicates(self):
        trace = self.write("ASCEND_PROFILER_OUTPUT/trace_view.json", "[]")
        self.assertEqual(clocks.discover_roots([trace, self.base, self.root]), [self.root])

    def test_discovery_of_renamed_root_stops_before_raw_capture(self):
        root = self.base / "renamed_capture"
        root.mkdir()
        (root / "profiler_info_0.json").write_text("{}")
        (root / "huge_subdirectory").mkdir()
        with mock.patch.object(clocks, "MAX_ENTRIES", 4):
            self.assertEqual(clocks.discover_roots([root]), [root])

    def test_bounded_directory_scan(self):
        for index in range(3):
            (self.base / f"extra{index}").mkdir()
        with mock.patch.object(clocks, "MAX_ENTRIES", 2), self.assertRaisesRegex(ValueError, "scan exceeds"):
            clocks.discover_roots([self.base])

    def test_no_capture_root_fails_explicitly(self):
        empty = self.base / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "no capture root"):
            clocks.discover_roots([empty])

    def test_only_known_small_metadata_paths_selected(self):
        expected = [
            self.write("profiler_info_0.json", "{}"),
            self.write("PROF_123_1_a/host/info.json", "{}"),
            self.write("PROF_123_1_a/host/host_start.log", "[Host]\ncntvct=123\n"),
            self.write("PROF_123_1_a/host/host_end.log", "[Host]\ncntvct=456\n"),
            self.write("PROF_123_1_a/device_0/start_info.0", "{'begin_time': 123}"),
        ]
        self.write("profiler_metadata.json", '{"secret": "not allowed"}')
        self.write("PROF_123_1_a/host/data/host_start.log", "nested raw not allowed")
        self.write("PROF_123_1_a/host/aclnn.bin", "raw binary not allowed")
        self.assertEqual(clocks.metadata_files(self.root), sorted(expected))

    def test_json_whitelist_preserves_epoch_and_excludes_secrets(self):
        data = {
            "start_info": {"start_monotonic": 1794000000000000001, "syscnt_enable": True, "freq": "100.000"},
            "end_info": {"collectionTimeEnd": 1794000000000000123, "MonotonicTimeEnd": 123},
            "CPU": [{"Frequency": "100.000", "hostname": "secret-host", "ip": "secret-ip"}],
            "environment": {"freq": 123, "http_proxy": "credential"},
            "config": {"proxy_settings": {"freq": 999}},
            "unrelated": 123,
            "torch_npu_version": "2.10.0.post4.dev20260715",
        }
        path = self.write("profiler_info.json", json.dumps(data))
        fields = clocks.clock_fields(clocks.parse_metadata(path))
        by_key = {row["key"]: row["value"] for row in fields}
        self.assertEqual(by_key["start_info.start_monotonic"], 1794000000000000001)
        self.assertEqual(by_key["CPU[0].Frequency"], "100.000")
        self.assertTrue(by_key["start_info.syscnt_enable"])
        printed = json.dumps(fields)
        self.assertNotIn("secret", printed)
        self.assertNotIn("environment", printed)
        self.assertNotIn("proxy", printed)
        self.assertNotIn("unrelated", printed)

    def test_ini_host_start_and_literal_start_info(self):
        ini = self.write(
            "PROF_a/host/host_start.log",
            "[Host]\nclock_monotonic_raw=1234567890123456789\ncntvct=543\nHostIP=private\n",
        )
        fields = clocks.clock_fields(clocks.parse_metadata(ini))
        self.assertEqual(
            {row["key"]: row["value"] for row in fields},
            {
                "Host.clock_monotonic_raw": "1234567890123456789",
                "Host.cntvct": "543",
            },
        )
        start = self.write("PROF_a/host/start_info", "{'collectionTimeBegin': 12345, 'clockMonotonicRaw': 12345678}")
        self.assertEqual(len(clocks.clock_fields(clocks.parse_metadata(start))), 2)

    def test_numbered_cann_names_and_device_colon_format(self):
        expected = [
            self.write(
                "PROF_a/device_7/info.json.7",
                '{"CPU":[{"Frequency":"2400 MHz"}],"DeviceInfo":[{"hwts_frequency":1000,"aic_frequency":500}]}',
            ),
            self.write(
                "PROF_a/device_7/host_start.log.7", "[Host]\nclock_monotonic_raw=1000\ncntvct=99\ncntvct_diff=-3\n"
            ),
            self.write(
                "PROF_a/device_7/dev_start.log.7",
                "clock_monotonic_raw: 2000\ncntvct: 399\nsysClockFreq: 2400\nHostIP: private\n",
            ),
        ]
        self.assertEqual(clocks.metadata_files(self.root), sorted(expected))
        self.assertEqual(
            clocks.parse_metadata(expected[2]), {"clock_monotonic_raw": "2000", "cntvct": "399", "sysClockFreq": "2400"}
        )
        fields = clocks.clock_fields(clocks.parse_metadata(expected[1]))
        self.assertEqual({row["key"]: row["value"] for row in fields}["Host.cntvct_diff"], "-3")
        fields = clocks.clock_fields(clocks.parse_metadata(expected[0]))
        self.assertEqual(fields[0]["type"], "unparsed_unit_string")
        self.assertEqual(fields[0]["value"], "2400 MHz")

    def test_clock_units_preserved_and_unknown_strings_redacted(self):
        fields = clocks.clock_fields(
            {"frequency": "2400.001MHz", "time_unit": "ns", "freq": "secret:credential@proxy", "cntvct": "UNKNOWN"}
        )
        values = {row["key"]: row for row in fields}
        self.assertEqual(values["frequency"]["type"], "unparsed_unit_string")
        self.assertEqual(values["time_unit"]["type"], "declared_unit")
        self.assertTrue(values["freq"]["value_omitted"])
        self.assertNotIn("secret", json.dumps(fields))

    def test_oversized_and_binary_metadata_not_read(self):
        path = self.write("profiler_info.json", "{" + " " * 20 + "}")
        with mock.patch.object(clocks, "MAX_FILE_BYTES", 16), self.assertRaisesRegex(ValueError, "exceeds"):
            clocks.parse_metadata(path)
        path.write_bytes(b"{\x00}")
        with self.assertRaisesRegex(ValueError, "binary"):
            clocks.parse_metadata(path)

    def test_parse_errors_never_echo_payload_and_continue(self):
        self.write("profiler_info.json", "SECRET_BAD_JSON")
        self.write("PROF_a/host/host_start.log", "SECRET_BAD_INI")
        self.write("PROF_a/host/info.json", '{"CPU":[{"Frequency":100}]}')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            clocks.inspect_root(self.root)
        self.assertNotIn("SECRET", output.getvalue())
        self.assertIn("CPU[0].Frequency", output.getvalue())
        self.assertEqual(output.getvalue().count("METADATA_UNREADABLE"), 2)

    def test_numeric_policy_rejects_invalid_scalar_types(self):
        fields = clocks.clock_fields(
            {"freq": False, "start_cnt": float("inf"), "cntvct": "nan", "start_monotonic": "bad"}
        )
        self.assertTrue(all(row["type"] == "unparsed_string" and row["value_omitted"] for row in fields))

    def test_field_structure_and_result_limits(self):
        with mock.patch.object(clocks, "MAX_FIELDS", 1):
            fields = clocks.clock_fields({"freq": 1, "cntvct": 2})
        self.assertEqual(fields[-1], {"key": "<field_limit>", "type": "truncation", "omitted_count": 1})
        with mock.patch.object(clocks, "MAX_DEPTH", 1), self.assertRaisesRegex(ValueError, "structure"):
            clocks.clock_fields({"a": {"b": {"freq": 1}}})

    def test_large_cpu_array_retains_first_cpu_and_device_clock(self):
        data = {
            "CPU": [{"Frequency": str(2400 + index)} for index in range(128)],
            "DeviceInfo": [{"hwts_frequency": 1000}],
        }
        fields = clocks.clock_fields(data)
        keys = {row["key"] for row in fields}
        self.assertIn("CPU[0].Frequency", keys)
        self.assertIn("DeviceInfo[0].hwts_frequency", keys)
        self.assertEqual(fields[-1]["omitted_count"], 29)

    def test_installed_source_inspected_without_importing_package(self):
        relative = clocks.SOURCE_FILES[0]
        self.write(relative, "ratio = time_fmt / self._freq\npassword = 'secret'\n")
        fake = mock.Mock(version="2.10.0")
        fake.locate_file.side_effect = lambda name: self.root / name
        output = io.StringIO()
        with (
            mock.patch.object(clocks.importlib.metadata, "distribution", return_value=fake),
            contextlib.redirect_stdout(output),
        ):
            clocks.inspect_installed_source()
        self.assertIn("ratio = time_fmt", output.getvalue())
        self.assertIn("does not prove capture-time version", output.getvalue())
        self.assertNotIn("secret", output.getvalue())

    def test_optional_clock_sampler_preserves_ns_and_brackets(self):
        values = {
            11: iter([1000000000, 1000000020, 1000000030, 1000000050, 2000000000, 2000000040, 2000000050, 2000000070]),
            12: iter([1000000110, 2000001120]),
            13: iter([3000000040, 4000002060]),
        }
        output = io.StringIO()
        with (
            mock.patch.object(clocks.time, "CLOCK_MONOTONIC_RAW", 11, create=True),
            mock.patch.object(clocks.time, "CLOCK_MONOTONIC", 12, create=True),
            mock.patch.object(clocks.time, "CLOCK_REALTIME", 13, create=True),
            mock.patch.object(
                clocks.time, "clock_gettime_ns", side_effect=lambda clock: next(values[clock]), create=True
            ),
            mock.patch.object(clocks.time, "monotonic", side_effect=[0, 0, 1]),
            mock.patch.object(clocks.time, "sleep"),
            contextlib.redirect_stdout(output),
        ):
            clocks.sample_host_clocks(1)
        rows = [
            json.loads(line.split(" ", 1)[1])
            for line in output.getvalue().splitlines()
            if line.startswith("HOST_CLOCK_SAMPLE ")
        ]
        self.assertEqual(rows[0]["offset_change_us"], 1.0)
        self.assertEqual(rows[1]["offset_change_us"], 2.0)
        self.assertEqual(rows[0]["max_bracket_ns"], 40)
        self.assertAlmostEqual(rows[0]["bracket_rate_uncertainty_us_per_second"], 0.03, places=6)
        self.assertAlmostEqual(rows[0]["change_us_per_second"], 1.0, places=6)

    def test_cann_source_discovery_is_fixed_path_and_methods_not_executed(self):
        relative = clocks.CANN_SOURCE_METHODS[0][0]
        path = self.write(
            "tools/profiler/profiler_tool/" + relative,
            'raise RuntimeError("do not execute")\nclass Clock:\n'
            '    def get_host_freq(self):\n        """not printed"""\n'
            '        return 2400 * 1000000\n    def unrelated(self):\n        return "secret"\n',
        )
        self.write("random/" + relative, "not selected")
        selected = clocks.cann_source_paths([self.root, self.root])
        self.assertEqual([item[0] for item in selected], [path.resolve()])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            clocks.inspect_cann_source([self.root])
        self.assertIn("return 2400 * 1000000", output.getvalue())
        self.assertIn("CANN_METHOD_MISSING=time_from_host_syscnt", output.getvalue())
        self.assertNotIn("not printed", output.getvalue())
        self.assertNotIn("secret", output.getvalue())

    def test_cann_source_line_limit_is_explicit(self):
        self.write("analysis/common_func/info_conf_reader.py", "def get_host_freq():\n    a = 1\n    return a\n")
        output = io.StringIO()
        with mock.patch.object(clocks, "MAX_CANN_SOURCE_LINES", 2), contextlib.redirect_stdout(output):
            clocks.inspect_cann_source([self.root])
        self.assertIn("shown=2; truncated=True", output.getvalue())
        self.assertNotIn("return a", output.getvalue())

    def test_sample_duration_validation(self):
        self.assertEqual(clocks.sample_seconds("5"), 5)
        for value in ("nan", "inf", "-1", "31"):
            with self.assertRaises(clocks.argparse.ArgumentTypeError):
                clocks.sample_seconds(value)


if __name__ == "__main__":
    unittest.main()
