# SPDX-License-Identifier: Apache-2.0
"""CPU ordering and failure checks for the opt-in captured prefetch experiment."""

import __future__

import ast
import importlib.util
import threading
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class Stream:
    def __init__(self, name, pointer, calls):
        self.name, self.npu_stream, self.calls = name, pointer, calls

    def record_event(self, event=None):
        event = object() if event is None else event
        self.calls.append((self.name, "record", event))
        return event

    def wait_event(self, event):
        self.calls.append((self.name, "wait", event))


class PrefetchJoinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location(
            "prefetch_join_host_sync", cls.root / "vllm_ascend/expert_offload/host_sync.py"
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        self.calls = []
        self.main = Stream("main", 11, self.calls)
        self.prefetch = Stream("prefetch", 22, self.calls)
        self.current = self.main

        @contextmanager
        def stream_context(stream):
            previous, self.current = self.current, stream
            try:
                yield
            finally:
                self.current = previous

        self.npu = SimpleNamespace(stream=stream_context, current_stream=lambda: self.current, Event=object)
        self.obj = self.module.MteHostDispatcher.__new__(self.module.MteHostDispatcher)
        self.obj.manager = SimpleNamespace(_prefetch_stream=self.prefetch, offload_config=SimpleNamespace())
        for name, value in {
            "closed": False,
            "next_slot": 1,
            "allocated_scan": False,
            "alias": 10000,
            "stride": 1024,
            "done_offset": 512,
            "state_base": 20000,
            "join_capture": None,
            "join_capture_failed": False,
            "join_capture_index": 0,
            "deferred_waits": {},
            "deferred_prefetch": {},
            "join_events": [],
            "join_published_slots": [],
            "joined_slots": [],
            "entries": [None] * 8,
            "entry_streams": [None] * 8,
        }.items():
            setattr(self.obj, name, value)
        self.obj.torch = SimpleNamespace(npu=self.npu)
        self.obj.publish = Mock(side_effect=lambda *args: self.calls.append(("publish", *args)))
        self.obj.wait = Mock(side_effect=lambda *args: self.calls.append(("strict_wait", *args)))
        self.source = object()
        self.destination = SimpleNamespace(
            copy_=Mock(side_effect=lambda source, **kwargs: self.calls.append(("mapping", source)))
        )
        self.print_patch = patch("builtins.print")
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)

    def manager_method(self, name):
        tree = ast.parse(
            (self.root / "vllm_ascend/expert_offload/expert_offload_manager.py").read_text(encoding="utf-8")
        )
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        namespace = {
            "torch_npu": SimpleNamespace(npu=self.npu),
            "_EXTRA_CTX": SimpleNamespace(capturing=True),
            "get_subscribed_compute_streams": lambda: set(),
            "callback_boundary_stage": lambda *args: nullcontext(),
        }
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]), name, "exec", flags=__future__.annotations.compiler_flag
            ),
            namespace,
        )
        return namespace[name]

    def test_same_slot_and_strict_wait_follow_compute_gate_before_mapping(self):
        callback = Mock()
        with self.obj.prefetch_join_capture():
            self.obj.defer_prefetch(self.prefetch, callback, "args", 1, self.destination, self.source)
            self.obj.wait.assert_not_called()
            callback.assert_not_called()
            self.calls.append(("main", "independent_compute"))
            done = self.obj.finish_prefetch_at_join(self.main, 1)
            self.main.wait_event(done)
        publish, _, gate, gate_wait, strict_wait, mapping, record_done, consumer_wait = self.calls
        self.assertEqual(gate[:2], ("main", "record"))
        self.assertEqual(gate_wait, ("prefetch", "wait", gate[2]))
        self.assertEqual(strict_wait[1:], (22, publish[2] + 512, publish[3]))
        self.assertEqual(mapping, ("mapping", self.source))
        self.assertEqual(record_done, ("prefetch", "record", done))
        self.assertEqual(consumer_wait, ("main", "wait", done))
        self.assertEqual(self.obj.join_events, [gate[2], done])
        self.assertEqual(self.obj.join_published_slots, self.obj.joined_slots)

    def test_wrong_stream_duplicate_and_foreign_tokens_cannot_enqueue_wait(self):
        with self.obj.prefetch_join_capture():
            token = self.obj.publish_for_join(self.prefetch, Mock(), ())
            with self.assertRaisesRegex(RuntimeError, "original prefetch"):
                self.obj.wait_for_join(self.main, token)
            with self.assertRaisesRegex(RuntimeError, "foreign"):
                self.obj.wait_for_join(self.prefetch, object())
            self.obj.wait.assert_not_called()
            self.obj.wait_for_join(self.prefetch, token)
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                self.obj.wait_for_join(self.prefetch, token)
            self.obj.wait.assert_called_once()

    def test_early_strict_wait_keeps_mapping_and_staging_behind_consumer_gate(self):
        self.obj.manager.offload_config.host_sync_diagnose_prefetch_wait_before_gate = True
        with self.obj.prefetch_join_capture():
            self.obj.defer_prefetch(self.prefetch, Mock(), (), 1, self.destination, self.source)
            self.obj.wait.assert_called_once()
            self.destination.copy_.assert_not_called()
            self.assertEqual(self.obj.deferred_waits, {})
            with self.assertRaisesRegex(RuntimeError, "staging"):
                self.obj.defer_prefetch(self.prefetch, Mock(), (), 2, self.destination, self.source)
            self.calls.append(("main", "independent_compute"))
            done = self.obj.finish_prefetch_at_join(self.main, 1)
            self.main.wait_event(done)
        publish, wait, compute, gate, gate_wait, mapping, record_done, consumer_wait = self.calls
        self.assertEqual(wait[1:], (22, publish[2] + 512, publish[3]))
        self.assertEqual(compute, ("main", "independent_compute"))
        self.assertEqual(gate[:2], ("main", "record"))
        self.assertEqual(gate_wait, ("prefetch", "wait", gate[2]))
        self.assertEqual(mapping, ("mapping", self.source))
        self.assertEqual(record_done, ("prefetch", "record", done))
        self.assertEqual(consumer_wait, ("main", "wait", done))
        self.obj.wait.assert_called_once()

    def test_early_wait_does_not_make_orphan_mapping_a_complete_capture(self):
        self.obj.manager.offload_config.host_sync_diagnose_prefetch_wait_before_gate = True
        with self.assertRaisesRegex(RuntimeError, "unjoined"), self.obj.prefetch_join_capture():
            self.obj.defer_prefetch(self.prefetch, Mock(), (), 1, self.destination, self.source)
        self.assertTrue(self.obj.join_capture_failed)
        self.destination.copy_.assert_not_called()

    def test_orphan_poison_prevents_reuse_in_next_capture(self):
        with self.assertRaisesRegex(RuntimeError, "unjoined"), self.obj.prefetch_join_capture():
            token = self.obj.publish_for_join(self.prefetch, Mock(), ())
        self.assertTrue(self.obj.join_capture_failed)
        with self.assertRaisesRegex(RuntimeError, "failed"), self.obj.prefetch_join_capture():
            self.fail("failed captures cannot be reused")
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.obj.wait_for_join(self.prefetch, token)
        self.obj.wait.assert_not_called()

    def test_buffer_reuse_wrong_layer_and_reentrant_capture_fail_closed(self):
        with self.obj.prefetch_join_capture():
            self.obj.defer_prefetch(self.prefetch, Mock(), (), 3, self.destination, self.source)
            for layer in (3, 4):
                with self.assertRaisesRegex(RuntimeError, "staging buffers"):
                    self.obj.defer_prefetch(self.prefetch, Mock(), (), layer, self.destination, self.source)
            with self.assertRaisesRegex(RuntimeError, "wrong consumer"):
                self.obj.finish_prefetch_at_join(self.main, 4)
            with self.assertRaisesRegex(RuntimeError, "already active"), self.obj.prefetch_join_capture():
                self.fail("must not nest captures")
            self.obj.finish_prefetch_at_join(self.main, 3)
        self.obj.publish.assert_called_once()

    def test_new_capture_allocates_new_slot_and_stale_token_is_rejected(self):
        with self.obj.prefetch_join_capture():
            old = self.obj.publish_for_join(self.prefetch, Mock(), ())
            self.obj.wait_for_join(self.prefetch, old)
        with self.obj.prefetch_join_capture():
            new = self.obj.publish_for_join(self.prefetch, Mock(), ())
            with self.assertRaisesRegex(RuntimeError, "stale"):
                self.obj.wait_for_join(self.prefetch, old)
            self.obj.wait_for_join(self.prefetch, new)
        self.assertEqual(self.obj.next_slot, 3)
        self.assertEqual(self.obj.joined_slots, [2])

    def test_manager_dispatch_defers_mapping_and_consumer_joins_before_early_return(self):
        layer = SimpleNamespace(log2phy=self.destination)
        config = SimpleNamespace(host_sync_diagnose_prefetch_wait_at_join=True)
        manager = SimpleNamespace(
            _host_dispatcher=self.obj,
            _prefetch_stream=self.prefetch,
            offload_config=config,
            enable_multi_card=False,
            _build_prefetch_call=lambda *args: (Mock(), ()),
            _prefetch_state_lock=threading.Lock(),
            _prefetch_layer_npu_event={},
            moe_layers=[layer],
            _finish_pending_predict=lambda index: None,
            _pf_wait_timing=False,
        )
        self.obj.manager = manager
        staged = (object(), object(), self.source, object(), layer, 0, None)
        with self.obj.prefetch_join_capture():
            self.manager_method("_dispatch_prefetch")(manager, staged, None)
            self.destination.copy_.assert_not_called()
            # log2phy=None takes the existing early return AFTER the join.
            self.manager_method("update_weights")(manager, layer, Mock(), None)
        self.destination.copy_.assert_called_once_with(self.source, non_blocking=True)
        self.assertEqual(self.calls[-1][:2], ("main", "wait"))
        self.assertEqual(manager._prefetch_layer_npu_event, {})

    def test_original_launch_still_publishes_then_strictly_waits(self):
        self.obj.launch(self.prefetch, Mock(), ())
        self.assertEqual([call[0] for call in self.calls], ["publish", "strict_wait"])
        self.assertEqual(self.obj.deferred_waits, {})

    def test_mapping_failure_poison_blocks_another_publish(self):
        self.destination.copy_.side_effect = RuntimeError("mapping failed")
        with self.assertRaisesRegex(RuntimeError, "mapping failed"), self.obj.prefetch_join_capture():
            self.obj.defer_prefetch(self.prefetch, Mock(), (), 1, self.destination, self.source)
            self.obj.finish_prefetch_at_join(self.main, 1)
        self.assertTrue(self.obj.join_capture_failed)
        with self.assertRaisesRegex(RuntimeError, "active full capture"):
            self.obj.publish_for_join(self.prefetch, Mock(), ())
        with self.assertRaisesRegex(RuntimeError, "failed MTE join"):
            self.obj.launch(self.prefetch, Mock(), ())
        self.obj.publish.assert_called_once()

    def test_consumer_cannot_be_prefetch_stream(self):
        with self.obj.prefetch_join_capture():
            self.obj.defer_prefetch(self.prefetch, Mock(), (), 1, self.destination, self.source)
            with self.assertRaisesRegex(RuntimeError, "distinct compute"):
                self.obj.finish_prefetch_at_join(self.prefetch, 1)
            self.obj.wait.assert_not_called()
            self.obj.finish_prefetch_at_join(self.main, 1)

    def test_manager_rejects_next_producer_before_shared_staging(self):
        layer = object()
        manager = SimpleNamespace(
            _host_dispatcher=self.obj,
            offload_config=SimpleNamespace(host_sync_diagnose_prefetch_wait_at_join=True, expert_prefetch_enabled=True),
            _skip_prefill=False,
            expert_predictor=None,
            moe_layers=[layer],
            _stage_predicted_topk=Mock(side_effect=AssertionError("must not stage")),
        )
        with self.obj.prefetch_join_capture():
            self.obj.defer_prefetch(self.prefetch, Mock(), (), 1, self.destination, self.source)
            with self.assertRaisesRegex(RuntimeError, "unjoined"):
                self.manager_method("trigger_next_layer_prefetch")(manager, layer)
            manager._stage_predicted_topk.assert_not_called()
            self.obj.finish_prefetch_at_join(self.main, 1)

    def test_eager_keeps_direct_callback_mapping_and_original_event(self):
        callback = Mock()
        source = SimpleNamespace(clone=Mock(return_value=object()))
        manager = SimpleNamespace(
            _host_dispatcher=self.obj,
            _prefetch_stream=self.prefetch,
            offload_config=SimpleNamespace(host_sync_diagnose_prefetch_wait_at_join=True),
            enable_multi_card=False,
            _build_prefetch_call=lambda *args: (callback, ()),
            _prefetch_state_lock=threading.Lock(),
            _prefetch_layer_npu_event={},
        )
        method = self.manager_method("_dispatch_prefetch")
        method.__globals__["_EXTRA_CTX"].capturing = False
        staged = (object(), object(), source, object(), SimpleNamespace(log2phy=self.destination), 1, None)
        method(manager, staged, None)
        callback.assert_called_once_with(())
        self.destination.copy_.assert_called_once_with(source, non_blocking=False)
        self.assertIn(1, manager._prefetch_layer_npu_event)
        self.obj.publish.assert_not_called()
        self.assertEqual(self.obj.deferred_prefetch, {})

    def test_configuration_rejects_mixed_or_ambiguous_experiments(self):
        valid = {
            "host_sync_backend": "mte",
            "expert_offload": True,
            "expert_prefetch_enabled": True,
            "host_sync_diagnose_graph_dump": True,
            "host_sync_diagnose_prefetch_wait_at_join": True,
            "host_sync_library": str(Path(__file__).resolve()),
        }
        self.module.validate_host_sync_config(valid)
        self.module.validate_host_sync_config({**valid, "host_sync_diagnose_prefetch_wait_before_gate": True})
        for changes in (
            {"host_sync_diagnose_prefetch_wait_before_gate": 1},
            {"host_sync_diagnose_prefetch_wait_before_gate": True, "host_sync_diagnose_prefetch_wait_at_join": False},
            {"host_sync_diagnose_prefetch_wait_before_gate": True, "host_sync_diagnose_graph_dump": False},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.module.validate_host_sync_config({**valid, **changes})
        self.module.validate_host_sync_config({**valid, "host_sync_diagnose_graph_dump": False})
        self.module.validate_host_sync_config(
            {key: value for key, value in valid.items() if key != "host_sync_diagnose_graph_dump"}
        )
        for changes in (
            {"host_sync_diagnose_prefetch_wait_at_join": 1},
            {"host_sync_backend": "callback"},
            {"expert_prefetch_enabled": False},
            {"expert_predictor": "trained"},
            {"host_sync_diagnose_prefetch_report_wait": True},
            {"enable_multi_card": True},
            {"host_sync_diagnose_compute_cube_cores": 14},
            {"host_sync_diagnose_split_scale": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.module.validate_host_sync_config({**valid, **changes})


if __name__ == "__main__":
    unittest.main()
