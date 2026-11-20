# SPDX-License-Identifier: Apache-2.0
"""Opt-in, GIL-free graph replay of single-card expert weight updates.

Python prepares immutable buffer descriptors once. Captured MTE tasks then
execute native planning, ACL H2D and completion without calling Python/Torch.
"""

import importlib


def validate_native_update_config(config):
    mode = config.get("host_update_backend", "python")
    if mode not in ("python", "native"):
        raise ValueError("host_update_backend must be python or native")
    if mode == "python":
        return
    required = {
        "host_sync_backend": "mte",
        "cache_policy_backend": "cpp",
        "cache_policy_enabled": True,
        "h2d_backend": "torch",
        "host_sync_load_stream": "ordinary",
    }
    for name, value in required.items():
        if config.get(name, value if name in ("h2d_backend", "host_sync_load_stream") else None) != value:
            raise ValueError(f"native updates require {name}={value}")
    incompatible = (
        "enable_multi_card",
        "expert_substitution_enabled",
        "experts_pruning_enabled",
        "callback_timing_port",
        "callback_mstx_enabled",
        "expert_trace_enabled",
        "moe_offload_debug",
        "expert_prefetch_wait_timing",
    )
    if any(config.get(key) for key in incompatible):
        raise ValueError(
            "native updates do not support multi-card, substitution/pruning or Python callback diagnostics"
        )
    for key, value in config.items():
        if (
            key.startswith("host_sync_diagnose_")
            and value
            and key
            not in (
                "host_sync_diagnose_prefetch_wait_at_join",
                "host_sync_diagnose_graph_dump_dir",
            )
        ):
            raise ValueError(f"native updates cannot combine {key}")
    if config.get("expert_prefetch_enabled") and not config.get("host_sync_diagnose_prefetch_wait_at_join"):
        raise ValueError("native prefetch requires the existing strict wait at consumer join")


def dense_span(shape, strides):
    """Return dense per-slot element count, rejecting holes and aliasing."""
    expected = 1
    for size, stride in sorted(zip(shape, strides), key=lambda pair: pair[1]):
        if size < 1 or stride < 0:
            raise ValueError("empty/negative-stride scale slot")
        if size == 1:
            continue
        if stride != expected:
            raise ValueError("native scale slot must be dense and non-overlapping")
        expected *= size
    return expected


def pack_scale(torch, source, destination):
    """Pack logical values into the exact dense physical order of an NPU slot."""
    dense_span(destination.shape, destination.stride())
    packed = torch.empty_strided(
        tuple(destination.shape),
        tuple(destination.stride()),
        dtype=destination.dtype,
        device="cpu",
        pin_memory=True,
    )
    packed.copy_(source.reshape(destination.shape))
    return packed


class NativeUpdateRuntime:
    def __init__(self, manager, expert_weight):
        # Optional imports: default paths require neither new extension APIs nor ACL.
        self.native = importlib.import_module("_vllm_ascend_lrc")
        self.torch = importlib.import_module("torch")
        self.npu = importlib.import_module("torch_npu")
        if not hasattr(self.native, "native_update_task"):
            raise RuntimeError("rebuild csrc/lrc_policy for native weight updates")
        self.manager = manager
        self.owners = []
        self.tables = []
        self.packed_bytes = 0
        # Drain Torch submissions before switching this load stream to direct ACL
        # on graph replay. Eager warmups still synchronize their Torch copies.
        manager.load_stream.synchronize()
        self.stream = int(manager.load_stream.npu_stream)
        for layer_idx, layer in enumerate(manager.moe_layers):
            self.tables.append(self._prepare_layer(layer_idx, layer, expert_weight))
        print(
            f"[MOE-NATIVE] prepared layers={len(self.tables)} scale_buffer_bytes={self.packed_bytes}; "
            "graph callbacks=GIL-free; eager/prefill=original; Python callback timers disabled",
            flush=True,
        )

    def _prepare_layer(self, index, layer, expert_weight):
        m = self.manager
        experts = m.num_total_experts
        slots = m.num_device_experts_for_layer(index)
        sources = [[] for _ in range(experts)]
        destinations = [[] for _ in range(slots)]
        sizes = []
        for name in ("w13", "w2"):
            size = getattr(m, f"{name}_expert_size_bytes")
            dst = expert_weight(layer, f"{name}_weight").data.untyped_storage()
            if dst.nbytes() < slots * size:
                raise ValueError("native weight destination too small")
            self.owners.append(dst)
            sizes.append(size)
            for slot in range(slots):
                destinations[slot].append(dst.data_ptr() + slot * size)
            for eid in range(experts):
                src = m._expert_src_storage(index, eid, name)
                if src.nbytes() < size:
                    raise ValueError("native weight source too small")
                self.owners.append(src)
                sources[eid].append(src.data_ptr())
        for buffers in (m.scale_cpu_buffers, m.offset_cpu_buffers, m.scale_bias_cpu_buffers):
            for name, layers in buffers.items():
                target = getattr(layer, name, None)
                if target is None or index >= len(layers):
                    continue
                local_ids = [m._shard_local(eid) for eid in range(experts)]
                if any(eid is None for eid in local_ids):
                    raise ValueError("native updates require all experts in local CPU buffers")
                cpu_sources = [layers[index][eid] for eid in local_ids]
                self._add_scale(target, cpu_sources, sources, destinations, sizes)
        if hasattr(layer, "w13_weight_scale_fp32"):
            # Match the old runtime .to(float32) on the logical scale values.
            cpu_sources = [
                m.scale_cpu_buffers["w13_weight_scale"][index][m._shard_local(eid)].to(layer.w13_weight_scale.dtype)
                for eid in range(experts)
            ]
            self._add_scale(layer.w13_weight_scale_fp32, cpu_sources, sources, destinations, sizes)
        return self.native.NativeTransfers(sources, destinations, sizes, self.stream, "", False)

    def _add_scale(self, target, cpu_sources, sources, destinations, sizes):
        if self.npu.get_npu_format(target) not in (0, 2):
            raise ValueError("native scales require base NCHW/ND storage; packed device formats are unsupported")
        if target.shape[0] < len(destinations):
            raise ValueError("native scale destination has too few slots")
        template = target[0]
        elements = dense_span(template.shape, template.stride())
        size = elements * template.element_size()
        if target.stride(0) < elements:
            raise ValueError("native scale expert slots overlap")
        storage_end = target.untyped_storage().data_ptr() + target.untyped_storage().nbytes()
        for slot, row in enumerate(destinations):
            dst = target[slot]
            if dst.data_ptr() + size > storage_end:
                raise ValueError("native scale write exceeds destination storage")
            row.append(dst.data_ptr())
        sizes.append(size)
        self.owners.append(target)
        for eid, src in enumerate(cpu_sources):
            packed = pack_scale(self.torch, src, template)
            if packed.untyped_storage().nbytes() != size:
                raise ValueError("packed scale storage contains unexpected padding")
            self.owners.append(packed)
            self.packed_bytes += size
            sources[eid].append(packed.data_ptr())

    def task(self, args):
        if len(args) not in (6, 9) or (len(args) == 9 and (args[6] or args[8] is not None)):
            raise ValueError("unsupported native callback arguments")
        ids, mapping, layer, index, scores, prefetch = args[:6]
        if layer is not self.manager.moe_layers[index]:
            raise ValueError("native layer identity mismatch")
        for tensor in (ids, scores):
            if tensor is not None and (tensor.device.type != "cpu" or not tensor.is_pinned()):
                raise ValueError("native request buffers must be pinned CPU tensors")
        return self.native.native_update_task(
            self.manager.cache_policy._native,
            self.tables[index],
            index,
            ids.numpy(),
            scores.numpy() if scores is not None else None,
            mapping,
            prefetch,
            self.manager.prefetch_topk,
            # Keep manager, staging tensors, source and destination storage alive
            # until Session.close() retires the registered capsules with the GIL.
            (self, ids, scores, mapping),
        )
