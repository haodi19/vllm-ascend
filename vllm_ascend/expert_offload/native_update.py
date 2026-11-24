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
        "expert_trace_enabled",
        "moe_offload_debug",
        "expert_prefetch_wait_timing",
    )
    if any(config.get(key) for key in incompatible):
        raise ValueError(
            "native updates do not support multi-card, substitution/pruning or Python callback diagnostics"
        )
    if bool(config.get("callback_timing_port")) != bool(config.get("callback_mstx_enabled")):
        raise ValueError("native diagnostics require both control port and native MSTX ranges")
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
    if source.device.type != "cpu" or source.numel() != destination.numel():
        raise ValueError("native scale source must be an equal-sized CPU tensor")
    packed = torch.empty_strided(
        tuple(destination.shape),
        tuple(destination.stride()),
        dtype=destination.dtype,
        device="cpu",
        pin_memory=True,
    )
    packed.copy_(source.reshape(destination.shape))
    return packed


def prepare_scale(torch, source, destination):
    """Reuse pinned physical bytes when safe; otherwise pack once at startup."""
    elements = dense_span(destination.shape, destination.stride())
    if source.device.type != "cpu" or source.numel() != elements:
        raise ValueError("native scale source must be an equal-sized CPU tensor")
    same_layout = tuple(source.shape) == tuple(destination.shape) and all(
        size == 1 or src_stride == dst_stride
        for size, src_stride, dst_stride in zip(destination.shape, source.stride(), destination.stride())
    )
    compatible = same_layout or (source.is_contiguous() and destination.is_contiguous())
    if source.dtype == destination.dtype and source.is_pinned() and compatible:
        # data_ptr includes storage_offset. A source may be a view of a slab.
        nbytes = elements * source.element_size()
        offset = source.storage_offset() * source.element_size()
        if offset < 0 or offset + nbytes > source.untyped_storage().nbytes():
            raise ValueError("native scale source range exceeds storage")
        return source, True
    return pack_scale(torch, source, destination), False


def tensor_layout(tensor):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.untyped_storage().nbytes(),
    )


class NativeUpdateRuntime:
    def __init__(self, manager, expert_weight):
        # Optional imports: default paths require neither new extension APIs nor ACL.
        self.native = importlib.import_module("_vllm_ascend_lrc")
        self.torch = importlib.import_module("torch")
        self.npu = importlib.import_module("torch_npu")
        if not hasattr(self.native, "native_update_task"):
            raise RuntimeError("rebuild csrc/lrc_policy for native weight updates")
        self.manager = manager
        self.expert_weight = expert_weight
        self.owners = []
        self.tables = []
        self.packed_bytes = 0
        self.reused_bytes = 0
        self.targets = []
        self.profiler = None
        if manager.offload_config.callback_mstx_enabled:
            if not hasattr(self.native, "NativeRanges"):
                raise RuntimeError("rebuild csrc/lrc_policy for native profiling")
            self.profiler = self.native.NativeRanges()
        # Drain Torch submissions before switching this load stream to direct ACL
        # on graph replay. Eager warmups still synchronize their Torch copies.
        manager.load_stream.synchronize()
        self.stream = int(manager.load_stream.npu_stream)
        for layer_idx, layer in enumerate(manager.moe_layers):
            self.tables.append(self._prepare_layer(layer_idx, layer, expert_weight))
        print(
            f"[MOE-NATIVE] prepared layers={len(self.tables)} scale_buffer_bytes={self.packed_bytes} "
            f"scale_reused_bytes={self.reused_bytes}; "
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
        targets = {}
        for name in ("w13", "w2"):
            size = getattr(m, f"{name}_expert_size_bytes")
            tensor = expert_weight(layer, f"{name}_weight")
            targets[f"{name}_weight"] = (tensor, tensor_layout(tensor))
            dst = tensor.data.untyped_storage()
            offset = tensor.storage_offset() * tensor.element_size()
            if offset + slots * size > dst.nbytes():
                raise ValueError("native weight destination too small")
            self.owners.append(dst)
            sizes.append(size)
            for slot in range(slots):
                destinations[slot].append(dst.data_ptr() + offset + slot * size)
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
                targets[name] = (target, tensor_layout(target))
                self._add_scale(target, cpu_sources, sources, destinations, sizes)
        if hasattr(layer, "w13_weight_scale_fp32"):
            # Match the old runtime .to(float32) on the logical scale values.
            cpu_sources = [
                m.scale_cpu_buffers["w13_weight_scale"][index][m._shard_local(eid)].to(layer.w13_weight_scale.dtype)
                for eid in range(experts)
            ]
            self._add_scale(layer.w13_weight_scale_fp32, cpu_sources, sources, destinations, sizes)
            targets["w13_weight_scale_fp32"] = (layer.w13_weight_scale_fp32, tensor_layout(layer.w13_weight_scale_fp32))
        self.targets.append(targets)
        table = self.native.NativeTransfers(sources, destinations, sizes, self.stream, "", False)
        if self.profiler is not None:
            table.set_profiler(self.profiler)
        return table

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
            packed, reused = prepare_scale(self.torch, src, template)
            self.owners.append(packed)
            if reused:
                self.reused_bytes += size
            else:
                self.packed_bytes += size
            sources[eid].append(packed.data_ptr())

    def task(self, args):
        if len(args) not in (6, 9) or (len(args) == 9 and (args[6] or args[8] is not None)):
            raise ValueError("unsupported native callback arguments")
        ids, mapping, layer, index, scores, prefetch = args[:6]
        if not 0 <= index < len(self.tables):
            raise ValueError("native layer was not prepared at finalization; cannot capture unregistered buffers")
        if layer is not self.manager.moe_layers[index]:
            raise ValueError("native layer identity mismatch")
        for name, (original, layout) in self.targets[index].items():
            current = self.expert_weight(layer, name) if name in ("w13_weight", "w2_weight") else getattr(layer, name)
            if current is not original or tensor_layout(current) != layout:
                raise RuntimeError("native destination changed after registration; restart with rebuilt descriptors")
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
