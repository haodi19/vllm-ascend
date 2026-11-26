# SPDX-License-Identifier: Apache-2.0
"""Early FATE prediction while current demand DMA is still outstanding.

Each captured request owns its route/mapping snapshot. Python only builds the
graph; planning, queued submission and device completion run without Python.
"""


def stage_route(runtime, index, ids, scores, prefetch):
    torch = runtime.torch
    layer = runtime.manager.moe_layers[index]
    ids_host = torch.empty(tuple(ids.shape), dtype=ids.dtype, pin_memory=True)
    ids_host.copy_(ids, non_blocking=True)
    scores_host = None
    if scores is not None:
        scores_host = torch.empty(tuple(scores.shape), dtype=torch.float32, pin_memory=True)
        scores_host.copy_(scores.to(torch.float32), non_blocking=True)
    mapping_host = torch.empty(tuple(layer.log2phy.shape), dtype=layer.log2phy.dtype, pin_memory=True)
    mapping_host.copy_(layer.log2phy, non_blocking=True)
    runtime.owners.extend((ids_host, scores_host, mapping_host))
    return (ids_host, mapping_host.numpy(), layer, index, scores_host, prefetch), mapping_host


def update_and_prefetch(runtime, index, ids, scores, hidden_states):
    manager = runtime.manager
    dispatcher = manager._host_dispatcher
    torch = runtime.torch
    main = torch.npu.current_stream()
    observed_scores = scores if manager.offload_config.cache_router_weight != 0 else None
    demand, _ = stage_route(runtime, index, ids, observed_scores, False)
    token = dispatcher.publish_pipeline_demand(main, demand)
    # Prediction uses the existing MoE input, not the result of current experts.
    # Only the weight consumer below must wait for the current demand DMA.
    if manager.offload_config.expert_prefetch_enabled and not manager._skip_prefill:
        prediction = manager.predict_next_layer_experts_npu(index, hidden_states)
        if prediction is not None:
            next_index = index + 1
            weights, predicted_ids = prediction
            args, mapping = stage_route(runtime, next_index, predicted_ids, weights, True)
            inputs_ready = main.record_event()
            dispatcher.join_events.append(inputs_ready)
            with torch.npu.stream(manager._prefetch_stream):
                manager._prefetch_stream.wait_event(inputs_ready)
                dispatcher.defer_prefetch(
                    manager._prefetch_stream,
                    None,
                    args,
                    next_index,
                    manager.moe_layers[next_index].log2phy,
                    mapping,
                    dependency=dispatcher.pipeline_indices[token[0]],
                )
    dispatcher.wait_pipeline_demand(token)
