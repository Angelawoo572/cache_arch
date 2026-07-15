#!/usr/bin/env python3
"""Source-input-only neural prefetch policies with no policy-shaped decoder.

The normal prefetcher and its neural student share only the normal source's
effective external inputs.  Normal actions are supervision, never inference
inputs.  The neural policy independently learns

* an unbounded Poisson request-count distribution; and
* an autoregressive mixture density over signed cache-line deltas.

There is no probability threshold, request budget, fixed degree, page-offset
table, same-page rule, or comparator candidate list in neural inference.
"""
import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# This is the width of ChampSim's uint64_t address interface, derived from the
# type rather than used as a prefetch-policy constant.  It is not a 64-entry
# page action space.
ADDRESS_BITS = np.iinfo(np.uint64).bits
ADDRESS_MASK = (1 << ADDRESS_BITS) - 1
CACHE_LINE_SHIFT = 6
CACHE_LINE_BYTES = 1 << CACHE_LINE_SHIFT
LINE_ADDRESS_BITS = ADDRESS_BITS - CACHE_LINE_SHIFT
LINE_ADDRESS_MASK = (1 << LINE_ADDRESS_BITS) - 1
LINE_ADDRESS_HALF_RANGE = 1 << (LINE_ADDRESS_BITS - 1)

# Model-architecture defaults.  These are ordinary configurable model
# hyperparameters, not normal-prefetcher thresholds or degree limits.  The CNN
# is intentionally shallow: two causal temporal filters, each with 17 taps.
CNN_KERNEL_SIZE = 17
CNN_STRIDE = 1
CNN_DILATIONS = (1, CNN_KERNEL_SIZE)
CNN_RECEPTIVE_FIELD = 1 + (CNN_KERNEL_SIZE - 1) * sum(CNN_DILATIONS)
DEFAULT_MIXTURE_COMPONENTS = 4


def _word_bits(value):
    value = int(value) & ADDRESS_MASK
    return [(value >> bit) & 1 for bit in range(ADDRESS_BITS)]


def runtime_bits(pcs, addresses, use_pc):
    """Losslessly encode the exact uint64_t values supplied by the source.

    Demand and fill addresses are line aligned in these ChampSim callbacks, so
    their low cache-line bits are naturally zero.  Keeping those bits in the
    representation makes the train/inference encoder literally an address
    encoder rather than a policy-shaped page/offset encoder.
    """
    if len(pcs) != len(addresses):
        raise RuntimeError("PC and address streams have different lengths")
    feature_count = ADDRESS_BITS * (2 if use_pc else 1)
    runtime = np.empty((len(addresses), feature_count), dtype=np.float32)
    for index, (pc, address) in enumerate(zip(pcs, addresses)):
        values = _word_bits(address)
        if use_pc:
            values = _word_bits(pc) + values
        runtime[index] = values
    return runtime


def signed_line_delta(base_line, target_line):
    """Return the shortest signed delta in the aligned uint64_t domain."""
    difference = (int(target_line) - int(base_line)) & LINE_ADDRESS_MASK
    if difference >= LINE_ADDRESS_HALF_RANGE:
        difference -= LINE_ADDRESS_MASK + 1
    return difference


def apply_signed_line_delta(base_line, delta):
    """Map a learned signed delta back to an aligned uint64_t address."""
    return (int(base_line) + int(delta)) & LINE_ADDRESS_MASK


def _delta_coordinate_numpy(delta):
    values = np.asarray(delta, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values))


def _coordinate_to_delta(coordinate):
    coordinate = float(coordinate)
    if not math.isfinite(coordinate):
        raise RuntimeError("neural delta coordinate is not finite")
    try:
        magnitude = math.expm1(abs(coordinate))
    except OverflowError as exc:
        raise RuntimeError("neural delta exceeds uint64_t address domain") from exc
    if not math.isfinite(magnitude) or magnitude > LINE_ADDRESS_HALF_RANGE:
        raise RuntimeError("neural delta exceeds uint64_t address domain")
    integer = int(round(magnitude))
    return -integer if coordinate < 0 else integer


class AutoregressiveActionDecoder(nn.Module):
    """Variable-cardinality direct-address decoder.

    The request count is Poisson, so its support is all non-negative integers.
    For each requested action, a recurrent decoder emits a signed log-delta
    mixture and, where the source output interface supports it, a fill class.
    """

    def __init__(
        self, context_size, decoder_size, fill_classes=0,
        mixture_components=DEFAULT_MIXTURE_COMPONENTS,
    ):
        super().__init__()
        if decoder_size < 1 or mixture_components < 1 or fill_classes < 0:
            raise ValueError("decoder dimensions must be positive")
        self.context_size = context_size
        self.decoder_size = decoder_size
        self.fill_classes = fill_classes
        self.mixture_components = mixture_components
        self.count_head = nn.Linear(context_size, 1)
        self.initial_state = nn.Linear(context_size, decoder_size)
        recurrent_inputs = 1 + fill_classes
        self.action_cell = nn.GRUCell(recurrent_inputs, decoder_size)
        self.delta_head = nn.Linear(decoder_size, 3 * mixture_components)
        self.fill_head = (
            nn.Linear(decoder_size, fill_classes) if fill_classes else None
        )

    def count_raw(self, context):
        return self.count_head(context).squeeze(-1)

    def begin(self, context):
        return torch.tanh(self.initial_state(context))

    def distribution(self, state):
        raw = self.delta_head(state)
        mix, mean, raw_scale = raw.chunk(3, dim=-1)
        scale = F.softplus(raw_scale) + torch.finfo(raw_scale.dtype).tiny
        fill = self.fill_head(state) if self.fill_head is not None else None
        return mix, mean, scale, fill

    def advance(self, state, delta_coordinate, fill_choice=None):
        parts = [delta_coordinate.reshape(-1, 1)]
        if self.fill_classes:
            if fill_choice is None:
                raise RuntimeError("fill choice is required by this decoder")
            parts.append(F.one_hot(
                fill_choice.to(torch.long), self.fill_classes
            ).to(delta_coordinate.dtype))
        return self.action_cell(torch.cat(parts, dim=-1), state)


class VariableActionLSTM(nn.Module):
    family = "lstm"

    def __init__(
        self, feature_count, hidden_size, fill_classes=0,
        mixture_components=DEFAULT_MIXTURE_COMPONENTS,
    ):
        super().__init__()
        self.feature_count = feature_count
        self.model_size = hidden_size
        self.fill_classes = fill_classes
        self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
        self.decoder = AutoregressiveActionDecoder(
            hidden_size, hidden_size, fill_classes, mixture_components
        )

    def encode(self, runtime, state=None):
        return self.lstm(runtime, state)


class VariableActionCNN(nn.Module):
    """Two-layer causal temporal CNN over the complete ordered stream."""

    family = "cnn"

    def __init__(
        self, feature_count, channels, fill_classes=0,
        mixture_components=DEFAULT_MIXTURE_COMPONENTS,
        kernel_size=CNN_KERNEL_SIZE, dilations=CNN_DILATIONS,
    ):
        super().__init__()
        if kernel_size < 2 or not dilations or any(d < 1 for d in dilations):
            raise ValueError("invalid causal CNN geometry")
        self.feature_count = feature_count
        self.model_size = channels
        self.fill_classes = fill_classes
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.receptive_field = 1 + (kernel_size - 1) * sum(self.dilations)
        self.input_projection = nn.Conv1d(feature_count, channels, 1)
        self.temporal_blocks = nn.ModuleList([
            nn.Conv1d(
                channels, channels, kernel_size=kernel_size,
                stride=CNN_STRIDE, dilation=dilation, padding=0,
            )
            for dilation in self.dilations
        ])
        self.decoder = AutoregressiveActionDecoder(
            channels, channels, fill_classes, mixture_components
        )

    def encode(self, runtime):
        x = runtime.transpose(1, 2)
        x = torch.tanh(self.input_projection(x))
        for dilation, convolution in zip(self.dilations, self.temporal_blocks):
            left_context = dilation * (self.kernel_size - 1)
            x = x + torch.tanh(convolution(F.pad(x, (left_context, 0))))
        return x.transpose(1, 2)


def build_model(
    family, feature_count, size, fill_classes=0,
    mixture_components=DEFAULT_MIXTURE_COMPONENTS,
):
    if family == "lstm":
        model = VariableActionLSTM(
            feature_count, size, fill_classes, mixture_components
        )
    elif family == "cnn":
        model = VariableActionCNN(
            feature_count, size, fill_classes, mixture_components
        )
    else:
        raise RuntimeError("unknown model family {}".format(family))
    return model, sum(parameter.numel() for parameter in model.parameters())


def targets_from_actions(lines, actions, fill_levels=()):
    """Convert teacher requests to variable-length delta supervision.

    The array width is the observed batch storage requirement, not an inference
    action cap.  No same-page or candidate-space restriction is applied.
    """
    if len(lines) != len(actions):
        raise RuntimeError("action rows do not match decision rows")
    counts = np.asarray([len(items) for items in actions], dtype=np.int64)
    storage_width = int(counts.max()) if len(counts) else 0
    deltas = np.zeros((len(lines), storage_width), dtype=np.float32)
    fills = np.full((len(lines), storage_width), -1, dtype=np.int64)
    fill_to_index = {value: index for index, value in enumerate(fill_levels)}
    for row_index, (line, items) in enumerate(zip(lines, actions)):
        for action_index, item in enumerate(items):
            if fill_levels:
                target_line, fill = item
                if fill not in fill_to_index:
                    raise RuntimeError("teacher action uses unknown fill level")
                fills[row_index, action_index] = fill_to_index[fill]
            else:
                target_line = item
            delta = signed_line_delta(line, target_line)
            deltas[row_index, action_index] = _delta_coordinate_numpy(delta)
    return counts, deltas, fills


def expand_targets(decision_targets, positions, context_count):
    """Place decision losses into a larger causal context event stream."""
    counts, deltas, fills = decision_targets
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) != len(counts):
        raise RuntimeError("decision positions and targets differ")
    expanded_counts = np.full(context_count, -1, dtype=np.int64)
    expanded_deltas = np.zeros(
        (context_count, deltas.shape[1]), dtype=np.float32
    )
    expanded_fills = np.full(
        (context_count, fills.shape[1]), -1, dtype=np.int64
    )
    expanded_counts[positions] = counts
    expanded_deltas[positions] = deltas
    expanded_fills[positions] = fills
    return expanded_counts, expanded_deltas, expanded_fills


def _structured_nll(model, context, counts, deltas, fills, reduction="mean"):
    flat_context = context.reshape(-1, context.shape[-1])
    flat_counts = counts.reshape(-1)
    flat_deltas = deltas.reshape(-1, deltas.shape[-1])
    flat_fills = fills.reshape(-1, fills.shape[-1])
    decision_rows = flat_counts >= 0
    decision_atoms = int(decision_rows.sum().detach().item())

    count_nll = context.new_zeros(())
    action_nll = context.new_zeros(())
    fill_nll = context.new_zeros(())
    action_atoms = 0
    fill_atoms = 0
    if decision_atoms:
        decision_context = flat_context[decision_rows]
        decision_counts = flat_counts[decision_rows]
        decision_deltas = flat_deltas[decision_rows]
        decision_fills = flat_fills[decision_rows]
        count_rate = F.softplus(model.decoder.count_raw(decision_context))
        positive_rate = count_rate + torch.finfo(count_rate.dtype).tiny
        count_targets = decision_counts.to(count_rate.dtype)
        count_nll = (
            positive_rate - count_targets * torch.log(positive_rate)
            + torch.lgamma(count_targets + 1.0)
        ).sum()

        state = model.decoder.begin(decision_context)
        for step in range(decision_deltas.shape[1]):
            active = decision_counts > step
            active_atoms = int(active.sum().detach().item())
            if not active_atoms:
                break
            indices = torch.nonzero(active, as_tuple=False).squeeze(1)
            active_state = state.index_select(0, indices)
            mix, mean, scale, fill_logits = model.decoder.distribution(
                active_state
            )
            target = decision_deltas[active, step]
            log_component = (
                -0.5 * ((target.unsqueeze(1) - mean) / scale).square()
                - torch.log(scale)
                - 0.5 * math.log(2.0 * math.pi)
            )
            log_probability = torch.logsumexp(
                F.log_softmax(mix, dim=-1) + log_component, dim=-1
            )
            action_nll = action_nll - log_probability.sum()
            action_atoms += active_atoms

            active_fill = None
            if model.fill_classes:
                active_fill = decision_fills[active, step]
                if torch.any(active_fill < 0):
                    raise RuntimeError("missing fill supervision")
                fill_nll = fill_nll + F.cross_entropy(
                    fill_logits, active_fill, reduction="sum"
                )
                fill_atoms += active_atoms
            # Free-running training: feed back the model's own deterministic
            # action, exactly as decode() does at inference. Teacher actions
            # supervise the NLL above but are never recurrent decoder inputs.
            predicted_component = mix.argmax(dim=-1, keepdim=True)
            predicted_coordinate = mean.gather(
                1, predicted_component
            ).squeeze(1)
            predicted_fill = (
                fill_logits.argmax(dim=-1)
                if fill_logits is not None else None
            )
            advanced = model.decoder.advance(
                active_state, predicted_coordinate, predicted_fill
            )
            state = state.index_copy(0, indices, advanced)

    total = count_nll + action_nll + fill_nll
    atoms = decision_atoms + action_atoms + fill_atoms
    components = (
        float(count_nll.detach().item()),
        float(action_nll.detach().item()),
        float(fill_nll.detach().item()),
    )
    if reduction == "sum":
        return total, atoms, components
    return total / float(max(1, atoms)), atoms, None


def _detach_state(state):
    return None if state is None else tuple(value.detach() for value in state)


def _iter_chunks(end, chunk_len):
    for start in range(0, end, chunk_len):
        yield start, min(end, start + chunk_len)


def _tensor_targets(counts, deltas, fills):
    return (
        torch.from_numpy(counts), torch.from_numpy(deltas),
        torch.from_numpy(fills),
    )


def _group_atoms(counts, fill_classes, group):
    decisions = 0
    actions = 0
    for start, stop in group:
        selected = counts[start:stop]
        decisions += int(np.count_nonzero(selected >= 0))
        actions += int(selected[selected >= 0].sum())
    return decisions + actions * (2 if fill_classes else 1)


def train_lstm(
    model, runtime, counts, deltas, fills, fit_end, device, epochs,
    chunk_len, accumulate_chunks, learning_rate,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    x = torch.from_numpy(runtime)
    count_tensor, delta_tensor, fill_tensor = _tensor_targets(
        counts, deltas, fills
    )
    chunks = list(_iter_chunks(fit_end, chunk_len))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        state = None
        totals = [0.0, 0.0, 0.0]
        total_nll = 0.0
        total_atoms = 0
        steps = 0
        for group_start in range(0, len(chunks), accumulate_chunks):
            group = chunks[group_start:group_start + accumulate_chunks]
            group_atoms = _group_atoms(counts, model.fill_classes, group)
            optimizer.zero_grad(set_to_none=True)
            for start, stop in group:
                xb = x[start:stop].unsqueeze(0).to(device)
                cb = count_tensor[start:stop].unsqueeze(0).to(device)
                db = delta_tensor[start:stop].unsqueeze(0).to(device)
                fb = fill_tensor[start:stop].unsqueeze(0).to(device)
                context, state = model.encode(xb, state)
                state = _detach_state(state)
                loss_sum, atoms, components = _structured_nll(
                    model, context, cb, db, fb, reduction="sum"
                )
                (loss_sum / float(max(1, group_atoms))).backward()
                total_nll += float(loss_sum.detach().item())
                total_atoms += atoms
                for position, value in enumerate(components):
                    totals[position] += value
            optimizer.step()
            steps += 1
        row = {
            "epoch": epoch,
            "nll_per_learned_decision": total_nll / max(1, total_atoms),
            "count_nll": totals[0],
            "delta_nll": totals[1],
            "fill_nll": totals[2],
            "chronological_chunks": len(chunks),
            "optimizer_steps": steps,
        }
        history.append(row)
        print("[train:lstm] epoch={} nll={:.8f}".format(
            epoch, row["nll_per_learned_decision"]
        ))
    return history


def train_cnn(
    model, runtime, counts, deltas, fills, fit_end, device, epochs,
    chunk_len, accumulate_chunks, learning_rate,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    x = torch.from_numpy(runtime)
    count_tensor, delta_tensor, fill_tensor = _tensor_targets(
        counts, deltas, fills
    )
    chunks = list(_iter_chunks(fit_end, chunk_len))
    context_width = model.receptive_field - 1
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = [0.0, 0.0, 0.0]
        total_nll = 0.0
        total_atoms = 0
        steps = 0
        for group_start in range(0, len(chunks), accumulate_chunks):
            group = chunks[group_start:group_start + accumulate_chunks]
            group_atoms = _group_atoms(counts, model.fill_classes, group)
            optimizer.zero_grad(set_to_none=True)
            for start, stop in group:
                context_start = max(0, start - context_width)
                offset = start - context_start
                xb = x[context_start:stop].unsqueeze(0).to(device)
                cb = count_tensor[start:stop].unsqueeze(0).to(device)
                db = delta_tensor[start:stop].unsqueeze(0).to(device)
                fb = fill_tensor[start:stop].unsqueeze(0).to(device)
                context = model.encode(xb)[:, offset:]
                loss_sum, atoms, components = _structured_nll(
                    model, context, cb, db, fb, reduction="sum"
                )
                (loss_sum / float(max(1, group_atoms))).backward()
                total_nll += float(loss_sum.detach().item())
                total_atoms += atoms
                for position, value in enumerate(components):
                    totals[position] += value
            optimizer.step()
            steps += 1
        row = {
            "epoch": epoch,
            "nll_per_learned_decision": total_nll / max(1, total_atoms),
            "count_nll": totals[0],
            "delta_nll": totals[1],
            "fill_nll": totals[2],
            "chronological_chunks": len(chunks),
            "optimizer_steps": steps,
        }
        history.append(row)
        print("[train:cnn] epoch={} nll={:.8f}".format(
            epoch, row["nll_per_learned_decision"]
        ))
    return history


def score_lstm(model, runtime, device, initial_state=None, chunk_len=8192):
    model.eval()
    context = np.empty(
        (len(runtime), model.decoder.context_size), dtype=np.float32
    )
    count_raw = np.empty(len(runtime), dtype=np.float32)
    state = initial_state
    with torch.no_grad():
        for start, stop in _iter_chunks(len(runtime), chunk_len):
            xb = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            encoded, state = model.encode(xb, state)
            state = _detach_state(state)
            encoded = encoded[0]
            context[start:stop] = encoded.cpu().numpy()
            count_raw[start:stop] = model.decoder.count_raw(
                encoded
            ).cpu().numpy()
    return (count_raw, context), state


def advance_lstm_state(
    model, runtime, device, initial_state=None, chunk_len=8192,
):
    model.eval()
    state = initial_state
    with torch.no_grad():
        for start, stop in _iter_chunks(len(runtime), chunk_len):
            xb = torch.from_numpy(runtime[start:stop]).unsqueeze(0).to(device)
            _, state = model.encode(xb, state)
            state = _detach_state(state)
    return state


def score_cnn(model, runtime, device, prefix_runtime=None, chunk_len=8192):
    model.eval()
    context_width = model.receptive_field - 1
    prefix_count = (
        0 if prefix_runtime is None else min(context_width, len(prefix_runtime))
    )
    all_runtime = (
        runtime if prefix_count == 0
        else np.concatenate([prefix_runtime[-prefix_count:], runtime], axis=0)
    )
    context = np.empty(
        (len(runtime), model.decoder.context_size), dtype=np.float32
    )
    count_raw = np.empty(len(runtime), dtype=np.float32)
    with torch.no_grad():
        for start, stop in _iter_chunks(len(runtime), chunk_len):
            global_start = prefix_count + start
            global_stop = prefix_count + stop
            context_start = max(0, global_start - context_width)
            offset = global_start - context_start
            xb = torch.from_numpy(
                all_runtime[context_start:global_stop]
            ).unsqueeze(0).to(device)
            encoded = model.encode(xb)[0, offset:]
            context[start:stop] = encoded.cpu().numpy()
            count_raw[start:stop] = model.decoder.count_raw(
                encoded
            ).cpu().numpy()
    return count_raw, context


def decode(model, encoded, base_lines, device, chunk_len=8192):
    """Decode learned count modes and direct address deltas without a cap."""
    count_raw, context = encoded
    if len(count_raw) != len(context) or len(count_raw) != len(base_lines):
        raise RuntimeError("decoder row counts differ")
    rates = np.logaddexp(0.0, np.asarray(count_raw, dtype=np.float64))
    if not np.all(np.isfinite(rates)):
        raise RuntimeError("neural request-count rate is not finite")
    if np.any(rates > np.iinfo(np.int64).max):
        raise RuntimeError("neural request count exceeds host index domain")
    counts = np.floor(rates).astype(np.int64)
    predicted_lines = [[] for _ in range(len(counts))]
    predicted_fills = [[] for _ in range(len(counts))]
    model.eval()
    with torch.no_grad():
        for start, stop in _iter_chunks(len(counts), chunk_len):
            local_context = torch.from_numpy(context[start:stop]).to(device)
            state = model.decoder.begin(local_context)
            local_counts = counts[start:stop]
            steps = int(local_counts.max()) if len(local_counts) else 0
            for step in range(steps):
                active_numpy = np.flatnonzero(local_counts > step)
                if not len(active_numpy):
                    break
                active = torch.from_numpy(active_numpy).to(
                    device=device, dtype=torch.long
                )
                active_state = state.index_select(0, active)
                mix, mean, _, fill_logits = model.decoder.distribution(
                    active_state
                )
                component = mix.argmax(dim=-1, keepdim=True)
                coordinate = mean.gather(1, component).squeeze(1)
                fill_choice = (
                    fill_logits.argmax(dim=-1)
                    if fill_logits is not None else None
                )
                coordinate_cpu = coordinate.cpu().numpy()
                fill_cpu = (
                    fill_choice.cpu().numpy() if fill_choice is not None
                    else np.full(len(active_numpy), -1, dtype=np.int64)
                )
                for local_position, value, fill in zip(
                    active_numpy, coordinate_cpu, fill_cpu
                ):
                    global_position = start + int(local_position)
                    delta = _coordinate_to_delta(value)
                    predicted_lines[global_position].append(
                        apply_signed_line_delta(
                            base_lines[global_position], delta
                        )
                    )
                    predicted_fills[global_position].append(int(fill))
                advanced = model.decoder.advance(
                    active_state, coordinate, fill_choice
                )
                state = state.index_copy(0, active, advanced)
    return counts, predicted_lines, predicted_fills


def behavior_metrics(
    predicted_counts, predicted_lines, predicted_fills,
    target_actions, fill_levels=(),
):
    if not (
        len(predicted_counts) == len(predicted_lines)
        == len(predicted_fills) == len(target_actions)
    ):
        raise RuntimeError("behavior metric row counts differ")
    true_positive = 0
    predicted_total = 0
    target_total = 0
    fill_matches = 0
    fill_total = 0
    exact_counts = 0
    for count, lines, fills, truth_items in zip(
        predicted_counts, predicted_lines, predicted_fills, target_actions
    ):
        truth_lines = [item[0] if fill_levels else item for item in truth_items]
        if int(count) == len(truth_lines):
            exact_counts += 1
        predicted_counter = Counter(int(line) for line in lines)
        truth_counter = Counter(int(line) for line in truth_lines)
        true_positive += sum((predicted_counter & truth_counter).values())
        predicted_total += len(lines)
        target_total += len(truth_lines)
        if fill_levels:
            truth_fill = {int(line): fill for line, fill in truth_items}
            for line, fill_index in zip(lines, fills):
                line = int(line)
                if line in truth_fill:
                    fill_total += 1
                    if (
                        0 <= int(fill_index) < len(fill_levels)
                        and fill_levels[int(fill_index)] == truth_fill[line]
                    ):
                        fill_matches += 1
    false_positive = predicted_total - true_positive
    false_negative = target_total - true_positive
    precision = (
        true_positive / float(predicted_total) if predicted_total else 0.0
    )
    recall = true_positive / float(target_total) if target_total else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "callbacks": len(target_actions),
        "predicted_actions": predicted_total,
        "normal_actions": target_total,
        "true_positive_actions": true_positive,
        "false_positive_actions": false_positive,
        "false_negative_actions": false_negative,
        "count_exact_match_rate": (
            exact_counts / float(len(target_actions)) if target_actions else 0.0
        ),
        "target_precision": precision,
        "target_recall": recall,
        "target_f1": f1,
        "fill_accuracy_on_matched_targets": (
            fill_matches / float(fill_total) if fill_total else None
        ),
    }


def self_test_cnn(feature_count, fill_classes=0):
    torch.manual_seed(123)
    model = VariableActionCNN(feature_count, 5, fill_classes).eval()
    layers = [module for module in model.modules() if isinstance(module, nn.Conv1d)]
    if len(layers) != 1 + len(CNN_DILATIONS):
        raise RuntimeError("CNN must contain one projection and two filters")
    projection, temporal_layers = layers[0], layers[1:]
    if projection.kernel_size != (1,) or projection.padding != (0,):
        raise RuntimeError("CNN input projection geometry changed")
    for dilation, convolution in zip(CNN_DILATIONS, temporal_layers):
        if (
            convolution.kernel_size != (CNN_KERNEL_SIZE,)
            or convolution.stride != (CNN_STRIDE,)
            or convolution.dilation != (dilation,)
            or convolution.padding != (0,)
        ):
            raise RuntimeError("CNN temporal geometry differs from contract")
    reachable_lags = {0}
    for dilation in CNN_DILATIONS:
        reachable_lags = {
            previous + tap * dilation
            for previous in reachable_lags
            for tap in range(CNN_KERNEL_SIZE)
        }
    if reachable_lags != set(range(CNN_RECEPTIVE_FIELD)):
        raise RuntimeError("CNN receptive field is not contiguous")
    runtime = torch.randn(1, CNN_RECEPTIVE_FIELD + 20, feature_count)
    pivot = CNN_RECEPTIVE_FIELD + 5
    with torch.no_grad():
        original = model.encode(runtime)
        future = runtime.clone()
        future[:, pivot + 1:] += 1000.0
        changed = model.encode(future)
    if not torch.allclose(
        original[:, :pivot + 1], changed[:, :pivot + 1],
        atol=1e-6, rtol=1e-6,
    ):
        raise RuntimeError("CNN future-input causality self-test failed")
    old = runtime.clone()
    old[:, pivot - CNN_RECEPTIVE_FIELD] += 1000.0
    with torch.no_grad():
        changed_old = model.encode(old)
    if not torch.allclose(
        original[:, pivot], changed_old[:, pivot], atol=1e-6, rtol=1e-6,
    ):
        raise RuntimeError("CNN receptive field exceeds its contract")


def self_test_variable_action_decoder(feature_count, family="lstm"):
    """Prove that inference has neither a 64-action cap nor a page boundary."""
    model, _ = build_model(family, feature_count, 3)
    encoded = (
        np.asarray([80.0], dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
    )
    counts, lines, _ = decode(
        model, encoded, [LINE_ADDRESS_HALF_RANGE - 1], torch.device("cpu")
    )
    if counts.tolist() != [80] or len(lines[0]) != 80:
        raise RuntimeError("variable-action decoder retained a hidden degree cap")
    if signed_line_delta(63, 64) != 1:
        raise RuntimeError("direct delta decoder retained a page boundary")


def self_test_free_running_decoder(
    feature_count, family="lstm", fill_classes=0,
):
    """Prove that teacher actions are losses, not decoder feedback inputs."""
    model, _ = build_model(
        family, feature_count, 3, fill_classes=fill_classes
    )
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    captured_inputs = []

    def capture(_module, arguments):
        captured_inputs.append(arguments[0].detach().clone())

    hook = model.decoder.action_cell.register_forward_pre_hook(capture)
    try:
        context = torch.zeros(1, 1, model.decoder.context_size)
        counts = torch.tensor([[2]], dtype=torch.long)
        deltas = torch.full((1, 1, 2), 3.0)
        fills = torch.full((1, 1, 2), -1, dtype=torch.long)
        if fill_classes:
            fills.fill_(fill_classes - 1)
        _structured_nll(model, context, counts, deltas, fills)
    finally:
        hook.remove()
    if len(captured_inputs) != 2:
        raise RuntimeError("decoder feedback self-test observed wrong step count")
    for observed in captured_inputs:
        if not torch.equal(observed[:, 0], torch.zeros_like(observed[:, 0])):
            raise RuntimeError("teacher delta leaked into decoder feedback")
        if fill_classes:
            expected = torch.zeros_like(observed[:, 1:])
            expected[:, 0] = 1
            if not torch.equal(observed[:, 1:], expected):
                raise RuntimeError("teacher fill leaked into decoder feedback")


__all__ = [
    "ADDRESS_BITS", "CACHE_LINE_BYTES", "CACHE_LINE_SHIFT",
    "CNN_DILATIONS", "CNN_KERNEL_SIZE",
    "CNN_RECEPTIVE_FIELD", "CNN_STRIDE", "DEFAULT_MIXTURE_COMPONENTS",
    "VariableActionCNN", "VariableActionLSTM", "advance_lstm_state",
    "apply_signed_line_delta",
    "behavior_metrics", "build_model", "decode", "expand_targets",
    "runtime_bits", "score_cnn", "score_lstm", "self_test_cnn",
    "self_test_free_running_decoder", "self_test_variable_action_decoder",
    "signed_line_delta",
    "targets_from_actions", "train_cnn", "train_lstm",
]
