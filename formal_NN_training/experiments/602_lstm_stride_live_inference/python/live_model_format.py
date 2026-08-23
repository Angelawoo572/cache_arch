#!/usr/bin/env python3
"""Versioned float32 model.bin format shared by exporter and validators."""

import struct
from pathlib import Path

import numpy as np


MAGIC = b"STRLSTM1"
FORMAT_VERSION = 1
ENDIAN_MARKER = 0x01020304
FLOAT_TYPE_FLOAT32 = 1
HEADER = struct.Struct("<8sIIIIIIQ")
TENSOR_HEADER = struct.Struct("<HHQ")
TENSOR_ORDER = [
    "input_projection.weight",
    "input_projection.bias",
    "encoder_lstm.weight_ih_l0",
    "encoder_lstm.weight_hh_l0",
    "encoder_lstm.bias_ih_l0",
    "encoder_lstm.bias_hh_l0",
    "emit_head.weight",
    "emit_head.bias",
    "log_count_mean.weight",
    "log_count_mean.bias",
    "action_decoder.action_cell.weight_ih",
    "action_decoder.action_cell.weight_hh",
    "action_decoder.action_cell.bias_ih",
    "action_decoder.action_cell.bias_hh",
    "action_decoder.delta_head.weight",
    "action_decoder.delta_head.bias",
]


def write_model(path, state_dict, hidden_size, feature_width):
    path = Path(path)
    missing = set(TENSOR_ORDER).difference(state_dict)
    extra = set(state_dict).difference(TENSOR_ORDER)
    if missing or extra:
        raise RuntimeError(
            "unexpected state_dict: missing={} extra={}".format(
                sorted(missing), sorted(extra)
            )
        )
    tensors = []
    parameter_count = 0
    for name in TENSOR_ORDER:
        value = (
            state_dict[name].detach().cpu().numpy().astype("<f4", copy=False)
        )
        value = np.ascontiguousarray(value)
        tensors.append((name, value))
        parameter_count += int(value.size)
    with path.open("wb") as handle:
        handle.write(HEADER.pack(
            MAGIC,
            FORMAT_VERSION,
            ENDIAN_MARKER,
            FLOAT_TYPE_FLOAT32,
            len(tensors),
            int(hidden_size),
            int(feature_width),
            int(parameter_count),
        ))
        for name, value in tensors:
            encoded = name.encode("utf-8")
            handle.write(TENSOR_HEADER.pack(
                len(encoded), value.ndim, value.size
            ))
            handle.write(struct.pack(
                "<{}I".format(value.ndim), *value.shape
            ))
            handle.write(encoded)
            handle.write(value.tobytes(order="C"))
    return parameter_count


def read_model(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise RuntimeError("truncated model header")
        (
            magic, version, endian, float_type, tensor_count,
            hidden_size, feature_width, parameter_count,
        ) = HEADER.unpack(raw)
        if (
            magic != MAGIC
            or version != FORMAT_VERSION
            or endian != ENDIAN_MARKER
            or float_type != FLOAT_TYPE_FLOAT32
        ):
            raise RuntimeError("unsupported model.bin header")
        tensors = {}
        for _ in range(tensor_count):
            raw = handle.read(TENSOR_HEADER.size)
            if len(raw) != TENSOR_HEADER.size:
                raise RuntimeError("truncated tensor header")
            name_length, rank, count = TENSOR_HEADER.unpack(raw)
            shape_raw = handle.read(4 * rank)
            if len(shape_raw) != 4 * rank:
                raise RuntimeError("truncated tensor shape")
            shape = struct.unpack("<{}I".format(rank), shape_raw)
            name = handle.read(name_length).decode("utf-8")
            data = handle.read(4 * count)
            if len(data) != 4 * count:
                raise RuntimeError("truncated tensor data: {}".format(name))
            value = np.frombuffer(data, dtype="<f4").copy().reshape(shape)
            tensors[name] = value
        if handle.read(1):
            raise RuntimeError("trailing bytes after model tensors")
    if list(tensors) != TENSOR_ORDER:
        raise RuntimeError("tensor order mismatch")
    if sum(value.size for value in tensors.values()) != parameter_count:
        raise RuntimeError("parameter count mismatch in model.bin")
    return {
        "format_version": version,
        "hidden_size": hidden_size,
        "feature_width": feature_width,
        "parameter_count": parameter_count,
        "tensors": tensors,
    }
