#!/usr/bin/env python3
"""Torch-free provenance helpers for the 602 prefix-sufficiency audit."""

import ast
import gzip
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_content_sha256(path):
    digest = hashlib.sha256()
    with gzip.open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def function_source(path, name):
    """Return the top-level function source as inspect.getsource would."""
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines(True)
    tree = ast.parse(text)
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                start = node.lineno - 1
                stop = (
                    tree.body[index + 1].lineno - 1
                    if index + 1 < len(tree.body) else len(lines)
                )
                source = "".join(lines[start:stop])
                return source.rstrip("\n") + "\n"
    raise RuntimeError("function {} not found in {}".format(name, path))


def _constant_expression(node, values):
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant):
        return node.value
    if sys.version_info < (3, 8) and isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_constant_expression(item, values) for item in node.elts)
    if isinstance(node, ast.BinOp):
        left = _constant_expression(node.left, values)
        right = _constant_expression(node.right, values)
        if isinstance(node.op, ast.LShift):
            return left << right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    if (
        isinstance(node, ast.Attribute) and node.attr == "bits"
        and isinstance(node.value, ast.Call)
    ):
        call = node.value
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "iinfo"
            and call.args
            and isinstance(call.args[0], ast.Attribute)
            and call.args[0].attr == "uint64"
        ):
            return 64
    raise ValueError("unsupported constant expression")


def literal_assignment(path, name):
    tree = ast.parse(Path(path).read_text())
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [
            target.id for target in node.targets
            if isinstance(target, ast.Name)
        ]
        if not targets:
            continue
        try:
            value = _constant_expression(node.value, values)
        except (KeyError, ValueError):
            continue
        for target in targets:
            values[target] = value
            if target == name:
                return value
    raise RuntimeError("literal {} not found in {}".format(name, path))


def authoritative_source_hashes(root):
    root = Path(root)
    trainer = (
        root
        / "formal_NN_training/experiments/602_offline_lstm_stride/python"
        / "train_and_offline_infer.py"
    )
    primitives = (
        root / "formal_NN_training/common/threshold_free_policy.py"
    )
    encoder_payload = {
        "entrypoint_source": function_source(trainer, "runtime_features"),
        "primitive_source": function_source(primitives, "runtime_bits"),
        "fields": ["pc", "cache_line_address"],
        "address_bits": literal_assignment(primitives, "ADDRESS_BITS"),
        "cache_line_bytes": literal_assignment(
            primitives, "CACHE_LINE_BYTES"
        ),
    }
    encoder = hashlib.sha256(
        json.dumps(encoder_payload, sort_keys=True).encode()
    ).hexdigest()
    router_payload = (
        function_source(trainer, "_group_indices_by_pc")
        + function_source(trainer, "_chunk_batches")
    )
    router = hashlib.sha256(router_payload.encode()).hexdigest()
    return {
        "runtime_encoder_sha256": encoder,
        "state_router_sha256": router,
        "trainer_source_sha256": sha256(trainer),
        "runtime_primitive_source_sha256": sha256(primitives),
    }


def replay_header(path):
    with Path(path).open("r") as handle:
        return handle.readline().strip()


def artifact(path, gzip_content=False):
    path = Path(path)
    if not path.is_file():
        return {
            "path": str(path),
            "available": False,
            "sha256": None,
        }
    result = {
        "path": str(path),
        "available": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if gzip_content:
        result["content_sha256"] = gzip_content_sha256(path)
    return result
