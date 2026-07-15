#!/usr/bin/env python3
"""Normal-policy mirrors used only for labels and comparator replay.

Nothing in this module is imported by the neural model or decoder.  The
constants below reproduce the configured normal Stride, Streamer, and AMPM
policies; they are intentionally isolated from neural inference.
"""
from collections import OrderedDict

import numpy as np


PAGE_LINES = 64
TRACKERS = 64
MAX_AMPM_DELTA = 16
NORMAL_POLICY_DEGREE = {"stride": 2, "streamer": 5, "ampm": 4}
POLICY_USES_PC = {"stride": True, "streamer": False, "ampm": False}


def stride_actions(rows, state=None):
    trackers = OrderedDict() if state is None else state
    actions = [[] for _ in rows]
    for index, (pc, line, _) in enumerate(rows):
        if pc not in trackers:
            if len(trackers) >= TRACKERS:
                trackers.popitem(last=True)
            trackers[pc] = (line, 0)
            trackers.move_to_end(pc, last=False)
            continue
        last_line, last_stride = trackers[pc]
        stride = line - last_line
        if stride == 0:
            continue
        if stride == last_stride:
            page = line // PAGE_LINES
            offset = line % PAGE_LINES
            for degree in range(NORMAL_POLICY_DEGREE["stride"]):
                target_offset = offset + stride * degree
                if target_offset < 0 or target_offset >= PAGE_LINES:
                    break
                actions[index].append(page * PAGE_LINES + target_offset)
        trackers[pc] = (line, stride)
        trackers.move_to_end(pc, last=False)
    return actions, trackers


def streamer_actions(rows, state=None):
    trackers = OrderedDict() if state is None else state
    actions = [[] for _ in rows]
    for index, (_, line, _) in enumerate(rows):
        page = line // PAGE_LINES
        offset = line % PAGE_LINES
        if page not in trackers:
            if len(trackers) >= TRACKERS:
                trackers.popitem(last=False)
            trackers[page] = (offset, 0)
            continue
        last_offset, last_direction = trackers[page]
        if offset == last_offset:
            continue
        direction = 1 if offset > last_offset else -1
        direction_match = direction == last_direction
        trackers.pop(page)
        trackers[page] = (offset, direction)
        if direction_match:
            for distance in range(1, NORMAL_POLICY_DEGREE["streamer"] + 1):
                target_offset = offset + direction * distance
                if target_offset < 0 or target_offset >= PAGE_LINES:
                    break
                actions[index].append(page * PAGE_LINES + target_offset)
    return actions, trackers


def ampm_actions(rows, state=None):
    pages = OrderedDict() if state is None else state
    actions = [[] for _ in rows]
    for index, (_, line, _) in enumerate(rows):
        page = line // PAGE_LINES
        offset = line % PAGE_LINES
        if page in pages:
            bitmap = pages.pop(page)
        else:
            if len(pages) >= TRACKERS:
                pages.popitem(last=False)
            bitmap = np.zeros(PAGE_LINES, dtype=np.bool_)
        bitmap[offset] = True
        pages[page] = bitmap
        selected = []
        for delta in range(MAX_AMPM_DELTA, 0, -1):
            one_hop = offset - delta
            two_hop = offset - 2 * delta
            if (
                one_hop >= 0 and two_hop >= 0
                and bitmap[one_hop] and bitmap[two_hop]
            ):
                target_offset = offset + delta
                if target_offset < PAGE_LINES:
                    selected.append(target_offset)
            if len(selected) >= NORMAL_POLICY_DEGREE["ampm"]:
                break
        if len(selected) < NORMAL_POLICY_DEGREE["ampm"]:
            for delta in range(MAX_AMPM_DELTA, 0, -1):
                one_hop = offset + delta
                two_hop = offset + 2 * delta
                if (
                    one_hop < PAGE_LINES and two_hop < PAGE_LINES
                    and bitmap[one_hop] and bitmap[two_hop]
                ):
                    target_offset = offset - delta
                    if target_offset >= 0:
                        selected.append(target_offset)
                if len(selected) >= NORMAL_POLICY_DEGREE["ampm"]:
                    break
        actions[index] = [
            page * PAGE_LINES + target_offset for target_offset in selected
        ]
    return actions, pages


def normal_actions(policy, rows, state=None):
    return {
        "stride": stride_actions,
        "streamer": streamer_actions,
        "ampm": ampm_actions,
    }[policy](rows, state)


def policy_self_test():
    base = 7 * PAGE_LINES
    stride_rows = [(1, base + 10, 0), (1, base + 12, 0), (1, base + 14, 0)]
    actions, _ = stride_actions(stride_rows)
    assert actions[2] == [base + 14, base + 16]
    stream_rows = [(1, base + 10, 0), (2, base + 12, 0), (3, base + 13, 0)]
    actions, _ = streamer_actions(stream_rows)
    assert actions[2] == [base + x for x in (14, 15, 16, 17, 18)]
    ampm_rows = [(1, base + 0, 0), (2, base + 2, 0), (3, base + 4, 0)]
    actions, _ = ampm_actions(ampm_rows)
    assert base + 6 in actions[2]


__all__ = [
    "POLICY_USES_PC", "ampm_actions", "normal_actions", "policy_self_test",
    "streamer_actions", "stride_actions",
]
