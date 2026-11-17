#!/usr/bin/env python3
"""Candidate scoring heads for the fixed-size LLC event encoder."""

import torch
from torch import nn

from train import CANDIDATE_FEATURES, HIDDEN, INPUTS, RANK


HEAD_WIDTHS = (4, 8, 16)
CHECKPOINT_VERSION = 3
AFFINE_MAGIC = 'LLC_LSTM_V1'
MLP_MAGIC = 'LLC_LSTM_V3'


class Model(nn.Module):
    """Use the same inputs, optional recurrent state, and scalar score in both heads."""

    def __init__(self, recurrent, head='affine', width=8):
        super().__init__()
        if head not in ('affine', 'mlp'):
            raise ValueError('unknown scoring head {}'.format(head))
        if head == 'mlp' and width not in HEAD_WIDTHS:
            raise ValueError('MLP head width must be one of {}'.format(HEAD_WIDTHS))
        self.recurrent = recurrent
        self.head = head
        self.width = RANK if head == 'affine' else width
        self.cell = nn.LSTM(INPUTS, HIDDEN, batch_first=True)
        if self.head == 'affine':
            # Keep parameter names and initialization compatible with the original scorer.
            self.query = nn.Linear(INPUTS + HIDDEN, RANK)
            self.key = nn.Linear(CANDIDATE_FEATURES, RANK)
            self.bias = nn.Linear(CANDIDATE_FEATURES, 1)
        else:
            self.event_projection = nn.Linear(INPUTS + HIDDEN, width)
            self.candidate_projection = nn.Linear(CANDIDATE_FEATURES, width, bias=False)
            self.reduction = nn.Linear(width, 1)

    def score_events(self, events, candidates, hidden):
        context = torch.cat((events, hidden), dim=-1)
        if self.head == 'affine':
            query = self.query(context).unsqueeze(-2)
            scores = (query * self.key(candidates)).sum(dim=-1)
            return scores + self.bias(candidates).squeeze(-1)

        event_projection = self.event_projection(context).unsqueeze(-2)
        # A separate candidate bias would duplicate the shared event bias before ReLU.
        activation = torch.relu(event_projection + self.candidate_projection(candidates))
        return self.reduction(activation).squeeze(-1)

    def forward(self, events, candidates, state=None):
        if self.recurrent:
            hidden, state = self.cell(events, state)
        else:
            hidden = events.new_zeros(events.shape[:-1] + (HIDDEN,))
        return self.score_events(events, candidates, hidden), state


def complexity(model, candidate_cap=16):
    """Account for portable coefficients and linear arithmetic before candidate filtering."""
    if candidate_cap < 1:
        raise ValueError('candidate cap must be positive')
    width = model.width
    cell_linear = 4 * HIDDEN * (INPUTS + HIDDEN)
    event_linear = width * (INPUTS + HIDDEN)
    if model.head == 'affine':
        candidate_linear = width * (CANDIDATE_FEATURES + 1) + CANDIDATE_FEATURES
        head_coefficients = (event_linear + width + width * CANDIDATE_FEATURES + width
                             + CANDIDATE_FEATURES + 1)
        # Compose the query with the key projection and bias once in the simulator.
        runtime_event = event_linear + width * (CANDIDATE_FEATURES + 1)
        runtime_candidate = CANDIDATE_FEATURES
        relu_comparisons = 0
    else:
        candidate_linear = width * (CANDIDATE_FEATURES + 1)
        head_coefficients = event_linear + width + candidate_linear + 1
        runtime_event = event_linear if model.recurrent else width * INPUTS
        runtime_candidate = candidate_linear
        relu_comparisons = width
    recurrent_linear = cell_linear if model.recurrent else 0
    exported = cell_linear + 4 * HIDDEN + head_coefficients
    return {
        'head': model.head,
        'head_width': width,
        'recurrent': model.recurrent,
        'candidate_cap': candidate_cap,
        'pytorch_parameter_count': sum(tensor.numel() for tensor in model.parameters()),
        'head_coefficients': head_coefficients,
        'exported_coefficients': exported,
        'exported_float32_bytes': 4 * exported,
        'persistent_state_floats_per_core': 2 * HIDDEN if model.recurrent else 0,
        'recurrent_matrix_macs_per_event': recurrent_linear,
        'head_event_macs': runtime_event,
        'head_candidate_macs': runtime_candidate,
        'max_linear_macs_per_event': recurrent_linear + runtime_event + candidate_cap * runtime_candidate,
        'straightforward_max_linear_macs_per_event': recurrent_linear + event_linear
                                                     + candidate_cap * candidate_linear,
        'relu_comparisons_per_candidate': relu_comparisons,
        'pre_relu_adds_per_candidate': relu_comparisons,
    }


def portable_tensors(model):
    cell = model.cell
    gates = (cell.weight_ih_l0, cell.weight_hh_l0, cell.bias_ih_l0 + cell.bias_hh_l0)
    if model.head == 'affine':
        return gates + (model.query.weight, model.query.bias, model.key.weight, model.key.bias,
                        model.bias.weight, model.bias.bias)
    return gates + (model.event_projection.weight, model.event_projection.bias,
                    model.candidate_projection.weight, model.reduction.weight, model.reduction.bias)


def export(model, candidate_cap, path):
    """Write a row-major ASCII model accepted by the ChampSim scorer."""
    cost = complexity(model, candidate_cap)
    tensors = [tensor.detach().cpu().float().reshape(-1) for tensor in portable_tensors(model)]
    if sum(tensor.numel() for tensor in tensors) != cost['exported_coefficients']:
        raise ValueError('export coefficient count does not match the scoring head')
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError('model contains nonfinite coefficients')
    mode = 'lstm' if model.recurrent else 'event'
    magic = AFFINE_MAGIC if model.head == 'affine' else MLP_MAGIC
    with path.open('w') as stream:
        stream.write('{} {} {} {} {} {} {} 0\n'.format(
            magic, mode, INPUTS, HIDDEN, model.width, CANDIDATE_FEATURES, candidate_cap,
        ))
        for tensor in tensors:
            stream.write(' '.join('{:.9g}'.format(value) for value in tensor.tolist()) + '\n')


def save_checkpoint(model, path, **metadata):
    """Save head type and width with weights so all arms share one loader."""
    reserved = {'model_version', 'state_dict', 'recurrent', 'head', 'width'}
    if reserved.intersection(metadata):
        raise ValueError('checkpoint metadata cannot override model fields')
    weights = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(dict(metadata, model_version=CHECKPOINT_VERSION, state_dict=weights,
                    recurrent=model.recurrent, head=model.head, width=model.width), path)


def load_checkpoint(path, device='cpu'):
    """Return the model and metadata. Older affine checkpoints keep their parameter names."""
    metadata = torch.load(path, map_location='cpu', weights_only=True)
    version = metadata.get('model_version')
    if version is not None and version != CHECKPOINT_VERSION:
        raise ValueError('unknown checkpoint version {}'.format(version))
    weights = metadata.pop('state_dict')
    model = Model(metadata['recurrent'], metadata.get('head', 'affine'), metadata.get('width', RANK))
    model.load_state_dict(weights)
    return model.to(device).eval(), metadata
