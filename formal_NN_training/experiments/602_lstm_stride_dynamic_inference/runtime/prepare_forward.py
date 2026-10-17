'''Generate the scalar forward in an isolated build; verify exact source bytes.

The live runtime remains canonical. These transformations retain the historical
four generated C++ files byte for byte; no generated arithmetic is maintained as
a second hand-edited source tree. Legacy completed builds can be checked without
requiring a marker that did not exist when they were built.
'''
from pathlib import Path
import argparse
import json

PROTOCOL = 'dynamic602_forward_generation_v1'
SOURCE = Path('formal_NN_training/experiments/602_lstm_stride_live_inference/runtime')
NAMES = ('stride_lstm_runtime.h', 'stride_lstm_runtime.cc',
         'stride_lstm_model_loader.h', 'stride_lstm_model_loader.cc')
MARKER = '.dynamic602_forward_prepared.json'


def replace_once(text, old, new, name):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{name}: expected exactly one source anchor, found {count}')
    return text.replace(old, new, 1)


def count_domain(text):
    # Current canonical live sources already contain the full int64 rounded-
    # boundary guard. The legacy generator's old .replace silently matched zero
    # times there. Accept exactly one explicitly supported representation.
    legacy = ('raw_count > static_cast<double>(\n'
              '            std::numeric_limits<std::uint64_t>::max())')
    inline = 'std::nearbyint(raw_count) >= 9223372036854775808.0'
    canonical = ('    const double rounded_count = std::nearbyint(raw_count);\n'
                 '    if (!std::isfinite(rounded_count) || rounded_count >= std::ldexp(1.0, 63)) {\n'
                 '      throw std::runtime_error("positive-count prediction exceeds host domain");\n'
                 '    }\n'
                 '    count = static_cast<std::uint64_t>(rounded_count);')
    counts = (text.count(legacy), text.count(inline), text.count(canonical))
    if counts == (1, 0, 0):
        return replace_once(text, legacy, inline, 'legacy count-domain guard')
    if counts in ((0, 1, 0), (0, 0, 1)):
        return text
    raise RuntimeError('count-domain guard: expected exactly one supported representation, '
                       f'found legacy/inline/canonical={counts}')


def generated_sources(repo):
    source = Path(repo) / SOURCE
    texts = {name: (source / name).read_text(encoding='utf-8') for name in NAMES}
    for name, token in [('stride_lstm_runtime.h', 'InferState('),
                        ('stride_lstm_runtime.h', 'output_limit_'),
                        ('stride_lstm_runtime.h', 'emit_margin'),
                        ('stride_lstm_runtime.cc', 'StrideLSTMRuntime::InferState(')]:
        if token in texts[name]:
            raise RuntimeError(f'{name}: canonical input is already extended ({token})')
    texts['stride_lstm_runtime.h'] = replace_once(
        texts['stride_lstm_runtime.h'], '  void Reset();', '''  void Reset();
  InferenceResult InferState(std::uint64_t pc, std::uint64_t line, const RecurrentState& prior);
  void SetOutputLimit(std::uint64_t limit) { output_limit_ = limit; }
''', 'runtime Reset declaration')
    texts['stride_lstm_runtime.h'] = replace_once(
        texts['stride_lstm_runtime.h'], '  FrozenModel model_;',
        '  std::uint64_t output_limit_ = 0;\n  FrozenModel model_;', 'runtime model member')
    texts['stride_lstm_runtime.cc'] = replace_once(
        texts['stride_lstm_runtime.cc'], 'void StrideLSTMRuntime::Reset()',
        '''InferenceResult StrideLSTMRuntime::InferState(std::uint64_t pc, std::uint64_t line, const RecurrentState& prior) {
  states_.clear();
  states_[pc] = prior;
  try { auto result = Infer(pc,line); states_.clear(); return result; }
  catch (...) { states_.clear(); throw; }
}

void StrideLSTMRuntime::Reset()''', 'runtime Reset definition')
    texts['stride_lstm_runtime.cc'] = replace_once(
        texts['stride_lstm_runtime.cc'], 'slot < count;',
        'slot < count && (output_limit_ == 0 || slot < output_limit_);', 'decoder slot bound')
    texts['stride_lstm_runtime.cc'] = replace_once(
        texts['stride_lstm_runtime.cc'], '  timings_ns_.push_back(result.nanoseconds);',
        '  // Per-callback host timing vectors are not needed by the dynamic inference engine.',
        'per-callback host timing capture')
    texts['stride_lstm_runtime.h'] = replace_once(
        texts['stride_lstm_runtime.h'], '  int emit = 0;',
        '  int emit = 0;\n  float emit_margin = 0.0F;', 'emit result field')
    texts['stride_lstm_runtime.cc'] = replace_once(
        texts['stride_lstm_runtime.cc'], '  result.emit = emit;',
        '  result.emit = emit;\n  result.emit_margin = emit_logits[1] - emit_logits[0];',
        'emit result assignment')
    texts['stride_lstm_runtime.cc'] = count_domain(texts['stride_lstm_runtime.cc'])
    # Preserve the historical emitted text, including a repeated <cmath> include
    # when canonical live sources already include it; arithmetic is unchanged.
    texts['stride_lstm_model_loader.cc'] = replace_once(
        texts['stride_lstm_model_loader.cc'], '#include <algorithm>',
        '#include <algorithm>\n#include <cmath>', 'loader algorithm include')
    texts['stride_lstm_model_loader.cc'] = replace_once(
        texts['stride_lstm_model_loader.cc'], '    tensors_.emplace(name, std::move(tensor));',
        '    for (float value : tensor.data) if (!std::isfinite(value)) throw std::runtime_error("nonfinite model tensor");\n    tensors_.emplace(name, std::move(tensor));',
        'loader tensor insertion')
    return {str(Path('inc' if name.endswith('.h') else 'prefetcher') / name): text.encode('utf-8')
            for name, text in texts.items()}


def marker_contents(expected):
    return dict(protocol=PROTOCOL, canonical_runtime=str(SOURCE),
                artifacts=sorted(expected),
                byte_lengths={name: len(expected[name]) for name in sorted(expected)})


def verify_prepared(repo, destination, require_marker=False):
    destination = Path(destination)
    expected = generated_sources(repo)
    records = []
    for name, content in expected.items():
        path = destination / name
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError('generated forward byte mismatch: ' + str(path))
        records.append(dict(path=name, bytes=len(content), byte_equal=True))
    marker = destination / MARKER
    if marker.exists():
        if json.loads(marker.read_text()) != marker_contents(expected):
            raise RuntimeError('generated forward marker does not match this source protocol')
    elif require_marker:
        raise RuntimeError('generated forward marker is missing: ' + str(marker))
    return dict(status='PASS', protocol=PROTOCOL, marker_present=marker.exists(),
                artifacts=records)


def prepare(repo, destination):
    destination = Path(destination)
    expected = generated_sources(repo)  # Validate every anchor before any write.
    for name, contents in expected.items():
        target = destination / name
        if not target.parent.is_dir():
            raise RuntimeError('isolated simulator target directory is missing: ' + str(target.parent))
    for name, contents in expected.items():
        (destination / name).write_bytes(contents)
    (destination / MARKER).write_text(json.dumps(marker_contents(expected), indent=2) + '\n')
    return verify_prepared(repo, destination, require_marker=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--verify-only', action='store_true',
                        help='Compare all four generated bytes without writing; legacy archives may lack the new marker.')
    parser.add_argument('--require-marker', action='store_true')
    args = parser.parse_args()
    result = (verify_prepared(args.repo, args.destination, args.require_marker)
              if args.verify_only else prepare(args.repo, args.destination))
    print(json.dumps(result, sort_keys=True))
