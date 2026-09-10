"""Extend the existing scalar forward in an isolated build, preserving arithmetic."""
from pathlib import Path


def prepare(repo, destination):
    source = Path(repo) / 'formal_NN_training/experiments/602_lstm_stride_live_inference/runtime'
    destination = Path(destination)
    texts = {name: (source / name).read_text() for name in (
        'stride_lstm_runtime.h', 'stride_lstm_runtime.cc',
        'stride_lstm_model_loader.h', 'stride_lstm_model_loader.cc')}
    texts['stride_lstm_runtime.h'] = texts['stride_lstm_runtime.h'].replace(
        '  void Reset();', '''  void Reset();
  InferenceResult InferState(std::uint64_t pc, std::uint64_t line, const RecurrentState& prior);
  void SetOutputLimit(std::uint64_t limit) { output_limit_ = limit; }
''').replace('  FrozenModel model_;', '  std::uint64_t output_limit_ = 0;\n  FrozenModel model_;')
    texts['stride_lstm_runtime.cc'] = texts['stride_lstm_runtime.cc'].replace(
        'void StrideLSTMRuntime::Reset()', '''InferenceResult StrideLSTMRuntime::InferState(std::uint64_t pc, std::uint64_t line, const RecurrentState& prior) {
  states_.clear();
  states_[pc] = prior;
  try { auto result = Infer(pc,line); states_.clear(); return result; }
  catch (...) { states_.clear(); throw; }
}

void StrideLSTMRuntime::Reset()''').replace(
        'slot < count;', 'slot < count && (output_limit_ == 0 || slot < output_limit_);').replace(
        '  timings_ns_.push_back(result.nanoseconds);',
        '  // Per-callback host timing vectors are not needed by the dynamic inference engine.')
    for name, contents in texts.items():
        target = destination / ('inc' if name.endswith('.h') else 'prefetcher') / name
        target.write_text(contents)


if __name__ == '__main__':
    import sys
    prepare(Path(__file__).resolve().parents[4], Path(sys.argv[1]))
