"""Extend the existing scalar forward in an isolated build, preserving arithmetic."""
from pathlib import Path


def prepare(repo, destination):
    source = Path(repo) / 'formal_NN_training/experiments/602_lstm_stride_live_inference/runtime'
    destination = Path(destination)
    texts = {name: (source / name).read_text() for name in (
        'stride_lstm_runtime.h', 'stride_lstm_runtime.cc',
        'stride_lstm_model_loader.h', 'stride_lstm_model_loader.cc')}
    texts['stride_lstm_model_loader.h'] = texts['stride_lstm_model_loader.h'].replace(
        '  void Load(const std::string& path);',
        '  void Load(const std::string& path);\n  void Publish(const std::vector<float>& values);')
    texts['stride_lstm_model_loader.cc'] = texts['stride_lstm_model_loader.cc'].replace(
        '#include <algorithm>', '#include <algorithm>\n#include <cmath>')
    publish = '''
void FrozenModel::Publish(const std::vector<float>& values) {
  if (values.size() != parameter_count_) throw std::runtime_error("publication size mismatch");
  for (float value : values) if (!std::isfinite(value)) throw std::runtime_error("nonfinite publication");
  std::size_t offset = 0;
  for (const char* name : kTensorNames) {
    Tensor& tensor = tensors_.at(name);
    std::copy(values.begin()+offset, values.begin()+offset+tensor.data.size(), tensor.data.begin());
    offset += tensor.data.size();
  }
}
'''
    texts['stride_lstm_model_loader.cc'] = texts['stride_lstm_model_loader.cc'].replace(
        'void FrozenModel::Load(', publish+'\nvoid FrozenModel::Load(')
    texts['stride_lstm_runtime.h'] = texts['stride_lstm_runtime.h'].replace(
        '  void Reset();', '''  void Reset();
  InferenceResult InferState(std::uint64_t pc, std::uint64_t line, const RecurrentState& prior);
  void Publish(const std::vector<float>& values) { model_.Publish(values); }
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
        '  // Per-callback host timing vectors are not needed by the online engine.')
    for name, contents in texts.items():
        target = destination / ('inc' if name.endswith('.h') else 'prefetcher') / name
        target.write_text(contents)


if __name__ == '__main__':
    import sys
    prepare(Path(__file__).resolve().parents[4], Path(sys.argv[1]))
