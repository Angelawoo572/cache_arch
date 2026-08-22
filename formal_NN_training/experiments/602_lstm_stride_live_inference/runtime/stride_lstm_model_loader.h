#ifndef STRIDE_LSTM_MODEL_LOADER_H
#define STRIDE_LSTM_MODEL_LOADER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace stride_lstm {

struct Tensor {
  std::vector<std::uint32_t> shape;
  std::vector<float> data;
};

class FrozenModel {
 public:
  void Load(const std::string& path);
  const Tensor& tensor(const std::string& name) const;

  std::uint32_t format_version() const { return format_version_; }
  std::uint32_t hidden_size() const { return hidden_size_; }
  std::uint32_t feature_width() const { return feature_width_; }
  std::uint64_t parameter_count() const { return parameter_count_; }
  std::uint64_t weight_bytes() const { return parameter_count_ * 4U; }

 private:
  std::uint32_t format_version_ = 0;
  std::uint32_t hidden_size_ = 0;
  std::uint32_t feature_width_ = 0;
  std::uint64_t parameter_count_ = 0;
  std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace stride_lstm

#endif
