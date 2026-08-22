#include "stride_lstm_model_loader.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace stride_lstm {
namespace {

const char kMagic[8] = {'S', 'T', 'R', 'L', 'S', 'T', 'M', '1'};
const std::uint32_t kFormatVersion = 1;
const std::uint32_t kEndianMarker = 0x01020304U;
const std::uint32_t kFloat32 = 1;

const char* const kTensorNames[] = {
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
};

template <typename T>
T Read(std::ifstream* input, const char* label) {
  T value;
  input->read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!*input) {
    throw std::runtime_error(std::string("truncated model while reading ") + label);
  }
  return value;
}

std::uint64_t ElementCount(const std::vector<std::uint32_t>& shape) {
  std::uint64_t count = 1;
  for (std::uint32_t dimension : shape) {
    if (dimension == 0 ||
        count > std::numeric_limits<std::uint64_t>::max() / dimension) {
      throw std::runtime_error("invalid tensor shape");
    }
    count *= dimension;
  }
  return count;
}

void ExpectShape(const FrozenModel& model, const std::string& name,
                 const std::vector<std::uint32_t>& shape) {
  if (model.tensor(name).shape != shape) {
    std::ostringstream message;
    message << "tensor shape mismatch: " << name;
    throw std::runtime_error(message.str());
  }
}

}  // namespace

const Tensor& FrozenModel::tensor(const std::string& name) const {
  const auto found = tensors_.find(name);
  if (found == tensors_.end()) {
    throw std::runtime_error("missing tensor: " + name);
  }
  return found->second;
}

void FrozenModel::Load(const std::string& path) {
  std::ifstream input(path.c_str(), std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open model: " + path);
  }
  char magic[8];
  input.read(magic, sizeof(magic));
  if (!input || std::memcmp(magic, kMagic, sizeof(magic)) != 0) {
    throw std::runtime_error("model magic mismatch");
  }
  format_version_ = Read<std::uint32_t>(&input, "format version");
  const std::uint32_t endian = Read<std::uint32_t>(&input, "endianness");
  const std::uint32_t float_type = Read<std::uint32_t>(&input, "float type");
  const std::uint32_t tensor_count = Read<std::uint32_t>(&input, "tensor count");
  hidden_size_ = Read<std::uint32_t>(&input, "hidden size");
  feature_width_ = Read<std::uint32_t>(&input, "feature width");
  parameter_count_ = Read<std::uint64_t>(&input, "parameter count");
  if (format_version_ != kFormatVersion || endian != kEndianMarker ||
      float_type != kFloat32) {
    throw std::runtime_error("unsupported model format");
  }
  const std::size_t expected_tensor_count =
      sizeof(kTensorNames) / sizeof(kTensorNames[0]);
  if (tensor_count != expected_tensor_count) {
    throw std::runtime_error("model tensor count mismatch");
  }
  tensors_.clear();
  std::uint64_t observed_parameters = 0;
  for (std::uint32_t index = 0; index < tensor_count; ++index) {
    const std::uint16_t name_length =
        Read<std::uint16_t>(&input, "tensor name length");
    const std::uint16_t rank = Read<std::uint16_t>(&input, "tensor rank");
    const std::uint64_t count = Read<std::uint64_t>(&input, "tensor count");
    if (name_length == 0 || rank == 0 || rank > 4) {
      throw std::runtime_error("invalid tensor header");
    }
    Tensor tensor;
    for (std::uint16_t axis = 0; axis < rank; ++axis) {
      tensor.shape.push_back(Read<std::uint32_t>(&input, "tensor shape"));
    }
    if (ElementCount(tensor.shape) != count ||
        count > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("invalid tensor element count");
    }
    std::string name(name_length, '\0');
    input.read(&name[0], name_length);
    if (!input || name != kTensorNames[index]) {
      throw std::runtime_error("model tensor name/order mismatch");
    }
    tensor.data.resize(static_cast<std::size_t>(count));
    input.read(reinterpret_cast<char*>(tensor.data.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
    if (!input || tensors_.count(name) != 0) {
      throw std::runtime_error("truncated or duplicate tensor");
    }
    tensors_.emplace(name, std::move(tensor));
    observed_parameters += count;
  }
  if (input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("trailing bytes after model tensors");
  }
  const std::uint64_t expected_parameters =
      11ULL * hidden_size_ * hidden_size_ +
      (static_cast<std::uint64_t>(feature_width_) + 22ULL) * hidden_size_ + 4ULL;
  if (observed_parameters != parameter_count_ ||
      parameter_count_ != expected_parameters || feature_width_ != 128U ||
      (hidden_size_ != 8U && hidden_size_ != 16U)) {
    throw std::runtime_error("model dimensions/parameter formula mismatch");
  }
  const std::uint32_t h = hidden_size_;
  const std::uint32_t f = feature_width_;
  ExpectShape(*this, "input_projection.weight", {h, f});
  ExpectShape(*this, "input_projection.bias", {h});
  ExpectShape(*this, "encoder_lstm.weight_ih_l0", {4U * h, h});
  ExpectShape(*this, "encoder_lstm.weight_hh_l0", {4U * h, h});
  ExpectShape(*this, "encoder_lstm.bias_ih_l0", {4U * h});
  ExpectShape(*this, "encoder_lstm.bias_hh_l0", {4U * h});
  ExpectShape(*this, "emit_head.weight", {2U, h});
  ExpectShape(*this, "emit_head.bias", {2U});
  ExpectShape(*this, "log_count_mean.weight", {1U, h});
  ExpectShape(*this, "log_count_mean.bias", {1U});
  ExpectShape(*this, "action_decoder.action_cell.weight_ih", {3U * h, 1U});
  ExpectShape(*this, "action_decoder.action_cell.weight_hh", {3U * h, h});
  ExpectShape(*this, "action_decoder.action_cell.bias_ih", {3U * h});
  ExpectShape(*this, "action_decoder.action_cell.bias_hh", {3U * h});
  ExpectShape(*this, "action_decoder.delta_head.weight", {1U, h});
  ExpectShape(*this, "action_decoder.delta_head.bias", {1U});
}

}  // namespace stride_lstm
