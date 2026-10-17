#include "stride_lstm_runtime.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string model;
  std::string events;
  std::string output;
  std::string stats;
  std::uint64_t max_events = 0;
};

void Usage(std::ostream& output) {
  output
      << "Usage: stride_lstm_standalone --model model.bin --events events.csv "
         "--output outputs.jsonl [--stats stats.json] [--max-events N]\n"
      << "Input CSV must contain pc and line columns. Only those columns are "
         "used by inference.\n";
}

Options Parse(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      Usage(std::cout);
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value for " + argument);
    }
    const std::string value(argv[++index]);
    if (argument == "--model") {
      options.model = value;
    } else if (argument == "--events") {
      options.events = value;
    } else if (argument == "--output") {
      options.output = value;
    } else if (argument == "--stats") {
      options.stats = value;
    } else if (argument == "--max-events") {
      options.max_events = std::stoull(value);
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  if (options.model.empty() || options.events.empty() ||
      options.output.empty()) {
    throw std::runtime_error("--model, --events, and --output are required");
  }
  return options;
}

std::vector<std::string> Split(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream input(line);
  std::string field;
  while (std::getline(input, field, ',')) {
    if (!field.empty() && field.back() == '\r') {
      field.pop_back();
    }
    fields.push_back(field);
  }
  return fields;
}

std::uint64_t ParseUnsigned(const std::string& value) {
  std::size_t used = 0;
  const std::uint64_t result = std::stoull(value, &used, 0);
  if (used != value.size()) {
    throw std::runtime_error("invalid integer: " + value);
  }
  return result;
}

template <typename T>
void WriteArray(std::ostream& output, const std::vector<T>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << values[index];
  }
  output << ']';
}

void WriteResult(std::ostream& output, std::uint64_t event_index,
                 const stride_lstm::InferenceResult& result) {
  output << "{\"event_index\":" << event_index
         << ",\"pc\":" << result.pc
         << ",\"line\":" << result.line
         << ",\"emit\":" << result.emit
         << ",\"k\":" << result.k
         << ",\"deltas\":";
  WriteArray(output, result.deltas);
  output << ",\"addresses\":";
  WriteArray(output, result.addresses);
  output << ",\"hidden\":";
  WriteArray(output, result.hidden);
  output << ",\"cell\":";
  WriteArray(output, result.cell);
  output << ",\"nanoseconds\":" << result.nanoseconds << "}\n";
}

void WriteStats(const std::string& path,
                const stride_lstm::StrideLSTMRuntime& runtime) {
  if (path.empty()) {
    return;
  }
  const stride_lstm::RuntimeStats stats = runtime.stats();
  std::ofstream output(path.c_str());
  if (!output) {
    throw std::runtime_error("cannot write stats: " + path);
  }
  output << std::setprecision(12)
         << "{\"live_inference_calls\":" << stats.calls
         << ",\"live_generated_addresses\":" << stats.generated_addresses
         << ",\"live_unique_pcs\":" << stats.unique_pcs
         << ",\"live_peak_pc_states\":" << stats.peak_pc_states
         << ",\"live_weight_bytes\":" << stats.weight_bytes
         << ",\"bytes_per_pc_state\":" << stats.bytes_per_pc_state
         << ",\"live_peak_recurrent_state_bytes\":"
         << stats.peak_recurrent_state_bytes
         << ",\"live_total_deployment_bytes\":"
         << stats.total_deployment_bytes
         << ",\"host_total_nanoseconds\":" << stats.total_nanoseconds
         << ",\"host_mean_nanoseconds\":" << stats.mean_nanoseconds
         << ",\"host_p50_nanoseconds\":" << stats.p50_nanoseconds
         << ",\"host_p95_nanoseconds\":" << stats.p95_nanoseconds
         << ",\"host_p99_nanoseconds\":" << stats.p99_nanoseconds
         << ",\"host_maximum_nanoseconds\":" << stats.maximum_nanoseconds
         << ",\"host_events_per_second\":" << stats.events_per_second
         << ",\"static_operations\":"
         << stride_lstm::StaticOperationCountsJson(
                runtime.model().hidden_size())
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = Parse(argc, argv);
    stride_lstm::StrideLSTMRuntime runtime(options.model);
    std::ifstream input(options.events.c_str());
    std::ofstream output(options.output.c_str());
    if (!input || !output) {
      throw std::runtime_error("cannot open events/output file");
    }
    output << std::setprecision(9);
    std::string header;
    if (!std::getline(input, header)) {
      throw std::runtime_error("empty event CSV");
    }
    const std::vector<std::string> columns = Split(header);
    std::size_t pc_column = columns.size();
    std::size_t line_column = columns.size();
    for (std::size_t index = 0; index < columns.size(); ++index) {
      if (columns[index] == "pc") {
        pc_column = index;
      } else if (columns[index] == "line") {
        line_column = index;
      }
    }
    if (pc_column == columns.size() || line_column == columns.size()) {
      throw std::runtime_error("event CSV requires pc and line columns");
    }
    std::string row;
    std::uint64_t event_index = 0;
    while (std::getline(input, row)) {
      if (row.empty()) {
        continue;
      }
      if (options.max_events != 0 && event_index >= options.max_events) {
        break;
      }
      const std::vector<std::string> fields = Split(row);
      if (pc_column >= fields.size() || line_column >= fields.size()) {
        throw std::runtime_error("short event CSV row");
      }
      const stride_lstm::InferenceResult result = runtime.Infer(
          ParseUnsigned(fields[pc_column]),
          ParseUnsigned(fields[line_column]));
      WriteResult(output, event_index, result);
      ++event_index;
    }
    WriteStats(options.stats, runtime);
    std::cout << "PASS events=" << event_index
              << " exact_pc_states=" << runtime.stats().unique_pcs << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 2;
  }
}
