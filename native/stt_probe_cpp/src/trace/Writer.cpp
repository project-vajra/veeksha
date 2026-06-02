#include "trace/Writer.h"

#include <stdexcept>

#include <nlohmann/json.hpp>

namespace veeksha::native_stt::trace {

JsonlWriter::JsonlWriter(const std::filesystem::path& path) : output_(path) {
  if (!output_) {
    throw std::runtime_error("failed to open jsonl output: " + path.string());
  }
}

void JsonlWriter::Write(const nlohmann::json& row) {
  std::lock_guard<std::mutex> lock(mutex_);
  output_ << row.dump() << '\n';
}

OutputWriters::OutputWriters(const std::filesystem::path& output_dir)
    : events_(output_dir / "native_events.jsonl"),
      request_metrics_(output_dir / "native_request_metrics.jsonl"),
      errors_(output_dir / "native_errors.jsonl") {}

void OutputWriters::WriteEvent(const nlohmann::json& row) {
  events_.Write(row);
}

void OutputWriters::WriteRequestMetric(const nlohmann::json& row) {
  request_metrics_.Write(row);
}

void OutputWriters::WriteError(const nlohmann::json& row) {
  errors_.Write(row);
}

}  // namespace veeksha::native_stt::trace
