#pragma once

#include <filesystem>
#include <fstream>
#include <mutex>

#include <nlohmann/json_fwd.hpp>

namespace veeksha::native_stt::trace {

class JsonlWriter {
 public:
  explicit JsonlWriter(const std::filesystem::path& path);

  void Write(const nlohmann::json& row);

 private:
  std::mutex mutex_;
  std::ofstream output_;
};

class OutputWriters {
 public:
  explicit OutputWriters(const std::filesystem::path& output_dir);

  void WriteEvent(const nlohmann::json& row);
  void WriteRequestMetric(const nlohmann::json& row);
  void WriteError(const nlohmann::json& row);

 private:
  JsonlWriter events_;
  JsonlWriter request_metrics_;
  JsonlWriter errors_;
};

}  // namespace veeksha::native_stt::trace
