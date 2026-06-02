#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace veeksha::native_stt::util {

struct CliOptions {
  std::filesystem::path run_config;
  std::filesystem::path manifest;
  std::filesystem::path output_dir;
  std::optional<std::string> provider;
  std::optional<std::int32_t> concurrency;
  std::optional<std::int32_t> duration_seconds;
  std::size_t max_requests = 0;
  bool dry_run = false;
  bool help = false;
};

CliOptions ParseCli(std::int32_t argc, char** argv);
std::string Usage();

}  // namespace veeksha::native_stt::util
