#include "util/Cli.h"

#include <stdexcept>

namespace veeksha::native_stt::util {
namespace {

std::string RequireValue(std::int32_t& i, const std::int32_t argc, char** argv,
                         const std::string& arg) {
  if (i + 1 >= argc) {
    throw std::runtime_error("missing value for " + arg);
  }
  ++i;
  return argv[i];
}

}  // namespace

CliOptions ParseCli(const std::int32_t argc, char** argv) {
  CliOptions options;
  for (std::int32_t i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      options.help = true;
    } else if (arg == "--run-config") {
      options.run_config = RequireValue(i, argc, argv, arg);
    } else if (arg == "--manifest") {
      options.manifest = RequireValue(i, argc, argv, arg);
    } else if (arg == "--output-dir") {
      options.output_dir = RequireValue(i, argc, argv, arg);
    } else if (arg == "--provider") {
      options.provider = RequireValue(i, argc, argv, arg);
    } else if (arg == "--concurrency") {
      options.concurrency = static_cast<std::int32_t>(
          std::stoi(RequireValue(i, argc, argv, arg)));
    } else if (arg == "--duration-seconds") {
      options.duration_seconds = static_cast<std::int32_t>(
          std::stoi(RequireValue(i, argc, argv, arg)));
    } else if (arg == "--max-requests") {
      options.max_requests = static_cast<std::size_t>(
          std::stoull(RequireValue(i, argc, argv, arg)));
    } else if (arg == "--dry-run") {
      options.dry_run = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
}

std::string Usage() {
  return R"(Usage:
  veeksha-native-stt \
    --run-config /path/to/native_run.json \
    --manifest /path/to/manifest.jsonl \
    --output-dir /path/to/run_dir \
    [--provider vajra|vllm_realtime] \
    [--concurrency N] \
    [--duration-seconds N] \
    [--max-requests N] \
    [--dry-run]
)";
}

}  // namespace veeksha::native_stt::util
