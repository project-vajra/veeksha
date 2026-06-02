#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "config/Config.h"
#include "health/Health.h"
#include "manifest/Manifest.h"
#include "runner/Scheduler.h"
#include "trace/Writer.h"
#include "util/Cli.h"

namespace native_stt = veeksha::native_stt;

int main(int argc, char** argv) {
  try {
    auto options =
        native_stt::util::ParseCli(static_cast<std::int32_t>(argc), argv);
    if (options.help) {
      std::cout << native_stt::util::Usage();
      return 0;
    }

    if (options.run_config.empty() || options.manifest.empty() ||
        options.output_dir.empty()) {
      std::cerr << native_stt::util::Usage();
      return 2;
    }

    auto run_config = native_stt::config::LoadRunConfig(options.run_config);
    native_stt::config::ApplyCliOverrides(run_config, options.provider,
                                          options.concurrency,
                                          options.duration_seconds);
    if (run_config.provider.empty()) {
      throw std::runtime_error("provider must be set in config or --provider");
    }

    std::filesystem::create_directories(options.output_dir);
    native_stt::trace::OutputWriters writers(options.output_dir);
    native_stt::health::HealthTracker health;

    const auto manifest = native_stt::manifest::LoadManifest(
        options.manifest, options.max_requests);
    native_stt::runner::Scheduler scheduler(std::move(run_config), manifest,
                                            writers, health);
    const std::int32_t result = scheduler.Run(options.dry_run);
    health.Write(options.output_dir);
    return result;
  } catch (const std::exception& e) {
    std::cerr << "veeksha-native-stt: " << e.what() << '\n';
    return 1;
  }
}
