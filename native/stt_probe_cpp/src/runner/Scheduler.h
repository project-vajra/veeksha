#pragma once

#include <memory>
#include <vector>

#include "config/Config.h"
#include "health/Health.h"
#include "manifest/Manifest.h"
#include "trace/Writer.h"

namespace veeksha::native_stt::runner {

class Scheduler {
 public:
  Scheduler(config::RunConfig run_config,
            std::vector<manifest::ManifestEntry> manifest,
            trace::OutputWriters& writers, health::HealthTracker& health);

  std::int32_t Run(bool is_dry_run);

 private:
  void RunDry();
  std::int32_t RunLive();

  config::RunConfig run_config_;
  std::vector<manifest::ManifestEntry> manifest_;
  trace::OutputWriters& writers_;
  health::HealthTracker& health_;
};

}  // namespace veeksha::native_stt::runner
