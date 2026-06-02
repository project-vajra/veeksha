#pragma once

#include <memory>
#include <stdexcept>

#include "config/Config.h"
#include "provider/Provider.h"
#include "provider/Vajra.h"
#include "provider/VllmRealtime.h"

namespace veeksha::native_stt::provider {

inline std::unique_ptr<Provider>
MakeProvider(const config::RunConfig& run_config) {
  if (run_config.provider == "vajra") {
    return std::make_unique<VajraProvider>();
  }
  if (run_config.provider == "vllm_realtime") {
    return std::make_unique<VllmRealtimeProvider>(run_config.model);
  }
  throw std::runtime_error("unsupported STT provider for native C++ probe: " +
                           run_config.provider);
}

}  // namespace veeksha::native_stt::provider
