#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>

namespace veeksha::native_stt::config {

struct AudioConfig {
  std::int32_t sample_rate_hz = 16000;
  std::int32_t channels = 1;
  std::string sample_format = "pcm16";
  std::int32_t chunk_ms = 100;
  bool is_realtime_pacing = true;
};

struct SessionConfig {
  std::int32_t connect_timeout_ms = 10000;
  std::int32_t request_timeout_ms = 120000;
  std::string turn_detection = "manual";
};

struct LoadConfig {
  std::int32_t concurrency = 1;
  std::int32_t ramp_seconds = 0;
  std::int32_t duration_seconds = 0;
  std::int32_t seed = 1234;
};

struct TraceConfig {
  bool should_write_events = true;
  bool should_write_raw_provider_events = false;
  bool should_profile = false;
};

struct RunConfig {
  std::string provider;
  std::string endpoint;
  std::string model;
  std::map<std::string, std::string> headers;
  AudioConfig audio;
  SessionConfig session;
  LoadConfig load;
  TraceConfig trace;
};

RunConfig LoadRunConfig(const std::filesystem::path& path);
void ApplyCliOverrides(RunConfig& config,
                       const std::optional<std::string>& provider,
                       const std::optional<std::int32_t>& concurrency,
                       const std::optional<std::int32_t>& duration_seconds);

}  // namespace veeksha::native_stt::config
