#include "config/Config.h"

#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace veeksha::native_stt::config {
namespace {

std::string ExpandEnvRefs(std::string value) {
  std::string out;
  out.reserve(value.size());

  for (std::size_t i = 0; i < value.size();) {
    if (value[i] == '$' && i + 1 < value.size() && value[i + 1] == '{') {
      const auto end = value.find('}', i + 2);
      if (end == std::string::npos) {
        out.append(value.substr(i));
        break;
      }
      const auto name = value.substr(i + 2, end - (i + 2));
      const char* env_value = std::getenv(name.c_str());
      if (env_value != nullptr) {
        out.append(env_value);
      }
      i = end + 1;
      continue;
    }
    out.push_back(value[i]);
    ++i;
  }

  return out;
}

template <typename T>
void read_if_present(const nlohmann::json& object, const char* key, T& target) {
  if (object.contains(key) && !object.at(key).is_null()) {
    target = object.at(key).get<T>();
  }
}

}  // namespace

RunConfig LoadRunConfig(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open run config: " + path.string());
  }

  nlohmann::json root;
  input >> root;

  RunConfig config;
  read_if_present(root, "provider", config.provider);
  read_if_present(root, "endpoint", config.endpoint);
  read_if_present(root, "model", config.model);

  if (root.contains("headers") && root.at("headers").is_object()) {
    for (const auto& [key, value] : root.at("headers").items()) {
      if (!value.is_null()) {
        config.headers[key] = ExpandEnvRefs(value.get<std::string>());
      }
    }
  }

  if (root.contains("audio") && root.at("audio").is_object()) {
    const auto& audio = root.at("audio");
    read_if_present(audio, "sample_rate_hz", config.audio.sample_rate_hz);
    read_if_present(audio, "channels", config.audio.channels);
    read_if_present(audio, "sample_format", config.audio.sample_format);
    read_if_present(audio, "chunk_ms", config.audio.chunk_ms);
    read_if_present(audio, "realtime_pacing", config.audio.is_realtime_pacing);
  }

  if (root.contains("session") && root.at("session").is_object()) {
    const auto& session = root.at("session");
    read_if_present(session, "connect_timeout_ms",
                    config.session.connect_timeout_ms);
    read_if_present(session, "request_timeout_ms",
                    config.session.request_timeout_ms);
    read_if_present(session, "turn_detection", config.session.turn_detection);
  }

  if (root.contains("load") && root.at("load").is_object()) {
    const auto& load = root.at("load");
    read_if_present(load, "concurrency", config.load.concurrency);
    read_if_present(load, "ramp_seconds", config.load.ramp_seconds);
    read_if_present(load, "duration_seconds", config.load.duration_seconds);
    read_if_present(load, "seed", config.load.seed);
  }

  if (root.contains("trace") && root.at("trace").is_object()) {
    const auto& trace = root.at("trace");
    read_if_present(trace, "write_events", config.trace.should_write_events);
    read_if_present(trace, "write_raw_provider_events",
                    config.trace.should_write_raw_provider_events);
    read_if_present(trace, "profile", config.trace.should_profile);
  }

  return config;
}

void ApplyCliOverrides(RunConfig& config,
                       const std::optional<std::string>& provider,
                       const std::optional<std::int32_t>& concurrency,
                       const std::optional<std::int32_t>& duration_seconds) {
  if (provider.has_value()) {
    config.provider = *provider;
  }
  if (concurrency.has_value()) {
    config.load.concurrency = *concurrency;
  }
  if (duration_seconds.has_value()) {
    config.load.duration_seconds = *duration_seconds;
  }
}

}  // namespace veeksha::native_stt::config
