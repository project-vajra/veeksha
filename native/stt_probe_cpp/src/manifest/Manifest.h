#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace veeksha::native_stt::manifest {

struct ManifestEntry {
  std::int32_t session_id = 0;
  std::string request_id;
  std::filesystem::path audio_file;
  std::string expected_transcript;
  std::string dataset = "unknown";
  double duration_s = 0.0;
  std::int32_t sample_rate = 0;
  std::optional<std::string> source_id;
  std::optional<std::string> sample_id;
  std::string reference_scope = "clip";
  std::optional<std::string> parent_id;
  std::optional<std::int32_t> chunk_index;
  nlohmann::json metadata;
};

std::vector<ManifestEntry>
LoadManifest(const std::filesystem::path& manifest_path, std::size_t max_rows);

}  // namespace veeksha::native_stt::manifest
