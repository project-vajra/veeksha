#include "manifest/Manifest.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace veeksha::native_stt::manifest {
namespace {

std::optional<std::string> OptionalString(const nlohmann::json& object,
                                          const char* key) {
  if (!object.contains(key) || object.at(key).is_null()) {
    return std::nullopt;
  }
  return object.at(key).get<std::string>();
}

std::optional<std::int32_t> OptionalInt(const nlohmann::json& object,
                                        const char* key) {
  if (!object.contains(key) || object.at(key).is_null()) {
    return std::nullopt;
  }
  return object.at(key).get<std::int32_t>();
}

}  // namespace

std::vector<ManifestEntry>
LoadManifest(const std::filesystem::path& manifest_path, std::size_t max_rows) {
  std::ifstream input(manifest_path);
  if (!input) {
    throw std::runtime_error("failed to open manifest: " +
                             manifest_path.string());
  }

  const auto manifest_dir = manifest_path.parent_path();
  std::vector<ManifestEntry> rows;
  std::string line;
  std::size_t line_number = 0;

  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) {
      continue;
    }
    if (max_rows != 0 && rows.size() >= max_rows) {
      break;
    }

    nlohmann::json raw;
    try {
      raw = nlohmann::json::parse(line);
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error("failed to parse manifest line " +
                               std::to_string(line_number) + ": " + e.what());
    }

    ManifestEntry entry;
    entry.metadata = raw;
    entry.session_id =
        raw.value("session_id", static_cast<std::int32_t>(rows.size()));
    entry.request_id = "r-" + std::to_string(rows.size());
    entry.audio_file = raw.at("audio_file").get<std::string>();
    if (entry.audio_file.is_relative()) {
      entry.audio_file = manifest_dir / entry.audio_file;
    }
    entry.expected_transcript =
        raw.at("expected_transcript").get<std::string>();
    entry.dataset = raw.value("dataset", "unknown");
    entry.duration_s = raw.value("duration_s", 0.0);
    entry.sample_rate = raw.value("sample_rate", 0);
    entry.source_id = OptionalString(raw, "source_id");
    entry.sample_id = OptionalString(raw, "sample_id");
    entry.reference_scope = raw.value("reference_scope", "clip");
    entry.parent_id = OptionalString(raw, "parent_id");
    entry.chunk_index = OptionalInt(raw, "chunk_index");
    rows.push_back(std::move(entry));
  }

  return rows;
}

}  // namespace veeksha::native_stt::manifest
