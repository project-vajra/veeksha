#include "health/Health.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace veeksha::native_stt::health {

void HealthTracker::RecordManifestRows(std::uint64_t count) {
  manifest_rows_.store(count, std::memory_order_relaxed);
}

void HealthTracker::RecordSessionStart() {
  const auto active =
      active_sessions_.fetch_add(1, std::memory_order_relaxed) + 1;
  auto observed = max_active_sessions_.load(std::memory_order_relaxed);
  while (active > observed &&
         !max_active_sessions_.compare_exchange_weak(
             observed, active, std::memory_order_relaxed)) {
  }
}

void HealthTracker::RecordSessionEnd() {
  active_sessions_.fetch_sub(1, std::memory_order_relaxed);
}

void HealthTracker::RecordConnect(bool is_success) {
  total_connects_.fetch_add(1, std::memory_order_relaxed);
  if (!is_success) {
    failed_connects_.fetch_add(1, std::memory_order_relaxed);
  }
}

void HealthTracker::RecordFrameReceived() {
  total_frames_received_.fetch_add(1, std::memory_order_relaxed);
}

void HealthTracker::RecordFrameSent() {
  total_frames_sent_.fetch_add(1, std::memory_order_relaxed);
}

void HealthTracker::RecordEvent() {
  events_written_.fetch_add(1, std::memory_order_relaxed);
}

void HealthTracker::RecordRequestMetric() {
  request_metrics_written_.fetch_add(1, std::memory_order_relaxed);
}

void HealthTracker::RecordError() {
  errors_written_.fetch_add(1, std::memory_order_relaxed);
}

void HealthTracker::Write(const std::filesystem::path& output_dir) const {
  nlohmann::json root = {
      {"schema_version", 1},
      {"manifest_rows", manifest_rows_.load(std::memory_order_relaxed)},
      {"events_written", events_written_.load(std::memory_order_relaxed)},
      {"request_metrics_written",
       request_metrics_written_.load(std::memory_order_relaxed)},
      {"errors_written", errors_written_.load(std::memory_order_relaxed)},
      {"active_sessions", active_sessions_.load(std::memory_order_relaxed)},
      {"max_active_sessions",
       max_active_sessions_.load(std::memory_order_relaxed)},
      {"total_connects", total_connects_.load(std::memory_order_relaxed)},
      {"failed_connects", failed_connects_.load(std::memory_order_relaxed)},
      {"total_frames_received",
       total_frames_received_.load(std::memory_order_relaxed)},
      {"total_frames_sent", total_frames_sent_.load(std::memory_order_relaxed)},
      {"event_writer_dropped_events", 0},
      {"status", "ok"}};

  const auto path = output_dir / "native_probe_health.json";
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to open health output: " + path.string());
  }
  output << root.dump(2) << '\n';
}

}  // namespace veeksha::native_stt::health
