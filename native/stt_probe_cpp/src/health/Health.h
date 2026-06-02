#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>

namespace veeksha::native_stt::health {

class HealthTracker {
 public:
  void RecordManifestRows(std::uint64_t count);
  void RecordSessionStart();
  void RecordSessionEnd();
  void RecordConnect(bool is_success);
  void RecordFrameReceived();
  void RecordFrameSent();
  void RecordEvent();
  void RecordRequestMetric();
  void RecordError();
  void Write(const std::filesystem::path& output_dir) const;

 private:
  std::atomic<std::uint64_t> manifest_rows_{0};
  std::atomic<std::uint64_t> active_sessions_{0};
  std::atomic<std::uint64_t> max_active_sessions_{0};
  std::atomic<std::uint64_t> total_connects_{0};
  std::atomic<std::uint64_t> failed_connects_{0};
  std::atomic<std::uint64_t> total_frames_received_{0};
  std::atomic<std::uint64_t> total_frames_sent_{0};
  std::atomic<std::uint64_t> events_written_{0};
  std::atomic<std::uint64_t> request_metrics_written_{0};
  std::atomic<std::uint64_t> errors_written_{0};
};

}  // namespace veeksha::native_stt::health
