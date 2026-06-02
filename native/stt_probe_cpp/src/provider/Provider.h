#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace veeksha::native_stt::provider {

struct OutboundMessage {
  std::string payload;
  bool binary = false;
  std::optional<std::int64_t> audio_offset_ms = std::nullopt;
};

struct ProviderEvent {
  std::string event_type;
  std::string provider_event_id;
  std::string provider_item_id;
  std::optional<std::string> text_delta;
  std::optional<std::string> final_text;
  std::optional<std::string> error_code;
  std::optional<std::string> error_message;
  bool is_partial = false;
  bool is_final = false;
  bool is_error = false;
};

class Provider {
 public:
  virtual ~Provider() = default;

  virtual std::string_view Name() const = 0;
  virtual std::vector<OutboundMessage> BuildStartSession() = 0;
  virtual OutboundMessage BuildAudioChunk(std::string_view chunk,
                                          std::int64_t audio_offset_ms) = 0;
  virtual std::optional<OutboundMessage> BuildCommit() = 0;
  virtual ProviderEvent ParseEvent(std::string_view frame) = 0;
};

}  // namespace veeksha::native_stt::provider
