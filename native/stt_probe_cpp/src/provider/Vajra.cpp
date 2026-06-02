#include "provider/Vajra.h"

#include <nlohmann/json.hpp>

namespace veeksha::native_stt::provider {

std::string_view VajraProvider::Name() const { return "vajra"; }

std::vector<OutboundMessage> VajraProvider::BuildStartSession() { return {}; }

OutboundMessage VajraProvider::BuildAudioChunk(std::string_view chunk,
                                               std::int64_t audio_offset_ms) {
  OutboundMessage message;
  message.binary = true;
  message.audio_offset_ms = audio_offset_ms;
  message.payload.assign(chunk.data(), chunk.size());
  return message;
}

std::optional<OutboundMessage> VajraProvider::BuildCommit() {
  return OutboundMessage{.payload = "end", .binary = false};
}

ProviderEvent VajraProvider::ParseEvent(std::string_view frame) {
  ProviderEvent event;
  try {
    const auto json = nlohmann::json::parse(frame);
    event.event_type = json.value("type", "");
    if (event.event_type == "delta") {
      event.text_delta = json.value("text", "");
      event.is_partial = true;
    } else if (event.event_type == "done") {
      event.final_text = json.value("text", "");
      event.is_final = true;
    } else if (event.event_type == "error") {
      event.error_message = json.value("message", "");
      event.is_error = true;
    }
  } catch (const nlohmann::json::exception& e) {
    event.event_type = "parse_error";
    event.error_code = "json_parse_error";
    event.error_message = e.what();
    event.is_error = true;
  }
  return event;
}

}  // namespace veeksha::native_stt::provider
