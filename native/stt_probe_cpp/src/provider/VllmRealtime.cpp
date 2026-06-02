#include "provider/VllmRealtime.h"

#include <array>

#include <nlohmann/json.hpp>

namespace veeksha::native_stt::provider {
namespace {

std::string Base64Encode(std::string_view bytes) {
  static constexpr std::array<char, 64> kAlphabet = {
      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
      'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};

  std::string out;
  out.reserve(((bytes.size() + 2) / 3) * 4);

  std::size_t i = 0;
  while (i + 3 <= bytes.size()) {
    const auto b0 = static_cast<unsigned char>(bytes[i]);
    const auto b1 = static_cast<unsigned char>(bytes[i + 1]);
    const auto b2 = static_cast<unsigned char>(bytes[i + 2]);
    out.push_back(kAlphabet[(b0 >> 2) & 0x3F]);
    out.push_back(kAlphabet[((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F)]);
    out.push_back(kAlphabet[((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03)]);
    out.push_back(kAlphabet[b2 & 0x3F]);
    i += 3;
  }

  const auto remaining = bytes.size() - i;
  if (remaining == 1) {
    const auto b0 = static_cast<unsigned char>(bytes[i]);
    out.push_back(kAlphabet[(b0 >> 2) & 0x3F]);
    out.push_back(kAlphabet[(b0 & 0x03) << 4]);
    out.push_back('=');
    out.push_back('=');
  } else if (remaining == 2) {
    const auto b0 = static_cast<unsigned char>(bytes[i]);
    const auto b1 = static_cast<unsigned char>(bytes[i + 1]);
    out.push_back(kAlphabet[(b0 >> 2) & 0x3F]);
    out.push_back(kAlphabet[((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F)]);
    out.push_back(kAlphabet[(b1 & 0x0F) << 2]);
    out.push_back('=');
  }

  return out;
}

}  // namespace

VllmRealtimeProvider::VllmRealtimeProvider(std::string model)
    : model_(std::move(model)) {}

std::string_view VllmRealtimeProvider::Name() const { return "vllm_realtime"; }

std::vector<OutboundMessage> VllmRealtimeProvider::BuildStartSession() {
  std::vector<OutboundMessage> messages;
  nlohmann::json update = {{"type", "session.update"}};
  if (!model_.empty()) {
    update["model"] = model_;
  }
  messages.push_back({.payload = update.dump(), .binary = false});
  messages.push_back(
      {.payload =
           nlohmann::json({{"type", "input_audio_buffer.commit"}}).dump(),
       .binary = false});
  return messages;
}

OutboundMessage
VllmRealtimeProvider::BuildAudioChunk(std::string_view chunk,
                                      std::int64_t audio_offset_ms) {
  const auto encoded = Base64Encode(chunk);
  const nlohmann::json payload = {
      {"type", "input_audio_buffer.append"},
      {"audio", encoded},
  };
  return OutboundMessage{
      .payload = payload.dump(),
      .binary = false,
      .audio_offset_ms = audio_offset_ms,
  };
}

std::optional<OutboundMessage> VllmRealtimeProvider::BuildCommit() {
  return OutboundMessage{
      .payload = nlohmann::json(
                     {{"type", "input_audio_buffer.commit"}, {"final", true}})
                     .dump(),
      .binary = false};
}

ProviderEvent VllmRealtimeProvider::ParseEvent(std::string_view frame) {
  ProviderEvent event;
  try {
    const auto json = nlohmann::json::parse(frame);
    event.event_type = json.value("type", "");
    if (event.event_type == "transcription.delta") {
      event.text_delta = json.value("delta", "");
      event.is_partial = true;
    } else if (event.event_type == "transcription.done") {
      event.final_text = json.value("text", "");
      event.is_final = true;
    } else if (event.event_type == "error") {
      event.error_message = json.value("error", "");
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
