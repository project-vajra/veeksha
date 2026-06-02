#pragma once

#include "provider/Provider.h"

namespace veeksha::native_stt::provider {

class VajraProvider final : public Provider {
 public:
  std::string_view Name() const override;
  std::vector<OutboundMessage> BuildStartSession() override;
  OutboundMessage BuildAudioChunk(std::string_view chunk,
                                  std::int64_t audio_offset_ms) override;
  std::optional<OutboundMessage> BuildCommit() override;
  ProviderEvent ParseEvent(std::string_view frame) override;
};

}  // namespace veeksha::native_stt::provider
