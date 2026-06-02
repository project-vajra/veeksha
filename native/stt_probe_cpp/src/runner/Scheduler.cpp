#include "runner/Scheduler.h"

#include <openssl/ssl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/ssl/context.hpp>
#include <boost/asio/ssl/stream_base.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/this_coro.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <nlohmann/json.hpp>

#include "provider/ProviderRegistry.h"

namespace veeksha::native_stt::runner {
namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace ssl = boost::asio::ssl;
namespace websocket = boost::beast::websocket;
using tcp = boost::asio::ip::tcp;
using Clock = std::chrono::steady_clock;

constexpr std::int32_t kBytesPerSample = 2;

struct Endpoint {
  std::string scheme;
  std::string host;
  std::string port;
  std::string target;
  std::string host_header;
  bool is_tls = false;
};

struct PcmAudio {
  std::string bytes;
  std::int32_t sample_rate_hz = 0;
  std::int32_t channels = 0;
};

struct RequestState {
  std::string request_id;
  std::string session_id;
  std::optional<std::int64_t> connect_start_ns;
  std::optional<std::int64_t> connect_done_ns;
  std::optional<std::int64_t> audio_send_start_ns;
  std::optional<std::int64_t> audio_send_done_ns;
  std::optional<std::int64_t> commit_sent_ns;
  std::optional<std::int64_t> first_rx_frame_ns;
  std::optional<std::int64_t> first_partial_ns;
  std::optional<std::int64_t> final_transcript_ns;
  std::optional<std::int64_t> close_ns;
  std::string partial_transcript;
  std::string final_transcript;
  std::string transcript_accumulator;
  std::string status = "ok";
  std::string error_message;
  std::vector<double> send_drift_ms;
  std::vector<double> rx_parse_delay_ms;
  std::size_t pcm_byte_count = 0;
  std::size_t chunk_count = 0;
  bool has_recorded_connect = false;
};

std::int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}

std::int64_t UnixNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::int64_t ToNs(Clock::time_point time_point) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             time_point.time_since_epoch())
      .count();
}

std::uint16_t ReadU16(const std::vector<char>& data, std::size_t offset) {
  return static_cast<std::uint16_t>(
      static_cast<unsigned char>(data.at(offset)) |
      (static_cast<unsigned char>(data.at(offset + 1)) << 8));
}

std::uint32_t ReadU32(const std::vector<char>& data, std::size_t offset) {
  return static_cast<std::uint32_t>(
      static_cast<unsigned char>(data.at(offset)) |
      (static_cast<unsigned char>(data.at(offset + 1)) << 8) |
      (static_cast<unsigned char>(data.at(offset + 2)) << 16) |
      (static_cast<unsigned char>(data.at(offset + 3)) << 24));
}

bool HasTag(const std::vector<char>& data, std::size_t offset,
            const char* expected) {
  return offset + 4 <= data.size() &&
         std::memcmp(data.data() + offset, expected, 4) == 0;
}

PcmAudio LoadPcm16Wav(const std::filesystem::path& path,
                      const config::AudioConfig& audio_config) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open audio file: " + path.string());
  }

  input.seekg(0, std::ios::end);
  const auto file_size = input.tellg();
  if (file_size < 0) {
    throw std::runtime_error("failed to size audio file: " + path.string());
  }
  input.seekg(0, std::ios::beg);

  std::vector<char> data(static_cast<std::size_t>(file_size));
  if (!data.empty()) {
    input.read(data.data(), file_size);
  }
  if (!input) {
    throw std::runtime_error("failed to read audio file: " + path.string());
  }
  if (data.size() < 44 || !HasTag(data, 0, "RIFF") ||
      !HasTag(data, 8, "WAVE")) {
    throw std::runtime_error("unsupported WAV file: " + path.string());
  }

  std::optional<std::uint16_t> audio_format;
  std::optional<std::uint16_t> num_channels;
  std::optional<std::uint32_t> sample_rate;
  std::optional<std::uint16_t> bits_per_sample;
  std::optional<std::size_t> data_offset;
  std::optional<std::size_t> data_size;

  std::size_t offset = 12;
  while (offset + 8 <= data.size()) {
    const std::uint32_t chunk_size = ReadU32(data, offset + 4);
    const std::size_t payload_offset = offset + 8;
    const std::size_t next_offset =
        payload_offset + chunk_size + (chunk_size % 2);
    if (payload_offset + chunk_size > data.size()) {
      throw std::runtime_error("truncated WAV chunk in: " + path.string());
    }

    if (HasTag(data, offset, "fmt ")) {
      if (chunk_size < 16) {
        throw std::runtime_error("invalid WAV fmt chunk: " + path.string());
      }
      audio_format = ReadU16(data, payload_offset);
      num_channels = ReadU16(data, payload_offset + 2);
      sample_rate = ReadU32(data, payload_offset + 4);
      bits_per_sample = ReadU16(data, payload_offset + 14);
    } else if (HasTag(data, offset, "data")) {
      data_offset = payload_offset;
      data_size = chunk_size;
    }
    offset = next_offset;
  }

  if (!audio_format || !num_channels || !sample_rate || !bits_per_sample ||
      !data_offset || !data_size) {
    throw std::runtime_error("missing WAV fmt/data chunk: " + path.string());
  }
  if (*audio_format != 1 || *bits_per_sample != 16) {
    throw std::runtime_error("only PCM16 WAV input is supported: " +
                             path.string());
  }
  if (static_cast<std::int32_t>(*num_channels) != audio_config.channels) {
    throw std::runtime_error("WAV channel count does not match run config: " +
                             path.string());
  }
  if (static_cast<std::int32_t>(*sample_rate) != audio_config.sample_rate_hz) {
    throw std::runtime_error("WAV sample rate does not match run config: " +
                             path.string());
  }

  PcmAudio audio;
  audio.sample_rate_hz = static_cast<std::int32_t>(*sample_rate);
  audio.channels = static_cast<std::int32_t>(*num_channels);
  audio.bytes.assign(data.data() + *data_offset, *data_size);
  return audio;
}

Endpoint ParseEndpoint(const std::string& raw_endpoint) {
  const auto scheme_end = raw_endpoint.find("://");
  if (scheme_end == std::string::npos) {
    throw std::runtime_error("endpoint must start with ws:// or wss://");
  }

  Endpoint endpoint;
  endpoint.scheme = raw_endpoint.substr(0, scheme_end);
  endpoint.is_tls = endpoint.scheme == "wss";
  if (endpoint.scheme != "ws" && endpoint.scheme != "wss") {
    throw std::runtime_error("unsupported websocket endpoint scheme: " +
                             endpoint.scheme);
  }

  const std::string rest = raw_endpoint.substr(scheme_end + 3);
  const auto target_start = rest.find('/');
  const std::string authority =
      target_start == std::string::npos ? rest : rest.substr(0, target_start);
  endpoint.target =
      target_start == std::string::npos ? "/" : rest.substr(target_start);
  if (authority.empty()) {
    throw std::runtime_error("endpoint host is empty");
  }

  const auto colon = authority.rfind(':');
  if (colon != std::string::npos && colon + 1 < authority.size()) {
    endpoint.host = authority.substr(0, colon);
    endpoint.port = authority.substr(colon + 1);
  } else {
    endpoint.host = authority;
    endpoint.port = endpoint.is_tls ? "443" : "80";
  }
  endpoint.host_header = endpoint.host + ":" + endpoint.port;
  return endpoint;
}

std::optional<double> Percentile(std::vector<double> values,
                                 double percentile) {
  if (values.empty()) {
    return std::nullopt;
  }
  std::sort(values.begin(), values.end());
  const double position = percentile * static_cast<double>(values.size() - 1);
  const auto index = static_cast<std::size_t>(position);
  return values.at(std::min(index, values.size() - 1));
}

std::string CleanErrorMessage(std::string message) {
  const auto detail_start = message.find(" [");
  if (detail_start != std::string::npos) {
    message = message.substr(0, detail_start);
  }
  return message;
}

nlohmann::json OptionalInt64(std::optional<std::int64_t> value) {
  if (!value) {
    return nullptr;
  }
  return *value;
}

nlohmann::json OptionalDouble(std::optional<double> value) {
  if (!value) {
    return nullptr;
  }
  return *value;
}

void WriteEvent(trace::OutputWriters& writers, health::HealthTracker& health,
                nlohmann::json event) {
  writers.WriteEvent(event);
  health.RecordEvent();
}

void WriteError(trace::OutputWriters& writers, health::HealthTracker& health,
                const RequestState& state, const std::string& error_code,
                const std::string& error_message) {
  writers.WriteError({
      {"schema_version", 1},
      {"event_type", "error"},
      {"request_id", state.request_id},
      {"session_id", state.session_id},
      {"monotonic_ns", SteadyNowNs()},
      {"wall_unix_ns", UnixNowNs()},
      {"error_code", error_code},
      {"error_message", error_message},
  });
  health.RecordError();
}

void ApplyProviderEvent(const provider::ProviderEvent& event,
                        RequestState& state) {
  if (event.is_partial && event.text_delta.has_value()) {
    state.transcript_accumulator += *event.text_delta;
    if (!event.text_delta->empty()) {
      ++state.chunk_count;
    }
    if (!state.first_partial_ns.has_value() &&
        !state.transcript_accumulator.empty()) {
      state.first_partial_ns = SteadyNowNs();
      state.partial_transcript = state.transcript_accumulator;
    }
  }
  if (event.is_final) {
    state.final_transcript =
        event.final_text.value_or(state.transcript_accumulator);
    if (state.final_transcript.empty()) {
      state.final_transcript = state.transcript_accumulator;
    }
    state.final_transcript_ns = SteadyNowNs();
  }
  if (event.is_error) {
    throw std::runtime_error(event.error_message.value_or("provider error"));
  }
}

nlohmann::json BuildMetricRow(const manifest::ManifestEntry& row,
                              const RequestState& state) {
  return {
      {"schema_version", 1},
      {"request_id", row.request_id},
      {"session_id", state.session_id},
      {"dataset", row.dataset},
      {"source_id", row.source_id.value_or("")},
      {"sample_id", row.sample_id.value_or("")},
      {"reference_scope", row.reference_scope},
      {"parent_id", row.parent_id.value_or("")},
      {"chunk_index", row.chunk_index.has_value()
                          ? nlohmann::json(*row.chunk_index)
                          : nlohmann::json(nullptr)},
      {"audio_file", row.audio_file.string()},
      {"audio_duration_ms", row.duration_s * 1000.0},
      {"sample_rate", row.sample_rate},
      {"connect_start_ns", OptionalInt64(state.connect_start_ns)},
      {"connect_done_ns", OptionalInt64(state.connect_done_ns)},
      {"audio_send_start_ns", OptionalInt64(state.audio_send_start_ns)},
      {"audio_send_done_ns", OptionalInt64(state.audio_send_done_ns)},
      {"commit_sent_ns", OptionalInt64(state.commit_sent_ns)},
      {"first_rx_frame_ns", OptionalInt64(state.first_rx_frame_ns)},
      {"first_partial_ns", OptionalInt64(state.first_partial_ns)},
      {"final_transcript_ns", OptionalInt64(state.final_transcript_ns)},
      {"close_ns", OptionalInt64(state.close_ns)},
      {"send_drift_p50_ms",
       OptionalDouble(Percentile(state.send_drift_ms, 0.5))},
      {"send_drift_p99_ms",
       OptionalDouble(Percentile(state.send_drift_ms, 0.99))},
      {"rx_parse_delay_p50_ms",
       OptionalDouble(Percentile(state.rx_parse_delay_ms, 0.5))},
      {"rx_parse_delay_p99_ms",
       OptionalDouble(Percentile(state.rx_parse_delay_ms, 0.99))},
      {"pcm_byte_count", state.pcm_byte_count},
      {"chunk_count", state.chunk_count},
      {"partial_transcript", state.partial_transcript.empty()
                                 ? nlohmann::json(nullptr)
                                 : nlohmann::json(state.partial_transcript)},
      {"final_transcript", state.final_transcript},
      {"expected_transcript", row.expected_transcript},
      {"error_message", state.error_message.empty()
                            ? nlohmann::json(nullptr)
                            : nlohmann::json(state.error_message)},
      {"status", state.status}};
}

template <typename WsStream>
void ConfigureWebSocket(WsStream& ws, const Endpoint& endpoint,
                        const config::RunConfig& run_config) {
  ws.set_option(
      websocket::stream_base::timeout::suggested(beast::role_type::client));
  ws.set_option(websocket::stream_base::decorator(
      [headers = run_config.headers](websocket::request_type& req) {
        for (const auto& [key, value] : headers) {
          req.set(key, value);
        }
      }));
  ws.binary(false);
  beast::get_lowest_layer(ws).expires_after(
      std::chrono::milliseconds(run_config.session.connect_timeout_ms));
  (void)endpoint;
}

template <typename WsStream>
asio::awaitable<void>
WriteMessage(std::shared_ptr<WsStream> ws, trace::OutputWriters& writers,
             health::HealthTracker& health, const RequestState& state,
             const provider::OutboundMessage& message,
             const std::string& event_type,
             std::optional<std::int64_t> scheduled_ns,
             std::optional<std::int64_t> actual_ns) {
  ws->binary(message.binary);
  co_await ws->async_write(asio::buffer(message.payload), asio::use_awaitable);
  health.RecordFrameSent();

  WriteEvent(writers, health,
             {{"schema_version", 1},
              {"request_id", state.request_id},
              {"session_id", state.session_id},
              {"event_type", event_type},
              {"monotonic_ns", SteadyNowNs()},
              {"wall_unix_ns", UnixNowNs()},
              {"send_scheduled_ns", OptionalInt64(scheduled_ns)},
              {"send_actual_ns", OptionalInt64(actual_ns)},
              {"audio_offset_ms", message.audio_offset_ms.has_value()
                                      ? nlohmann::json(*message.audio_offset_ms)
                                      : nlohmann::json(nullptr)}});
}

template <typename WsStream>
asio::awaitable<provider::ProviderEvent>
ReadProviderEvent(std::shared_ptr<WsStream> ws, provider::Provider& provider,
                  RequestState& state, trace::OutputWriters& writers,
                  health::HealthTracker& health) {
  beast::flat_buffer buffer;
  co_await ws->async_read(buffer, asio::use_awaitable);
  const auto rx_ns = SteadyNowNs();
  const auto wall_ns = UnixNowNs();
  if (!state.first_rx_frame_ns.has_value()) {
    state.first_rx_frame_ns = rx_ns;
  }
  health.RecordFrameReceived();

  const auto frame = beast::buffers_to_string(buffer.data());
  const auto parsed_event = provider.ParseEvent(frame);
  const auto parsed_ns = SteadyNowNs();
  state.rx_parse_delay_ms.push_back(static_cast<double>(parsed_ns - rx_ns) /
                                    1'000'000.0);

  WriteEvent(writers, health,
             {{"schema_version", 1},
              {"request_id", state.request_id},
              {"session_id", state.session_id},
              {"event_type", "rx_frame"},
              {"monotonic_ns", rx_ns},
              {"wall_unix_ns", wall_ns},
              {"rx_frame_at_ns", rx_ns}});

  WriteEvent(
      writers, health,
      {{"schema_version", 1},
       {"request_id", state.request_id},
       {"session_id", state.session_id},
       {"event_type", "parsed_event"},
       {"monotonic_ns", parsed_ns},
       {"wall_unix_ns", UnixNowNs()},
       {"rx_frame_at_ns", rx_ns},
       {"parsed_at_ns", parsed_ns},
       {"provider_event_type", parsed_event.event_type},
       {"provider_event_id", parsed_event.provider_event_id},
       {"provider_item_id", parsed_event.provider_item_id},
       {"text_delta", parsed_event.text_delta.has_value()
                          ? nlohmann::json(*parsed_event.text_delta)
                          : nlohmann::json(nullptr)},
       {"final_text", parsed_event.final_text.has_value()
                          ? nlohmann::json(*parsed_event.final_text)
                          : nlohmann::json(nullptr)},
       {"error_code", parsed_event.error_code.has_value()
                          ? nlohmann::json(*parsed_event.error_code)
                          : nlohmann::json(nullptr)},
       {"error_message", parsed_event.error_message.has_value()
                             ? nlohmann::json(*parsed_event.error_message)
                             : nlohmann::json(nullptr)}});

  co_return parsed_event;
}

template <typename WsStream>
asio::awaitable<void>
SendAudio(std::shared_ptr<WsStream> ws,
          std::shared_ptr<provider::Provider> provider,
          std::shared_ptr<RequestState> state, trace::OutputWriters& writers,
          health::HealthTracker& health, const config::RunConfig& run_config,
          const PcmAudio& audio) {
  try {
    const std::int64_t bytes_per_second =
        static_cast<std::int64_t>(audio.sample_rate_hz) * audio.channels *
        kBytesPerSample;
    const std::size_t chunk_size = std::max<std::size_t>(
        1, static_cast<std::size_t>(bytes_per_second *
                                    run_config.audio.chunk_ms / 1000));
    const auto send_start_time = Clock::now();
    state->audio_send_start_ns = ToNs(send_start_time);
    state->pcm_byte_count = audio.bytes.size();

    asio::steady_timer timer(co_await asio::this_coro::executor);
    for (std::size_t offset = 0; offset < audio.bytes.size();
         offset += chunk_size) {
      const auto remaining = audio.bytes.size() - offset;
      const auto size = std::min(chunk_size, remaining);
      const std::int64_t audio_offset_ms =
          static_cast<std::int64_t>(offset) * 1000 / bytes_per_second;
      const auto scheduled_time =
          send_start_time +
          std::chrono::microseconds(static_cast<std::int64_t>(offset) *
                                    1'000'000 / bytes_per_second);
      const auto scheduled_ns = ToNs(scheduled_time);
      if (run_config.audio.is_realtime_pacing &&
          Clock::now() < scheduled_time) {
        timer.expires_at(scheduled_time);
        co_await timer.async_wait(asio::use_awaitable);
      }

      const std::string_view chunk(audio.bytes.data() + offset, size);
      const auto message = provider->BuildAudioChunk(chunk, audio_offset_ms);
      const auto actual_ns = SteadyNowNs();
      state->send_drift_ms.push_back(
          static_cast<double>(actual_ns - scheduled_ns) / 1'000'000.0);
      co_await WriteMessage(ws, writers, health, *state, message, "send_audio",
                            scheduled_ns, actual_ns);
    }

    state->audio_send_done_ns = SteadyNowNs();
    const auto commit = provider->BuildCommit();
    if (commit.has_value()) {
      const auto actual_ns = SteadyNowNs();
      co_await WriteMessage(ws, writers, health, *state, *commit, "send_commit",
                            std::nullopt, actual_ns);
      state->commit_sent_ns = actual_ns;
    }
  } catch (const std::exception& e) {
    state->status = "send_error";
    state->error_message = CleanErrorMessage(e.what());
    WriteError(writers, health, *state, "send_error", state->error_message);
    beast::get_lowest_layer(*ws).cancel();
  }
}

template <typename WsStream>
asio::awaitable<void>
RunConversation(std::shared_ptr<WsStream> ws,
                std::shared_ptr<provider::Provider> provider,
                std::shared_ptr<RequestState> state,
                trace::OutputWriters& writers, health::HealthTracker& health,
                const config::RunConfig& run_config, const PcmAudio& audio) {
  beast::get_lowest_layer(*ws).expires_after(
      std::chrono::milliseconds(run_config.session.request_timeout_ms));

  auto initial_event =
      co_await ReadProviderEvent(ws, *provider, *state, writers, health);
  ApplyProviderEvent(initial_event, *state);

  for (const auto& message : provider->BuildStartSession()) {
    co_await WriteMessage(ws, writers, health, *state, message, "send_control",
                          std::nullopt, SteadyNowNs());
  }

  asio::co_spawn(
      co_await asio::this_coro::executor,
      SendAudio(ws, provider, state, writers, health, run_config, audio),
      asio::detached);

  while (!state->final_transcript_ns.has_value()) {
    auto event =
        co_await ReadProviderEvent(ws, *provider, *state, writers, health);
    ApplyProviderEvent(event, *state);
    if (event.is_final) {
      break;
    }
  }

  boost::system::error_code close_ec;
  co_await ws->async_close(websocket::close_code::normal,
                           asio::redirect_error(asio::use_awaitable, close_ec));
  state->close_ns = SteadyNowNs();
  WriteEvent(writers, health,
             {{"schema_version", 1},
              {"request_id", state->request_id},
              {"session_id", state->session_id},
              {"event_type", "close"},
              {"monotonic_ns", *state->close_ns},
              {"wall_unix_ns", UnixNowNs()},
              {"error_code", close_ec ? close_ec.message() : ""}});
}

asio::awaitable<void>
RunPlainSession(const Endpoint& endpoint,
                std::shared_ptr<provider::Provider> provider,
                std::shared_ptr<RequestState> state,
                trace::OutputWriters& writers, health::HealthTracker& health,
                const config::RunConfig& run_config, const PcmAudio& audio) {
  const auto executor = co_await asio::this_coro::executor;
  tcp::resolver resolver(executor);
  state->connect_start_ns = SteadyNowNs();
  auto results = co_await resolver.async_resolve(endpoint.host, endpoint.port,
                                                 asio::use_awaitable);

  auto ws = std::make_shared<websocket::stream<beast::tcp_stream>>(executor);
  ConfigureWebSocket(*ws, endpoint, run_config);
  co_await beast::get_lowest_layer(*ws).async_connect(results,
                                                      asio::use_awaitable);
  co_await ws->async_handshake(endpoint.host_header, endpoint.target,
                               asio::use_awaitable);
  state->connect_done_ns = SteadyNowNs();
  state->has_recorded_connect = true;
  health.RecordConnect(true);

  co_await RunConversation(ws, provider, state, writers, health, run_config,
                           audio);
}

asio::awaitable<void>
RunTlsSession(const Endpoint& endpoint, ssl::context& ssl_context,
              std::shared_ptr<provider::Provider> provider,
              std::shared_ptr<RequestState> state,
              trace::OutputWriters& writers, health::HealthTracker& health,
              const config::RunConfig& run_config, const PcmAudio& audio) {
  const auto executor = co_await asio::this_coro::executor;
  tcp::resolver resolver(executor);
  state->connect_start_ns = SteadyNowNs();
  auto results = co_await resolver.async_resolve(endpoint.host, endpoint.port,
                                                 asio::use_awaitable);

  auto ws =
      std::make_shared<websocket::stream<beast::ssl_stream<beast::tcp_stream>>>(
          executor, ssl_context);
  ConfigureWebSocket(*ws, endpoint, run_config);
  if (!SSL_set_tlsext_host_name(ws->next_layer().native_handle(),
                                endpoint.host.c_str())) {
    throw std::runtime_error("failed to set TLS SNI host name");
  }
  co_await beast::get_lowest_layer(*ws).async_connect(results,
                                                      asio::use_awaitable);
  co_await ws->next_layer().async_handshake(ssl::stream_base::client,
                                            asio::use_awaitable);
  co_await ws->async_handshake(endpoint.host_header, endpoint.target,
                               asio::use_awaitable);
  state->connect_done_ns = SteadyNowNs();
  state->has_recorded_connect = true;
  health.RecordConnect(true);

  co_await RunConversation(ws, provider, state, writers, health, run_config,
                           audio);
}

asio::awaitable<void> RunOneSession(const config::RunConfig& run_config,
                                    const manifest::ManifestEntry& row,
                                    ssl::context& ssl_context,
                                    trace::OutputWriters& writers,
                                    health::HealthTracker& health) {
  auto state = std::make_shared<RequestState>();
  state->request_id = row.request_id;
  state->session_id = "s-" + std::to_string(row.session_id);
  health.RecordSessionStart();

  try {
    const auto endpoint = ParseEndpoint(run_config.endpoint);
    const auto audio = LoadPcm16Wav(row.audio_file, run_config.audio);
    auto provider =
        std::shared_ptr<provider::Provider>(provider::MakeProvider(run_config));

    WriteEvent(writers, health,
               {{"schema_version", 1},
                {"request_id", state->request_id},
                {"session_id", state->session_id},
                {"event_type", "connect"},
                {"provider", run_config.provider},
                {"monotonic_ns", SteadyNowNs()},
                {"wall_unix_ns", UnixNowNs()},
                {"endpoint", run_config.endpoint}});

    if (endpoint.is_tls) {
      co_await RunTlsSession(endpoint, ssl_context, provider, state, writers,
                             health, run_config, audio);
    } else {
      co_await RunPlainSession(endpoint, provider, state, writers, health,
                               run_config, audio);
    }
  } catch (const std::exception& e) {
    if (state->status == "ok") {
      state->status = state->connect_done_ns.has_value() ? "protocol_error"
                                                         : "connect_error";
      state->error_message = CleanErrorMessage(e.what());
    }
    if (!state->has_recorded_connect) {
      state->has_recorded_connect = true;
      health.RecordConnect(false);
    }
    WriteError(writers, health, *state, state->status, state->error_message);
  }

  writers.WriteRequestMetric(BuildMetricRow(row, *state));
  health.RecordRequestMetric();
  health.RecordSessionEnd();
}

asio::awaitable<void>
RunWorker(const config::RunConfig& run_config,
          const std::vector<manifest::ManifestEntry>& rows,
          std::atomic<std::size_t>& next_idx, ssl::context& ssl_context,
          trace::OutputWriters& writers, health::HealthTracker& health) {
  while (true) {
    const auto row_idx = next_idx.fetch_add(1, std::memory_order_relaxed);
    if (row_idx >= rows.size()) {
      co_return;
    }
    co_await RunOneSession(run_config, rows.at(row_idx), ssl_context, writers,
                           health);
  }
}

}  // namespace

Scheduler::Scheduler(config::RunConfig run_config,
                     std::vector<manifest::ManifestEntry> manifest,
                     trace::OutputWriters& writers,
                     health::HealthTracker& health)
    : run_config_(std::move(run_config)),
      manifest_(std::move(manifest)),
      writers_(writers),
      health_(health) {}

std::int32_t Scheduler::Run(bool is_dry_run) {
  health_.RecordManifestRows(manifest_.size());
  if (is_dry_run) {
    RunDry();
    return 0;
  }
  return RunLive();
}

std::int32_t Scheduler::RunLive() {
  if (manifest_.empty()) {
    return 0;
  }
  if (run_config_.endpoint.empty()) {
    throw std::runtime_error(
        "run config endpoint is required for live native STT");
  }
  if (run_config_.audio.sample_format != "pcm16") {
    throw std::runtime_error("only pcm16 native STT audio is supported");
  }

  asio::io_context io_context;
  ssl::context ssl_context(ssl::context::tls_client);
  ssl_context.set_default_verify_paths();
  ssl_context.set_verify_mode(ssl::verify_peer);

  std::atomic<std::size_t> next_idx{0};
  const auto positive_concurrency =
      std::max<std::int32_t>(1, run_config_.load.concurrency);
  const auto num_workers = std::min<std::size_t>(
      static_cast<std::size_t>(positive_concurrency), manifest_.size());

  for (std::size_t worker_idx = 0; worker_idx < num_workers; ++worker_idx) {
    asio::co_spawn(io_context,
                   RunWorker(run_config_, manifest_, next_idx, ssl_context,
                             writers_, health_),
                   asio::detached);
  }
  io_context.run();
  return 0;
}

void Scheduler::RunDry() {
  writers_.WriteEvent({
      {"schema_version", 1},
      {"event_type", "probe_start"},
      {"provider", run_config_.provider},
      {"monotonic_ns", SteadyNowNs()},
      {"wall_unix_ns", UnixNowNs()},
      {"concurrency", run_config_.load.concurrency},
      {"duration_seconds", run_config_.load.duration_seconds},
  });
  health_.RecordEvent();

  writers_.WriteEvent({
      {"schema_version", 1},
      {"event_type", "manifest_loaded"},
      {"provider", run_config_.provider},
      {"monotonic_ns", SteadyNowNs()},
      {"wall_unix_ns", UnixNowNs()},
      {"row_count", manifest_.size()},
  });
  health_.RecordEvent();

  for (std::size_t i = 0; i < manifest_.size(); ++i) {
    const auto& row = manifest_[i];
    RequestState state;
    state.request_id = row.request_id;
    state.session_id = "s-" + std::to_string(row.session_id);
    state.status = "dry_run";
    writers_.WriteRequestMetric(BuildMetricRow(row, state));
    health_.RecordRequestMetric();
  }
}

}  // namespace veeksha::native_stt::runner
