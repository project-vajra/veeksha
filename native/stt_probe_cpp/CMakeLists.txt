cmake_minimum_required(VERSION 3.27)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

project(veeksha_native_stt VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost REQUIRED COMPONENTS system)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

add_executable(
  veeksha-native-stt
  src/Main.cpp
  src/config/Config.cpp
  src/health/Health.cpp
  src/manifest/Manifest.cpp
  src/provider/Vajra.cpp
  src/provider/VllmRealtime.cpp
  src/runner/Scheduler.cpp
  src/trace/Writer.cpp
  src/util/Cli.cpp)

target_include_directories(veeksha-native-stt PRIVATE src)

target_link_libraries(
  veeksha-native-stt
  PRIVATE Boost::system
          OpenSSL::SSL
          OpenSSL::Crypto
          nlohmann_json::nlohmann_json
          Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(
    veeksha-native-stt
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
endif()
