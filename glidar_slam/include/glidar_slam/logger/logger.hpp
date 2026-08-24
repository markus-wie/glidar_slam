#pragma once

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace glidar_slam::log {

enum class LogLevel
{
  Debug,
  Info,
  Warn,
  Error,
  None
};

using LogCallback = std::function<void(LogLevel, std::string_view)>;

namespace detail {
inline LogLevel current_level = LogLevel::Info;

inline LogCallback active_sink = [](LogLevel level, std::string_view msg) {
  if (level >= LogLevel::Error) {
    std::cerr << "[SAM ERR] " << msg << '\n';
  } else {
    std::cout << "[SAM] " << msg << '\n';
  }
};

// Helper to replace the first occurrence of "{}" with a value
template <typename T>
inline void format_helper(std::ostringstream & oss, std::string_view & str, const T & value)
{
  std::size_t pos = str.find("{}");
  if (pos != std::string_view::npos) {
    oss << str.substr(0, pos) << value;
    str = str.substr(pos + 2);
  }
}

// Variadic template using C++17 fold expressions to process all arguments
template <typename... Args>
inline std::string format(std::string_view fmt, const Args &... args)
{
  if constexpr (sizeof...(args) == 0) {
    return std::string(fmt);
  } else {
    std::ostringstream oss;
    (format_helper(oss, fmt, args), ...);  // Fold expression over arguments
    oss << fmt;                            // Append any remaining string
    return oss.str();
  }
}
}  // namespace detail

inline void set_log_level(LogLevel level) noexcept
{
  detail::current_level = level;
}

inline void set_log_callback(LogCallback callback) noexcept
{
  if (callback) {
    detail::active_sink = std::move(callback);
  }
}

inline void log_dispatch(LogLevel level, std::string_view msg)
{
  detail::active_sink(level, msg);
}
}  // namespace glidar_slam::log

// __VA_ARGS__ cleanly captures BOTH the format string and the formatting arguments.
#define SAM_LOG(level, ...)                                                                 \
  ((static_cast<int>(level) >= static_cast<int>(glidar_slam::log::detail::current_level))   \
     ? glidar_slam::log::log_dispatch(level, glidar_slam::log::detail::format(__VA_ARGS__)) \
     : (void)0)

#define SAM_TRACE(...) SAM_LOG(glidar_slam::log::LogLevel::Trace, __VA_ARGS__)
#define SAM_DEBUG(...) SAM_LOG(glidar_slam::log::LogLevel::Debug, __VA_ARGS__)
#define SAM_INFO(...) SAM_LOG(glidar_slam::log::LogLevel::Info, __VA_ARGS__)
#define SAM_WARN(...) SAM_LOG(glidar_slam::log::LogLevel::Warn, __VA_ARGS__)
#define SAM_ERROR(...) SAM_LOG(glidar_slam::log::LogLevel::Error, __VA_ARGS__)