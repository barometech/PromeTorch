#pragma once

#include <exception>
#include <string>
#include <sstream>
#include <vector>
#include "c10/macros/Macros.h"

namespace c10 {

// ============================================================================
// Base Exception Class
// ============================================================================

class PT_API Error : public std::exception {
public:
    Error(std::string msg, std::string file, int line)
        : msg_(std::move(msg))
        , file_(std::move(file))
        , line_(line)
    {
        what_str_ = formatMessage();
    }

    const char* what() const noexcept override {
        return what_str_.c_str();
    }

    const std::string& msg() const noexcept {
        return msg_;
    }

    const std::string& file() const noexcept {
        return file_;
    }

    int line() const noexcept {
        return line_;
    }

private:
    std::string formatMessage() const {
        std::ostringstream oss;
        oss << file_ << ":" << line_ << ": " << msg_;
        return oss.str();
    }

    std::string msg_;
    std::string file_;
    int line_;
    std::string what_str_;
};

// ============================================================================
// Specialized Exceptions
// ============================================================================

class PT_API IndexError : public Error {
public:
    using Error::Error;
};

class PT_API ValueError : public Error {
public:
    using Error::Error;
};

class PT_API TypeError : public Error {
public:
    using Error::Error;
};

class PT_API NotImplementedError : public Error {
public:
    using Error::Error;
};

class PT_API OutOfMemoryError : public Error {
public:
    using Error::Error;
};

class PT_API DeviceError : public Error {
public:
    using Error::Error;
};

// ============================================================================
// Exception Throwing Macros
// ============================================================================

#define PT_THROW_ERROR(ExceptionType, ...) \
    throw ExceptionType( \
        ::c10::detail::formatErrorMessage(__VA_ARGS__), \
        __FILE__, \
        __LINE__ \
    )

#define PT_ERROR(...) PT_THROW_ERROR(::c10::Error, __VA_ARGS__)
#define PT_INDEX_ERROR(...) PT_THROW_ERROR(::c10::IndexError, __VA_ARGS__)
#define PT_VALUE_ERROR(...) PT_THROW_ERROR(::c10::ValueError, __VA_ARGS__)
#define PT_TYPE_ERROR(...) PT_THROW_ERROR(::c10::TypeError, __VA_ARGS__)
#define PT_NOT_IMPLEMENTED(...) PT_THROW_ERROR(::c10::NotImplementedError, __VA_ARGS__)
#define PT_OOM_ERROR(...) PT_THROW_ERROR(::c10::OutOfMemoryError, __VA_ARGS__)
#define PT_DEVICE_ERROR(...) PT_THROW_ERROR(::c10::DeviceError, __VA_ARGS__)

// ============================================================================
// Format Helper
// ============================================================================

namespace detail {

inline std::string formatErrorMessage() {
    return "";
}

inline std::string formatErrorMessage(const std::string& msg) {
    return msg;
}

inline std::string formatErrorMessage(const char* msg) {
    return std::string(msg);
}

template<typename... Args>
std::string formatErrorMessage(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << std::forward<Args>(args));
    return oss.str();
}

} // namespace detail

// ============================================================================
// Warning System
// ============================================================================

enum class WarningType {
    UserWarning,
    DeprecationWarning,
    RuntimeWarning
};

class PT_API Warning {
public:
    static void warn(
        const std::string& message,
        WarningType type = WarningType::UserWarning
    );

    static void set_enabled(bool enabled);
    static bool is_enabled();

private:
    static bool enabled_;
};

#define PT_WARN(...) \
    ::c10::Warning::warn(::c10::detail::formatErrorMessage(__VA_ARGS__))

#define PT_WARN_DEPRECATION(...) \
    ::c10::Warning::warn( \
        ::c10::detail::formatErrorMessage(__VA_ARGS__), \
        ::c10::WarningType::DeprecationWarning \
    )

} // namespace c10
