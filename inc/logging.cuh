#ifndef LOG_H
#define LOG_H

#include <iostream>

#define LOG_LEVEL_ERROR 0
#define LOG_LEVEL_WARN 1
#define LOG_LEVEL_INFO 2
#define LOG_LEVEL_DEBUG 3

#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_INFO
#endif

#if LOG_LEVEL >= LOG_LEVEL_ERROR
#define LOG_ERROR(msg, ...) \
    std::printf("[ERROR]: %s:%d - " msg, __FILE__, __LINE__, __VA_ARGS__);
#else
#define LOG_ERROR(msg)
#endif

#if LOG_LEVEL >= LOG_LEVEL_WARN
#define LOG_WARN(msg, ...) \
    std::printf("[WARN]: %s:%d - " msg, __FILE__, __LINE__, __VA_ARGS__);
#else
#define LOG_WARN(msg)
#endif

#if LOG_LEVEL >= LOG_LEVEL_INFO
#define LOG_INFO(msg, ...) \
    std::printf("[INFO]: %s:%d - " msg, __FILE__, __LINE__, __VA_ARGS__);
#else
#define LOG_INFO(msg)
#endif

#if LOG_LEVEL >= LOG_LEVEL_DEBUG
#define LOG_DEBUG(msg, ...) \
    std::printf("[DEBUG]: %s:%d - " msg, __FILE__, __LINE__, __VA_ARGS__);
#else
#define LOG_DEBUG(msg)
#endif

#endif