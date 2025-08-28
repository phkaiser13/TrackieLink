#ifndef OBJC_CORE_SYSTEM_INFO_H
#define OBJC_CORE_SYSTEM_INFO_H

// This header provides a C-style API to access macOS-specific system information.
// By using a C API, we create a stable ABI that can be easily called from
// the main C++ application core without exposing any Objective-C types or syntax.

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Gets the name of the operating system (e.g., "macOS").
 * @return A C-string containing the OS name. The caller is responsible for freeing this string.
 *         Returns NULL on error.
 */
const char* objc_get_os_name();

/**
 * @brief Gets the version of the operating system as a string (e.g., "14.5").
 * @return A C-string with the OS version. The caller is responsible for freeing this string.
 *         Returns NULL on error.
 */
const char* objc_get_os_version();

/**
 * @brief Checks if the current process is running under the Rosetta 2 translation layer.
 * This is useful for determining if an x86_64 binary is running on Apple Silicon.
 * @return 1 if running under Rosetta, 0 if running natively, -1 on error.
 */
int objc_is_running_under_rosetta();


#ifdef __cplusplus
}
#endif

#endif // OBJC_CORE_SYSTEM_INFO_H
