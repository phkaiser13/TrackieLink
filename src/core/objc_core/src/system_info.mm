#include "objc_core/system_info.h"

#import <Foundation/Foundation.h>
#include <sys/sysctl.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// This file uses Objective-C++ to implement the C-style API defined in the header.
// This is a common pattern for bridging a C++ application with native Apple APIs.

/**
 * @brief Helper function to convert an NSString to a C-string that the caller must free.
 */
static const char* copy_nsstring_to_cstring(NSString* ns_string) {
    if (ns_string == nil) {
        return NULL;
    }
    const char* utf8_string = [ns_string UTF8String];
    if (utf8_string == NULL) {
        return NULL;
    }
    // strdup allocates new memory for the C-string.
    return strdup(utf8_string);
}

const char* objc_get_os_name() {
    // In modern Objective-C, it's better to check for the platform at compile time.
    // We return a hardcoded value as NSProcessInfo.operatingSystemName is not very descriptive.
    #if TARGET_OS_IPHONE
        return strdup("iOS");
    #elif TARGET_OS_MAC
        return strdup("macOS");
    #else
        return strdup("Unknown Apple OS");
    #endif
}

const char* objc_get_os_version() {
    // NSProcessInfo is the standard way to get information about the current process
    // and the operating system environment.
    NSOperatingSystemVersion version = [[NSProcessInfo processInfo] operatingSystemVersion];

    NSString* versionString = [NSString stringWithFormat:@"%ld.%ld.%ld",
                               (long)version.majorVersion,
                               (long)version.minorVersion,
                               (long)version.patchVersion];

    return copy_nsstring_to_cstring(versionString);
}

int objc_is_running_under_rosetta() {
    // This uses the sysctl system call to check for the 'proc_translated' property,
    // which indicates if the process is running under Rosetta 2 translation.
    int ret = 0;
    size_t size = sizeof(ret);
    if (sysctlbyname("sysctl.proc_translated", &ret, &size, NULL, 0) == -1) {
        // If the sysctl call fails, we check the error number.
        // ENOENT means the 'proc_translated' key doesn't exist, which is true
        // for Intel-based Macs. On Apple Silicon, the key always exists.
        if (errno == ENOENT) {
            return 0; // Not an Apple Silicon Mac, so definitely not running under Rosetta.
        }
        return -1; // Some other unexpected error occurred.
    }
    // If the call succeeds, 'ret' will be 1 if translated, 0 if not.
    return ret;
}
