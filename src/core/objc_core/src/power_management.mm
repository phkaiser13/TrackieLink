#include "objc_core/power_management.h"

#import <Foundation/Foundation.h>
#import <IOKit/ps/IOPowerSources.h>
#import <IOKit/ps/IOPSKeys.h>

// This file demonstrates interacting with the lower-level IOKit framework
// to get detailed power source information. This is a common requirement for
// applications that need to adjust their behavior based on power state.

double objc_get_battery_level() {
    // IOPSCopyPowerSourcesInfo() returns a blob of power source information.
    // IOPSCopyPowerSourcesList() returns an array of power sources.
    // We get a description of the list of sources.
    CFTypeRef powerSourcesInfo = IOPSGetPowerSourceDescription(NULL, IOPSCopyPowerSourcesList());
    if (powerSourcesInfo == NULL) {
        return -1.0;
    }

    // The returned object is a CFArray. We must manage its memory with CFRelease.
    CFArrayRef powerSources = (CFArrayRef)powerSourcesInfo;
    if (CFArrayGetCount(powerSources) == 0) {
        CFRelease(powerSourcesInfo);
        return -1.0; // No power sources found (e.g., on a Mac Pro or iMac).
    }

    // We assume the first power source is the primary one (the internal battery).
    CFDictionaryRef psDescription = IOPSGetPowerSourceDescription(powerSources, CFArrayGetValueAtIndex(powerSources, 0));
    if (psDescription == NULL) {
        CFRelease(powerSourcesInfo);
        return -1.0;
    }

    double currentCapacity = 0;
    double maxCapacity = 0;

    // Extract the numeric values for current and max capacity from the dictionary.
    CFNumberRef currentCapNum = (CFNumberRef)CFDictionaryGetValue(psDescription, CFSTR(kIOPSCurrentCapacityKey));
    CFNumberRef maxCapNum = (CFNumberRef)CFDictionaryGetValue(psDescription, CFSTR(kIOPSMaxCapacityKey));

    if (currentCapNum && maxCapNum) {
        CFNumberGetValue(currentCapNum, kCFNumberDoubleType, &currentCapacity);
        CFNumberGetValue(maxCapNum, kCFNumberDoubleType, &maxCapacity);
    }

    // It's crucial to release Core Foundation objects that you created or copied.
    CFRelease(powerSourcesInfo);

    if (maxCapacity > 0) {
        return (currentCapacity / maxCapacity) * 100.0;
    }

    return -1.0; // No battery found or max capacity is zero.
}

int objc_is_on_ac_power() {
    CFTypeRef powerSourcesInfo = IOPSGetPowerSourceDescription(NULL, IOPSCopyPowerSourcesList());
    if (powerSourcesInfo == NULL) {
        return -1;
    }

    CFArrayRef powerSources = (CFArrayRef)powerSourcesInfo;
    if (CFArrayGetCount(powerSources) == 0) {
        CFRelease(powerSourcesInfo);
        return -1;
    }

    CFDictionaryRef psDescription = IOPSGetPowerSourceDescription(powerSources, CFArrayGetValueAtIndex(powerSources, 0));
    if (psDescription == NULL) {
        CFRelease(powerSourcesInfo);
        return -1;
    }

    // The power source state is a string (e.g., "AC Power", "Battery Power").
    CFStringRef powerSourceState = (CFStringRef)CFDictionaryGetValue(psDescription, CFSTR(kIOPSPowerSourceStateKey));
    if (powerSourceState == NULL) {
        CFRelease(powerSourcesInfo);
        return -1;
    }

    // Compare the current state string with the constant for AC Power.
    bool isOnAC = (CFStringCompare(powerSourceState, CFSTR(kIOPSACPowerValue), 0) == kCFCompareEqualTo);

    CFRelease(powerSourcesInfo);

    return isOnAC ? 1 : 0;
}
