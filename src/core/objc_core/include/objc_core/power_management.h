#ifndef OBJC_CORE_POWER_MANAGEMENT_H
#define OBJC_CORE_POWER_MANAGEMENT_H

// This header provides a C-style API to access macOS-specific
// power management information, such as battery status.

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Gets the current battery level as a percentage.
 *
 * This function queries the system's power sources to find the current
 * charge capacity relative to the maximum capacity.
 *
 * @return The battery level as a percentage (e.g., 95.5 for 95.5%).
 *         Returns -1.0 if the information is not available (e.g., on a desktop Mac)
 *         or if an error occurs.
 */
double objc_get_battery_level();

/**
 * @brief Checks if the device is currently connected to an AC power source.
 * @return 1 if connected to AC power, 0 if running on battery, -1 on error.
 */
int objc_is_on_ac_power();

#ifdef __cplusplus
}
#endif

#endif // OBJC_CORE_POWER_MANAGEMENT_H
