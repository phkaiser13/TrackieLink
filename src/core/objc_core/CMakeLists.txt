# Defines the objc_core library, which contains Objective-C++ code
# for native integration with Apple platforms (macOS, iOS).

add_library(objc_core STATIC
    src/system_info.mm
    src/power_management.mm
)

# Expose the public include directory.
target_include_directories(objc_core PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# This library needs to link against native Apple frameworks.
# find_library is the standard CMake command to locate system frameworks.
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(IOKIT_FRAMEWORK IOKit REQUIRED)

target_link_libraries(objc_core PUBLIC
    ${FOUNDATION_FRAMEWORK}
    ${IOKIT_FRAMEWORK}
)
