# Defines the metal_renderer library.
# This module is only configured and built on Apple platforms.

# 1. Compile the .metal shader source into a default.metallib file.
#    On Apple platforms, shaders are compiled into a library, similar to code.
#    This command uses the `xcrun` tool to invoke the Metal shader compiler.
#    It's a two-step process: compile to an intermediate representation (.air),
#    then create a library (.metallib) from the .air file.
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
    COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.metal -o ${CMAKE_CURRENT_BINARY_DIR}/air.air
    COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/air.air -o ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.metal
    COMMENT "Compiling Metal Shaders into default.metallib"
)

# We create a custom target to ensure the custom command is always executed.
add_custom_target(MetalShaders DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

# 2. Define the library from the Objective-C++ source files.
add_library(metal_renderer STATIC
    src/renderer.mm
)

# Add a dependency to ensure shaders are compiled before the renderer library,
# as the renderer code will expect the .metallib to exist.
add_dependencies(metal_renderer MetalShaders)

# 3. Expose the public include directory.
target_include_directories(metal_renderer PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# 4. Find and link against required Apple frameworks for rendering.
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED) # Needed for CAMetalLayer
find_library(APPKIT_FRAMEWORK AppKit REQUIRED)       # Needed for NSWindow/NSView integration

target_link_libraries(metal_renderer PUBLIC
    ${METAL_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
    ${APPKIT_FRAMEWORK}
)
