#ifndef METAL_RENDERER_RENDERER_H
#define METAL_RENDERER_RENDERER_H

#include <stdint.h>

// This header defines the C-style API for controlling the Metal renderer.
// This allows the C++ core of the application to create and manage a Metal
// rendering surface without needing to know any Objective-C.

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief An opaque handle to the internal MetalRenderer Objective-C++ object.
 * The C++ code will only ever interact with this pointer.
 */
typedef struct renderer_t renderer_t;

/**
 * @brief Creates and initializes a new Metal renderer and attaches it to a window's view.
 * @param view_handle A platform-specific view handle. On macOS, this should be a `void*`
 *                    pointing to an `NSView` instance.
 * @return A handle to the renderer, or NULL on failure.
 */
renderer_t* renderer_create(void* view_handle);

/**
 * @brief Destroys the Metal renderer and frees all associated Metal and Objective-C resources.
 * @param renderer The renderer handle to destroy.
 */
void renderer_destroy(renderer_t* renderer);

/**
 * @brief Draws a single frame using the configured Metal pipeline.
 * @param renderer The renderer handle.
 */
void renderer_draw_frame(renderer_t* renderer);


#ifdef __cplusplus
}
#endif

#endif // METAL_RENDERER_RENDERER_H
