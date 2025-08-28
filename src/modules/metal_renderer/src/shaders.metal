#include <metal_stdlib>

using namespace metal;

// This file contains the source code for our GPU shaders, written in the
// Metal Shading Language (MSL), which is based on C++14.

/**
 * @struct Vertex
 * @brief Defines the data layout for a single vertex that will be passed
 *        to the vertex shader. The `[[position]]` attribute is a special
 *        qualifier that tells Metal this member is the vertex's position
 *        in clip space.
 */
struct Vertex {
    // The position is a float4 (x, y, z, w) for 3D homogeneous coordinates.
    float4 position [[position]];
};

/**
 * @brief The vertex shader for our rendering pipeline.
 *
 * A vertex shader's primary job is to process each vertex and determine its
 * final position on the screen. This is a simple "pass-through" shader.
 *
 * @param vertex_array A pointer to the array of vertices in device memory. `[[buffer(0)]]`
 *                     maps this to the first buffer set by the CPU.
 * @param vertex_id The unique ID of the current vertex being processed. `[[vertex_id]]`
 *                  is a system-generated value.
 * @return The processed vertex.
 */
vertex Vertex vertex_main(const device Vertex* vertex_array [[buffer(0)]],
                         uint vertex_id [[vertex_id]])
{
    // We simply look up the vertex in the array and return it without modification.
    return vertex_array[vertex_id];
}

/**
 * @brief The fragment shader for our rendering pipeline.
 *
 * A fragment shader (or pixel shader) runs for every pixel that is part of
 * a primitive (like our triangle) and determines its final color.
 *
 * @return A float4 representing the color (Red, Green, Blue, Alpha).
 */
fragment float4 fragment_main()
{
    // This shader ignores all input and simply returns a constant color.
    // In this case, a nice purple.
    return float4(0.5, 0.2, 0.8, 1.0);
}
