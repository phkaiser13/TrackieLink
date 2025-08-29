#include <metal_stdlib>

using namespace metal;

/**
 * @brief A compute kernel that inverts the colors of an image.
 *
 * @param inTexture The input texture to read from. `[[texture(0)]]`
 * @param outTexture The output texture to write to. `[[texture(1)]]`
 * @param gid The grid-based thread position, identifies the pixel to process. `[[thread_position_in_grid]]`
 */
kernel void invert_colors(texture2d<half, access::read> inTexture [[texture(0)]],
                          texture2d<half, access::write> outTexture [[texture(1)]],
                          uint2 gid [[thread_position_in_grid]])
{
    // Read the color of the pixel from the input texture.
    half4 color = inTexture.read(gid);

    // Invert the RGB components, leaving alpha unchanged.
    half3 inverted_rgb = 1.0 - color.rgb;

    // Write the new color to the output texture.
    outTexture.write(half4(inverted_rgb, color.a), gid);
}

kernel void box_blur(texture2d<half, access::read> inTexture [[texture(0)]],
                     texture2d<half, access::write> outTexture [[texture(1)]],
                     uint2 gid [[thread_position_in_grid]])
{
    float3 color = float3(0.0, 0.0, 0.0);
    int count = 0;

    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            uint2 sample_pos = uint2(gid.x + i, gid.y + j);
            if (sample_pos.x < inTexture.get_width() && sample_pos.y < inTexture.get_height()) {
                color += inTexture.read(sample_pos).rgb;
                count++;
            }
        }
    }

    color /= count;
    outTexture.write(half4(half3(color), 1.0), gid);
}

kernel void sharpen(texture2d<half, access::read> inTexture [[texture(0)]],
                      texture2d<half, access::write> outTexture [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    half4 center = inTexture.read(gid);
    half4 top = inTexture.read(uint2(gid.x, gid.y - 1));
    half4 bottom = inTexture.read(uint2(gid.x, gid.y + 1));
    half4 left = inTexture.read(uint2(gid.x - 1, gid.y));
    half4 right = inTexture.read(uint2(gid.x + 1, gid.y));

    half4 sharpened = center * 5.0 - (top + bottom + left + right);
    outTexture.write(sharpened, gid);
}
