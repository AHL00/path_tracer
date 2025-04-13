#version 460
#extension GL_EXT_ray_tracing : require

#include "shared.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_as;
layout(set = 1, binding = 0, rgba32f) uniform image2D image;

layout(location = 0) rayPayloadEXT RayPayload payload;

layout(push_constant) uniform PushConstants {
    RendererUniforms uniforms;
    Camera camera;    
} push_constants;

const uint MAX_DEPTH = 16;
const uint SAMPLES_PER_PIXEL = 1;

void main() {
    const vec2 pixel_center = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 in_uv = pixel_center / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = in_uv * 2.0 - 1.0;

    vec4 origin = push_constants.camera.view_inverse * vec4(0, 0, 0, 1);
    vec4 target = push_constants.camera.proj_inverse * vec4(d.x, d.y, 1, 1);
    vec4 direction_vec4 = push_constants.camera.view_inverse * vec4(normalize(target.xyz), 0);

    vec3 direction = normalize(direction_vec4.xyz);

    uint ray_flags = gl_RayFlagsOpaqueEXT;
    float t_min = 0.001;
    float t_max = 10000.0;
    
    vec3 accumulated_value = vec3(0.0, 0.0, 0.0);

    for (uint i = 0; i < SAMPLES_PER_PIXEL; i++)
    {
        payload.depth = 0;
        payload.hit_value = vec3(0.0, 0.0, 0.0);
        payload.attenuation = vec3(1.0, 1.0, 1.0);
        payload.done = 1;
        payload.origin = origin.xyz;

        // Randomize the ray direction
        // vec3 random = random_unit_vector(in_uv, i);
        // vec3 randomised_direction = normalize(direction + random * 0.1);

        // payload.direction = randomised_direction; 
        payload.direction = direction.xyz;
        payload.in_uv = in_uv;

        vec3 hit_value = vec3(0.0, 0.0, 0.0);
        
        for(;;)
        {
            traceRayEXT(
                top_level_as,  // acceleration structure
                ray_flags,     // rayFlags
                0xFF,          // cullMask
                0,             // sbtRecordOffset
                0,             // sbtRecordStride
                0,             // missIndex
                origin.xyz,    // ray origin
                t_min,         // ray min range
                direction.xyz, // ray direction
                t_max,         // ray max range
                0              // payload (location 0)
            );      
            hit_value += payload.hit_value * payload.attenuation;
            // hit_value = payload.hit_value;

            payload.depth++;
            if(payload.done == 1 || payload.depth >= MAX_DEPTH)
                break;

            origin.xyz = payload.origin;
            direction.xyz = payload.direction;
            payload.done = 1;  // Will stop if a reflective material isn't hit
        }

        // Accumulate the hit value
        accumulated_value += hit_value / SAMPLES_PER_PIXEL;
        // accumulated_value = randomised_direction;
    }

    // uint u32_max = 0xFFFFFFFF;
    // uint uniform_seed_normalized = push_constants.uniforms.seed % u32_max;
    // float seed_normalized = float(uniform_seed_normalized) / float(u32_max);
    // accumulated_value = vec3(seed_normalized, seed_normalized, seed_normalized);

    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(accumulated_value, 1.0));
    // imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(uniforms.seed, 1.0));
    // imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(payload.attenuation, 1.0));
    // imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(payload.depth / 16.0));
}
