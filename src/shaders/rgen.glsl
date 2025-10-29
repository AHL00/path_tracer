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

void main() {
    // const vec2 pixel_center = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    // const vec2 in_uv = pixel_center / vec2(gl_LaunchSizeEXT.xy);

    // Flip the Y coordinate
    const vec2 pixel_center = vec2(gl_LaunchIDEXT.x, gl_LaunchSizeEXT.y - gl_LaunchIDEXT.y - 1) + vec2(0.5);
    const vec2 in_uv = pixel_center / vec2(gl_LaunchSizeEXT.xy);
    
    vec2 d = in_uv * 2.0 - 1.0;

    vec4 origin = push_constants.camera.view_inverse * vec4(0, 0, 0, 1);
    vec4 target = push_constants.camera.proj_inverse * vec4(d.x, d.y, 1, 1);
    vec4 direction_vec4 = push_constants.camera.view_inverse * vec4(normalize(target.xyz), 0);

    vec3 direction = normalize(direction_vec4.xyz);

    uint ray_flags = gl_RayFlagsOpaqueEXT;
    float t_min = 0.001;
    float t_max = 10000.0;


    payload.depth = 0;
    payload.hit_value = vec3(0.0, 0.0, 0.0);
    payload.attenuation = vec3(1.0, 1.0, 1.0);
    payload.done = 1;
    payload.origin = origin.xyz;
    payload.in_uv = in_uv;
    payload.hdri_enabled = push_constants.uniforms.hdri_enabled;

    payload.direction = direction;

    // Apply temporal anti-aliasing instead of per-pixel jitter
    // This helps with distance blurring by being more stable
    // The jitter is now encoded in the random sampling at hit points
    // rather than perturbing rays at the source
    if (push_constants.uniforms.accumulated_count > 0) {
        // Use a proper cosine-weighted hemisphere sample for anti-aliasing
        vec3 jitter = cosine_hemisphere_sample(
            in_uv, 
            0,
            push_constants.uniforms.accumulated_count,
            push_constants.uniforms.seed, 
            direction
        );
        // Apply smaller, smoother jitter that still provides anti-aliasing
        payload.direction = normalize(direction + jitter * 0.0001);
    }

    vec3 attenuation = vec3(1.0);
    
    for(;;)
    {
        traceRayEXT(
            top_level_as,      // acceleration structure
            ray_flags,         // rayFlags
            0xFF,              // cullMask
            0,                 // sbtRecordOffset
            0,                 // sbtRecordStride
            0,                 // missIndex
            payload.origin,    // Use payload's current origin
            t_min,             // ray min range
            payload.direction, // Use payload's current direction
            t_max,             // ray max range
            0                  // payload (location 0)
        );      
        
        attenuation *= payload.attenuation;
        

        payload.depth++;
        if(payload.done == 1 || payload.depth >= MAX_DEPTH)
            break;

        origin.xyz = payload.origin;
        direction.xyz = payload.direction;
        payload.done = 1;  // Will stop if a reflective material isn't hit
    }


    // Rand visualiser
    uint u32_max = 0xFFFFFFFF;
    // accumulated_value = vec3(rand(in_uv, 0, push_constants.uniforms.seed));

    // vec3 normal = vec3(0.0, 1.0, 0.0);
    // accumulated_value = 
    // cosine_hemisphere_sample(
    //         in_uv, 
    //         float(0), 
    //         push_constants.uniforms.seed, 
    //         normal
    //     );

    float accumulated_count = float(push_constants.uniforms.accumulated_count);

    vec3 current_accumulated_value = imageLoad(image, ivec2(gl_LaunchIDEXT.xy)).xyz;
    
    // Debug mode: visualize bounce count
    if (push_constants.uniforms.debug_mode == 4u) {
        // Normalize depth to [0, 1] range with max at 16 bounces
        float normalized_depth = float(payload.depth) / 16.0;
        attenuation = vec3(normalized_depth);
    }
    
    // Proper running average: (old_sum + new_sample) / (n + 1)
    // where old_sum = current_accumulated_value (already averaged n times)
    // So: (current_accumulated_value * n + attenuation) / (n + 1)
    vec3 new_average_value = (current_accumulated_value * accumulated_count + attenuation) / (accumulated_count + 1.0);

    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(new_average_value, 1.0));


    // imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(cosine_hemisphere_sample(
    //     in_uv, 
    //     0, 
    //     push_constants.uniforms.accumulated_count,
    //     push_constants.uniforms.seed, 
    //     vec3(0.5, 0.5, 0.5)
    // ), 1.0));

    // imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(vec3(float(push_constants.uniforms.seed) / float(u32_max)), 1.0));
    // imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(rand(in_uv, 0, seed_normalized), 1.0));
    // imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(payload.attenuation, 1.0));
    // imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(payload.depth / 16.0));
}
