#version 460
#extension GL_EXT_ray_tracing : require

#include "shared.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

// HDRI texture and sampler (optional, may be unbound)
layout(set = 4, binding = 0) uniform sampler2D hdri_sampler;

vec3 sky_color(vec3 direction) {
    // Realistic daytime sky with sun (HDR)
    // Values can exceed 1.0 for bright light sources
    
    // Base sky gradient - more saturated blue
    vec3 sky;
    if (direction.y > 0.0f) {
        // Horizon color (warmer): yellowish at horizon due to atmospheric scattering
        vec3 horizon = vec3(0.95f, 0.8f, 0.6f);
        // Sky color (cooler): deep blue at top
        vec3 sky_top = vec3(0.2f, 0.5f, 1.0f);
        // Smooth interpolation with power curve for more blue at top
        float t = pow(direction.y, 0.4f);
        sky = mix(horizon, sky_top, t);
    }
    else {
        // Dark blue-gray below horizon
        sky = vec3(0.1f, 0.1f, 0.15f);
    }
    
    // Add sun - approximate position at upper right
    vec3 sun_dir = normalize(vec3(0.3f, 0.8f, 0.5f));
    float sun_dot = max(0.0f, dot(direction, sun_dir));
    float sun_size = 0.025f; // Angular size of sun
    
    // HDR sun contribution: bright hot spot fading to glow
    // Warm yellow-orange sun color
    vec3 sun_color = vec3(1.0f, 0.85f, 0.5f);
    
    // Smooth falloff for sun disk with soft edges
    if (sun_dot > (1.0f - sun_size * 2.0f)) {
        // Smooth falloff from center to edge
        float sun_factor = (sun_dot - (1.0f - sun_size * 2.0f)) / (sun_size * 2.0f);
        sun_factor = smoothstep(0.0f, 1.0f, sun_factor);
        
        // Direct sun disk - very bright (HDR)
        sky += sun_color * 150.0f * sun_factor;
    }
    
    // Sun glow/atmosphere halo - softer and larger
    if (sun_dot > (1.0f - sun_size * 5.0f)) {
        float glow_factor = pow(sun_dot, 8.0f);
        sky += sun_color * 10.0f * glow_factor;
    }
    
    return sky;
}

/// Sample HDRI texture
vec3 sample_hdri(vec3 direction) {
    // Convert to UV coordinates (no rotation for now, kept simple)
    vec2 uv = world_to_hdri_uv(direction);
    
    // Sample HDRI texture
    vec3 hdri_color = texture(hdri_sampler, uv).rgb;
    
    return hdri_color;
}


void main() {
    // Get the direction of the ray
    vec3 direction = gl_WorldRayDirectionEXT;
    
    // Use HDRI if enabled, otherwise use procedural sky
    vec3 sky;
    if (payload.hdri_enabled != 0u) {
        sky = sample_hdri(direction);
    } else {
        sky = sky_color(direction);
    }
    
    payload.attenuation = sky;
    payload.done = 1;
}