#version 460
#extension GL_EXT_ray_tracing : require

#include "shared.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

vec3 sky_color(vec3 direction) {
    // Realistic daytime sky with sun (HDR)
    // Values can exceed 1.0 for bright light sources
    
    // Base sky gradient
    vec3 sky;
    if (direction.y > 0.0f) {
        // Interpolate from light gray (0.6, 0.6, 0.6) at horizon to bright blue (0.5, 0.7, 1.0) at top
        sky = mix(vec3(0.6f, 0.6f, 0.6f), vec3(0.5f, 0.7f, 1.0f), direction.y);
    }
    else {
        // Dark gray below horizon
        sky = vec3(0.2f);
    }
    
    // Add sun - approximate position at upper right
    vec3 sun_dir = normalize(vec3(0.3f, 0.8f, 0.5f));
    float sun_dot = max(0.0f, dot(direction, sun_dir));
    float sun_size = 0.02f; // Angular size of sun
    
    // HDR sun contribution: bright hot spot fading to glow
    vec3 sun_color = vec3(1.0f, 0.95f, 0.8f);
    if (sun_dot > (1.0f - sun_size)) {
        // Direct sun disk - very bright (HDR)
        sky += sun_color * 50.0f * (1.0f - (1.0f - sun_dot) / sun_size);
    }
    else if (sun_dot > (1.0f - sun_size * 3.0f)) {
        // Sun glow halo - medium bright (HDR)
        sky += sun_color * 5.0f * pow(sun_dot, 16.0f);
    }
    
    return sky;
}


void main() {
    // // Get the direction of the ray
    vec3 direction = gl_WorldRayDirectionEXT;
    
    // // Sample the skybox using the ray direction
    // hit_value = texture(skybox, direction).rgb;

    // Light sky color
    vec3 sky = sky_color(direction);
    payload.attenuation = sky;
    // payload.attenuation = vec3(1.0, 1.0, 1.0);
    payload.done = 1;
}