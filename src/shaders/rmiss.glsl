#version 460
#extension GL_EXT_ray_tracing : require

#include "shared.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

vec3 sky_color(vec3 direction) {
    if (direction.y > 0.0f) {
        return mix(vec3(0.8f, 0.8f, 0.8f), vec3(0.25f, 0.5f, 1.0f), direction.y);
    }
    else {
        return vec3(0.03f);
    }
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