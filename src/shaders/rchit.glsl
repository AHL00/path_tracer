#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 hit_value;
hitAttributeEXT vec2 attribs;

#include "shared.glsl"

layout(binding = 0, set = 2) readonly buffer offsets_buffer {
    Offsets offsets_array[];
};

layout(binding = 1, set = 2) readonly buffer vertex_buffer {
    Vertex vertices[];
};

layout(binding = 2, set = 2) readonly buffer material_buffer {
    Material materials[];
};

void main() {
    vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    hit_value = barycentrics;

    Offsets offsets = offsets_array[gl_InstanceCustomIndexEXT];
}
