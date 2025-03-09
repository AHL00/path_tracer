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

layout(binding = 3, set = 2) readonly buffer index_buffer {
    uint indices[];
};

void main() {
    Offsets offsets = offsets_array[gl_InstanceCustomIndexEXT];

    // Get indices for the triangle
    uint index_start = offsets.index_offset;
    uint i0 = indices[index_start + gl_PrimitiveID * 3 + 0];
    uint i1 = indices[index_start + gl_PrimitiveID * 3 + 1];
    uint i2 = indices[index_start + gl_PrimitiveID * 3 + 2];
    
    // Get vertices using the indices
    Vertex v0 = vertices[offsets.vertex_offset + i0];
    Vertex v1 = vertices[offsets.vertex_offset + i1];
    Vertex v2 = vertices[offsets.vertex_offset + i2];
    
    // Interpolate using barycentric coordinates
    vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 normal = normalize(
        v0.normal * barycentrics.x +
        v1.normal * barycentrics.y +
        v2.normal * barycentrics.z
    );
    
    hit_value = normal;
}
