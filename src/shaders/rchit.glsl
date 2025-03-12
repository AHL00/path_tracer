#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

#include "shared.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT bool is_shadowed;
hitAttributeEXT vec2 attribs;

const int MAX_BOUNCES = 5; 

layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_as;

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

layout(binding = 0, set = 3) uniform sampler2D textures[];

// PCG (Permuted Congruential Generator) - better randomness than the basic sin hash
float rand(vec2 co) {
    uint state = uint(co.x * 12345.0) + uint(co.y * 67890.0);
    state = state * 747796405u + 2891336453u;
    state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    state = (state >> 22u) ^ state;
    return float(state) / 4294967295.0;
}

// Hash function for generating random direction vectors
vec3 random_unit_vector(vec2 seed, float depth) {
    // Create different seeds for each component
    float x = rand(seed + vec2(depth, 0.13));
    float y = rand(seed + vec2(0.71, depth));
    float z = rand(seed + vec2(depth * 0.39, depth * 0.57));
    
    // Map from [0,1] to [-1,1]
    vec3 v = 2.0 * vec3(x, y, z) - 1.0;
    return normalize(v);
}

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
    vec2 uv = 
        v0.uv * barycentrics.x +
        v1.uv * barycentrics.y +
        v2.uv * barycentrics.z;
    vec3 position =
        v0.position * barycentrics.x +
        v1.position * barycentrics.y +
        v2.position * barycentrics.z;

    Material material = materials[offsets.material_offset];
    
    vec3 base_color;
    if (material.has_base_texture) {
        vec4 texture_color = texture(textures[nonuniformEXT(material.base_texture_indice)], uv);
        base_color = texture_color.xyz;
    } else {
        base_color = material.base_color.xyz;
    }
    
    // Get incoming ray direction and world position
    vec3 ray_dir = gl_WorldRayDirectionEXT;
    vec3 world_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    
    // Calculate reflection direction
    vec3 reflected = reflect(ray_dir, normal);

    vec3 random = random_unit_vector(payload.in_uv, float(payload.depth));

    // Apply random perturbation to the reflection direction
    reflected += random / 2.0 * material.roughness;
    reflected = normalize(reflected);

    // Make sure it's in the same hemisphere as the normal
    if (dot(reflected, normal) < 0.0) {
        reflected = -reflected;
    }

    vec3 reflect_origin = world_pos + normal * 0.001; // Offset to avoid self-intersection

    // Update payload for next bounce
    payload.origin = reflect_origin;
    payload.direction = reflected;
    payload.done = 0;


    // Store diffuse contribution as the hit value
    payload.hit_value = base_color;
    
    // Apply reflection contribution to attenuation for the next bounce
    payload.attenuation *= 0.5;
}
