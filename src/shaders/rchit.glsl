#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

#include "shared.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT bool is_shadowed;
hitAttributeEXT vec2 attribs;

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
    
    // // Calculate reflection direction
    // vec3 reflected = reflect(ray_dir, normal);

    // vec3 random = random_unit_vector(payload.in_uv, float(payload.depth));

    // // Apply random perturbation to the reflection direction
    // reflected += random / 2.0;
    // reflected = normalize(reflected);

    // // Make sure it's in the same hemisphere as the normal
    // if (dot(reflected, normal) < 0.0) {
    //     reflected = -reflected;
    // }

    // vec3 reflect_origin = world_pos + normal * 0.001; // Offset to avoid self-intersection

    // Update payload for next bounce
    // payload.origin = reflect_origin;
    // payload.direction = reflected;
    // payload.done = 0;


    // Store diffuse contribution as the hit value
    // payload.hit_value = base_color;
    
    // Apply reflection contribution to attenuation for the next bounce
    // payload.attenuation *= 0.33;

    float roughness = material.roughness;
    float metallic = material.metallic;

    // Generate random values
    vec2 random_values = random_float2(payload.in_uv, float(payload.depth));

    vec3 scatter_direction;
    vec3 attenuation;

    // Specular reflection with roughness
    if (rand(payload.in_uv + vec2(0.0, float(payload.depth) * 0.17)) < metallic) {
        // Specular/metallic path
        vec3 reflected = reflect(ray_dir, normal);
        
        // Add roughness-controlled perturbation
        vec3 random_perturbation = random_unit_vector(payload.in_uv, float(payload.depth));
        scatter_direction = normalize(mix(reflected, random_perturbation, roughness * roughness));
        
        // Make sure we're in the right hemisphere
        if (dot(scatter_direction, normal) <= 0.0) {
            scatter_direction = reflected;
        }
        
        // Metallic surfaces tint the reflection with their color
        attenuation = base_color;
    } else {
        // Diffuse path - use cosine weighted sampling
        scatter_direction = cosine_hemisphere_sample(random_values, normal);
        
        // Diffuse surfaces lose energy based on their color
        attenuation = base_color * (1.0 - metallic);
    }

    // Add small offset to avoid self-intersection
    vec3 scatter_origin = world_pos + normal * 0.001;

    // Update payload for next bounce
    payload.origin = scatter_origin;
    payload.direction = scatter_direction;
    payload.done = 0;

    // Set the contribution for this bounce
    payload.hit_value = vec3(0.0);  // Direct lighting would go here

    // Update attenuation for the next bounce
    payload.attenuation *= attenuation;
}
