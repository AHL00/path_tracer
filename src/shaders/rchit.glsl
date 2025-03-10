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

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
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

    // // Add some roughness-based perturbation
    // if (material.roughness > 0.0) {
    //     // Simple roughness implementation - perturbs reflection based on roughness
    //     float roughness_factor = material.roughness * material.roughness; // Square for physically-based response
    //     vec3 random_dir = normalize(vec3(
    //         rand(world_pos.xy + gl_PrimitiveID) * 2.0 - 1.0,
    //         rand(world_pos.yz + gl_PrimitiveID) * 2.0 - 1.0,
    //         rand(world_pos.zx + gl_PrimitiveID) * 2.0 - 1.0
    //     ));
    //     reflected = normalize(mix(reflected, random_dir, roughness_factor));
    // }

    // // Ensure the reflected direction is on the correct hemisphere
    // if (dot(reflected, normal) < 0.0) {
    //     reflected = reflected - 2.0 * dot(reflected, normal) * normal;
    // }
    

    // Calculate Fresnel factor for metals and non-metals
    // For metals (metallic=1.0): Fresnel reflects the base_color
    // For non-metals (metallic=0.0): Fresnel uses standard dielectric F0=0.04
    vec3 F0 = mix(vec3(0.04), base_color, material.metallic);
    vec3 fresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(normalize(-ray_dir), normal), 0.0), 5.0);
    
    // Avoid self-intersection by offsetting the origin slightly
    vec3 reflect_origin = world_pos + normal * 0.001;

    // Update payload for next bounce
    payload.origin = reflect_origin;
    payload.direction = reflected;
    payload.done = 0;

    // Materials with higher metallic values reflect more of their base color
    vec3 metallic_reflection = mix(vec3(1.0), base_color, material.metallic);
    
    // Update attenuation based on material properties
    payload.attenuation *= metallic_reflection;
    
    // Direct lighting contribution
    float diffuse_factor = max(dot(normal, -ray_dir), 0.0);
    vec3 ambient = base_color * 0.02 + vec3(0.05); // Simple ambient term
    
    // Non-metallic surfaces have diffuse contribution, metallic surfaces don't
    vec3 diffuse = mix(base_color * diffuse_factor, vec3(0.0), material.metallic);
    
    // Combine lighting for this bounce
    payload.hit_value = (ambient + diffuse) * 15.0;
}
