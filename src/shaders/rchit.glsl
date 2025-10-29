

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

#include "shared.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT bool is_shadowed;
hitAttributeEXT vec2 attribs;

layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_as;

layout(push_constant) uniform RchitPushConstants { RendererUniforms uniforms; }
push_constants;

layout(binding = 0, set = 2) readonly buffer offsets_buffer {
  Offsets offsets_array[];
};

layout(binding = 1, set = 2) readonly buffer vertex_buffer {
  Vertex vertices[];
};

layout(binding = 2, set = 2) readonly buffer material_buffer {
  Material materials[];
};

layout(binding = 3, set = 2) readonly buffer index_buffer { uint indices[]; };

layout(binding = 0, set = 3) uniform sampler2D textures[];

// Helper functions for texture flag checks
bool has_base_color_texture(uint flags) { return (flags & (1u << 0u)) != 0u; }

bool has_metallic_roughness_texture(uint flags) {
  return (flags & (1u << 1u)) != 0u;
}

bool has_normal_texture(uint flags) { return (flags & (1u << 2u)) != 0u; }

bool has_emissive_texture(uint flags) { return (flags & (1u << 3u)) != 0u; }

bool has_ao_texture(uint flags) { return (flags & (1u << 4u)) != 0u; }

void main() {
  // Retrieve geometry data
  Offsets offsets = offsets_array[gl_InstanceCustomIndexEXT];

  // Get triangle indices
  uint index_start = offsets.index_offset;
  uint i0 = indices[index_start + gl_PrimitiveID * 3 + 0];
  uint i1 = indices[index_start + gl_PrimitiveID * 3 + 1];
  uint i2 = indices[index_start + gl_PrimitiveID * 3 + 2];

  // Fetch vertices
  Vertex v0 = vertices[offsets.vertex_offset + i0];
  Vertex v1 = vertices[offsets.vertex_offset + i1];
  Vertex v2 = vertices[offsets.vertex_offset + i2];

  // Interpolate using barycentric coordinates
  vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
  vec3 normal =
      normalize(v0.normal * barycentrics.x + v1.normal * barycentrics.y +
                v2.normal * barycentrics.z);
  vec2 uv =
      v0.uv * barycentrics.x + v1.uv * barycentrics.y + v2.uv * barycentrics.z;

  // Calculate hit position in world space
  vec3 world_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
  vec3 ray_dir = normalize(gl_WorldRayDirectionEXT);

  // Get material
  Material material = materials[offsets.material_offset];

  // Sample base color/albedo from texture or use constant
  vec3 albedo = material.base_color.xyz;
  if (has_base_color_texture(material.texture_flags)) {
    vec4 texture_color =
        texture(textures[nonuniformEXT(material.base_color_texture_index)], uv);
    albedo = texture_color.xyz;
  }

  // Sample metallic and roughness
  float metallic;
  float roughness;

  if (has_metallic_roughness_texture(material.texture_flags)) {
    vec4 metal_rough = texture(
        textures[nonuniformEXT(material.metallic_roughness_texture_index)], uv);
    metallic = metal_rough.b;  // Blue channel = metallic
    roughness = metal_rough.g; // Green channel = roughness
  } else {
    // No texture, use base values
    metallic = material.metallic;
    roughness = material.roughness;
  }

  // Clamp roughness to avoid fireflies
  roughness = clamp(roughness, 0.05, 0.99);

  // Debug mode visualization
  if (push_constants.uniforms.debug_mode == 1u) {
    // Metallic visualization: grayscale (black=0, white=1)
    payload.attenuation = vec3(metallic);
    payload.done = 1;
    return;
  } else if (push_constants.uniforms.debug_mode == 2u) {
    // Roughness visualization: grayscale (black=smooth, white=rough)
    payload.attenuation = vec3(roughness);
    payload.done = 1;
    return;
  }

  // Sample normal map if available
  vec3 surface_normal = normal;
  if (has_normal_texture(material.texture_flags)) {
    vec3 normal_sample =
        texture(textures[nonuniformEXT(material.normal_texture_index)], uv).xyz;
    // Unpack normal map (assuming standard DX-style encoding)
    normal_sample = normalize(normal_sample * 2.0 - 1.0);
    // Transform from tangent space to world space
    mat3 basis = create_basis(normal);
    surface_normal = normalize(basis * normal_sample);
  }

  // Sample emissive texture and compute emissive contribution
  vec3 emissive = material.emissive_color.rgb * material.emissive_strength;
  if (has_emissive_texture(material.texture_flags)) {
    vec3 emissive_sample = 
        texture(textures[nonuniformEXT(material.emissive_texture_index)], uv).rgb;
    emissive *= emissive_sample;
  }

  // Add emissive light directly to the accumulated color
  // This is the light emitted by this surface
  if (length(emissive) > 0.0) {
    payload.attenuation *= emissive;
    payload.done = 1;  // Terminate ray path at emissive surface
    return;
  }

  // Determine scattering behavior based on material type
  vec3 scatter_direction;
  vec3 attenuation;
  bool should_scatter = true;
  
  if (material.material_type == 1u) {
    // Metallic material
    vec3 reflected = reflect(ray_dir, surface_normal);

    // Add roughness-based perturbation
    vec3 random_perturb = cosine_hemisphere_sample(
        payload.in_uv, payload.depth, push_constants.uniforms.accumulated_count,
        push_constants.uniforms.seed, surface_normal);

    scatter_direction =
        normalize(mix(reflected, random_perturb, roughness * roughness));

    if (dot(scatter_direction, surface_normal) <= 0.0) {
      scatter_direction = reflected;
    }

    // Metallic surfaces lose less energy - use higher attenuation
    attenuation =
        mix(vec3(1.0), albedo, metallic); // Blend white reflection with color

  } else if (material.material_type == 2u) {
    // Glass/Dielectric material
    float eta = 1.0 / material.ior;

    // Check if ray is entering or exiting
    float cos_theta = -dot(ray_dir, surface_normal);
    vec3 surface_normal_adjusted = surface_normal;

    if (cos_theta < 0.0) {
      // Ray is exiting the material
      cos_theta = -cos_theta;
      surface_normal_adjusted = -surface_normal;
      eta = material.ior;
    }

    // Compute transmission direction
    float discriminant = 1.0 - eta * eta * (1.0 - cos_theta * cos_theta);

    if (discriminant >= 0.0) {
      // Refraction is possible - compute both refracted and reflected rays
      vec3 refracted = eta * ray_dir + (eta * cos_theta - sqrt(discriminant)) *
                                           surface_normal_adjusted;
      vec3 reflected = reflect(ray_dir, surface_normal_adjusted);

      // Schlick's approximation for Fresnel effect
      float r0 = (1.0 - material.ior) / (1.0 + material.ior);
      r0 = r0 * r0;
      float fresnel = r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);

      // Randomly choose between refraction and reflection based on Fresnel
      float rand_val = rand(payload.in_uv, payload.depth,
                            push_constants.uniforms.accumulated_count,
                            push_constants.uniforms.seed);
      if (rand_val < fresnel) {
        scatter_direction = reflected;
      } else {
        scatter_direction = refracted;
      }
    } else {
      // Total internal reflection
      scatter_direction = reflect(ray_dir, surface_normal_adjusted);
    }

    attenuation = vec3(1.0); // Glass doesn't absorb

  } else {
    // Diffuse material (material_type == 0)
    scatter_direction = cosine_hemisphere_sample(
        payload.in_uv, payload.depth, push_constants.uniforms.accumulated_count,
        push_constants.uniforms.seed, surface_normal);

    // Diffuse surfaces lose energy based on their color
    attenuation = albedo * (1.0 - metallic);
  }

  // Add small offset to avoid self-intersection
  vec3 scatter_origin = world_pos + surface_normal * 0.001;

  // Update payload for next bounce
  payload.origin = scatter_origin;
  payload.direction = scatter_direction;
  payload.attenuation *= attenuation;

  payload.done = should_scatter ? 0 : 1;
  payload.hit_value = vec3(0.0); // Direct lighting would be computed separately

  // Depth increment
  payload.depth += 1u;
}





// #version 460
// #extension GL_EXT_ray_tracing : require
// #extension GL_EXT_nonuniform_qualifier : require

// #include "shared.glsl"

// layout(location = 0) rayPayloadInEXT RayPayload payload;
// layout(location = 1) rayPayloadEXT bool is_shadowed;
// hitAttributeEXT vec2 attribs;

// layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_as;

// layout(push_constant) uniform RchitPushConstants { RendererUniforms uniforms; }
// push_constants;

// layout(binding = 0, set = 2) readonly buffer offsets_buffer {
//   Offsets offsets_array[];
// };

// layout(binding = 1, set = 2) readonly buffer vertex_buffer {
//   Vertex vertices[];
// };

// layout(binding = 2, set = 2) readonly buffer material_buffer {
//   Material materials[];
// };

// layout(binding = 3, set = 2) readonly buffer index_buffer { uint indices[]; };

// layout(binding = 0, set = 3) uniform sampler2D textures[];

// // Helper functions for texture flag checks
// bool has_base_color_texture(uint flags) { return (flags & (1u << 0u)) != 0u; }

// bool has_metallic_roughness_texture(uint flags) {
//   return (flags & (1u << 1u)) != 0u;
// }

// bool has_normal_texture(uint flags) { return (flags & (1u << 2u)) != 0u; }

// bool has_emissive_texture(uint flags) { return (flags & (1u << 3u)) != 0u; }

// bool has_ao_texture(uint flags) { return (flags & (1u << 4u)) != 0u; }

// void main() {
//   // Retrieve geometry data
//   Offsets offsets = offsets_array[gl_InstanceCustomIndexEXT];

//   // Get triangle indices
//   uint index_start = offsets.index_offset;
//   uint i0 = indices[index_start + gl_PrimitiveID * 3 + 0];
//   uint i1 = indices[index_start + gl_PrimitiveID * 3 + 1];
//   uint i2 = indices[index_start + gl_PrimitiveID * 3 + 2];

//   // Fetch vertices
//   Vertex v0 = vertices[offsets.vertex_offset + i0];
//   Vertex v1 = vertices[offsets.vertex_offset + i1];
//   Vertex v2 = vertices[offsets.vertex_offset + i2];

//   // Interpolate using barycentric coordinates
//   vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
//   vec3 normal =
//       normalize(v0.normal * barycentrics.x + v1.normal * barycentrics.y +
//                 v2.normal * barycentrics.z);
//   vec2 uv =
//       v0.uv * barycentrics.x + v1.uv * barycentrics.y + v2.uv * barycentrics.z;

//   // Calculate hit position in world space
//   vec3 world_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
//   vec3 ray_dir = normalize(gl_WorldRayDirectionEXT);

//   // Get material
//   Material material = materials[offsets.material_offset];

//   // Sample base color/albedo from texture or use constant
//   vec3 albedo = material.base_color.xyz;
//   if (has_base_color_texture(material.texture_flags)) {
//     vec4 texture_color =
//         texture(textures[nonuniformEXT(material.base_color_texture_index)], uv);
//     albedo = texture_color.xyz;
//   }

//   // Sample metallic and roughness
//   float metallic;
//   float roughness;

//   if (has_metallic_roughness_texture(material.texture_flags)) {
//     vec4 metal_rough = texture(
//         textures[nonuniformEXT(material.metallic_roughness_texture_index)], uv);
//     metallic = metal_rough.b;  // Blue channel = metallic
//     roughness = metal_rough.g; // Green channel = roughness
//   } else {
//     // No texture, use base values
//     metallic = material.metallic;
//     roughness = material.roughness;
//   }

//   // Clamp roughness to avoid fireflies
//   roughness = clamp(roughness, 0.05, 0.99);

//   // Debug mode visualization
//   if (push_constants.uniforms.debug_mode == 1u) {
//     // Metallic visualization: grayscale (black=0, white=1)
//     payload.attenuation = vec3(metallic);
//     payload.done = 1;
//     return;
//   } else if (push_constants.uniforms.debug_mode == 2u) {
//     // Roughness visualization: grayscale (black=smooth, white=rough)
//     payload.attenuation = vec3(roughness);
//     payload.done = 1;
//     return;
//   }

//   // Sample normal map if available
//   vec3 surface_normal = normal;
//   if (has_normal_texture(material.texture_flags)) {
//     vec3 normal_sample =
//         texture(textures[nonuniformEXT(material.normal_texture_index)], uv).xyz;
//     // Unpack normal map (assuming standard DX-style encoding)
//     normal_sample = normalize(normal_sample * 2.0 - 1.0);
//     // Transform from tangent space to world space
//     mat3 basis = create_basis(normal);
//     surface_normal = normalize(basis * normal_sample);
//   }

//   // Sample emissive texture and compute emissive contribution
//   vec3 emissive = material.emissive_color.rgb * material.emissive_strength;
//   if (has_emissive_texture(material.texture_flags)) {
//     vec3 emissive_sample = 
//         texture(textures[nonuniformEXT(material.emissive_texture_index)], uv).rgb;
//     emissive *= emissive_sample;
//   }

//   // Add emissive light directly to the accumulated color
//   // This is the light emitted by this surface
//   if (length(emissive) > 0.0) {
//     payload.attenuation *= emissive;
//     payload.done = 1;  // Terminate ray path at emissive surface
//     return;
//   }

//   // Unified PBR scattering - no material type assumptions
  
//   // Compute view-dependent Fresnel using Schlick's approximation
//   float cos_theta = max(dot(-ray_dir, surface_normal), 0.0);
  
//   // Base reflectivity (F0) depends on metallic
//   // Dielectrics: ~0.04 (4% reflectance at normal incidence)
//   // Metals: use albedo as F0, which is much higher
//   vec3 f0 = mix(vec3(0.04), albedo, metallic);
  
//   // Schlick's approximation for Fresnel
//   float fresnel_factor = pow(1.0 - cos_theta, 5.0);
//   vec3 fresnel = f0 + (vec3(1.0) - f0) * fresnel_factor;
  
//   // For path tracing, we need a scalar probability
//   // Use the luminance of the Fresnel color
//   float fresnel_luminance = dot(fresnel, vec3(0.299, 0.587, 0.114));
  
//   // Roughness reduces specular lobe sharpness AND probability for dielectrics
//   // Very rough dielectric surfaces should be almost entirely diffuse
//   // We use a more aggressive falloff for rough dielectrics
//   float roughness_factor = 1.0 - roughness;
  
//   // For dielectrics, roughness dramatically reduces specular reflection
//   // For metals, roughness only affects lobe width, not probability
//   float dielectric_specular = fresnel_luminance * roughness_factor * roughness_factor * roughness_factor;
//   float metal_specular = 1.0; // Metals always take specular path
  
//   float specular_probability = mix(dielectric_specular, metal_specular, metallic);
  
//   // Perturb based on roughness
//   vec3 random_perturb = cosine_hemisphere_sample(
//       payload.in_uv, payload.depth, push_constants.uniforms.accumulated_count,
//       push_constants.uniforms.seed, surface_normal);
  
//   // Specular and diffuse directions
//   vec3 reflected = reflect(ray_dir, surface_normal);
//   vec3 specular_dir = normalize(mix(reflected, random_perturb, roughness * roughness));
//   vec3 diffuse_dir = random_perturb;
  
//   // Blend between specular and diffuse based on modified Fresnel
//   float rand_val = rand(payload.in_uv, payload.depth,
//                         push_constants.uniforms.accumulated_count,
//                         push_constants.uniforms.seed);
  
//   vec3 scatter_direction;
//   if (rand_val < specular_probability) {
//     scatter_direction = specular_dir;
//   } else {
//     scatter_direction = diffuse_dir;
//   }
  
//   // Ensure ray is above surface
//   if (dot(scatter_direction, surface_normal) <= 0.0) {
//     scatter_direction = diffuse_dir;
//   }

//   // Energy conservation for PBR:
//   // - Metals (metallic=1): Use albedo as reflectance tint, no diffuse absorption
//   // - Dielectrics (metallic=0): Use albedo for diffuse, white for specular
//   // The attenuation should account for the path taken (specular vs diffuse)
//   vec3 attenuation;
//   if (rand_val < specular_probability) {
//     // Specular path: metals tint by albedo, dielectrics reflect white
//     attenuation = mix(vec3(1.0), albedo, metallic);
//   } else {
//     // Diffuse path: use albedo, but metals don't have diffuse so darken less
//     attenuation = albedo;
//   }

//   // Add small offset to avoid self-intersection
//   vec3 scatter_origin = world_pos + surface_normal * 0.001;

//   // Update payload for next bounce
//   payload.origin = scatter_origin;
//   payload.direction = scatter_direction;
//   payload.attenuation *= attenuation;

//   payload.done = 0;
//   payload.hit_value = vec3(0.0);

//   // Depth increment
//   payload.depth += 1u;
// }