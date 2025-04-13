struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 uv;
    vec2 padding;
};

struct Offsets {
    uint vertex_offset;
    uint vertex_count;
    uint material_offset;
    uint padding;
    uint index_offset;
    uint index_count;
};

struct Camera {
    mat4 view_proj;    // Camera view * projection
    mat4 view_inverse; // Camera inverse view matrix
    mat4 proj_inverse; // Camera inverse projection matrix
};

// struct Texture {
//     sampler2D texture;
// };

struct Material {
    vec4 base_color;
    // TODO: Make this a bitflag type
    // to support all textures
    // with just one 32-bit integer
    bool has_base_texture;
    uint base_texture_indice;
    float metallic;
    float roughness;
};

struct RayPayload {
    vec3 hit_value;
    uint depth;
    vec3 attenuation;
    int done;
    vec3 origin;
    vec3 direction;
    vec2 in_uv;
};

struct RendererUniforms {
    uint samples_per_pixel;
    uint seed;
    uint padding[2];
};

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

// Create an orthonormal basis with Z axis aligned to normal
mat3 create_basis(vec3 normal) {
    vec3 tangent_x, tangent_y;
    
    // Find least dominant axis of normal for stable cross product
    if (abs(normal.x) < abs(normal.y) && abs(normal.x) < abs(normal.z))
        tangent_x = vec3(1, 0, 0);
    else if (abs(normal.y) < abs(normal.z))
        tangent_x = vec3(0, 1, 0);
    else
        tangent_x = vec3(0, 0, 1);
        
    tangent_x = normalize(cross(tangent_x, normal));
    tangent_y = normalize(cross(normal, tangent_x));
    
    return mat3(tangent_x, tangent_y, normal);
}

// Cosine-weighted hemisphere sampling 
vec3 cosine_hemisphere_sample(vec2 u, vec3 normal) {
    float phi = 2.0 * 3.1415926 * u.x;
    float cos_theta = sqrt(u.y);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    
    vec3 direction = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return create_basis(normal) * direction;
}

// Generate two random floats for sampling
vec2 random_float2(vec2 seed, float depth) {
    return vec2(
        rand(seed + vec2(depth, 0.13)),
        rand(seed + vec2(0.71, depth + 0.29))
    );
}