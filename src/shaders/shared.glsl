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

    vec3 normal;
    float dist;
};

struct RendererUniforms {
    uint samples_per_pixel;
    uint seed;
    // The number of frames already accumulated in the accumulation buffer
    uint accumulated_count;
    // For depth normalisation
    float z_near;
    float z_far;
    uint padding[3];
};

// float rand(vec2 uv, float depth, uint frame_seed) {
//     // Convert uint seed to something we can use in calculations
//     float seed_float = float(frame_seed) * 0.0000000234; // Small multiplier to avoid precision issues
    
//     // Create a better seed distribution from spatial coordinates
//     float x = uv.x * 12.9898 + depth * 78.233 + seed_float * 43.7191;
//     float y = uv.y * 39.7319 + depth * 68.7302 + seed_float * 23.4239;
    
//     // Add non-linear mixing to break grid patterns
//     x += sin(y * 0.0013) * 43.12;
//     y += cos(x * 0.0023) * 27.32;
    
//     // Convert to uint for bit manipulation
//     uvec2 p = uvec2(abs(x * 1000000.0), abs(y * 1000000.0));
    
//     // Apply non-linear bit mixing
//     p.x ^= p.y * 1013904223u;
//     p.y ^= p.x * 1664525u;
//     p.x ^= p.y * 3014900209u;
//     p.y ^= p.x * 2949673269u;
    
//     // Mix in the original seed bits more directly
//     p.x += frame_seed * 373587883u;
//     p.y ^= frame_seed * 1453127827u;
    
//     // Apply golden ratio multiplication for better distribution
//     uint state = p.x + p.y * 2654435761u;
    
//     // Apply numeric reciprocal hashing 
//     state ^= state >> 16;
//     state *= 0x85ebca6bu;
//     state ^= state >> 13;
//     state *= 0xc2b2ae35u;
//     state ^= state >> 16;
    
//     // Improved Wang hash with additional passes
//     state = (state ^ 61) ^ (state >> 16);
//     state *= 9;
//     state = state ^ (state >> 4);
//     state *= 0x27d4eb2d;
//     state = state ^ (state >> 15);
    
//     // Convert to normalized float
//     return float(state) / 4294967296.0;
// }

// // Hash function for generating random direction vectors
// vec3 random_unit_vector(vec2 uv, float depth, uint frame_seed) {
//     // Create different seeds for each component
//     float x = rand(uv, depth, frame_seed);
//     float y = rand(uv, depth, frame_seed + 1u);
//     float z = rand(uv, depth, frame_seed + 2u);
    
//     // Map from [0,1] to [-1,1]
//     vec3 v = 2.0 * vec3(x, y, z) - 1.0;
//     return normalize(v);
// }

// // Create an orthonormal basis with Z axis aligned to normal
// mat3 create_basis(vec3 normal) {
//     vec3 tangent_x, tangent_y;
    
//     // Find least dominant axis of normal for stable cross product
//     if (abs(normal.x) < abs(normal.y) && abs(normal.x) < abs(normal.z))
//         tangent_x = vec3(1, 0, 0);
//     else if (abs(normal.y) < abs(normal.z))
//         tangent_x = vec3(0, 1, 0);
//     else
//         tangent_x = vec3(0, 0, 1);
        
//     tangent_x = normalize(cross(tangent_x, normal));
//     tangent_y = normalize(cross(normal, tangent_x));
    
//     return mat3(tangent_x, tangent_y, normal);
// }

// // Cosine-weighted hemisphere sampling with uint frame_seed
// vec3 cosine_hemisphere_sample(vec2 uv, float depth, uint frame_seed, vec3 normal) {
//     // Use the rand function with frame_seed to generate stable random values
//     float u1 = rand(uv, depth, frame_seed);
//     float u2 = rand(uv, depth, frame_seed + 42u); // Different seed offset for variety
    
//     float phi = 2.0 * 3.1415926 * u1;
//     float cos_theta = sqrt(u2);
//     float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    
//     vec3 direction = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
//     return create_basis(normal) * direction;
// }




// Simple, fast integer hash function (PCG output function variant)
uint hash(uint state) {
    state ^= state >> 16; state *= 0x7feb352dU;
    state ^= state >> 15; state *= 0x846ca68bU;
    state ^= state >> 16;
    return state;
}

// Combine two uints into one hash value
uint hash2(uvec2 p) {
    p.x = hash(p.x);
    p.y = hash(p.y);
    return hash(p.x ^ (p.y * 7919u));
}

// Combine three uints into one hash value
uint hash3(uvec3 p) {
    p.x = hash(p.x);
    p.y = hash(p.y);
    p.z = hash(p.z);
    return hash(p.x ^ (p.y * 31u) ^ (p.z * 71u));
}

// Combine four uints into one hash value
uint hash4(uvec4 p) {
    p.x = hash(p.x);
    p.y = hash(p.y);
    p.z = hash(p.z);
    p.w = hash(p.w);
    // Combine hashed components (using prime multipliers)
    return hash(p.x ^ (p.y * 31u) ^ (p.z * 71u) ^ (p.w * 101u));
}


// --- Pseudo-Random Number Generation (PRNG) ---

// Generates a pseudo-random float in [0, 1) based on multiple inputs.
// Uses integer hashing for better statistical properties in parallel execution.
// Now includes accumulated_count for better temporal decorrelation.
float rand(vec2 uv, uint depth, uint accumulated_count, uint seed) {
    // Convert uv float bits directly to uint for robust integer hashing
    uvec2 u_uv = floatBitsToUint(uv);

    // Combine inputs into a single state using hashing.
    // Hash uv coords first, then combine with depth, count, and seed.
    uint state = hash2(u_uv);
    state = hash(state ^ depth);             // Mix in depth
    state = hash(state ^ accumulated_count); // Mix in accumulated_count
    state = hash(state ^ seed);              // Mix in seed

    // Convert final hash to float in [0, 1) range
    return float(state) / 4294967296.0;
}


// --- Sampling Functions ---

// Generates a random direction vector uniformly distributed *within* the unit sphere volume.
// Uses rejection sampling. Now includes accumulated_count.
vec3 random_in_unit_sphere(vec2 uv, uint depth, uint accumulated_count, uint seed) {
    vec3 p;
    uint current_seed = seed; // Use a local copy to modify for retries
    uint seed_offset = 0u;    // Offset for generating multiple numbers

    do {
        // Generate 3 independent random numbers using seed offsets
        // Pass accumulated_count to rand
        float x = rand(uv, depth, accumulated_count, current_seed + seed_offset + 0u);
        float y = rand(uv, depth, accumulated_count, current_seed + seed_offset + 1u);
        float z = rand(uv, depth, accumulated_count, current_seed + seed_offset + 2u);
        // Map to [-1, 1] cube
        p = 2.0 * vec3(x, y, z) - 1.0;
        // Increase offset significantly for next attempt if needed, to avoid correlation
        seed_offset += 3u;
    } while (dot(p, p) >= 1.0); // Reject if outside unit sphere

    return p; // Already within the sphere, no need to normalize for volume sampling
}

// Creates an orthonormal basis (tangent, bitangent, normal) from a normal vector.
// Uses the "Building an Orthonormal Basis, Revisited" method by Tom Duff et al.
mat3 create_basis(vec3 normal) {
    float sign_val = sign(normal.z);
    if (normal.z == 0.0) { sign_val = 1.0; } // Consistent handling for normal.z == 0
    float a = -1.0 / (sign_val + normal.z);
    float b = normal.x * normal.y * a;
    vec3 tangent = vec3(1.0 + sign_val * normal.x * normal.x * a, sign_val * b, -sign_val * normal.x);
    vec3 bitangent = vec3(b, sign_val + normal.y * normal.y * a, -normal.y);
    return mat3(tangent, bitangent, normal);
}


// Generates a random direction vector on the hemisphere oriented by 'normal',
// weighted by the cosine of the angle to the normal (Lambertian distribution).
// Now includes accumulated_count.
// TODO: Remove depth, keep accumulated_count and seed only
vec3 cosine_hemisphere_sample(vec2 uv, uint depth, uint accumulated_count, uint seed, vec3 normal) {
    // Generate 2 independent random numbers using different seed offsets
    // Use large prime offsets to reduce correlation risk
    // Pass accumulated_count to rand
    float u1 = rand(uv, depth, accumulated_count, seed + 7919u);
    float u2 = rand(uv, depth, accumulated_count, seed + 16381u);

    // Map uniform random numbers to cosine-weighted hemisphere direction (local space, Z up)
    // Malley's method / Shirley's concentric mapping
    float r = sqrt(u1);
    float phi = 2.0 * 3.1415926535 * u2;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0, 1.0 - u1)); // z = sqrt(1 - r^2) = sqrt(1 - u1)

    vec3 direction_local = vec3(x, y, z);

    // Create basis and transform local direction to world space
    mat3 basis = create_basis(normal);
    return basis * direction_local;
}