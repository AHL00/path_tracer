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