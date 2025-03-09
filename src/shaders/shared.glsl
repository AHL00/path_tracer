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
    float metallic;
    float roughness;
    float padding_a;
    float padding_b;
};