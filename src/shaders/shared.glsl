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
    uint material_count;
    uint index_offset;
    uint index_count;
};

// struct Texture {
//     sampler2D texture;
// };

struct Material {
    vec3 albedo;
    float metallic;
    float roughness;
    float alpha;
    vec3 emissive;
};