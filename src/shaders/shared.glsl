struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 uv;
};

struct Offsets {
    uint vertex_offset;
    uint material_offset;
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