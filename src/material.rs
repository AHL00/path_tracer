use std::sync::atomic::AtomicU64;

use crate::{
    graphics::Texture,
    renderer::{Renderer, shaders},
};

static MATERIAL_ID: AtomicU64 = AtomicU64::new(0);

/// A PBR material compatible with GLTF.
/// Linked to a specific scene due to shared buffers.
#[derive(Debug)]
pub struct Material {
    _id: u64,
    _shared_buffer_index: u64,

    pub name: String,
    pub base_color: glam::Vec4,
    pub base_texture: Option<Texture>,
    pub metallic: f32,
    pub roughness: f32,
}

impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        self._id == other._id
    }
}
impl Material {
    pub fn from_gltf<'a>(
        gltf_mat: gltf::Material<'a>,
        renderer: &mut Renderer,
        context: &crate::graphics::VulkanContext,
        buffers: &[gltf::buffer::Data],
        gltf_path: &std::path::Path,
    ) -> Self {
        let name = gltf_mat.name().unwrap_or("Unnamed").to_string();
        let base_color = gltf_mat.pbr_metallic_roughness().base_color_factor();
        let base_texture_info = gltf_mat.pbr_metallic_roughness().base_color_texture();
        let metallic = gltf_mat.pbr_metallic_roughness().metallic_factor();
        let roughness = gltf_mat.pbr_metallic_roughness().roughness_factor();

        let texture = base_texture_info.map(|info| {
            let image_source = info.texture().source().source();
            let sampler = info.texture().sampler();

            let data =
                gltf::image::Data::from_source(image_source, Some(gltf_path), buffers).unwrap();

            Texture::from_gltf(data, sampler, renderer, context)
        });

        let shader_mat = shaders::Material {
            base_color,
            has_base_texture: texture.is_some() as u32,
            base_texture_indice: texture.as_ref().map_or(0, |t| t.bindless_indice),
            metallic: metallic,
            roughness: roughness,
        };

        let buffer_index = renderer.scene.add_material(&shader_mat, context);

        Self {
            _id: MATERIAL_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            _shared_buffer_index: buffer_index,

            name,
            base_color: glam::Vec4::from_array(shader_mat.base_color),
            base_texture: texture,
            metallic: shader_mat.metallic,
            roughness: shader_mat.roughness,
        }
    }

    pub fn get_shared_buffer_offset(&self) -> u64 {
        self._shared_buffer_index
    }
}
