use std::sync::atomic::AtomicU64;

use crate::{graphics::Texture, renderer::shaders};

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

impl Into<shaders::Material> for Material {
    fn into(self) -> shaders::Material {
        shaders::Material {
            base_color: [
                self.base_color.x,
                self.base_color.y,
                self.base_color.z,
                self.base_color.w,
            ],
            metallic: self.metallic,
            roughness: self.roughness,
            padding_a: 0.0,
            padding_b: 0.0,
        }
    }
}

impl<'a> Into<shaders::Material> for gltf::Material<'a> {
    fn into(self) -> shaders::Material {
        let pbr_mr = self.pbr_metallic_roughness();
        pbr_mr.base_color_texture();
        shaders::Material {
            base_color: [
                pbr_mr.base_color_factor()[0],
                pbr_mr.base_color_factor()[1],
                pbr_mr.base_color_factor()[2],
                pbr_mr.base_color_factor()[3],
            ],
            metallic: pbr_mr.metallic_factor(),
            roughness: pbr_mr.roughness_factor(),
            padding_a: 0.0,
            padding_b: 0.0,
        }
    }
}

impl Material {
    pub fn from_gltf<'a>(
        gltf_mat: gltf::Material<'a>,
        scene: &mut crate::scene::Scene,
        context: &crate::graphics::VulkanContext,
        buffers: &[gltf::buffer::Data],
        gltf_path: &std::path::Path,
    ) -> Self {
        let name = gltf_mat.name().unwrap_or("Unnamed").to_string();
        let base_texture_info = gltf_mat.pbr_metallic_roughness().base_color_texture();

        let shader_mat = gltf_mat.into();

        let buffer_index = scene.add_material(&shader_mat, context);

        Self {
            _id: MATERIAL_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            _shared_buffer_index: buffer_index,

            name,
            base_color: glam::Vec4::new(
                shader_mat.base_color[0],
                shader_mat.base_color[1],
                shader_mat.base_color[2],
                shader_mat.base_color[3],
            ),
            base_texture: base_texture_info.map(|info| {
                let image_source = info.texture().source().source();
                let sampler = info.texture().sampler();

                let data = gltf::image::Data::from_source(
                    image_source,
                    Some(gltf_path),
                    buffers,
                )
                .unwrap();

                Texture::from_gltf(data, sampler, context)
            }),
            metallic: shader_mat.metallic,
            roughness: shader_mat.roughness,
        }
    }

    pub fn get_shared_buffer_offset(&self) -> u64 {
        self._shared_buffer_index
    }
}
