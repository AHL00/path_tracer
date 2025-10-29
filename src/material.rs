use std::{
    cell::LazyCell,
    sync::{Arc, Mutex, OnceLock, Weak, atomic::AtomicU64},
};

use gltf::json::extensions::texture;

use crate::{
    graphics::{Texture, VulkanContextID},
    renderer::{Renderer, RendererID, shaders},
    scene_resources::{SceneResources, TextureWeakCache},
};

static MATERIAL_ID: AtomicU64 = AtomicU64::new(0);

/// A PBR material compatible with GLTF.
/// Uses GGX/Trowbridge-Reitz BRDF for physically accurate rendering.
/// Linked to a specific scene due to shared buffers.
#[derive(Debug)]
pub struct Material {
    _id: u64,
    _shared_buffer_index: u64,

    pub name: String,

    // Base color
    pub base_color: glam::Vec4,
    pub base_color_texture: Option<Arc<Texture>>,

    // PBR Parameters
    pub metallic: f32,
    pub roughness: f32,

    // PBR Textures
    pub metallic_roughness_texture: Option<Arc<Texture>>,
    pub normal_texture: Option<Arc<Texture>>,
    pub emissive_texture: Option<Arc<Texture>>,
    pub ao_texture: Option<Arc<Texture>>,

    // Emissive
    pub emissive_color: glam::Vec4,
    pub emissive_strength: f32,

    // IOR for dielectric materials (glass, plastic, etc.)
    pub ior: f32,

    // Material type: 0=Diffuse, 1=Metallic, 2=Glass/Dielectric
    pub material_type: MaterialType,
}

impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        self._id == other._id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MaterialType {
    Diffuse = 0,
    Metallic = 1,
    Glass = 2,
}

// NOTE: Check shared.glsl for texture flag usage
fn generate_texture_flags(
    base_color_texture: &Option<Arc<Texture>>,
    metallic_roughness_texture: &Option<Arc<Texture>>,
    normal_texture: &Option<Arc<Texture>>,
    emissive_texture: &Option<Arc<Texture>>,
    ao_texture: &Option<Arc<Texture>>,
) -> u32 {
    let mut flags = 0;
    if base_color_texture.is_some() {
        flags |= 1 << 0;
    }
    if metallic_roughness_texture.is_some() {
        flags |= 1 << 1;
    }
    if normal_texture.is_some() {
        flags |= 1 << 2;
    }
    if emissive_texture.is_some() {
        flags |= 1 << 3;
    }
    if ao_texture.is_some() {
        flags |= 1 << 4;
    }
    flags
}

impl Material {
    pub fn from_gltf<'a>(
        gltf_mat: gltf::Material<'a>,
        scene_resources: Arc<Mutex<SceneResources>>,
        context: &crate::graphics::VulkanContext,
        buffers: &[gltf::buffer::Data],
        gltf_path: &std::path::Path,
    ) -> Self {
        let name = gltf_mat.name().unwrap_or("Unnamed").to_string();
        let base_color = gltf_mat.pbr_metallic_roughness().base_color_factor();
        let base_texture_info = gltf_mat.pbr_metallic_roughness().base_color_texture();
        let metallic = gltf_mat.pbr_metallic_roughness().metallic_factor();
        let roughness = gltf_mat.pbr_metallic_roughness().roughness_factor();

        let texture_cache = TextureWeakCache::singleton();

        let base_color_texture = base_texture_info.map(|info| {
            let image_source = info.texture().source().source();
            let sampler = info.texture().sampler();
            texture_cache.get_or_load(
                image_source,
                gltf_path,
                buffers,
                sampler,
                scene_resources.clone(),
                context,
            )
        });

        let metallic_roughness_texture = gltf_mat
            .pbr_metallic_roughness()
            .metallic_roughness_texture()
            .map(|info| {
                let image_source = info.texture().source().source();
                let sampler = info.texture().sampler();
                texture_cache.get_or_load(
                    image_source,
                    gltf_path,
                    buffers,
                    sampler,
                    scene_resources.clone(),
                    context,
                )
            });

        let normal_texture = gltf_mat.normal_texture().map(|info| {
            let image_source = info.texture().source().source();
            let sampler = info.texture().sampler();
            texture_cache.get_or_load(
                image_source,
                gltf_path,
                buffers,
                sampler,
                scene_resources.clone(),
                context,
            )
        });

        let emissive_texture = gltf_mat.emissive_texture().map(|info| {
            let image_source = info.texture().source().source();
            let sampler = info.texture().sampler();
            texture_cache.get_or_load(
                image_source,
                gltf_path,
                buffers,
                sampler,
                scene_resources.clone(),
                context,
            )
        });

        let ao_texture = gltf_mat.occlusion_texture().map(|info| {
            let image_source = info.texture().source().source();
            let sampler = info.texture().sampler();
            texture_cache.get_or_load(
                image_source,
                gltf_path,
                buffers,
                sampler,
                scene_resources.clone(),
                context,
            )
        });

        let emissive_color = gltf_mat.emissive_factor();
        let emissive_strength = gltf_mat.emissive_strength().unwrap_or(1.0);

        // Determine IOR and material type
        let ior = gltf_mat.ior().unwrap_or(1.5);

        let material_type = if let Some(_) = gltf_mat.transmission() {
            MaterialType::Glass
        } else if metallic > 0.1 {
            // Lower threshold - most metallic materials
            MaterialType::Metallic
        } else {
            MaterialType::Diffuse
        };

        let shader_mat = shaders::Material {
            base_color: [base_color[0], base_color[1], base_color[2], base_color[3]],
            base_color_texture_index: base_color_texture
                .as_ref()
                .map_or(0, |t| t.bindless_indice()),

            metallic,
            roughness,

            ior,

            metallic_roughness_texture_index: metallic_roughness_texture
                .as_ref()
                .map_or(0, |t| t.bindless_indice()),
            normal_texture_index: normal_texture.as_ref().map_or(0, |t| t.bindless_indice()),
            emissive_texture_index: emissive_texture.as_ref().map_or(0, |t| t.bindless_indice()),
            ao_texture_index: ao_texture.as_ref().map_or(0, |t| t.bindless_indice()),

            emissive_color: [emissive_color[0], emissive_color[1], emissive_color[2], 1.0],
            emissive_strength,

            texture_flags: generate_texture_flags(
                &base_color_texture,
                &metallic_roughness_texture,
                &normal_texture,
                &emissive_texture,
                &ao_texture,
            ),

            material_type: material_type as u32,

            _padding: 0,
        };

        let buffer_index = scene_resources
            .lock()
            .unwrap()
            .add_material(&shader_mat, context);

        // let buffer_index = scene_resources
        //     .lock()
        //     .unwrap()
        //     .

        Self {
            _id: MATERIAL_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            _shared_buffer_index: buffer_index,

            name,

            base_color: glam::Vec4::from_array(shader_mat.base_color),
            base_color_texture,

            metallic,
            roughness,
            metallic_roughness_texture,

            normal_texture,
            emissive_texture,
            ao_texture,

            emissive_color: glam::Vec4::from_array(shader_mat.emissive_color),
            emissive_strength,
            ior,
            material_type,
        }
    }

    pub fn get_shared_buffer_offset(&self) -> u64 {
        self._shared_buffer_index
    }
}
