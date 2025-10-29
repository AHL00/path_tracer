use crate::{
    graphics::{Texture, VulkanContext, VulkanContextID},
    renderer::shaders,
    scene::geometry,
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock, Weak},
};
use vulkano::{
    acceleration_structure::AccelerationStructure,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    device::DeviceOwnedVulkanObject,
    image::{sampler::Sampler, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::GpuFuture,
};

/// SceneResources contains all GPU resources that are tied to a specific Scene.
///
/// These resources are scene-dependent and must be managed together:
/// - `tlas` (Top-Level Acceleration Structure) is built from scene geometries
/// - Texture pool (`bindless_textures`, `loaded_textures_map`) stores scene textures
///
/// When swapping scenes, you must swap both the Scene AND its corresponding SceneResources
/// to maintain consistency. This struct ensures they're conceptually grouped together.
pub struct SceneResources {
    /// Top-Level Acceleration Structure for ray tracing.
    /// Built from the scene's geometries and transforms.
    /// Must be rebuilt whenever scene geometry changes.
    pub tlas: Arc<AccelerationStructure>,

    /// Descriptor set for bindless texture sampling in shaders.
    /// Points to all loaded textures for this scene.
    pub bindless_textures_descriptor_set: Arc<DescriptorSet>,

    /// CPU-side copy of bindless textures for updating the descriptor set.
    /// The entire array must be updated at once via WriteDescriptorSet.
    bindless_textures: Vec<(Arc<ImageView>, Arc<Sampler>)>,

    /// Current index for adding new textures to the bindless pool.
    bindless_texture_index: u32,

    /// Lookup map for textures by identifier (e.g., filename).
    /// Allows quick access to already-loaded textures without reloading.
    ///
    /// TODO: These should be weak pointers to avoid keeping textures alive
    /// when they're no longer referenced by geometries.
    pub loaded_textures_map: std::collections::HashMap<String, Texture>,

    pub rhit_descriptor_set: Arc<DescriptorSet>,

    /// Don't care about updating data in this buffer.
    /// It will be updated every frame as it's small
    /// enough to not care.
    pub shared_offsets_buffer: Subbuffer<[shaders::Offsets]>,

    pub shared_vertex_buffer: Subbuffer<[shaders::Vertex]>,
    pub vertex_offset: u64,
    pub shared_material_buffer: Subbuffer<[shaders::Material]>,
    pub material_offset: u64,
    pub shared_index_buffer: Subbuffer<[u32]>,
    pub index_offset: u64,

    // A data structure to store the hashes of loaded textures
    pub texture_weak_cache: TextureWeakCache,

    pub geometries_map: HashMap<String, crate::scene::geometry::Geometry>,

    _is_dirty_for_renderer: bool,
}

impl SceneResources {
    const MAX_TEXTURE_COUNT: u32 = 5000;

    // For prototyping, constant shared buffer sizes
    /// In elements
    const SHARED_OFFSETS_BUFFER_SIZE: u64 = 10000;
    /// In elements
    const SHARED_VERTEX_BUFFER_SIZE: u64 = 2500000;
    /// In elements
    const SHARED_MATERIAL_BUFFER_SIZE: u64 = 10000;
    /// In elements
    const SHARED_INDEX_BUFFER_SIZE: u64 = 5000000;

    pub fn check_dirty_and_reset(&mut self) -> bool {
        if self._is_dirty_for_renderer {
            self._is_dirty_for_renderer = false;
            true
        } else {
            false
        }
    }

    pub fn new(
        context: &VulkanContext,
        pipeline_layout: &vulkano::pipeline::PipelineLayout,
    ) -> Self {
        // layout(binding = 0, set = 2) readonly buffer offsets_buffer {
        //     Offsets offsets_array[];
        // };

        log::info!(
            "Max buffer size: {:?}",
            context
                .device
                .physical_device()
                .properties()
                .max_buffer_size
        );

        // Create initial scene resources
        let tlas =
            unsafe { crate::renderer::build_top_level_acceleration_structure(vec![], &context) };
        let bindless_textures_descriptor_set = vulkano::descriptor_set::DescriptorSet::new(
            context.descriptor_set_allocator_update_after_bind.clone(),
            pipeline_layout.set_layouts().get(3).unwrap().clone(),
            [],
            [],
        )
        .unwrap();

        log::info!(
            "Shared offsets buffer size: {:?} bytes",
            Self::SHARED_OFFSETS_BUFFER_SIZE as u64
                * std::mem::size_of::<shaders::Offsets>() as u64
        );
        let shared_offsets_buffer: Subbuffer<[shaders::Offsets]> = Buffer::new_slice(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                ..Default::default()
            },
            AllocationCreateInfo {
                ..Default::default()
            },
            Self::SHARED_OFFSETS_BUFFER_SIZE,
        )
        .unwrap();

        // layout(binding = 1, set = 2) readonly buffer vertex_buffer {
        //     Vertex vertices[];
        // };

        log::info!(
            "Shared vertex buffer size: {:?} bytes",
            Self::SHARED_VERTEX_BUFFER_SIZE as u64 * std::mem::size_of::<shaders::Vertex>() as u64
        );
        let shared_vertex_buffer: Subbuffer<[shaders::Vertex]> = Buffer::new_slice(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                ..Default::default()
            },
            AllocationCreateInfo {
                ..Default::default()
            },
            Self::SHARED_VERTEX_BUFFER_SIZE as u64,
        )
        .unwrap();

        shared_vertex_buffer
            .buffer()
            .clone()
            .set_debug_utils_object_name(Some("shared_vertex_buffer"))
            .unwrap();

        // layout(binding = 2, set = 2) readonly buffer material_buffer {
        //     Material materials[];
        // };

        log::info!(
            "Shared material buffer size: {:?} bytes",
            Self::SHARED_MATERIAL_BUFFER_SIZE as u64
                * std::mem::size_of::<shaders::Material>() as u64
        );
        let shared_material_buffer: Subbuffer<[shaders::Material]> = Buffer::new_slice(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                ..Default::default()
            },
            AllocationCreateInfo {
                ..Default::default()
            },
            Self::SHARED_MATERIAL_BUFFER_SIZE as u64,
        )
        .unwrap();

        shared_material_buffer
            .buffer()
            .clone()
            .set_debug_utils_object_name(Some("shared_material_buffer"))
            .unwrap();

        // layout(binding = 3, set = 2) readonly buffer index_buffer {
        //     uint indices[];
        // };

        log::info!(
            "Shared index buffer size: {:?} bytes",
            Self::SHARED_INDEX_BUFFER_SIZE as u64 * std::mem::size_of::<u32>() as u64
        );
        let shared_index_buffer: Subbuffer<[u32]> = Buffer::new_slice(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                ..Default::default()
            },
            AllocationCreateInfo {
                ..Default::default()
            },
            Self::SHARED_INDEX_BUFFER_SIZE as u64,
        )
        .unwrap();

        shared_index_buffer
            .buffer()
            .clone()
            .set_debug_utils_object_name(Some("shared_index_buffer"))
            .unwrap();

        let rhit_descriptor_set = DescriptorSet::new(
            context.descriptor_set_allocator.clone(),
            pipeline_layout.set_layouts()[2].clone(),
            [
                WriteDescriptorSet::buffer(0, shared_offsets_buffer.clone()),
                WriteDescriptorSet::buffer(1, shared_vertex_buffer.clone()),
                WriteDescriptorSet::buffer(2, shared_material_buffer.clone()),
                WriteDescriptorSet::buffer(3, shared_index_buffer.clone()),
            ],
            [],
        )
        .unwrap();

        Self {
            tlas,
            bindless_textures_descriptor_set,
            bindless_textures: vec![],
            bindless_texture_index: 0,
            loaded_textures_map: std::collections::HashMap::new(),

            rhit_descriptor_set,
            geometries_map: HashMap::new(),

            texture_weak_cache: TextureWeakCache::new(),

            shared_offsets_buffer,
            shared_vertex_buffer,
            vertex_offset: 0,
            shared_material_buffer,
            material_offset: 0,
            shared_index_buffer,
            index_offset: 0,

            _is_dirty_for_renderer: true,
        }
    }

    pub fn lookup_texture(&self, image_ident: &str) -> Option<Texture> {
        self.loaded_textures_map.get(image_ident).cloned()
    }

    pub fn add_texture_to_lookup_map(&mut self, image_ident: String, texture: Texture) {
        self.loaded_textures_map.insert(image_ident, texture);
    }

    /// Add a texture to the global pool, returning the index of the texture
    /// in the bindless texture descriptor set.
    /// NOTE: Does not check whether the texture is already loaded or add it to the lookup map.
    pub fn add_texture(
        &mut self,
        image: Arc<ImageView>,
        sampler: Arc<Sampler>,
        context: &crate::graphics::VulkanContext,
    ) -> u32 {
        let index = self.bindless_texture_index;
        self.bindless_textures
            .push((image.clone(), sampler.clone()));

        let write_descriptor_set =
            vulkano::descriptor_set::WriteDescriptorSet::image_view_sampler_array(
                0,
                0,
                self.bindless_textures.clone(),
            );

        unsafe {
            self.bindless_textures_descriptor_set
                .update_by_ref([write_descriptor_set], [])
                .unwrap()
        };

        self.bindless_texture_index += 1;

        if self.bindless_texture_index >= Self::MAX_TEXTURE_COUNT {
            panic!("Bindless texture pool size exceeded");
        }

        index
    }

    pub fn bindless_textures_descriptor_set(&self) -> &Arc<DescriptorSet> {
        &self.bindless_textures_descriptor_set
    }

    pub fn tlas(&self) -> &Arc<AccelerationStructure> {
        &self.tlas
    }

    pub fn set_tlas(&mut self, new_tlas: Arc<AccelerationStructure>) {
        self.tlas = new_tlas;
    }

    pub fn add_geometry(
        &mut self,
        geometry_ident: String,
        geometry: geometry::Geometry,
    ) -> Option<geometry::Geometry> {
        self.geometries_map.insert(geometry_ident, geometry)
    }

    pub fn get_geometry(&self, geometry_ident: &str) -> Option<&geometry::Geometry> {
        self.geometries_map.get(geometry_ident)
    }

    /// Adds vertices to the shared vertex buffer and returns the start and end offset.
    /// NOTE: MAKE SURE A FRAME IS NOT IN FLIGHT.
    /// Returns (start, end) offsets in the shared vertex buffer.
    pub fn add_vertices(
        &mut self,
        vertices: &[shaders::Vertex],
        context: &VulkanContext,
    ) -> (u64, u64) {
        self._is_dirty_for_renderer = true;

        let start_offset = self.vertex_offset;
        let end_offset = start_offset + vertices.len() as u64;

        // Check if we have enough space
        if end_offset > Self::SHARED_VERTEX_BUFFER_SIZE {
            panic!("Vertex buffer overflow");
        }

        // Create a host-visible staging buffer
        let staging_buffer = Buffer::from_iter(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices.iter().cloned(),
        )
        .unwrap();

        // Create a command buffer to copy from staging to device-local buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            context.command_buffer_allocator.clone(),
            context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                staging_buffer,
                self.shared_vertex_buffer
                    .clone()
                    .slice(start_offset..end_offset),
            ))
            .unwrap();

        // Execute the command buffer and wait for completion
        builder
            .build()
            .unwrap()
            .execute(context.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        // Update offset for next write
        self.vertex_offset = end_offset;
        (start_offset, end_offset)
    }

    pub fn get_vertices_subbuffer(&self, start: u64, end: u64) -> Subbuffer<[shaders::Vertex]> {
        self.shared_vertex_buffer.clone().slice(start..end)
    }

    /// Adds a material to the shared material buffer and returns the offset in the buffer.
    /// NOTE: MAKE SURE A FRAME IS NOT IN FLIGHT.
    pub fn add_material(&mut self, materials: &shaders::Material, context: &VulkanContext) -> u64 {
        self._is_dirty_for_renderer = true;

        let start_offset = self.material_offset;
        let end_offset = start_offset + 1;

        // Check if we have enough space
        if end_offset > Self::SHARED_MATERIAL_BUFFER_SIZE {
            panic!("Material buffer overflow");
        }

        // Create a host-visible staging buffer
        let staging_buffer = Buffer::from_iter(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            std::iter::once(*materials),
        )
        .unwrap();

        // Create a command buffer to copy from staging to device-local buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            context.command_buffer_allocator.clone(),
            context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                staging_buffer,
                self.shared_material_buffer
                    .clone()
                    .slice(start_offset..end_offset),
            ))
            .unwrap();

        // Execute the command buffer and wait for completion
        builder
            .build()
            .unwrap()
            .execute(context.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        // Update offset for next write
        self.material_offset = end_offset;
        start_offset
    }

    pub fn get_materials_subbuffer(&self, start: u64, end: u64) -> Subbuffer<[shaders::Material]> {
        self.shared_material_buffer.clone().slice(start..end)
    }

    /// Returns (start, end) offsets in the shared index buffer.
    pub fn add_indices(&mut self, indices: &[u32], context: &VulkanContext) -> (u64, u64) {
        self._is_dirty_for_renderer = true;

        let start_offset = self.index_offset;
        let end_offset = start_offset + indices.len() as u64;

        // Check if we have enough space
        if end_offset > Self::SHARED_INDEX_BUFFER_SIZE {
            panic!("Index buffer overflow");
        }

        // Create a host-visible staging buffer
        let staging_buffer = Buffer::from_iter(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices.iter().cloned(),
        )
        .unwrap();

        // Create a command buffer to copy from staging to device-local buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            context.command_buffer_allocator.clone(),
            context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                staging_buffer,
                self.shared_index_buffer
                    .clone()
                    .slice(start_offset..end_offset),
            ))
            .unwrap();

        // Execute the command buffer and wait for completion
        builder
            .build()
            .unwrap()
            .execute(context.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        // Update offset for next write
        self.index_offset = end_offset;
        (start_offset, end_offset)
    }

    pub fn get_indices_subbuffer(&self, start: u64, end: u64) -> Subbuffer<[u32]> {
        self.shared_index_buffer.clone().slice(start..end)
    }

    pub fn update_shared_offsets_buffer(
        &mut self,
        offsets: &[shaders::Offsets],
        context: &VulkanContext,
    ) {
        log::debug!("update_shared_offsets_buffer: starting");
        if offsets.len() == 0 {
            // TODO: Is it okay to just return?
            return;
        }

        log::debug!("update_shared_offsets_buffer: creating staging buffer");
        // Create a host-visible staging buffer
        let staging_buffer = Buffer::from_iter(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            offsets.iter().cloned(),
        )
        .unwrap();

        log::debug!("update_shared_offsets_buffer: creating command builder");
        // Create a command buffer to copy from staging to device-local buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            context.command_buffer_allocator.clone(),
            context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        log::debug!("update_shared_offsets_buffer: copying buffer");
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                staging_buffer,
                self.shared_offsets_buffer.clone(),
            ))
            .unwrap();

        log::debug!("update_shared_offsets_buffer: building and executing command buffer");
        // Execute the command buffer and wait for completion
        builder
            .build()
            .unwrap()
            .execute(context.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        log::debug!("update_shared_offsets_buffer: done");
    }
}

// SAFETY: SceneResources can be safely sent between threads.
// All GPU resources are Arc-wrapped (thread-safe reference counting).
// Texture lookup operations are read-only or protected by Mutex at the RenderApp level.
unsafe impl Send for SceneResources {}
unsafe impl Sync for SceneResources {}

pub struct TextureWeakCache {
    // Still store context id for validation even though it's hashed together with the data
    by_hash: Mutex<std::collections::HashMap<u64, (Weak<Texture>, VulkanContextID)>>,
}

impl TextureWeakCache {
    pub fn new() -> Self {
        Self {
            by_hash: Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn singleton() -> &'static TextureWeakCache {
        static INSTANCE: OnceLock<TextureWeakCache> = OnceLock::new();
        INSTANCE.get_or_init(TextureWeakCache::new)
    }

    pub fn get_or_load(
        &self,
        image_source: gltf::image::Source,
        gltf_path: &std::path::Path,
        buffers: &[gltf::buffer::Data],
        sampler: gltf::texture::Sampler,
        scene_resources: Arc<Mutex<SceneResources>>,
        context: &crate::graphics::VulkanContext,
    ) -> Arc<Texture> {
        let image_hash;
        let mut data = None;
        let mut uri_ = "?";

        match &image_source {
            gltf::image::Source::View { .. } => {
                data = Some(
                    gltf::image::Data::from_source(image_source.clone(), Some(gltf_path), buffers)
                        .unwrap(),
                );
                image_hash = xxhash_rust::xxh64::xxh64(&data.as_ref().unwrap().pixels, 0);
            }
            gltf::image::Source::Uri { uri, .. } => {
                image_hash = xxhash_rust::xxh64::xxh64(uri.as_bytes(), 0);
                uri_ = uri;
            }
        }

        // Combine context ID and image hash
        let context_id = context.id();
        let combined_hash = xxhash_rust::xxh64::xxh64(&context_id.to_le_bytes(), image_hash);

        log::info!("Loading texture [{}] with hash {:x}", uri_, combined_hash);

        // Check if we have a weak reference and if it's still valid
        {
            let cache = self.by_hash.lock().unwrap();
            // log::info!("Cache: {:#x?}", cache);
            if let Some((weak_texture, cached_context_id)) = cache.get(&combined_hash) {
                // If the renderer ID matches, we can use the cached texture
                if *cached_context_id == context.id() {
                    // If we can upgrade the weak reference, return the strong Arc.
                    // Texture is still loaded on the renderer being used.
                    if let Some(texture) = weak_texture.upgrade() {
                        log::info!("Repeated texture detected with hash {:x}", combined_hash);
                        return texture;
                    }
                }
                // Weak pointer is dead or wrong renderer, we'll reload below
            }
        }

        let image_ident = format!("gltf_texture_{:x}", combined_hash);
        let texture = Arc::new(Texture::from_gltf(
            data.unwrap_or_else(|| {
                gltf::image::Data::from_source(image_source, Some(gltf_path), buffers).unwrap()
            }),
            image_ident,
            sampler,
            scene_resources,
            context,
        ));

        // Store weak reference
        self.by_hash
            .lock()
            .expect("Failed to lock texture weak cache mutex")
            .insert(combined_hash, (Arc::downgrade(&texture), context.id()));
        texture
    }
}
