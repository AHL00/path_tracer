use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateFlags, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
    },
    descriptor_set::{CopyDescriptorSet, DescriptorSet, WriteDescriptorSet},
    device::DeviceOwnedVulkanObject,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::GpuFuture,
};

use crate::{
    graphics::VulkanContext,
    renderer::{Renderer, shaders},
};

pub mod geometry;
pub mod gltf;

pub struct Scene {
    pub world: legion::World,
    pub resources: legion::Resources,

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
}

impl Scene {
    // For prototyping, constant shared buffer sizes
    /// In elements
    const SHARED_OFFSETS_BUFFER_SIZE: u64 = 10000;
    /// In elements
    const SHARED_VERTEX_BUFFER_SIZE: u64 = 500000;
    /// In elements
    const SHARED_MATERIAL_BUFFER_SIZE: u64 = 1024;
    /// In elements
    const SHARED_INDEX_BUFFER_SIZE: u64 = 1000000;

    pub fn new(
        context: &VulkanContext,
        pipeline_layout: &vulkano::pipeline::PipelineLayout,
    ) -> Self {
        let world = legion::World::default();
        let resources = legion::Resources::default();

        // layout(binding = 0, set = 2) readonly buffer offsets_buffer {
        //     Offsets offsets_array[];
        // };

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
            Self::SHARED_OFFSETS_BUFFER_SIZE as u64
                * std::mem::size_of::<shaders::Offsets>() as u64,
        )
        .unwrap();

        // layout(binding = 1, set = 2) readonly buffer vertex_buffer {
        //     Vertex vertices[];
        // };

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
            Self::SHARED_VERTEX_BUFFER_SIZE as u64 * std::mem::size_of::<shaders::Vertex>() as u64,
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

        let shared_material_buffer: Subbuffer<[shaders::Material]> = Buffer::new_slice(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: vulkano::buffer::BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                ..Default::default()
            },
            Self::SHARED_MATERIAL_BUFFER_SIZE as u64
                * std::mem::size_of::<shaders::Material>() as u64,
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
            Self::SHARED_INDEX_BUFFER_SIZE as u64 * std::mem::size_of::<u32>() as u64,
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
            world,
            resources,
            rhit_descriptor_set,
            shared_offsets_buffer,
            shared_vertex_buffer,
            vertex_offset: 0,
            shared_material_buffer,
            material_offset: 0,
            shared_index_buffer,
            index_offset: 0,
        }
    }

    /// Adds vertices to the shared vertex buffer and returns the start and end offset.
    /// NOTE: MAKE SURE A FRAME IS NOT IN FLIGHT.
    pub fn add_vertices(
        &mut self,
        vertices: &[shaders::Vertex],
        context: &VulkanContext,
    ) -> (u64, u64) {
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

    /// Adds materials to the shared material buffer and returns the start and end offset.
    /// NOTE: MAKE SURE A FRAME IS NOT IN FLIGHT.
    pub fn add_materials(
        &mut self,
        materials: &[shaders::Material],
        context: &VulkanContext,
    ) -> (u64, u64) {
        todo!("Implement add_materials");
    }

    pub fn get_materials_subbuffer(&self, start: u64, end: u64) -> Subbuffer<[shaders::Material]> {
        self.shared_material_buffer.clone().slice(start..end)
    }

    pub fn add_indices(&mut self, indices: &[u32], context: &VulkanContext) -> (u64, u64) {
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
                self.shared_offsets_buffer.clone(),
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
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,

    dirty: bool,
    matrix: glam::Mat4,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            dirty: true,
            matrix: glam::Mat4::IDENTITY,
        }
    }
}

impl Transform {
    pub fn new(position: glam::Vec3, rotation: glam::Quat, scale: glam::Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,

            dirty: true,
            matrix: glam::Mat4::IDENTITY,
        }
    }

    pub fn get_matrix(&mut self) -> glam::Mat4 {
        if self.dirty {
            self.dirty = false;
            self.matrix = glam::Mat4::from_scale_rotation_translation(
                self.scale,
                self.rotation,
                self.position,
            );
        }
        self.matrix
    }
}
