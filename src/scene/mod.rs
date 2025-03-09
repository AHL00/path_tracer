use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateFlags, BufferCreateInfo, Subbuffer},
    descriptor_set::{CopyDescriptorSet, DescriptorSet, WriteDescriptorSet},
    memory::allocator::AllocationCreateInfo,
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
    pub shared_offsets_buffer: Subbuffer<[shaders::Offsets]>,
    pub shared_vertex_buffer: Subbuffer<[shaders::Vertex]>,
    pub shared_material_buffer: Subbuffer<[shaders::Material]>,
}

impl Scene {
    // For prototyping, constant shared buffer sizes
    /// In elements
    const SHARED_OFFSETS_BUFFER_SIZE: usize = 10000;
    /// In elements
    const SHARED_VERTEX_BUFFER_SIZE: usize = 500000;
    /// In elements
    const SHARED_MATERIAL_BUFFER_SIZE: usize = 1024;

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
                usage: vulkano::buffer::BufferUsage::STORAGE_BUFFER,
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
                usage: vulkano::buffer::BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                ..Default::default()
            },
            Self::SHARED_VERTEX_BUFFER_SIZE as u64 * std::mem::size_of::<shaders::Vertex>() as u64,
        )
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

        let rhit_descriptor_set = DescriptorSet::new(
            context.descriptor_set_allocator.clone(),
            pipeline_layout.set_layouts()[2].clone(),
            [
                WriteDescriptorSet::buffer(0, shared_offsets_buffer.clone()),
                WriteDescriptorSet::buffer(1, shared_vertex_buffer.clone()),
                WriteDescriptorSet::buffer(2, shared_material_buffer.clone()),
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
            shared_material_buffer,
        }
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
