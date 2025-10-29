use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::GpuFuture,
};

use crate::{graphics::VulkanContext, renderer::shaders};

pub mod geometry;
pub mod gltf;

pub struct Scene {
    world: std::sync::Arc<std::sync::Mutex<legion::World>>,
    resources: std::sync::Arc<std::sync::Mutex<legion::Resources>>,

    _is_dirty_for_renderer: bool,
}

// SAFETY: Scene contains legion::World which uses raw pointers internally,
// but all its data is accessed through safe abstractions and the actual
// data it manages (components) can be safely sent between threads.
// The Arc<Mutex<>> wrapping ensures thread-safe access.
unsafe impl Send for Scene {}
unsafe impl Sync for Scene {}

impl Scene {
    pub fn new() -> Self {
        let world = legion::World::default();
        let resources = legion::Resources::default();

        Self {
            world: std::sync::Arc::new(std::sync::Mutex::new(world)),
            resources: std::sync::Arc::new(std::sync::Mutex::new(resources)),

            _is_dirty_for_renderer: true,
        }
    }

    pub fn check_dirty_and_reset(&mut self) -> bool {
        if self._is_dirty_for_renderer {
            self._is_dirty_for_renderer = false;
            true
        } else {
            false
        }
    }

    pub fn world(&self) -> &Arc<std::sync::Mutex<legion::World>> {
        &self.world
    }

    /// This is sketchy, but needed to mark the scene as dirty. If
    /// editing the world, the renderer needs to know to update its structures.
    /// In the future, we might want to have a better system for tracking changes.
    pub fn world_mut(&mut self) -> &Arc<std::sync::Mutex<legion::World>> {
        self._is_dirty_for_renderer = true;

        &mut self.world
    }

    // Don't ask, long story
    pub fn world_mut_dont_mark_dirty(&mut self) -> &Arc<std::sync::Mutex<legion::World>> {
        &mut self.world
    }

    pub fn resources(&self) -> &Arc<std::sync::Mutex<legion::Resources>> {
        &self.resources
    }

    pub fn resources_mut(&mut self) -> &Arc<std::sync::Mutex<legion::Resources>> {
        self._is_dirty_for_renderer = true;

        &mut self.resources
    }

    // Again, long story
    pub fn resources_mut_dont_mark_dirty(&mut self) -> &Arc<std::sync::Mutex<legion::Resources>> {
        &mut self.resources
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

    pub fn forward(&self) -> glam::Vec3 {
        self.rotation * glam::Vec3::NEG_Z
    }

    pub fn right(&self) -> glam::Vec3 {
        self.rotation * glam::Vec3::X
    }

    pub fn up(&self) -> glam::Vec3 {
        self.rotation * glam::Vec3::Y
    }
}
