use std::sync::Arc;

use vulkano::{
    NonNullDeviceAddress,
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
        AccelerationStructureCreateInfo, AccelerationStructureGeometries,
        AccelerationStructureGeometryTrianglesData, AccelerationStructureType,
        BuildAccelerationStructureFlags, BuildAccelerationStructureMode, GeometryFlags,
    },
    buffer::{Buffer, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract},
    format::Format,
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    sync::GpuFuture,
};

use crate::renderer::{
    CustomVertex,
    shaders::{self, Vertex},
};

#[derive(Clone)]
/// A component representing a geometry object in the scene
pub struct Geometry {
    pub blas: Arc<AccelerationStructure>,
    pub vertex_count: u32,
    pub index_count: u32,
    pub dynamic: Option<DynamicGeometryData>,

    _shared_vertex_buffer_offset: u64,
    _shared_vertex_buffer_count: u64,
    _shared_index_buffer_offset: u64,
    _shared_index_buffer_count: u64,
}

#[derive(Clone)]
pub struct DynamicGeometryData {
    index_buffer: Option<Subbuffer<[u32]>>,
}

impl Geometry {
    pub fn get_blas(&self) -> Arc<AccelerationStructure> {
        self.blas.clone()
    }

    pub fn get_blas_device_address(&self) -> NonNullDeviceAddress {
        self.blas.device_address()
    }

    /// Creates a new Geometry object with its own vertex and index buffers.
    /// Geometry is linked to a specific scene due to shared buffers.
    pub fn create(
        geometry_ident: String,
        vertices: Vec<CustomVertex>,
        indices: Vec<u32>,
        scene: &mut crate::scene::Scene,
        context: &crate::graphics::VulkanContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Check if the geometry already exists
        if scene.get_geometry(&geometry_ident).is_some() {
            log::debug!("Geometry {} already exists, returning existing geometry", geometry_ident);
            return Ok(scene.get_geometry(&geometry_ident).unwrap().clone());
        }

        // log::debug!("Creating geometry [{}] with {} vertices and {} indices", geometry_ident, vertices.len(), indices.len());

        // Upload vertex buffer
        let (vertices_start, vertices_end) = scene.add_vertices(
            &vertices
                .iter()
                .map(|v| Vertex::from_custom_vertex(*v))
                .collect::<Vec<_>>(),
            context,
        );

        let vertices_buffer = scene.get_vertices_subbuffer(vertices_start, vertices_end);

        // Upload index buffer
        let (index_start, index_end) = scene.add_indices(&indices, context);

        let index_buffer = scene.get_indices_subbuffer(index_start, index_end);

        // Build BLAS for this geometry
        let blas = unsafe { build_blas_from_mesh(vertices_buffer, Some(index_buffer), context) };

        let geometry = Self {
            blas,
            vertex_count: vertices.len() as u32,
            index_count: indices.len() as u32,
            _shared_vertex_buffer_offset: vertices_start,
            _shared_vertex_buffer_count: vertices_end - vertices_start,
            _shared_index_buffer_offset: index_start,
            _shared_index_buffer_count: index_end - index_start,
            dynamic: None,
        };

        // Add the geometry to the scene
        scene.add_geometry(geometry_ident, geometry.clone());
        
        Ok(geometry)
    }

    /// Returns (vertex_offset, vertex_count)
    pub fn get_shared_vertex_buffer_offsets(&self) -> (u64, u64) {
        (
            self._shared_vertex_buffer_offset,
            self._shared_vertex_buffer_count,
        )
    }

    /// Returns (index_offset, index_count)
    pub fn get_shared_index_buffer_offsets(&self) -> (u64, u64) {
        (
            self._shared_index_buffer_offset,
            self._shared_index_buffer_count,
        )
    }

    pub fn find_model_vertex_bounds(
        vertices: &[CustomVertex]
    ) -> (glam::Vec3, glam::Vec3) {
        let mut min = glam::Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = glam::Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for vertex in vertices {
            min = glam::Vec3::min(min, vertex.position);
            max = glam::Vec3::max(max, vertex.position);
        }

        (min, max)
    }
}

unsafe fn build_blas_from_mesh(
    vertex_buffer: Subbuffer<[shaders::Vertex]>,
    index_buffer: Option<Subbuffer<[u32]>>,
    context: &crate::graphics::VulkanContext,
) -> Arc<AccelerationStructure> {
    let primitive_count = if index_buffer.is_some() {
        index_buffer.as_ref().unwrap().len() / 3
    } else {
        vertex_buffer.len() / 3
    };

    let as_geometry_triangles_data = AccelerationStructureGeometryTrianglesData {
        // transform_data: todo!(),
        max_vertex: vertex_buffer.len() as _,
        vertex_data: Some(vertex_buffer.into_bytes()),
        vertex_stride: size_of::<shaders::Vertex>() as _,
        flags: GeometryFlags::OPAQUE,
        index_data: index_buffer.map(|buffer| IndexBuffer::U32(buffer)),
        // TODO: Transform here???
        transform_data: None,
        ..AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT)
    };

    let geometries = AccelerationStructureGeometries::Triangles(vec![as_geometry_triangles_data]);

    unsafe {
        build_acceleration_structure_common(
            geometries,
            primitive_count,
            AccelerationStructureType::BottomLevel,
            BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
            context,
        )
    }
}

/// Builds an AS and blocks until it is built.
unsafe fn build_acceleration_structure_common(
    geometries: AccelerationStructureGeometries,
    primitive_count: u64,
    ty: AccelerationStructureType,
    flags: BuildAccelerationStructureFlags,
    context: &crate::graphics::VulkanContext,
) -> Arc<AccelerationStructure> {
    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        mode: BuildAccelerationStructureMode::Build,
        flags,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let as_build_sizes_info = context
        .device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &as_build_geometry_info,
            &[primitive_count as u32],
        )
        .unwrap();

    let min_as_scratch_offset_alignment = context
        .device
        .physical_device()
        .properties()
        .min_acceleration_structure_scratch_offset_alignment
        .expect("Failed to get min acceleration structure scratch offset alignment")
        as u64;

    // We create a new scratch buffer for each acceleration structure for simplicity. You may want
    // to reuse scratch buffers if you need to build many acceleration structures.
    let scratch_buffer = Buffer::new(
        context.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
        DeviceLayout::from_size_alignment(
            as_build_sizes_info.build_scratch_size,
            min_as_scratch_offset_alignment,
        )
        .expect("Failed to create device layout for scratch buffer"),
    )
    .unwrap();

    as_build_geometry_info.scratch_data = Some(scratch_buffer.into());

    let as_create_info = AccelerationStructureCreateInfo {
        ty,
        ..AccelerationStructureCreateInfo::new(
            Buffer::new_slice::<u8>(
                context.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE
                        | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
                as_build_sizes_info.acceleration_structure_size,
            )
            .unwrap(),
        )
    };

    let acceleration =
        unsafe { AccelerationStructure::new(context.device.clone(), as_create_info) }.unwrap();

    as_build_geometry_info.dst_acceleration_structure = Some(acceleration.clone());

    let as_build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count: primitive_count as u32,
        ..Default::default()
    };

    // For simplicity, we build a single command buffer that builds the acceleration structure,
    // then waits for its execution to complete.
    let mut builder = AutoCommandBufferBuilder::primary(
        context.command_buffer_allocator.clone(),
        context.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    unsafe {
        builder
            .build_acceleration_structure(
                as_build_geometry_info,
                std::iter::once(as_build_range_info).collect(),
            )
            .unwrap()
    };

    builder
        .build()
        .unwrap()
        .execute(context.queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    acceleration
}
