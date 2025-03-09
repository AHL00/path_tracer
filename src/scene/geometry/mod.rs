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

use crate::renderer::CustomVertex;

use super::Transform;

#[derive(Clone, Debug, PartialEq)]
/// A component representing a geometry object in the scene
pub struct Geometry {
    blas: Arc<AccelerationStructure>,
    vertex_count: u32,
    index_count: u32,
    dynamic: Option<DynamicGeometryData>,
}

#[derive(Clone, Debug, PartialEq)]
struct DynamicGeometryData {
    // Store cause the BLAS needs to be rebuilt
    vertex_buffer: Subbuffer<[CustomVertex]>,
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
    pub fn create(
        vertices: Vec<CustomVertex>,
        indices: Vec<u32>,
        context: &crate::graphics::VulkanContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create vertex buffer
        let vertex_buffer = Buffer::from_iter(
            context.memory_allocator.clone(),
            vulkano::buffer::BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER
                    | BufferUsage::STORAGE_BUFFER
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices.iter().cloned(),
        )?;

        // Create index buffer (if indices are provided)
        let (index_buffer, index_count) = if !indices.is_empty() {
            let buffer = Buffer::from_iter(
                context.memory_allocator.clone(),
                vulkano::buffer::BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER
                        | BufferUsage::STORAGE_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                indices.iter().cloned(),
            )?;
            (Some(buffer), indices.len() as u32)
        } else {
            (None, 0)
        };

        // Build BLAS for this geometry
        let blas =
            unsafe { build_blas_from_mesh(vertex_buffer.clone(), index_buffer.clone(), context) };

        Ok(Self {
            blas,
            vertex_count: vertices.len() as u32,
            index_count,
            dynamic: None,
        })
    }
}

unsafe fn build_blas_from_mesh(
    vertex_buffer: Subbuffer<[CustomVertex]>,
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
        vertex_stride: size_of::<CustomVertex>() as _,
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
        ).expect("Failed to create device layout for scratch buffer"),
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
