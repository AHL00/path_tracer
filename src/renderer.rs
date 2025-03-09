use std::sync::Arc;

use vulkano::{
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
        AccelerationStructureCreateInfo, AccelerationStructureGeometries,
        AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
        AccelerationStructureGeometryTrianglesData, AccelerationStructureInstance,
        AccelerationStructureType, BuildAccelerationStructureFlags, BuildAccelerationStructureMode,
    },
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
    },
    format::Format,
    image::{Image, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::vertex_input::Vertex,
        layout::PipelineLayoutCreateInfo,
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
            ShaderBindingTable,
        },
    },
    shader::ShaderStages,
    sync::GpuFuture,
};

mod shaders {
    pub mod raygen {
        vulkano_shaders::shader! {
            ty: "raygen",
            path: "src/shaders/rgen.glsl",
            vulkan_version: "1.3",
        }
    }

    pub mod miss {
        vulkano_shaders::shader! {
            ty: "miss",
            path: "src/shaders/rmiss.glsl",
            vulkan_version: "1.3",
        }
    }

    pub mod closest_hit {
        vulkano_shaders::shader! {
            ty: "closesthit",
            path: "src/shaders/rchit.glsl",
            vulkan_version: "1.3",
        }
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

pub struct Renderer {
    pub descriptor_set: Arc<DescriptorSet>,
    pub swapchain_image_sets: Vec<(Arc<ImageView>, Arc<DescriptorSet>)>,
    pub pipeline_layout: Arc<PipelineLayout>,
    pub pipeline: Arc<RayTracingPipeline>,
    pub shader_binding_table: ShaderBindingTable,

    // Just a base blas for the tlas
    // Idk how to make a tlas without a blas for now
    _blas: Arc<AccelerationStructure>,
    pub tlas: Arc<AccelerationStructure>,
    pub scene: crate::scene::Scene,
}

impl Renderer {
    pub fn new(context: &crate::graphics::VulkanContext) -> Self {
        let scene = crate::scene::Scene::new();

        let pipeline_layout = PipelineLayout::new(
            context.device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![
                    DescriptorSetLayout::new(
                        context.device.clone(),
                        DescriptorSetLayoutCreateInfo {
                            bindings: [
                                (
                                    0,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::RAYGEN,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::AccelerationStructure,
                                        )
                                    },
                                ),
                                (
                                    1,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::RAYGEN,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::UniformBuffer,
                                        )
                                    },
                                ),
                            ]
                            .into_iter()
                            .collect(),
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                    DescriptorSetLayout::new(
                        context.device.clone(),
                        DescriptorSetLayoutCreateInfo {
                            bindings: [(
                                0,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::RAYGEN,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::StorageImage,
                                    )
                                },
                            )]
                            .into_iter()
                            .collect(),
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                ],
                ..Default::default()
            },
        )
        .unwrap();

        let pipeline = {
            let raygen = shaders::raygen::load(context.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let closest_hit = shaders::closest_hit::load(context.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let miss = shaders::miss::load(context.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            // Make a list of the shader stages that the pipeline will have.
            let stages = [
                PipelineShaderStageCreateInfo::new(raygen),
                PipelineShaderStageCreateInfo::new(miss),
                PipelineShaderStageCreateInfo::new(closest_hit),
            ];

            // Define the shader groups that will eventually turn into the shader binding table.
            // The numbers are the indices of the stages in the `stages` array.
            let groups = [
                RayTracingShaderGroupCreateInfo::General { general_shader: 0 },
                RayTracingShaderGroupCreateInfo::General { general_shader: 1 },
                RayTracingShaderGroupCreateInfo::TrianglesHit {
                    closest_hit_shader: Some(2),
                    any_hit_shader: None,
                },
            ];

            RayTracingPipeline::new(
                context.device.clone(),
                None,
                RayTracingPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    groups: groups.into_iter().collect(),
                    max_pipeline_ray_recursion_depth: 1,
                    ..RayTracingPipelineCreateInfo::layout(pipeline_layout.clone())
                },
            )
            .unwrap()
        };

        log::info!("Ray tracing pipeline created");

        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 4.0 / 3.0, 0.01, 100.0);
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.0, 1.0),
            glam::Vec3::new(0.0, 0.0, 0.0),
            glam::Vec3::new(0.0, -1.0, 0.0),
        );

        let uniform_buffer = Buffer::from_data(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            shaders::raygen::Camera {
                view_proj: (proj * view).to_cols_array_2d(),
                view_inverse: view.inverse().to_cols_array_2d(),
                proj_inverse: proj.inverse().to_cols_array_2d(),
            },
        )
        .unwrap();

        log::info!("Uniform buffer created");

        let vertices = [
            MyVertex {
                position: [-0.5, -0.25, 0.0],
            },
            MyVertex {
                position: [0.0, 0.5, 0.0],
            },
            MyVertex {
                position: [0.25, -0.1, 0.0],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        let blas = unsafe { build_acceleration_structure_triangles(vertex_buffer, context) };

        let tlas = unsafe {
            build_top_level_acceleration_structure(
                vec![AccelerationStructureInstance {
                    acceleration_structure_reference: blas.device_address().into(),
                    ..Default::default()
                }],
                context,
            )
        };

        log::info!("Top-level acceleration structure created");

        let descriptor_set = DescriptorSet::new(
            context.descriptor_set_allocator.clone(),
            pipeline_layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::acceleration_structure(0, tlas.clone()),
                WriteDescriptorSet::buffer(1, uniform_buffer.clone()),
            ],
            [],
        )
        .unwrap();

        log::info!("Descriptor set created");

        let swapchain_image_sets = context
            .images
            .iter()
            .map(|image| {
                let image_view = ImageView::new_default(image.clone()).unwrap();
                let descriptor_set = DescriptorSet::new(
                    context.descriptor_set_allocator.clone(),
                    pipeline_layout.set_layouts()[1].clone(),
                    [WriteDescriptorSet::image_view(0, image_view.clone())],
                    [],
                )
                .unwrap();

                (image_view, descriptor_set)
            })
            .collect();

        log::info!("Swapchain image descriptor sets created");

        let shader_binding_table =
            ShaderBindingTable::new(context.memory_allocator.clone(), &pipeline).unwrap();

        log::info!("Shader binding table created");

        Self {
            pipeline_layout,
            pipeline,
            descriptor_set,
            swapchain_image_sets,
            shader_binding_table,

            _blas: blas.clone(),
            tlas,
            scene,
        }
    }

    pub fn resize(
        &mut self,
        context: &crate::graphics::VulkanContext,
        new_images: &Vec<Arc<Image>>,
    ) {
        // Recreate the swapchain images and their descriptor sets
        self.swapchain_image_sets = new_images
            .iter()
            .map(|image| {
                let image_view = ImageView::new_default(image.clone()).unwrap();
                let descriptor_set = DescriptorSet::new(
                    context.descriptor_set_allocator.clone(),
                    self.pipeline_layout.set_layouts()[1].clone(),
                    [WriteDescriptorSet::image_view(0, image_view.clone())],
                    [],
                )
                .unwrap();

                (image_view, descriptor_set)
            })
            .collect();
    }

    pub fn record_commands(
        &self,
        image_index: u32,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::RayTracing,
                self.pipeline_layout.clone(),
                0,
                vec![
                    self.descriptor_set.clone(),
                    self.swapchain_image_sets[image_index as usize].1.clone(),
                ],
            )
            .unwrap()
            .bind_pipeline_ray_tracing(self.pipeline.clone())
            .unwrap();

        let extent = self.swapchain_image_sets[0].0.image().extent();

        unsafe { builder.trace_rays(self.shader_binding_table.addresses().clone(), extent) }
            .unwrap();
    }
}

unsafe fn build_top_level_acceleration_structure(
    as_instances: Vec<AccelerationStructureInstance>,
    context: &crate::graphics::VulkanContext,
) -> Arc<AccelerationStructure> {
    let primitive_count = as_instances.len() as u32;

    // For an empty TLAS, we still need a buffer (even if empty)
    let instance_buffer = if !as_instances.is_empty() {
        Buffer::from_iter(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            as_instances,
        )
        .unwrap()
    } else {
        // Create a minimal empty buffer when no instances are provided
        Buffer::new_slice::<AccelerationStructureInstance>(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            1, // Minimum size
        )
        .unwrap()
    };

    let as_geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
        AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer)),
    );

    let geometries = AccelerationStructureGeometries::Instances(as_geometry_instances_data);

    unsafe {
        build_acceleration_structure_common(
            geometries,
            primitive_count,
            AccelerationStructureType::TopLevel,
            context,
        )
    }
}

/// A helper function to build a acceleration structure and wait for its completion.
///
/// # Safety
///
/// - If you are referencing a bottom-level acceleration structure in a top-level acceleration
///   structure, you must ensure that the bottom-level acceleration structure is kept alive.
unsafe fn build_acceleration_structure_common(
    geometries: AccelerationStructureGeometries,
    primitive_count: u32,
    ty: AccelerationStructureType,
    context: &crate::graphics::VulkanContext,
) -> Arc<AccelerationStructure> {
    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        mode: BuildAccelerationStructureMode::Build,
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let as_build_sizes_info = context
        .device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &as_build_geometry_info,
            &[primitive_count],
        )
        .unwrap();

    // We create a new scratch buffer for each acceleration structure for simplicity. You may want
    // to reuse scratch buffers if you need to build many acceleration structures.
    let scratch_buffer = Buffer::new_slice::<u8>(
        context.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
        as_build_sizes_info.build_scratch_size,
    )
    .unwrap();

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
    as_build_geometry_info.scratch_data = Some(scratch_buffer);

    let as_build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count,
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

unsafe fn build_acceleration_structure_triangles(
    vertex_buffer: Subbuffer<[MyVertex]>,
    context: &crate::graphics::VulkanContext,
) -> Arc<AccelerationStructure> {
    let primitive_count = (vertex_buffer.len() / 3) as u32;
    let as_geometry_triangles_data = AccelerationStructureGeometryTrianglesData {
        max_vertex: vertex_buffer.len() as _,
        vertex_data: Some(vertex_buffer.into_bytes()),
        vertex_stride: size_of::<MyVertex>() as _,
        ..AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT)
    };

    let geometries = AccelerationStructureGeometries::Triangles(vec![as_geometry_triangles_data]);

    unsafe {
        build_acceleration_structure_common(
            geometries,
            primitive_count,
            AccelerationStructureType::BottomLevel,
            context,
        )
    }
}
