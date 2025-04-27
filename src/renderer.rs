use std::{fs::File, path::Path, ptr::NonNull, sync::Arc, u32};

use bytes::Buf;
use image::{
    DynamicImage, ImageFormat, Pixel, Rgb, Rgb32FImage, RgbImage, Rgba, buffer::ConvertBuffer,
};

use nrd_sys::NrdDenoiser;
use rand::Rng;
use vulkano::{
    Packed24_8,
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
        AccelerationStructureCreateInfo, AccelerationStructureGeometries,
        AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
        AccelerationStructureGeometryTrianglesData, AccelerationStructureInstance,
        AccelerationStructureType, BuildAccelerationStructureFlags, BuildAccelerationStructureMode,
        GeometryFlags,
    },
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyImageToBufferInfo,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        layout::{
            DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutBinding,
            DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, DescriptorType,
        },
    },
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageLayout, ImageSubresourceLayers, ImageUsage,
        sampler::{self, Sampler},
        view::ImageView,
    },
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::vertex_input::Vertex,
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
            ShaderBindingTable,
        },
    },
    shader::ShaderStages,
    sync::GpuFuture,
};

use crate::{
    camera::Camera,
    graphics::{Texture, VulkanContext},
    material::Material,
    scene::{Transform, geometry::Geometry},
};

pub mod shaders {
    use super::CustomVertex;

    pub(super) mod raygen {
        vulkano_shaders::shader! {
            ty: "raygen",
            path: "src/shaders/rgen.glsl",
            vulkan_version: "1.3",
        }
    }

    pub(super) mod miss {
        vulkano_shaders::shader! {
            ty: "miss",
            path: "src/shaders/rmiss.glsl",
            vulkan_version: "1.3",
        }
    }

    pub(super) mod closest_hit {
        vulkano_shaders::shader! {
            ty: "closesthit",
            path: "src/shaders/rchit.glsl",
            vulkan_version: "1.3",
        }
    }

    pub type Offsets = closest_hit::Offsets;
    pub type Material = closest_hit::Material;
    pub type Vertex = closest_hit::Vertex;

    impl From<super::CustomVertex> for Vertex {
        fn from(v: super::CustomVertex) -> Self {
            Self {
                position: [v.position.x, v.position.y, v.position.z].into(),
                normal: [v.normal.x, v.normal.y, v.normal.z].into(),
                uv: v.tex_coords.into(),
                padding: [0.0; 2],
            }
        }
    }

    impl Vertex {
        pub fn from_custom_vertex(v: CustomVertex) -> Self {
            Self {
                position: [v.position.x, v.position.y, v.position.z].into(),
                normal: [v.normal.x, v.normal.y, v.normal.z].into(),
                uv: v.tex_coords.into(),
                padding: [0.0; 2],
            }
        }
    }
}

#[derive(BufferContents, Vertex, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
/// NOTE: Position must be the first field as AccelerationStructure creation requires position and
/// it reads data from the start of each vertex.
pub struct CustomVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: glam::Vec3,

    #[format(R32G32B32_SFLOAT)]
    pub normal: glam::Vec3,

    #[format(R32G32_SFLOAT)]
    pub tex_coords: glam::Vec2,
}

pub struct PushConstantRequirements {
    closest_hit: PushConstantRange,
    raygen: PushConstantRange,
}

pub struct Renderer {
    pub rgen_descriptor_set: Arc<DescriptorSet>,

    pub pipeline_layout: Arc<PipelineLayout>,
    pub pipeline: Arc<RayTracingPipeline>,
    pub shader_binding_table: ShaderBindingTable,
    pub push_constant_requirements: PushConstantRequirements,

    pub albedo_texture: Arc<ImageView>,
    pub normal_texture: Arc<ImageView>,
    pub depth_texture: Arc<ImageView>,
    render_resolution: [u32; 2],
    pub render_textures_descriptor_set: Arc<DescriptorSet>,

    pub z_near: f32,
    pub z_far: f32,

    pub tlas: Arc<AccelerationStructure>,

    // Bindless textures descriptor set
    pub bindless_textures_descriptor_set: Arc<DescriptorSet>,
    /// This is basically a CPU side
    /// version of the bindless textures descriptor set.
    /// It's used when updating the descriptor set because
    /// the entire array must be updated at once.
    bindless_textures: Vec<(Arc<ImageView>, Arc<Sampler>)>,
    bindless_texture_index: u32,
    // TODO: These should be weak pointers
    // to avoid keeping them alive
    // Also in geometry and material
    pub loaded_textures_map: std::collections::HashMap<String, Texture>,

    pub scene: crate::scene::Scene,

    pub camera: crate::camera::Camera,
    pub samples_per_pixel: u32,

    pub oidn_cpu: oidn::Device,

    pub prev_cam_state: crate::camera::Camera,
    pub resized_dirty: bool,
    pub accumulated_count: u32,
    pub accumulation_limit: u32,
}

impl Renderer {
    const MAX_TEXTURE_COUNT: u32 = 5000; // Maximum number of textures
    const RAY_RECURSION_DEPTH: u32 = 2;

    pub fn new(context: &crate::graphics::VulkanContext, render_resolution: [u32; 2]) -> Self {
        let push_constant_requirements = PushConstantRequirements {
            closest_hit: shaders::closest_hit::load(context.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap()
                .info()
                .push_constant_requirements
                .expect("Failed to get push constant requirements"),
            raygen: shaders::raygen::load(context.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap()
                .info()
                .push_constant_requirements
                .expect("Failed to get push constant requirements"),
        };

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
                                        stages: ShaderStages::RAYGEN
                                            | ShaderStages::CLOSEST_HIT
                                            | ShaderStages::ANY_HIT
                                            | ShaderStages::INTERSECTION,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::AccelerationStructure,
                                        )
                                    },
                                ),
                                // (
                                //     1,
                                //     DescriptorSetLayoutBinding {
                                //         stages: ShaderStages::RAYGEN,
                                //         ..DescriptorSetLayoutBinding::descriptor_type(
                                //             DescriptorType::UniformBuffer,
                                //         )
                                //     },
                                // ),
                                // (
                                //     2,
                                //     DescriptorSetLayoutBinding {
                                //         stages: ShaderStages::RAYGEN,
                                //         ..DescriptorSetLayoutBinding::descriptor_type(
                                //             DescriptorType::UniformBuffer,
                                //         )
                                //     },
                                // ),
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
                            bindings: [
                                // Albedo
                                (
                                    0,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::RAYGEN,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::StorageImage,
                                        )
                                    },
                                ),
                                // Normal
                                (
                                    1,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::RAYGEN,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::StorageImage,
                                        )
                                    },
                                ),
                                // Depth
                                (
                                    2,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::RAYGEN,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::StorageImage,
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
                    // layout(binding = 0, set = 2) readonly buffer offsets_buffer {
                    //     Offsets offsets_array[];
                    // };

                    // layout(binding = 1, set = 2) readonly buffer vertex_buffer {
                    //     Vertex vertices[];
                    // };

                    // layout(binding = 2, set = 2) readonly buffer material_buffer {
                    //     Material materials[];
                    // };

                    // layout(binding = 3, set = 2) readonly buffer index_buffer {
                    //     uint indices[];
                    // };
                    DescriptorSetLayout::new(
                        context.device.clone(),
                        DescriptorSetLayoutCreateInfo {
                            bindings: [
                                (
                                    0,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::CLOSEST_HIT
                                            | ShaderStages::ANY_HIT
                                            | ShaderStages::INTERSECTION,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::StorageBuffer,
                                        )
                                    },
                                ),
                                (
                                    1,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::CLOSEST_HIT
                                            | ShaderStages::ANY_HIT
                                            | ShaderStages::INTERSECTION,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::StorageBuffer,
                                        )
                                    },
                                ),
                                (
                                    2,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::CLOSEST_HIT
                                            | ShaderStages::ANY_HIT
                                            | ShaderStages::INTERSECTION,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::StorageBuffer,
                                        )
                                    },
                                ),
                                (
                                    3,
                                    DescriptorSetLayoutBinding {
                                        stages: ShaderStages::CLOSEST_HIT
                                            | ShaderStages::ANY_HIT
                                            | ShaderStages::INTERSECTION,
                                        ..DescriptorSetLayoutBinding::descriptor_type(
                                            DescriptorType::StorageBuffer,
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
                    // Bindless texture descriptor set
                    DescriptorSetLayout::new(
                        context.device.clone(),
                        DescriptorSetLayoutCreateInfo {
                            bindings: [(
                                0,
                                DescriptorSetLayoutBinding {
                                    stages: ShaderStages::CLOSEST_HIT | ShaderStages::ANY_HIT,
                                    descriptor_count: Self::MAX_TEXTURE_COUNT,
                                    binding_flags: DescriptorBindingFlags::UPDATE_AFTER_BIND
                                        | DescriptorBindingFlags::PARTIALLY_BOUND,
                                    ..DescriptorSetLayoutBinding::descriptor_type(
                                        DescriptorType::CombinedImageSampler,
                                    )
                                },
                            )]
                            .into_iter()
                            .collect(),
                            flags: DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                ],
                push_constant_ranges: vec![
                    push_constant_requirements.closest_hit,
                    push_constant_requirements.raygen,
                ],
                ..Default::default()
            },
        )
        .unwrap();

        let scene = crate::scene::Scene::new(context, &pipeline_layout);

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
                    max_pipeline_ray_recursion_depth: Self::RAY_RECURSION_DEPTH,
                    ..RayTracingPipelineCreateInfo::layout(pipeline_layout.clone())
                },
            )
            .unwrap()
        };

        log::info!("Ray tracing pipeline created");

        // For reference
        // let proj = glam::Mat4::perspective_rh(90.0_f32.to_radians(), 1280.0 / 720.0, 0.01, 10000.0);
        // let view = glam::Mat4::look_to_rh(
        //     // glam::Vec3::new(-3.0, 2.0, -2.0),
        //     glam::Vec3::new(-0.0, 1.0, 2.2),
        //     glam::Vec3::new(0.0, 0.0, -1.0),
        //     glam::Vec3::new(0.0, -1.0, 0.0),
        // );

        let tlas = unsafe { build_top_level_acceleration_structure(vec![], context) };

        log::info!("Top-level acceleration structure created");

        let rgen_descriptor_set = DescriptorSet::new(
            context.descriptor_set_allocator.clone(),
            pipeline_layout.set_layouts()[0].clone(),
            [WriteDescriptorSet::acceleration_structure(0, tlas.clone())],
            [],
        )
        .unwrap();

        log::info!("Descriptor sets created");

        let shader_binding_table =
            ShaderBindingTable::new(context.memory_allocator.clone(), &pipeline).unwrap();

        log::info!("Shader binding table created");

        let bindless_textures_descriptor_set = DescriptorSet::new(
            context.descriptor_set_allocator_update_after_bind.clone(),
            pipeline_layout.set_layouts().get(3).unwrap().clone(),
            [],
            [],
        )
        .unwrap();

        // Recreate the render texture and its descriptor set
        let albedo_texture = ImageView::new_default(
            Image::new(
                context.memory_allocator.clone(),
                ImageCreateInfo {
                    format: vulkano::format::Format::R32G32B32A32_SFLOAT,
                    extent: [render_resolution[0], render_resolution[1], 1].into(),
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let normal_texture = ImageView::new_default(
            Image::new(
                context.memory_allocator.clone(),
                ImageCreateInfo {
                    format: vulkano::format::Format::R32G32B32A32_SFLOAT,
                    extent: [render_resolution[0], render_resolution[1], 1].into(),
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let depth_texture = ImageView::new_default(
            Image::new(
                context.memory_allocator.clone(),
                ImageCreateInfo {
                    format: Format::R32_SFLOAT,
                    extent: [render_resolution[0], render_resolution[1], 1].into(),
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let render_textures_descriptor_set = DescriptorSet::new(
            context.descriptor_set_allocator.clone(),
            pipeline_layout.set_layouts()[1].clone(),
            [
                WriteDescriptorSet::image_view(0, albedo_texture.clone()),
                WriteDescriptorSet::image_view(1, normal_texture.clone()),
                WriteDescriptorSet::image_view(2, depth_texture.clone()),
            ],
            [],
        )
        .unwrap();

        let oidn_cpu = oidn::Device::cpu();

        let nrd =
            nrd_sys::vulkano::VulkanoNrdInstance::new(&[NrdDenoiser::REBLUR_DIFFUSE]).unwrap();

        let mut res = Self {
            pipeline_layout,
            pipeline,
            rgen_descriptor_set,
            push_constant_requirements,

            albedo_texture,
            normal_texture,
            depth_texture,
            render_resolution,
            // TODO: Configurable?
            render_textures_descriptor_set,

            shader_binding_table,
            bindless_textures_descriptor_set,
            bindless_textures: vec![],
            bindless_texture_index: 0,
            loaded_textures_map: std::collections::HashMap::new(),

            tlas,
            scene,

            samples_per_pixel: 1,
            camera: Camera::default(),

            oidn_cpu,

            z_near: 0.1,
            z_far: 100.0,

            prev_cam_state: Camera::default(),
            resized_dirty: false,
            accumulated_count: 0,
            accumulation_limit: u32::MAX,
        };

        res.resize_render_buffer(context, render_resolution);

        res
    }

    pub fn render_resolution(&self) -> [u32; 2] {
        self.render_resolution
    }

    pub fn resize_render_buffer(
        &mut self,
        context: &crate::graphics::VulkanContext,
        new_resolution: [u32; 2],
    ) {
        // Recreate the render texture and its descriptor set
        self.albedo_texture = ImageView::new_default(
            Image::new(
                context.memory_allocator.clone(),
                ImageCreateInfo {
                    format: vulkano::format::Format::R32G32B32A32_SFLOAT,
                    extent: [new_resolution[0], new_resolution[1], 1].into(),
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        self.normal_texture = ImageView::new_default(
            Image::new(
                context.memory_allocator.clone(),
                ImageCreateInfo {
                    format: vulkano::format::Format::R32G32B32A32_SFLOAT,
                    extent: [new_resolution[0], new_resolution[1], 1].into(),
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        self.depth_texture = ImageView::new_default(
            Image::new(
                context.memory_allocator.clone(),
                ImageCreateInfo {
                    format: Format::R32_SFLOAT,
                    extent: [new_resolution[0], new_resolution[1], 1].into(),
                    usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        self.render_resolution = new_resolution;

        self.render_textures_descriptor_set = DescriptorSet::new(
            context.descriptor_set_allocator.clone(),
            self.pipeline_layout.set_layouts()[1].clone(),
            [
                WriteDescriptorSet::image_view(0, self.albedo_texture.clone()),
                WriteDescriptorSet::image_view(1, self.normal_texture.clone()),
                WriteDescriptorSet::image_view(2, self.depth_texture.clone()),
            ],
            [],
        )
        .unwrap();

        // Resize window to match the render aspect
        // Resize handler will handle the swapchain resize
        // For now, take the height of the window as the constant
        // and calculate the width based on the render aspect
        let render_aspect = self.render_resolution[0] as f32 / self.render_resolution[1] as f32;
        let window_size = context.winit.inner_size();
        let height = window_size.height as f32;
        let width = (height * render_aspect) as u32;

        let resize_res = context
            .winit
            .request_inner_size(winit::dpi::LogicalSize::new(width as f64, height as f64));

        if let None = resize_res {
            log::warn!("Failed to resize window to match render aspect ratio");
        }

        self.resized_dirty = true;
    }

    fn copy_image_to_host_vec(
        image_view: Arc<ImageView>,
        context: &crate::graphics::VulkanContext,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let image = image_view.image();
        // This should be a length 1 slice according to the docs
        let memory_requirements = image.memory_requirements()[0];

        let staging_buffer = Buffer::new(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST,
                ..Default::default()
            },
            memory_requirements.layout,
        )?;

        let staging_subbuffer = Subbuffer::new(staging_buffer);

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            context.command_buffer_allocator.clone(),
            context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        command_buffer_builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            staging_subbuffer.clone(),
        ))?;

        let copy_future = command_buffer_builder
            .build()?
            .execute(context.queue.clone())?;

        copy_future.then_signal_fence_and_flush()?.wait(None)?;

        let data = staging_subbuffer.read()?;
        Ok(data.to_vec())
    }

    fn copy_image_to_host_vec_f32(
        image_view: Arc<ImageView>,
        context: &crate::graphics::VulkanContext,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let data_u8 = Self::copy_image_to_host_vec(image_view, context)?;

        if data_u8.len() % size_of::<f32>() != 0 {
            return Err(format!(
                "Byte data length ({}) is not a multiple of f32 size ({})",
                data_u8.len(),
                size_of::<f32>()
            )
            .into());
        }

        let num_floats = data_u8.len() / size_of::<f32>();
        // Vec<f32> guarantees correct alignment.
        let mut data_f32_vec = Vec::<f32>::with_capacity(num_floats);

        unsafe {
            std::ptr::copy_nonoverlapping(
                data_u8.as_ptr() as *const f32,
                data_f32_vec.as_mut_ptr(),
                num_floats,
            );

            data_f32_vec.set_len(num_floats);
        }

        Ok(data_f32_vec)
    }

    pub fn save_render_texture_to_file(
        &self,
        context: &crate::graphics::VulkanContext,
        file_path: &Path,
        format: image::ImageFormat,
        oidn: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // If image exists, error
        if file_path.exists() {
            return Err("File already exists".into());
        }

        // Create an RGBA8 image from the data
        let albedo = image::Rgba32FImage::from_raw(
            self.render_resolution[0],
            self.render_resolution[1],
            Self::copy_image_to_host_vec_f32(self.albedo_texture.clone(), context)?,
        )
        .ok_or("Failed to create image from data")?;

        let mut file = File::create(file_path)?;

        let albedo = if oidn {
            let normal = image::Rgba32FImage::from_raw(
                self.render_resolution[0],
                self.render_resolution[1],
                Self::copy_image_to_host_vec_f32(self.normal_texture.clone(), context)?,
            )
            .ok_or("Failed to create image from data")?;

            // Convert normal and albedo to RGB8
            let normal = DynamicImage::from(normal).into_rgb32f();
            let albedo = DynamicImage::from(albedo).into_rgb32f();

            let mut oidn_ray_tracing = oidn::RayTracing::new(&self.oidn_cpu);

            oidn_ray_tracing.albedo_normal(albedo.as_raw(), normal.as_raw());
            oidn_ray_tracing.image_dimensions(
                self.render_resolution[0] as usize,
                self.render_resolution[1] as usize,
            );
            oidn_ray_tracing.filter_quality(oidn::Quality::High);

            let mut albedo_raw = albedo.into_raw();

            oidn_ray_tracing
                .filter_in_place(&mut albedo_raw)
                .map_err(|e| format!("Failed to filter image: {:?}", e))?;

            let albedo = Rgb32FImage::from_raw(
                self.render_resolution[0],
                self.render_resolution[1],
                albedo_raw,
            )
            .ok_or("Failed to create image from data")?;

            // Convert to RGB8
            let albedo = DynamicImage::from(albedo).into_rgb8();

            albedo
        } else {
            let albedo = DynamicImage::from(albedo).into_rgb8();

            albedo
        };

        albedo
            .write_to(&mut file, format)
            .map_err(|e| format!("Failed to write image to file: {}", e))?;

        log::info!("Saved render texture to file: {:?}", file_path);

        Ok(())
    }

    // TODO This is no longer required, maybe
    // pub fn resize_window_swapchain(
    //     &mut self,
    //     context: &crate::graphics::VulkanContext,
    //     new_images: &Vec<Arc<Image>>,
    // ) {
    //     let current_swap_extent: [u32; 2] = context.swapchain.image_extent();
    //     let window_size: [u32; 2] = context.winit.inner_size().into();

    //     // Check if the window size is different from the swapchain size
    //     if current_swap_extent[0] == window_size[0] && current_swap_extent[1] == window_size[1] {
    //         return;
    //     }

    //     // If so, resize, but constrain it to the render aspect
    //     let render_aspect = self.render_resolution[0] as f32 / self.render_resolution[1] as f32;

    //     let new_width = (window_size[1] as f32 * render_aspect) as u32;
    //     let new_height = window_size[1];

    //     // Resize window to match the render aspect
    //     let resize_res = context
    //         .winit
    //         .request_inner_size(winit::dpi::LogicalSize::new(new_width, new_height));

    //     if let None = resize_res {
    //         log::warn!("Failed to resize window to match render aspect ratio");
    //     }

    //     self.resized_dirty = true;
    // }

    /// Update the descriptor set with self.tlas and self.uniform_buffer
    pub fn update_descriptor_set(&mut self, context: &crate::graphics::VulkanContext) {
        let descriptor_set = DescriptorSet::new(
            context.descriptor_set_allocator.clone(),
            self.pipeline_layout.set_layouts()[0].clone(),
            [WriteDescriptorSet::acceleration_structure(
                0,
                self.tlas.clone(),
            )],
            [],
        )
        .unwrap();

        self.rgen_descriptor_set = descriptor_set;
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
        context: &VulkanContext,
    ) -> u32 {
        let index = self.bindless_texture_index;
        self.bindless_textures
            .push((image.clone(), sampler.clone()));

        let write_descriptor_set =
            WriteDescriptorSet::image_view_sampler_array(0, 0, self.bindless_textures.clone());

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

    pub fn record_commands(
        &mut self,
        image_index: u32,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        context: &crate::graphics::VulkanContext,
    ) {
        let mut blas_instances = vec![];

        use legion::IntoQuery;
        let mut query = <(&mut Geometry, &mut Transform, &Material)>::query();
        let mut offsets_vec = vec![];
        // TODO: If the number of meshes doesn't change, we can just update the TLAS
        for (geometry, transform, material) in
            query.iter_mut(self.scene.world_mut_dont_mark_dirty())
        {
            let show_percentage = 100.0;
            let rng = rand::rng().random_range(0.0..100.0);
            if rng > show_percentage {
                continue;
            }

            let offset_idx = offsets_vec.len();
            let (vertex_offset, vertex_count) = geometry.get_shared_vertex_buffer_offsets();
            let (index_offset, index_count) = geometry.get_shared_index_buffer_offsets();
            let material_offset = material.get_shared_buffer_offset();

            let offsets = shaders::Offsets {
                vertex_offset: vertex_offset as u32,
                vertex_count: vertex_count as u32,
                material_offset: material_offset as u32,
                padding: 0,
                index_offset: index_offset as u32,
                index_count: index_count as u32,
            };
            offsets_vec.push(offsets);

            // TODO Only build if dynamic and necessary
            // Find a way to update and not rebuild the whole thing
            blas_instances.push(AccelerationStructureInstance {
                acceleration_structure_reference: geometry.get_blas_device_address().into(),
                transform: mat4_to_as_instance_array(transform.get_matrix()),
                instance_custom_index_and_mask: Packed24_8::new(
                    // The index into the offset buffer
                    offset_idx as u32,
                    0xFF,
                ),
                // instance_shader_binding_table_record_offset_and_flags: todo!(),
                ..Default::default()
            });
        }

        // Update the offsets buffer with the new offsets
        self.scene
            .update_shared_offsets_buffer(&offsets_vec, context);

        self.tlas = unsafe { build_top_level_acceleration_structure(blas_instances, context) };

        // let proj = glam::Mat4::perspective_rh(90.0_f32.to_radians(), 1280.0 / 720.0, 0.01, 10000.0);
        // let view = glam::Mat4::look_to_rh(
        //     // glam::Vec3::new(-3.0, 2.0, -2.0),
        //     glam::Vec3::new(-0.0, 1.0, 2.2),
        //     glam::Vec3::new(0.0, 0.0, -1.0),
        //     glam::Vec3::new(0.0, -1.0, 0.0),
        // );
        let proj = glam::Mat4::perspective_rh(
            self.camera.fov_y,
            context.swapchain.image_extent()[0] as f32 / context.swapchain.image_extent()[1] as f32,
            self.camera.near,
            self.camera.far,
        );
        let view = glam::Mat4::look_to_rh(
            self.camera.transform.position,
            self.camera.transform.forward(),
            self.camera.transform.up(),
        );
        // println!(
        //     "Eye: {:#?}\nDir: {:#?}\nUp: {:#?}",
        //     self.camera.transform.position,
        //     self.camera.transform.forward(),
        //     self.camera.transform.up()
        // );

        let state_dirty = {
            let cam_dirty = self.camera != self.prev_cam_state;
            let resized_dirty = self.resized_dirty;

            self.resized_dirty = false;
            self.prev_cam_state = self.camera.clone();

            cam_dirty || self.scene.check_dirty_and_reset() || resized_dirty
        };

        if state_dirty {
            self.accumulated_count = 0;
        }

        let raygen_push_constants = shaders::raygen::PushConstants {
            uniforms: shaders::raygen::RendererUniforms {
                samples_per_pixel: self.samples_per_pixel,
                seed: rand::random(),
                accumulated_count: self.accumulated_count,
                z_near: self.z_near,
                z_far: self.z_far,
                padding: [0; 3],
            },
            camera: shaders::raygen::Camera {
                view_proj: (proj * view).to_cols_array_2d(),
                view_inverse: view.inverse().to_cols_array_2d(),
                proj_inverse: proj.inverse().to_cols_array_2d(),
            },
        };

        let closest_hit_push_constants = shaders::closest_hit::RchitPushConstants {
            uniforms: shaders::closest_hit::RendererUniforms {
                samples_per_pixel: self.samples_per_pixel,
                seed: rand::random(),
                accumulated_count: self.accumulated_count,
                z_near: self.z_near,
                z_far: self.z_far,
                padding: [0; 3],
            },
        };

        if !state_dirty && self.accumulated_count < self.accumulation_limit {
            self.accumulated_count += 1;
        }

        builder
            .push_constants(
                self.pipeline_layout.clone(),
                0, // Offset 0 for the entire block
                raygen_push_constants,
            )
            .unwrap();

        builder
            .push_constants(self.pipeline_layout.clone(), 0, closest_hit_push_constants)
            .unwrap();

        // Update the descriptor set with the new TLAS
        self.update_descriptor_set(context);

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::RayTracing,
                self.pipeline_layout.clone(),
                0,
                vec![
                    self.rgen_descriptor_set.clone(),
                    self.render_textures_descriptor_set.clone(),
                    self.scene.rhit_descriptor_set.clone(),
                    self.bindless_textures_descriptor_set.clone(),
                ],
            )
            .unwrap()
            .bind_pipeline_ray_tracing(self.pipeline.clone())
            .unwrap();

        let extent = self.albedo_texture.image().extent();

        unsafe { builder.trace_rays(self.shader_binding_table.addresses().clone(), extent) }
            .expect("Failed to trace rays");

        // Draw the render texture to the swapchain image
        // They are not guaranteed to be the same size, so we need to use a blit command
        let swapchain_image = context.swapchain_images[image_index as usize].clone();
        let render_image_view = self.albedo_texture.clone();
        // For debugging:
        // let render_image_view = self.normal_texture.clone();
        // let render_image_view = self.depth_texture.clone();

        let blit_image_info =
            BlitImageInfo::images(render_image_view.image().clone(), swapchain_image.clone());

        builder
            .blit_image(blit_image_info)
            .expect("Failed to blit image from render texture to swapchain image");
    }
}

pub unsafe fn update_top_level_acceleration_structure(
    tlas: Arc<AccelerationStructure>,
    as_instances: Vec<AccelerationStructureInstance>,
    context: &crate::graphics::VulkanContext,
) {
    log::info!("Updating TLAS with {} instances", as_instances.len());
    let instance_count = as_instances.len() as u32;

    let instance_buffer = Buffer::from_iter(
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
    .unwrap();

    let as_geometry_instances_data = AccelerationStructureGeometryInstancesData {
        flags: GeometryFlags::OPAQUE,
        ..AccelerationStructureGeometryInstancesData::new(
            AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer)),
        )
    };

    let geometries = AccelerationStructureGeometries::Instances(as_geometry_instances_data);

    let mut as_update_geometry_info = AccelerationStructureBuildGeometryInfo {
        mode: BuildAccelerationStructureMode::Update(tlas.clone()),
        dst_acceleration_structure: Some(tlas),
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE
            | BuildAccelerationStructureFlags::ALLOW_UPDATE,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let as_build_sizes_info = context
        .device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &as_update_geometry_info,
            &[instance_count],
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
            as_build_sizes_info.update_scratch_size,
            min_as_scratch_offset_alignment,
        )
        .expect("Failed to create device layout for scratch buffer"),
    )
    .unwrap();

    as_update_geometry_info.scratch_data = Some(scratch_buffer.into());

    let as_build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count: instance_count,
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
            .build_acceleration_structure(as_update_geometry_info, vec![as_build_range_info].into())
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
}

/// Builds a top-level acceleration structure (TLAS) from a list of instances.
/// Blocks until the TLAS is built.
pub unsafe fn build_top_level_acceleration_structure(
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
            BuildAccelerationStructureFlags::PREFER_FAST_TRACE
            // Important, we will only maintain one TLAS and update it throughout
                | BuildAccelerationStructureFlags::ALLOW_UPDATE,
            context,
            None,
        )
    }
}

/// A helper function to build a acceleration structure and wait for its completion.
///
/// # Safety
///
/// - If you are referencing a bottom-level acceleration structure in a top-level acceleration
///   structure, you must ensure that the bottom-level acceleration structure is kept alive.
///
/// Size:
/// If smaller than the minimum, will warn.
/// If None, will use minimum size.
pub unsafe fn build_acceleration_structure_common(
    geometries: AccelerationStructureGeometries,
    primitive_count: u32,
    ty: AccelerationStructureType,
    flags: BuildAccelerationStructureFlags,
    context: &crate::graphics::VulkanContext,
    size: Option<u64>,
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
            &[primitive_count],
        )
        .unwrap();

    // We create a new scratch buffer for each acceleration structure for simplicity. You may want
    // to reuse scratch buffers if you need to build many acceleration structures.
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
                if let Some(size) = size {
                    if size < as_build_sizes_info.acceleration_structure_size {
                        log::warn!(
                            "Acceleration structure size is smaller than the minimum size, using \
                             minimum size instead"
                        );

                        as_build_sizes_info.acceleration_structure_size
                    } else {
                        size
                    }
                } else {
                    // Minimum size
                    as_build_sizes_info.acceleration_structure_size
                },
            )
            .unwrap(),
        )
    };

    let acceleration =
        unsafe { AccelerationStructure::new(context.device.clone(), as_create_info) }.unwrap();

    as_build_geometry_info.dst_acceleration_structure = Some(acceleration.clone());
    as_build_geometry_info.scratch_data = Some(scratch_buffer.into());

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

pub unsafe fn build_acceleration_structure_triangles(
    vertex_buffer: Subbuffer<[CustomVertex]>,
    context: &crate::graphics::VulkanContext,
) -> Arc<AccelerationStructure> {
    let primitive_count = (vertex_buffer.len() / 3) as u32;
    let as_geometry_triangles_data = AccelerationStructureGeometryTrianglesData {
        max_vertex: vertex_buffer.len() as _,
        vertex_data: Some(vertex_buffer.into_bytes()),
        vertex_stride: size_of::<CustomVertex>() as _,
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
            None,
        )
    }
}

pub(crate) fn mat4_to_as_instance_array(mat: glam::Mat4) -> [[f32; 4]; 3] {
    let arr: [f32; 12] = mat.transpose().to_cols_array()[..12].try_into().unwrap();
    [
        [arr[0], arr[1], arr[2], arr[3]],
        [arr[4], arr[5], arr[6], arr[7]],
        [arr[8], arr[9], arr[10], arr[11]],
    ]
}
