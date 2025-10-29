use std::sync::{Arc, Mutex};

use vulkano::sync::GpuFuture;

use crate::{
    graphics::VulkanContext, renderer::Renderer, scene::Scene, scene_resources::SceneResources,
};

/// RenderApp is the main application struct that manages rendering.
///
/// It owns:
/// - VulkanContext: GPU resources and synchronization (Arc-wrapped for thread safety)
/// - Renderer: Ray tracing pipeline and rendering state
/// - Scene: ECS world with loaded meshes and entities
/// - SceneResources: GPU resources tied to the Scene (TLAS, texture pools)
///
/// # Invariant: Scene and SceneResources Must Stay Together
///
/// SceneResources (TLAS, texture pools) are built from Scene data. When swapping scenes,
/// you MUST also swap their corresponding resources. These are grouped together in this struct
/// to make that requirement explicit.
///
/// # Thread Safety
///
/// - VulkanContext is Arc<> and thread-safe
/// - Renderer is Send+Sync (all Arcs internally)
/// - Scene is Arc<Mutex<>> (can be modified from parallel loaders)
/// - SceneResources is Arc<Mutex<>> (tied to Scene)
/// - Safe to load scenes from background threads
pub struct RenderContext {
    /// Shared Vulkan context - all scenes and renderers must use this same context.
    /// Stored here to ensure scene swaps don't accidentally use wrong context.
    /// This acts as an invariant: "all GPU resources in this app belong to this context".
    pub vulkan_context: VulkanContext,

    pub renderer: Renderer,

    /// Scenes can be swapped, but must always use the same VulkanContext.
    /// Wrapped in Mutex to allow loading from background threads.
    pub scene: Scene,

    /// GPU resources tied to the current scene (TLAS, texture pool).
    /// MUST be swapped together with scene - if you swap scene without swapping
    /// scene_resources, rendering will use wrong TLAS and textures.
    pub scene_resources: Arc<Mutex<SceneResources>>,

    /// Previous frame's GPU future for synchronization.
    /// Not stored in VulkanContext to keep it Send+Sync.
    /// GpuFuture is not Send+Sync, so it lives in RenderContext which doesn't need to be.
    pub previous_frame_end: Arc<Mutex<Option<Box<dyn vulkano::sync::GpuFuture>>>>,
}

impl RenderContext {
    pub fn new(vulkan_context: VulkanContext, render_resolution: [u32; 2]) -> Self {
        let renderer = Renderer::new(&vulkan_context, render_resolution);
        let scene = Scene::new();
        let scene_resources = Arc::new(Mutex::new(SceneResources::new(
            &vulkan_context,
            &renderer.pipeline_layout(),
        )));

        let previous_frame_end =
            Arc::new(Mutex::new(Some(vulkano::sync::now(vulkan_context.device.clone()).boxed())));

        Self {
            vulkan_context,
            renderer,
            scene,
            scene_resources,
            previous_frame_end,
        }
    }

    pub fn wait_for_previous_frame_end(&self) {
        if let Ok(mut frame_end) = self.previous_frame_end.lock() {
            log::debug!("Waiting for previous frame to finish...");
            if let Some(future) = frame_end.as_mut() {
                future.cleanup_finished();
                log::debug!("Finished");
            }
        }
    }

    /// Swap the current scene with a new one.
    ///
    /// # Safety Considerations
    ///
    /// Both the new scene AND its scene_resources must have been created with the same VulkanContext.
    /// If scenes are loaded from different contexts, rendering will fail silently or produce validation errors.
    ///
    pub fn swap_scene_and_resources(
        &mut self,
        new_scene: Scene,
        new_resources: Arc<Mutex<SceneResources>>,
    ) {
        // TODO: Consider adding validation that both use same context
        self.scene = new_scene;
        self.scene_resources = new_resources;
    }
}
