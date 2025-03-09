pub mod geometry;

pub struct Scene {
    pub world: legion::World,
    pub resources: legion::Resources,
}

impl Scene {
    pub fn new() -> Self {
        let world = legion::World::default();
        let resources = legion::Resources::default();

        Self { world, resources }
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
