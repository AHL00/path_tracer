use crate::scene::Transform;


#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    pub transform: Transform,
    /// Field of view in radians
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        let transform = Transform::default();
        // transform.rotation = glam::Quat::from_rotation_y(PI);
        Self {
            transform,
            fov_y: 70.0_f32.to_radians(),
            near: 0.1,
            far: 1000.0,
        }
    }
}