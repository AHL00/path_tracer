use std::sync::Arc;

use vulkano::acceleration_structure::AccelerationStructure;



#[derive(Clone, Debug, PartialEq)]
/// A component representing a geometry object in the scene

pub struct Geometry {
    blas: Option<Arc<AccelerationStructure>>,
}

impl Geometry {
    pub fn new() -> Self {
        Self { blas: None }
    }

    pub fn set_blas(&mut self, blas: Arc<AccelerationStructure>) {
        self.blas = Some(blas);
    }

    pub fn get_blas(&self) -> Option<Arc<AccelerationStructure>> {
        self.blas.clone()
    }
}

