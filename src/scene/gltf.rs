use crate::{graphics::VulkanContext, renderer::CustomVertex};

use super::{Transform, geometry::Geometry};

impl super::Scene {
    /// Imports a GLTF file and adds it to the scene.
    pub fn import_gltf(
        &mut self,
        path: &std::path::Path,
        context: &VulkanContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (document, buffers, images) = gltf::import(path)?;

        for scene in document.scenes() {
            for node in scene.nodes() {
                if let Some(mesh) = node.mesh() {
                    let mut transform = Transform::default();

                    let (position, rotation, scale) = node.transform().decomposed();

                    transform.position = glam::Vec3::from(position);
                    transform.rotation = glam::Quat::from_array(rotation);
                    transform.scale = glam::Vec3::from(scale);

                    for primitive in mesh.primitives() {
                        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                        let indices = if let Some(indices_reader) = reader.read_indices() {
                            indices_reader.into_u32().collect::<Vec<_>>()
                        } else {
                            // Generate sequential indices if no indices are provided
                            let vertex_count = reader
                                .read_positions()
                                .map_or(0, |positions| positions.count());
                            (0..vertex_count as u32).collect()
                        };

                        // Read vertex positions (required)
                        let positions = if let Some(positions) = reader.read_positions() {
                            positions.collect::<Vec<_>>()
                        } else {
                            return Err("Mesh has no positions".into());
                        };

                        // Read normals (optional, default to [0,0,1])
                        let normals = if let Some(normals) = reader.read_normals() {
                            normals.collect::<Vec<_>>()
                        } else {
                            log::warn!(
                                "Mesh in Node [{:?}] has no normals, generating normals",
                                node.name()
                            );

                            let position_vecs = positions
                                .iter()
                                .map(|p| glam::Vec3::from(*p))
                                .collect::<Vec<_>>();
                            calculate_normals(&position_vecs, &indices)
                        };

                        // Read texture coordinates (optional, default to [0,0])
                        let tex_coords = if let Some(tex_coords) = reader.read_tex_coords(0) {
                            tex_coords.into_f32().collect::<Vec<_>>()
                        } else {
                            vec![[0.0, 0.0]; positions.len()]
                        };

                        // Combine data into vertices
                        let vertices = positions
                            .iter()
                            .zip(normals.iter())
                            .zip(tex_coords.iter())
                            .map(|((pos, norm), tex)| CustomVertex {
                                position: glam::Vec3::from(*pos),
                                normal: glam::Vec3::from(*norm),
                                tex_coords: glam::Vec2::from(*tex),
                            })
                            .collect();

                        let geometry = Geometry::create(vertices, indices, context)?;

                        // Create a new entity with the geometry and transform components
                        self.world.push((geometry, transform));
                    }
                }
            }
        }

        Ok(())
    }
}

fn calculate_normals(positions: &[glam::Vec3], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![glam::Vec3::ZERO; positions.len()];

    for i in (0..indices.len()).step_by(3) {
        let a = positions[indices[i] as usize];
        let b = positions[indices[i + 1] as usize];
        let c = positions[indices[i + 2] as usize];

        let normal = (b - a).cross(c - a).normalize();

        normals[indices[i] as usize] += normal;
        normals[indices[i + 1] as usize] += normal;
        normals[indices[i + 2] as usize] += normal;
    }

    for normal in &mut normals {
        *normal = normal.normalize();
    }

    normals.iter().map(|n| [n.x, n.y, n.z]).collect()
}
