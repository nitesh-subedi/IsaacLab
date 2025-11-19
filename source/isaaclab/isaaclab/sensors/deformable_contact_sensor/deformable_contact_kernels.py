# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for deformable body contact detection."""

import warp as wp


@wp.kernel
def mesh_proximity_detection_kernel(
    deformable_vertices: wp.array2d(dtype=wp.vec3),  # (num_envs, num_vertices, 3)
    rigid_mesh_id: wp.uint64,
    rigid_mesh_transforms: wp.array2d(dtype=wp.transform),  # (num_envs, 1)
    contact_threshold: float,
    # Outputs
    contact_detected: wp.array2d(dtype=wp.int32),  # (num_envs, num_vertices)
    contact_points: wp.array2d(dtype=wp.vec3),  # (num_envs, num_vertices) - vec3
    contact_normals: wp.array2d(dtype=wp.vec3),  # (num_envs, num_vertices) - vec3
    contact_distances: wp.array2d(dtype=wp.float32),  # (num_envs, num_vertices)
):
    """Detect contacts between deformable mesh vertices and a rigid body mesh.

    This kernel checks each vertex of the deformable body against the rigid body mesh
    to determine if they are within the contact threshold distance.

    Args:
        deformable_vertices: Vertex positions of the deformable mesh in world frame.
        rigid_mesh_id: Warp mesh ID of the rigid body collision mesh.
        rigid_mesh_transforms: Transforms of the rigid body meshes per environment.
        contact_threshold: Distance threshold for contact detection.
        contact_detected: Output array indicating if vertex is in contact (1) or not (0).
        contact_points: Output array of contact points on the rigid mesh surface.
        contact_normals: Output array of contact normals pointing away from rigid mesh.
        contact_distances: Output array of signed distances (negative = penetration).
    """
    env_idx, vertex_idx = wp.tid()

    # Get the vertex position in world frame
    vertex_pos = deformable_vertices[env_idx, vertex_idx]

    # Get the rigid body transform for this environment
    rigid_transform = rigid_mesh_transforms[env_idx, 0]

    # Transform vertex to rigid body local frame
    vertex_local = wp.transform_point(wp.transform_inverse(rigid_transform), vertex_pos)

    # Query closest point on mesh
    query = wp.mesh_query_point(rigid_mesh_id, vertex_local, contact_threshold)

    if query.result:
        # Contact detected - get the closest point and normal using barycentric coordinates
        closest_point_local = wp.mesh_eval_position(rigid_mesh_id, query.face, query.u, query.v)
        face_normal_local = wp.mesh_eval_face_normal(rigid_mesh_id, query.face)

        # Transform back to world frame
        closest_point_world = wp.transform_point(rigid_transform, closest_point_local)
        face_normal_world = wp.transform_vector(rigid_transform, face_normal_local)

        # Calculate signed distance (negative = penetration)
        diff = vertex_pos - closest_point_world
        distance = wp.dot(diff, face_normal_world)

        # Store results
        contact_detected[env_idx, vertex_idx] = 1
        contact_points[env_idx, vertex_idx] = closest_point_world
        contact_normals[env_idx, vertex_idx] = face_normal_world
        contact_distances[env_idx, vertex_idx] = distance
    else:
        # No contact
        contact_detected[env_idx, vertex_idx] = 0
        contact_points[env_idx, vertex_idx] = wp.vec3(0.0, 0.0, 0.0)
        contact_normals[env_idx, vertex_idx] = wp.vec3(0.0, 0.0, 0.0)
        contact_distances[env_idx, vertex_idx] = contact_threshold


@wp.kernel
def multi_rigid_mesh_proximity_kernel(
    deformable_vertices: wp.array2d(dtype=wp.vec3),  # (num_envs, num_vertices, 3)
    rigid_mesh_ids: wp.array(dtype=wp.uint64),  # (num_rigid_bodies,)
    rigid_mesh_transforms: wp.array3d(dtype=wp.transform),  # (num_envs, num_rigid_bodies, 1)
    contact_threshold: float,
    num_rigid_bodies: int,
    # Outputs
    contact_detected: wp.array3d(dtype=wp.int32),  # (num_envs, num_vertices, num_rigid_bodies)
    contact_points: wp.array3d(dtype=wp.vec3),  # (num_envs, num_vertices, num_rigid_bodies) - vec3
    contact_normals: wp.array3d(dtype=wp.vec3),  # (num_envs, num_vertices, num_rigid_bodies) - vec3
    contact_distances: wp.array3d(dtype=wp.float32),  # (num_envs, num_vertices, num_rigid_bodies)
):
    """Detect contacts between deformable mesh vertices and multiple rigid body meshes.

    This kernel checks each vertex of the deformable body against multiple rigid body meshes
    to determine if they are within the contact threshold distance.

    Args:
        deformable_vertices: Vertex positions of the deformable mesh in world frame.
        rigid_mesh_ids: Array of Warp mesh IDs for rigid body collision meshes.
        rigid_mesh_transforms: Transforms of the rigid body meshes per environment.
        contact_threshold: Distance threshold for contact detection.
        num_rigid_bodies: Number of rigid bodies to check.
        contact_detected: Output array indicating if vertex is in contact with each rigid body.
        contact_points: Output array of contact points on rigid mesh surfaces.
        contact_normals: Output array of contact normals.
        contact_distances: Output array of signed distances.
    """
    env_idx, vertex_idx = wp.tid()

    # Get the vertex position in world frame
    vertex_pos = deformable_vertices[env_idx, vertex_idx]

    # Check against all rigid bodies
    for rigid_idx in range(num_rigid_bodies):
        rigid_mesh_id = rigid_mesh_ids[rigid_idx]
        rigid_transform = rigid_mesh_transforms[env_idx, rigid_idx, 0]

        # Transform vertex to rigid body local frame
        vertex_local = wp.transform_point(wp.transform_inverse(rigid_transform), vertex_pos)

        # Query closest point on mesh
        query = wp.mesh_query_point(rigid_mesh_id, vertex_local, contact_threshold)

        if query.result:
            # Contact detected - get the closest point and normal using barycentric coordinates
            closest_point_local = wp.mesh_eval_position(rigid_mesh_id, query.face, query.u, query.v)
            face_normal_local = wp.mesh_eval_face_normal(rigid_mesh_id, query.face)

            # Transform back to world frame
            closest_point_world = wp.transform_point(rigid_transform, closest_point_local)
            face_normal_world = wp.transform_vector(rigid_transform, face_normal_local)

            # Calculate signed distance
            diff = vertex_pos - closest_point_world
            distance = wp.dot(diff, face_normal_world)

            # Store results
            contact_detected[env_idx, vertex_idx, rigid_idx] = 1
            contact_points[env_idx, vertex_idx, rigid_idx] = closest_point_world
            contact_normals[env_idx, vertex_idx, rigid_idx] = face_normal_world
            contact_distances[env_idx, vertex_idx, rigid_idx] = distance
        else:
            # No contact
            contact_detected[env_idx, vertex_idx, rigid_idx] = 0
            contact_points[env_idx, vertex_idx, rigid_idx] = wp.vec3(0.0, 0.0, 0.0)
            contact_normals[env_idx, vertex_idx, rigid_idx] = wp.vec3(0.0, 0.0, 0.0)
            contact_distances[env_idx, vertex_idx, rigid_idx] = contact_threshold


@wp.kernel
def aggregate_contact_forces_kernel(
    contact_detected: wp.array3d(dtype=wp.int32),  # (num_envs, num_vertices, num_rigid_bodies)
    contact_normals_x: wp.array3d(dtype=wp.float32),  # (num_envs, num_vertices, num_rigid_bodies)
    contact_normals_y: wp.array3d(dtype=wp.float32),  # (num_envs, num_vertices, num_rigid_bodies)
    contact_normals_z: wp.array3d(dtype=wp.float32),  # (num_envs, num_vertices, num_rigid_bodies)
    contact_distances: wp.array3d(dtype=wp.float32),  # (num_envs, num_vertices, num_rigid_bodies)
    vertex_velocities_x: wp.array2d(dtype=wp.float32),  # (num_envs, num_vertices)
    vertex_velocities_y: wp.array2d(dtype=wp.float32),  # (num_envs, num_vertices)
    vertex_velocities_z: wp.array2d(dtype=wp.float32),  # (num_envs, num_vertices)
    stiffness: float,
    damping: float,
    # Outputs
    net_contact_forces_x: wp.array2d(dtype=wp.float32),  # (num_envs, num_rigid_bodies)
    net_contact_forces_y: wp.array2d(dtype=wp.float32),  # (num_envs, num_rigid_bodies)
    net_contact_forces_z: wp.array2d(dtype=wp.float32),  # (num_envs, num_rigid_bodies)
    num_contacts: wp.array2d(dtype=wp.int32),  # (num_envs, num_rigid_bodies)
):
    """Aggregate contact forces from all vertices for each rigid body.

    This kernel computes estimated contact forces using a spring-damper model
    based on penetration depth and vertex velocity.

    Note: Uses separate X,Y,Z arrays to avoid Warp vec3 indexing issues.

    Args:
        contact_detected: Array indicating contact status.
        contact_normals_x/y/z: Components of contact normals.
        contact_distances: Array of signed distances (negative = penetration).
        vertex_velocities_x/y/z: Components of vertex velocities.
        stiffness: Spring stiffness for force estimation.
        damping: Damping coefficient for force estimation.
        net_contact_forces_x/y/z: Output net contact force components per rigid body.
        num_contacts: Output number of contacts per rigid body.
    """
    env_idx, rigid_idx = wp.tid()

    total_force_x = float(0.0)
    total_force_y = float(0.0)
    total_force_z = float(0.0)
    contact_count = int(0)

    # Get number of vertices from the contact_detected array
    num_vertices = contact_detected.shape[1]

    for vertex_idx in range(num_vertices):
        if contact_detected[env_idx, vertex_idx, rigid_idx] == 1:
            # Get contact information
            nx = contact_normals_x[env_idx, vertex_idx, rigid_idx]
            ny = contact_normals_y[env_idx, vertex_idx, rigid_idx]
            nz = contact_normals_z[env_idx, vertex_idx, rigid_idx]
            distance_val = contact_distances[env_idx, vertex_idx, rigid_idx]

            vx = vertex_velocities_x[env_idx, vertex_idx]
            vy = vertex_velocities_y[env_idx, vertex_idx]
            vz = vertex_velocities_z[env_idx, vertex_idx]

            # Only apply force if penetrating (negative distance)
            if distance_val < 0.0:
                # Spring force (proportional to penetration)
                spring_force = -distance_val * stiffness

                # Damping force (proportional to velocity along normal)
                vel_normal = vx * nx + vy * ny + vz * nz
                damping_force = -vel_normal * damping

                # Total force along normal
                force_magnitude = spring_force + damping_force

                total_force_x = total_force_x + nx * force_magnitude
                total_force_y = total_force_y + ny * force_magnitude
                total_force_z = total_force_z + nz * force_magnitude
                contact_count = contact_count + 1

    net_contact_forces_x[env_idx, rigid_idx] = total_force_x
    net_contact_forces_y[env_idx, rigid_idx] = total_force_y
    net_contact_forces_z[env_idx, rigid_idx] = total_force_z
    num_contacts[env_idx, rigid_idx] = contact_count
