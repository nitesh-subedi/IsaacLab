# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import warp as wp
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.warp import convert_to_warp_mesh

from ..sensor_base import SensorBase
from .deformable_contact_kernels import (
    aggregate_contact_forces_kernel,
    multi_rigid_mesh_proximity_kernel,
)
from .deformable_contact_sensor_data import DeformableContactSensorData

if TYPE_CHECKING:
    from .deformable_contact_sensor_cfg import DeformableContactSensorCfg


class DeformableContactSensor(SensorBase):
    """A contact sensor for detecting contacts between deformable and rigid bodies.

    This sensor uses mesh proximity detection to identify contacts between a deformable body
    and one or more rigid bodies. It leverages Warp kernels for efficient GPU-accelerated
    computation.

    The sensor works by:
    1. Accessing the collision mesh vertices of the deformable body from PhysX
    2. Converting rigid body collision meshes to Warp meshes
    3. Using Warp's mesh query functions to detect proximity between deformable vertices
       and rigid body surfaces
    4. Estimating contact forces using a spring-damper model

    Note:
        This sensor requires that the deformable body has been created with PhysX deformable
        body simulation enabled, and the rigid bodies must have valid collision meshes.
    """

    cfg: DeformableContactSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: DeformableContactSensorCfg):
        """Initializes the deformable contact sensor.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data: DeformableContactSensorData = DeformableContactSensorData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Deformable contact sensor:\n"
            f"\tDeformable body  : {self.cfg.deformable_body_prim_path}\n"
            f"\tRigid bodies     : {self.cfg.rigid_body_prim_paths}\n"
            f"\tupdate period (s): {self.cfg.update_period}\n"
            f"\tcontact threshold: {self.cfg.contact_threshold}\n"
            f"\tnum environments : {self._num_envs}\n"
            f"\tnum rigid bodies : {self._num_rigid_bodies}\n"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._num_envs

    @property
    def data(self) -> DeformableContactSensorData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset contact data
        self._data.contact_detected[env_ids] = 0
        self._data.num_contacts[env_ids] = 0
        self._data.net_contact_forces_w[env_ids] = 0.0
        # reset time tracking if enabled
        if self.cfg.track_contact_time:
            self._data.current_air_time[env_ids] = 0.0
            self._data.last_air_time[env_ids] = 0.0
            self._data.current_contact_time[env_ids] = 0.0
            self._data.last_contact_time[env_ids] = 0.0

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()

        # Initialize Warp device (now self._device is available from parent)
        self._wp_device = wp.device_from_torch(self._device)

        # Obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # Validate deformable body path
        if not self.cfg.deformable_body_prim_path:
            raise ValueError("deformable_body_prim_path must be specified in the configuration!")

        # Validate rigid body paths
        if not self.cfg.rigid_body_prim_paths:
            raise ValueError("At least one rigid body path must be specified in rigid_body_prim_paths!")

        # Initialize deformable body view
        self._initialize_deformable_body()

        # Initialize rigid body views and meshes
        self._initialize_rigid_bodies()

        # Create data buffers
        self._create_buffers()

        omni.log.info(f"Deformable contact sensor initialized at: {self.cfg.prim_path}")
        omni.log.info(f"Monitoring {self._num_rigid_bodies} rigid bodies against deformable body")
        omni.log.info(f"Deformable mesh has {self._max_vertices} collision vertices per instance")

    def _initialize_deformable_body(self):
        """Initialize the deformable body view."""
        from pxr import PhysxSchema

        # Find the template deformable object prim
        template_prim = sim_utils.find_first_matching_prim(self.cfg.deformable_body_prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.deformable_body_prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        # Find deformable root prims (child prims with PhysxDeformableBodyAPI)
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI),
            traverse_instance_prims=False,
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a deformable body when resolving '{self.cfg.deformable_body_prim_path}'."
                " Please ensure that the prim has 'PhysxSchema.PhysxDeformableBodyAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single deformable body when resolving '{self.cfg.deformable_body_prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one deformable body in the prim path tree."
            )

        # Get the actual deformable body prim
        root_prim = root_prims[0]
        root_prim_path = root_prim.GetPath().pathString
        root_prim_path_expr = self.cfg.deformable_body_prim_path + root_prim_path[len(template_prim_path) :]

        # Create soft body view
        deformable_prim_path_glob = root_prim_path_expr.replace(".*", "*")
        self._deformable_view = self._physics_sim_view.create_soft_body_view(deformable_prim_path_glob)

        if self._deformable_view._backend is None:
            raise RuntimeError(
                f"Failed to create deformable body view at: {root_prim_path_expr}. "
                "Please check PhysX logs."
            )

        # Get mesh information
        self._max_vertices = self._deformable_view.max_vertices_per_body
        omni.log.info(
            f"Deformable body initialized: {root_prim_path_expr} "
            f"with {self._max_vertices} collision vertices"
        )

    def _initialize_rigid_bodies(self):
        """Initialize rigid body views and convert their meshes to Warp format."""
        self._rigid_body_views = []
        self._rigid_meshes: list[wp.Mesh] = []
        self._num_rigid_bodies = len(self.cfg.rigid_body_prim_paths)

        for rigid_path in self.cfg.rigid_body_prim_paths:
            # Create PhysX view for transforms
            rigid_path_glob = rigid_path.replace(".*", "*")
            try:
                rigid_view = self._physics_sim_view.create_rigid_body_view(rigid_path_glob)
                self._rigid_body_views.append(rigid_view)
            except Exception as e:
                raise RuntimeError(f"Failed to create rigid body view for {rigid_path}: {e}")

            # Get collision mesh from first instance (assuming all instances have same mesh)
            template_path = rigid_path.replace("env_.*", "env_0")
            mesh_prim = self._find_collision_mesh(template_path)

            if mesh_prim is None:
                raise RuntimeError(f"Could not find collision mesh for rigid body: {rigid_path}")

            # Convert to Warp mesh
            wp_mesh = self._convert_usd_mesh_to_warp(mesh_prim)
            self._rigid_meshes.append(wp_mesh)

            omni.log.info(f"Initialized rigid body mesh for: {rigid_path}")

    def _find_collision_mesh(self, prim_path: str) -> UsdGeom.Mesh | None:
        """Find the collision mesh for a rigid body prim."""
        # First, try to find the prim itself
        prim = sim_utils.find_first_matching_prim(prim_path)
        if prim is None:
            return None

        # Check if the prim itself is a mesh
        if prim.GetTypeName() == "Mesh":
            return UsdGeom.Mesh(prim)

        # Otherwise, search children for mesh
        mesh_prim = sim_utils.get_first_matching_child_prim(prim_path, lambda p: p.GetTypeName() == "Mesh")
        if mesh_prim is not None:
            return UsdGeom.Mesh(mesh_prim)

        return None

    def _convert_usd_mesh_to_warp(self, mesh_prim: UsdGeom.Mesh) -> wp.Mesh:
        """Convert a USD mesh to a Warp mesh."""
        # Get mesh vertices
        points = np.asarray(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)

        # Get mesh faces
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        face_vertex_counts = mesh_prim.GetFaceVertexCountsAttr().Get()

        # Triangulate if necessary
        if face_vertex_counts is not None:
            face_vertex_counts = np.asarray(face_vertex_counts)
            if not np.all(face_vertex_counts == 3):
                indices = self._triangulate_mesh(indices, face_vertex_counts)

        # Keep mesh in local coordinates - we'll apply transforms at runtime
        # The runtime transforms from PhysX will handle world positioning

        # Convert to Warp mesh
        return convert_to_warp_mesh(points, indices, device=self._wp_device)

    def _triangulate_mesh(self, indices: np.ndarray, face_vertex_counts: np.ndarray) -> np.ndarray:
        """Convert polygon mesh to triangle mesh using fan triangulation."""
        triangulated = []
        idx = 0
        for count in face_vertex_counts:
            if count < 3:
                idx += count
                continue
            elif count == 3:
                triangulated.extend(indices[idx : idx + 3])
            else:
                face_indices = indices[idx : idx + count]
                for i in range(1, count - 1):
                    triangulated.extend([face_indices[0], face_indices[i], face_indices[i + 1]])
            idx += count
        return np.array(triangulated, dtype=np.int32)

    def _create_buffers(self):
        """Create buffers for storing sensor data."""
        # Basic contact detection
        self._data.contact_detected = torch.zeros(
            self._num_envs, self._num_rigid_bodies, dtype=torch.int32, device=self._device
        )
        self._data.num_contacts = torch.zeros(
            self._num_envs, self._num_rigid_bodies, dtype=torch.int32, device=self._device
        )
        self._data.net_contact_forces_w = torch.zeros(
            self._num_envs, self._num_rigid_bodies, 3, device=self._device
        )

        # Optional: detailed contact information
        if self.cfg.track_contact_points:
            self._data.contact_points_w = torch.zeros(
                self._num_envs, self._max_vertices, self._num_rigid_bodies, 3, device=self._device
            )
            self._data.contact_normals_w = torch.zeros(
                self._num_envs, self._max_vertices, self._num_rigid_bodies, 3, device=self._device
            )
            self._data.contact_distances = torch.full(
                (self._num_envs, self._max_vertices, self._num_rigid_bodies),
                self.cfg.contact_threshold,
                device=self._device,
            )
            # Internal buffers for Warp kernels
            self._contact_detected_wp = torch.zeros(
                self._num_envs, self._max_vertices, self._num_rigid_bodies, dtype=torch.int32, device=self._device
            )

        # Optional: deformable pose tracking
        if self.cfg.track_deformable_pose:
            self._data.deformable_pos_w = torch.zeros(self._num_envs, 3, device=self._device)

        # Optional: contact time tracking
        if self.cfg.track_contact_time:
            self._data.last_air_time = torch.zeros(self._num_envs, self._num_rigid_bodies, device=self._device)
            self._data.current_air_time = torch.zeros(self._num_envs, self._num_rigid_bodies, device=self._device)
            self._data.last_contact_time = torch.zeros(self._num_envs, self._num_rigid_bodies, device=self._device)
            self._data.current_contact_time = torch.zeros(self._num_envs, self._num_rigid_bodies, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # Default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        # Get deformable body vertex positions and velocities
        deformable_vertices = self._deformable_view.get_sim_nodal_positions()  # (num_envs, max_vertices, 3)
        deformable_velocities = self._deformable_view.get_sim_nodal_velocities()  # (num_envs, max_vertices, 3)

        # Get rigid body transforms
        rigid_transforms_list = []
        for rigid_view in self._rigid_body_views:
            transforms = rigid_view.get_transforms()  # (num_envs, 7) - [pos(3), quat(4)]
            rigid_transforms_list.append(transforms)

        # Stack transforms: (num_envs, num_rigid_bodies, 7)
        rigid_transforms = torch.stack(rigid_transforms_list, dim=1)

        # Debug: Print first time to see what we're getting
        if not hasattr(self, '_debug_printed'):
            print(f"\n[DEBUG] Deformable Contact Sensor Update:")
            print(f"  Deformable vertices shape: {deformable_vertices.shape}")
            print(f"  Rigid transforms shape: {rigid_transforms.shape}")
            print(f"  Rigid transforms[0]: {rigid_transforms[0]}")
            print(f"  Num rigid meshes: {len(self._rigid_meshes)}")
            print(f"  Contact threshold: {self.cfg.contact_threshold}")
            self._debug_printed = True

        # Convert transforms to Warp format (wp.transform)
        rigid_transforms_wp = self._convert_to_warp_transforms(rigid_transforms)

        # Prepare Warp mesh IDs
        mesh_ids_wp = wp.array([mesh.id for mesh in self._rigid_meshes], dtype=wp.uint64, device=self._wp_device)

        # Run proximity detection kernel
        if self.cfg.track_contact_points:
            # Use detailed kernel
            deformable_vertices_wp = wp.from_torch(deformable_vertices, dtype=wp.vec3)
            contact_detected_wp = wp.from_torch(self._contact_detected_wp, dtype=wp.int32)
            contact_points_wp = wp.from_torch(self._data.contact_points_w, dtype=wp.vec3)
            contact_normals_wp = wp.from_torch(self._data.contact_normals_w, dtype=wp.vec3)
            contact_distances_wp = wp.from_torch(self._data.contact_distances, dtype=wp.float32)

            wp.launch(
                kernel=multi_rigid_mesh_proximity_kernel,
                dim=(self._num_envs, self._max_vertices),
                inputs=[
                    deformable_vertices_wp,
                    mesh_ids_wp,
                    rigid_transforms_wp,
                    float(self.cfg.contact_threshold),
                    int(self._num_rigid_bodies),
                ],
                outputs=[contact_detected_wp, contact_points_wp, contact_normals_wp, contact_distances_wp],
                device=self._wp_device,
            )

            # Aggregate forces - split vec3 into components to avoid Warp type issues
            # Extract X, Y, Z components from contact normals and velocities
            contact_normals_x = self._data.contact_normals_w[..., 0].contiguous()
            contact_normals_y = self._data.contact_normals_w[..., 1].contiguous()
            contact_normals_z = self._data.contact_normals_w[..., 2].contiguous()

            velocity_x = deformable_velocities[..., 0].contiguous()
            velocity_y = deformable_velocities[..., 1].contiguous()
            velocity_z = deformable_velocities[..., 2].contiguous()

            # Create output buffers
            force_x = self._data.net_contact_forces_w[..., 0].contiguous()
            force_y = self._data.net_contact_forces_w[..., 1].contiguous()
            force_z = self._data.net_contact_forces_w[..., 2].contiguous()

            wp.launch(
                kernel=aggregate_contact_forces_kernel,
                dim=(self._num_envs, self._num_rigid_bodies),
                inputs=[
                    contact_detected_wp,
                    wp.from_torch(contact_normals_x, dtype=wp.float32),
                    wp.from_torch(contact_normals_y, dtype=wp.float32),
                    wp.from_torch(contact_normals_z, dtype=wp.float32),
                    contact_distances_wp,
                    wp.from_torch(velocity_x, dtype=wp.float32),
                    wp.from_torch(velocity_y, dtype=wp.float32),
                    wp.from_torch(velocity_z, dtype=wp.float32),
                    float(self.cfg.force_stiffness),
                    float(self.cfg.force_damping),
                ],
                outputs=[
                    wp.from_torch(force_x, dtype=wp.float32),
                    wp.from_torch(force_y, dtype=wp.float32),
                    wp.from_torch(force_z, dtype=wp.float32),
                    wp.from_torch(self._data.num_contacts, dtype=wp.int32),
                ],
                device=self._wp_device,
            )

            # Copy results back to the main tensor
            self._data.net_contact_forces_w[..., 0] = force_x
            self._data.net_contact_forces_w[..., 1] = force_y
            self._data.net_contact_forces_w[..., 2] = force_z
        else:
            # Simplified version: just detect contact and estimate forces
            # For now, fall back to torch-based simple proximity check
            self._simple_contact_detection(deformable_vertices, rigid_transforms)

        # Update contact detection flag
        self._data.contact_detected[env_ids] = (self._data.num_contacts[env_ids] > 0).int()

        # Track deformable pose if enabled
        if self.cfg.track_deformable_pose:
            self._data.deformable_pos_w[env_ids] = deformable_vertices[env_ids].mean(dim=1)

        # Track contact time if enabled
        if self.cfg.track_contact_time:
            elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
            is_contact = self._data.contact_detected[env_ids] > 0
            is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact
            is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact

            # Update air time
            self._data.last_air_time[env_ids] = torch.where(
                is_first_contact,
                self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
                self._data.last_air_time[env_ids],
            )
            self._data.current_air_time[env_ids] = torch.where(
                ~is_contact, self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0
            )

            # Update contact time
            self._data.last_contact_time[env_ids] = torch.where(
                is_first_detached,
                self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
                self._data.last_contact_time[env_ids],
            )
            self._data.current_contact_time[env_ids] = torch.where(
                is_contact, self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0
            )

    def _convert_to_warp_transforms(self, transforms: torch.Tensor) -> wp.array:
        """Convert PyTorch transforms to Warp transform array.

        Args:
            transforms: Tensor of shape (num_envs, num_rigid_bodies, 7) with [pos(3), quat(4)]

        Returns:
            Warp array of transforms
        """
        # Warp expects transforms as (x, y, z, w) quaternions. The PhysX views already provide them
        # in this layout, so we only need to ensure a singleton dimension exists before the last axis
        # to match the Warp array3d shape requirements used by the kernels.
        if transforms.ndim == 3:
            transforms = transforms.unsqueeze(2)
        elif transforms.ndim != 4 or transforms.shape[-1] != 7:
            raise ValueError(
                f"Expected transforms tensor of shape (num_envs, num_bodies, 7) or "
                f"(num_envs, num_bodies, 1, 7). Received shape: {tuple(transforms.shape)}"
            )

        # Ensure the tensor lives on the same device as the sensor and is contiguous before mapping
        transforms = transforms.to(self._device).contiguous()
        return wp.from_torch(transforms, dtype=wp.transform)

    def _simple_contact_detection(self, deformable_vertices: torch.Tensor, rigid_transforms: torch.Tensor):
        """Simplified contact detection without detailed tracking.

        This is a fallback method that uses simple distance checks.
        """
        # For each rigid body, check minimum distance to any deformable vertex
        for rigid_idx in range(self._num_rigid_bodies):
            # Get rigid body position
            rigid_pos = rigid_transforms[:, rigid_idx, :3]  # (num_envs, 3)

            # Compute distances from rigid body center to all vertices
            distances = torch.norm(
                deformable_vertices - rigid_pos.unsqueeze(1), dim=-1
            )  # (num_envs, max_vertices)

            # Check if any vertex is within threshold
            min_distances = distances.min(dim=1)[0]  # (num_envs,)
            in_contact = min_distances < self.cfg.contact_threshold

            # Update contact detection
            self._data.contact_detected[:, rigid_idx] = in_contact.int()
            self._data.num_contacts[:, rigid_idx] = (distances < self.cfg.contact_threshold).sum(dim=1)

            # Estimate simple contact force (proportional to penetration)
            penetration = torch.clamp(self.cfg.contact_threshold - min_distances, min=0.0)
            force_magnitude = penetration * self.cfg.force_stiffness
            # Direction from deformable center to rigid body
            deformable_center = deformable_vertices.mean(dim=1)
            direction = rigid_pos - deformable_center
            direction_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            direction_normalized = direction / direction_norm
            self._data.net_contact_forces_w[:, rigid_idx] = direction_normalized * force_magnitude.unsqueeze(-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        if debug_vis:
            if not hasattr(self, "contact_visualizer"):
                self.contact_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.contact_visualizer.set_visibility(True)
        else:
            if hasattr(self, "contact_visualizer"):
                self.contact_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Visualize contact points on deformable body
        if self.cfg.track_deformable_pose and self._data.deformable_pos_w is not None:
            # marker indices: 0 = contact, 1 = no contact
            marker_indices = torch.where(self._data.contact_detected.any(dim=1), 0, 1)
            self.contact_visualizer.visualize(self._data.deformable_pos_w, marker_indices=marker_indices)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
        self._deformable_view = None
        self._rigid_body_views = []
