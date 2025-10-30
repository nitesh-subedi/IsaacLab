# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import gymnasium as gym
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from collections import deque
from collections.abc import Sequence
from typing import Tuple

import omni
import isaacsim.core.utils.stage as stage_utils
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import Gf, PhysxSchema, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    DeformableObject,
    DeformableObjectCfg,
    RigidObject,
    RigidObjectCfg)

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg, save_images_to_file
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from pxr import Usd, UsdShade, Sdf

import omni.usd


MASK_SIZE = 30

@configclass
class MyBuddyEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 1.0
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(gpu_temp_buffer_capacity=2**26,
                       gpu_found_lost_pairs_capacity=2**27,
                    #    gpu_found_lost_aggregate_pairs_capacity=2**30,
                       ),
        gravity=(0.0, 0.0, 9.81),
    )
    # robot
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/mybuddy",
        spawn=sim_utils.UsdFileCfg(
            usd_path="mybuddy_rotate.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        actuators={
            ".*": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=1000.0,
                velocity_limit=10.0,
                stiffness=800.0,
                damping=4.0,
            ),
        },
    )

    arm_dof_name = [
        "bc2bl",
        "left_arm_j1",
        "left_arm_j2",
        "left_arm_j3",
        "left_arm_j4",
        "left_arm_j5",
    ]  # , "right_arm_j1", "right_arm_j2", "right_arm_j3", "right_arm_j4", "right_arm_j5"]
    link_dof_name = ["left_arm_l1", "left_arm_l3", "left_arm_l4", "left_arm_l5", "left_arm_l6"]

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5), rot=(0.0, 0.0, 0.53833, 0.84274), convention="opengl"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=4.0,
            focus_distance=0.6,
            horizontal_aperture=5.760000228881836,
            vertical_aperture=3.5999999046325684,
            clipping_range=(0.076, 10.0),
            f_stop=240.0,
        ),
        width=256,
        height=256,
    )

    # goals
    num_goal_cubes: int = 2  # number of goal cubes to spawn and render masks for

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(arm_dof_name),))
    observation_space = {
        "joints": gym.spaces.Box(low=-20.00, high=20.0, shape=(len(arm_dof_name),), dtype=np.float32),
        "rgb": gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(tiled_camera.height, tiled_camera.width, 5),
            dtype=np.float32,
        ),
        "ee_position": gym.spaces.Box(low=-20.00, high=20.0, shape=(3,), dtype=np.float32),
    }

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=50, env_spacing=2.5, replicate_physics=False)

    # reset
    max_y_pos = 0.0
    maxlen = 10

    # reward scales
    rew_scale_alive = -0.1


class MyBuddyEnv(DirectRLEnv):
    cfg: MyBuddyEnvCfg

    def __init__(self, cfg: MyBuddyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_dof_idx, _ = self._robot.find_joints(self.cfg.arm_dof_name)
        self._arm_pos_idx = self._robot.find_bodies(["left_arm_l6"])  # end effector
        print(f"Arm DOF Index: {self._arm_dof_idx}")
        print(f"Arm Position Index: {self._arm_pos_idx}")
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel

        self.image_sequences = deque(maxlen=20)
        self.cube_detected_time = None  # , 90, 0, -120, -120, 95
        self.root_init_angles = (
            torch.deg2rad(torch.tensor([0, -90, 0, 120, -120, 180], device=self.device))
            .repeat(self.num_envs, 1)
            .view(-1, len(self.cfg.arm_dof_name))
        )  # torch.deg2rad(torch.tensor([-90, 0, 120, -120], device=self.device)).repeat(self.num_envs, 1).view(-1, len(self.cfg.arm_dof_name))
        self.last_angles = self.root_init_angles.clone()

        self.camera_intrinsics = self._tiled_camera.data.intrinsic_matrices
        camera_rotation = self._tiled_camera.data.quat_w_opengl
        camera_position = self._tiled_camera.data.pos_w
        rotation_matrix = self.quaternion_to_rotation_matrix(camera_rotation)
        translation_vector = -torch.bmm(rotation_matrix, camera_position.unsqueeze(-1))
        self.extrinsic_matrix = torch.cat((rotation_matrix, translation_vector), dim=2)  # [2, 3, 4]
        self.projection_matrix = torch.bmm(self.camera_intrinsics, self.extrinsic_matrix)

        # Multi-goal support: cache initial poses for all cubes and create buffers
        self.root_cube_positions = torch.stack(
            [cube.data.root_com_state_w.clone() for cube in self.goal_cubes], dim=1
        )  # [N, K, state]
        self.target_cube_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cube_maskas = torch.zeros(
            (self.num_envs, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, self.cfg.num_goal_cubes),
            device=self.device,
            dtype=torch.float32,
        )
        self.rgb_image = torch.zeros(
            (self.num_envs, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self.occlusion_time_below_threshold = torch.zeros(self.num_envs, 1, device=self.device)
        self.joint_readings = self._robot.data.joint_pos[:, self._arm_dof_idx]

    @staticmethod
    def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of quaternions to rotation matrices.

        Args:
            quat (torch.Tensor): A tensor of shape (N, 4), where each row is (w, x, y, z).

        Returns:
            torch.Tensor: A tensor of shape (N, 3, 3) representing rotation matrices.
        """
        # Ensure the input tensor has the correct shape
        if quat.dim() == 1:
            quat = quat.unsqueeze(0)  # Reshape to (1, 4) if it's a single quaternion
        # Normalize the quaternions to ensure valid rotations
        quat = quat / quat.norm(dim=1, keepdim=True)

        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Compute the rotation matrix elements
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rot_mats = torch.stack(
            [
                1 - 2 * (yy + zz),
                2 * (xy - wz),
                2 * (xz + wy),
                2 * (xy + wz),
                1 - 2 * (xx + zz),
                2 * (yz - wx),
                2 * (xz - wy),
                2 * (yz + wx),
                1 - 2 * (xx + yy),
            ],
            dim=1,
        ).reshape(-1, 3, 3)

        return rot_mats

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _add_plant(self, n_plants=1):
        self.deformable_objects_list = []
        for plant_number in range(n_plants):
            plant_number += 1

            deformable_plant_cfg = DeformableObjectCfg(
                prim_path=f"/World/envs/env_.*/Plant{plant_number}/stalk",
                debug_vis=True,
            )

            plant_object = DeformableObject(deformable_plant_cfg)
            self.deformable_objects_list.append(plant_object)

            # Deformable Objects
            for i in range(1, 6):
                deformable_plant_cfg = DeformableObjectCfg(
                    prim_path=f"/World/envs/env_.*/Plant{plant_number}/stalk{i}",
                    debug_vis=True,
                )

                plant_object = DeformableObject(deformable_plant_cfg)
                self.deformable_objects_list.append(plant_object)

            for env_path in self.scene.env_prim_paths:
                plant_prim = self.stage.GetPrimAtPath(f"{env_path}/Plant{plant_number}")
                # plant_prim = UsdGeom.Mesh.Define(self.stage, f"{env_path}/Plant/stalk")
                plant_meshes = plant_prim.GetAllChildren()
                plant_meshes = [mesh.GetAllChildren()[0] for mesh in plant_meshes]
                plant_meshes = plant_meshes[1:]
                # print("Plant Prim Children: ", plant_prim.GetAllChildrenNames()[1:])
                prim_dict = dict(zip(plant_prim.GetAllChildrenNames()[1:], plant_meshes))
                self.make_deformable(prim_dict)
                self.attach_cylinder_to_ground(prim_dict, f"{env_path}/Plant{plant_number}")
    
    def create_background_material(self):
        material_path = "/World/Looks/ImageMaterial"
        material = UsdShade.Material.Define(self.stage, material_path)

        # Create a shader
        shader = UsdShade.Shader.Define(self.stage, material_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")

        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        # Create a texture reader
        diffuse_texture = UsdShade.Shader.Define(self.stage, material_path + "/diffuse_tex")
        diffuse_texture.CreateIdAttr("UsdUVTexture")

        # Point to your image file (absolute path or relative to USD)
        diffuse_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("/home/nitesh/IsaacSim5/IsaacLab/textures/bushes.jpg")

        # Texture outputs RGB -> shader diffuse color
        diffuse_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuse_texture, "rgb")

        # Apply the material to the background plane
        # UsdShade.MaterialBindingAPI(bg_plane).Bind(material)
        return material, shader, diffuse_texture


    def _setup_scene(self):
        self.stage = stage_utils.get_current_stage()
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        euler = torch.tensor([0.0, 0.0, 90.0], device=self.device)
        quat = quat_from_euler_xyz(*euler).cpu().numpy()
        plant_cfg = sim_utils.UsdFileCfg(usd_path="plant_v21.usd", scale=(0.05, 0.05, 0.05))
        plant_cfg.func(
            "/World/envs/env_.*/Plant1",
            plant_cfg,
            translation=(0.06, -0.25, 0.0),
            orientation=(quat[0], quat[1], quat[2], quat[3]),
        )

        # euler = torch.tensor([0.0, 0.0, 90.0], device=self.device)
        # quat = quat_from_euler_xyz(*euler).cpu().numpy()
        # plant_cfg = sim_utils.UsdFileCfg(usd_path="plant_v21.usd", scale=(0.05, 0.05, 0.05))
        # plant_cfg.func(
        #     "/World/envs/env_.*/Plant2",
        #     plant_cfg,
        #     translation=(-0.07, -0.25, 0.0),
        #     orientation=(quat[0], quat[1], quat[2], quat[3]),
        # )

        # add light
        light_cfg = sim_utils.SphereLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0), radius=1.0)
        light_cfg.func("/World/envs/env_.*/Light", light_cfg, translation=(0.0, -0.4, 2.0))

        # add background plane
        intensity = 0.6
        background_color = (intensity, intensity, intensity)

        euler = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        quat = quat_from_euler_xyz(*euler).cpu().numpy()

        background_plane_cfg = sim_utils.UsdFileCfg(usd_path="/home/nitesh/Desktop/isaac_sim_assets/background.usd", scale=(1.0, 1.0, 1.0))
        background_plane_cfg.func(
            "/World/envs/env_.*/BackgroundPlane", background_plane_cfg, translation=(0.0, 0.2, 0.0), orientation=(quat[0], quat[1], quat[2], quat[3])
        )

        # background_plane_cfg = sim_utils.CuboidCfg(
        #     size=(1.25, 0.01, 1.25), 
        #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1))
        # )

        # background_plane_cfg.func(
        #     "/World/envs/env_.*/BackgroundPlane", background_plane_cfg, translation=(0.0, -0.45, 0.0)
        # )

        # add ground plane
        background_plane_cfg = sim_utils.CuboidCfg(
            size=(1.25, 0.25, 0.01), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=background_color)
        )
        background_plane_cfg.func(
            "/World/envs/env_.*/GroundPlane2", background_plane_cfg, translation=(0.0, -0.35, 0.0)
        )

        # Spawn multiple goal cubes under a common prim
        self.cube_size = (0.003, 0.003, 0.003)
        self.goal_cubes = []
        fruit_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        for i in range(self.cfg.num_goal_cubes):
            cube_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/GoalCube_{i}",
                spawn=sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=fruit_colors[i % len(fruit_colors)]),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.4, 0.28)),
            )
            self.goal_cubes.append(RigidObject(cube_cfg))
        contact_sensors_list = []
        for link_names in self.cfg.link_dof_name:
            contact_forces = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/mybuddy/{link_names}",
                update_period=0.0,
                history_length=6,
                debug_vis=True,
                # filter_prim_paths_expr=["/World/envs/env_.*/GoalCubes/.*"],
            )
            contact_sensor = ContactSensor(contact_forces)
            contact_sensors_list.append(contact_sensor)

        self.deformable_material_path = omni.usd.get_stage_next_free_path(self.stage, "/plant_material", True)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=[])

        self._add_plant(1)

        # add articulation to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        for i, contact_sensor in enumerate(contact_sensors_list):
            self.scene.sensors[f"contact_sensor_{i}"] = contact_sensor
        for i, cube in enumerate(self.goal_cubes):
            self.scene.rigid_objects[f"goal_cube_{i}"] = cube
        for i, deformable_object in enumerate(self.deformable_objects_list):
            self.scene.deformable_objects[f"stalk_{i}"] = deformable_object

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # self.camera_rotation = self._tiled_camera.data.quat_w_world[0]
        self.detection_sequence = deque(maxlen=self.cfg.maxlen)

        # set all the elements of detection sequence to False
        for i in range(self.cfg.maxlen):
            self.detection_sequence.append(torch.zeros(self.num_envs, device=self.device))

        self.done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def attach_cylinder_to_ground(self, prim_dict, prim_name):
        key, value = list(prim_dict.items())[0]
        attachment_path = value.GetPath().AppendElementString(f"attachment_{key}")
        stalk_attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment_path)
        stalk_attachment.GetActor0Rel().SetTargets([value.GetPath()])
        stalk_attachment.GetActor1Rel().SetTargets(["/World/ground/GroundPlane/CollisionPlane"])
        auto_attachment_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(stalk_attachment.GetPrim())
        # Set attributes to reduce initial movement and gap
        auto_attachment_api.GetPrim().GetAttribute("physxAutoAttachment:deformableVertexOverlapOffset").Set(0.005)
        # auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:rigidSurfaceSamplingDistance').Set(0.01)
        auto_attachment_api.GetPrim().GetAttribute("physxAutoAttachment:enableDeformableVertexAttachments").Set(True)
        auto_attachment_api.GetPrim().GetAttribute("physxAutoAttachment:enableRigidSurfaceAttachments").Set(True)
        auto_attachment_api.GetPrim().GetAttribute("physxAutoAttachment:enableCollisionFiltering").Set(True)
        auto_attachment_api.GetPrim().GetAttribute("physxAutoAttachment:collisionFilteringOffset").Set(0.01)
        auto_attachment_api.GetPrim().GetAttribute("physxAutoAttachment:enableDeformableFilteringPairs").Set(True)

        for key, value in list(prim_dict.items())[1:]:
            attachment_path = value.GetPath().AppendElementString(f"attachment_{key}")
            stalk_attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment_path)
            stalk_attachment.GetActor0Rel().SetTargets([value.GetPath()])
            stalk_attachment.GetActor1Rel().SetTargets([f"{prim_name}/stalk/plant_023"])
            auto_attachment_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(stalk_attachment.GetPrim())

    def make_deformable(self, prim_dict, simulation_resolution=10):
        key, value = list(prim_dict.items())[0]

        # Create the material
        deformableUtils.add_deformable_body_material(
            self.stage,
            self.deformable_material_path,
            youngs_modulus=7.5e15,
            poissons_ratio=0.1,
            damping_scale=1.0,
            dynamic_friction=0.5,
            density=10,
        )
        deformableUtils.add_physx_deformable_body(
            self.stage,
            value.GetPath(),
            collision_simplification=True,
            simulation_hexahedral_resolution=simulation_resolution,
            self_collision=False,
        )
        # physicsUtils.add_physics_material_to_prim(self.stage, value.GetPrim(), self.deformable_material_path)

        for key, value in list(prim_dict.items())[1:]:
            deformableUtils.add_physx_deformable_body(
                self.stage,
                value.GetPath(),
                collision_simplification=True,
                simulation_hexahedral_resolution=simulation_resolution,
                self_collision=False,
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Clamp and scale actions
        self.raw_actions = actions.clone()
        self.actions = (
            torch.clamp(self.raw_actions, -1, 1) * self.dt * self.action_scale
            + self._robot.data.joint_pos[:, self._arm_dof_idx]
        )
        # Apply limits to specific action dimensions
        self.actions[:, 0].clamp_(-1.0, 0.7)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.actions, joint_ids=self._arm_dof_idx)

    def _get_observations(self) -> dict:
        # Joint readings
        joint_readings = self._robot.data.joint_pos[:, self._arm_dof_idx]

        # RGB
        self.rgb_image = self._tiled_camera.data.output["rgb"].clone()
        self.rgb_image = self.rgb_image / 255.0
        # save_images_to_file(self.rgb_image, "rgb_obs.png")
        self.rgb_image_raw = self.rgb_image.clone()
        mean_tensor = torch.mean(self.rgb_image, dim=(1, 2), keepdim=True)
        self.rgb_image -= mean_tensor
        self.rgb_image = self.rgb_image.to(dtype=torch.float32)

        self.depth_image = self._tiled_camera.data.output["depth"].clone()
        self.depth_image[self.depth_image == float("inf")] = 0

        # End Effector Position
        self.ee_position = (
            torch.squeeze(self._robot.data.body_com_pos_w[:, self._arm_pos_idx[0]]) - self._robot.data.root_com_pos_w
        )
        # Compose observation image: RGB + K masks + depth
        # Select the correct cube mask per environment using torch.gather
        cube_masks_selected = torch.gather(
            self.cube_maskas, 3, self.target_cube_idx.view(-1, 1, 1, 1).expand(-1, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, 1)
        )
        # save_images_to_file(cube_masks_selected, "cube_masks.png")
        
        image_obs = torch.cat((self.rgb_image, cube_masks_selected, self.depth_image), dim=3)
        # print(image_obs.shape)
        # one-hot encoding of target cube index per env
        # goal_one_hot = F.one_hot(self.target_cube_idx, num_classes=self.cfg.num_goal_cubes).float()
        # Observations
        obs = {
            "rgb": image_obs.to(dtype=torch.float32),
            "joints": joint_readings.to(dtype=torch.float32),
            "ee_position": self.ee_position.to(dtype=torch.float32),
        }
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        plant_mask = calculate_plant_mask(self.depth_image)

        # Occlusion Reward: use selected cube's mask via one-hot selection
        # self.cube_maskas: [N, W, H, K], plant_mask: [N, W, H, 1]
        one_hot = (
            F.one_hot(self.target_cube_idx, num_classes=self.cfg.num_goal_cubes).float().unsqueeze(1).unsqueeze(1)
        )
        # broadcast one_hot: [N, 1, 1, K]
        selected_mask = torch.sum(self.cube_maskas * one_hot, dim=3, keepdim=True)
        occlusion_mask = selected_mask * plant_mask
        occlusion_pixels = torch.sum(occlusion_mask, dim=(1, 2, 3)).reshape(-1, 1)
        occlusion_reward = (1 - occlusion_pixels / (MASK_SIZE * MASK_SIZE)) * 10.0

        full_visibility_reward = torch.where(
            occlusion_pixels <= 160, torch.tensor(3.0, device=self.device), torch.tensor(0.0, device=self.device)
        ).reshape(self.num_envs)

        # Create mask where full visibility is achieved
        full_visibility_mask = full_visibility_reward > 0

        self.detection_sequence.append(full_visibility_mask)

        # print(f"Detection Sequence: {self.detection_sequence}")
        visibility_history = torch.stack(list(self.detection_sequence))
        sustained_detection = torch.all(visibility_history, dim=0)
        sustained_reward = sustained_detection.float() * 20.0  # High reward

        # Penalize action inputs when visibility is achieved (using action magnitude)
        action_penalty = torch.norm(self.raw_actions.clamp_(-1.0, 1.0), dim=1) * -0.06
        action_penalty *= full_visibility_mask.float()

        self.done = sustained_detection

        occlusion_reward = occlusion_reward.reshape(self.num_envs)
        total_reward = self.contact_reward + full_visibility_reward + sustained_reward + action_penalty + occlusion_reward

        return total_reward

    # def _get_hard_constraints(self) -> torch.Tensor:
    #     pose_cons, vel_cons = self._get_deformation_reward()
    #     return pose_cons.reshape(self.num_envs)

    # def get_deformation_reward(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     # Deformation Reward
    #     deformable_object: DeformableObject = self.deformable_objects_list[0]
    #     default_nodal_state = deformable_object.data.default_nodal_state_w.clone()
    #     default_nodel_pose = default_nodal_state[:, :, :3]
    #     nodal_position = deformable_object.data.nodal_pos_w.clone()
    #     nodal_velocity = deformable_object.data.nodal_vel_w.clone()
    #     deformation = torch.norm(nodal_position - default_nodel_pose, dim=2)
    #     # print(f"Deformation: {deformation.shape}")
    #     velocity_deformation = torch.norm(nodal_velocity, dim=2)
    #     return torch.sum(deformation, dim=1), torch.sum(velocity_deformation, dim=1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Collision Reward
        self.contact_reward = self.calculate_contact_reward(len(self.cfg.link_dof_name))
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.in_contact | self.done, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Handle default environment indices
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids][:, self._arm_dof_idx] + self.root_init_angles[env_ids]
        # Sample random only second joint
        if np.random.uniform() < 0.5:
            joint_pos[:, 2] = sample_uniform(-0.52, -0.17, joint_pos[:, 2].shape, self.device)
            # joint_pos[:, 0] = sample_uniform(-0.5, 0.0, joint_pos[:, 0].shape, self.device)
        else:
            joint_pos[:, 2] = sample_uniform(0.2, 0.3, joint_pos[:, 2].shape, self.device)
            # joint_pos[:, 0] = sample_uniform(0.0, 0.5, joint_pos[:, 0].shape, self.device)

        # Store initial angles reference
        self.last_angles[env_ids] = joint_pos.clone()

        # Configure root state with environment offsets
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Update internal state buffers
        joint_vel = torch.zeros_like(joint_pos, device=self.device)

        # Write states to physics simulation
        self._robot.set_joint_position_target(joint_pos, self._arm_dof_idx, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, self._arm_dof_idx, env_ids)

        # Reset the nodal state of the deformable objects
        for deformable_object in self.deformable_objects_list:
            deformable_object: DeformableObject
            nodal_state = deformable_object.data.default_nodal_state_w.clone()
            nodal_state = nodal_state[env_ids, :, :]
            deformable_object.write_nodal_state_to_sim(nodal_state, env_ids)

    # Randomly sample target cube index per env and reposition all cubes
        num_envs_reset = len(env_ids)
        rand_idx = torch.randint(0, self.cfg.num_goal_cubes, (num_envs_reset,), device=self.device)
        self.target_cube_idx[env_ids] = rand_idx

        # Base positions for cubes
        base_states = self.root_cube_positions[env_ids]  # [E, K, state]
        base_positions = base_states[:, :, :3]

        # Jitter x for all cubes
        x_jitter = sample_uniform(0.00, 0.15, base_positions[..., 0].shape, self.device)
        new_positions = base_positions.clone()
        new_positions[..., 0] += x_jitter

        # Write all cubes
        for k, cube in enumerate(self.goal_cubes):
            # Construct full root state for this cube (pos + quat + lin/ang vel)
            states_k = base_states[:, k, :].clone()
            states_k[:, :3] = new_positions[:, k, :3]
            cube.write_root_pose_to_sim(states_k[:, :7], env_ids)
            cube.write_root_velocity_to_sim(torch.zeros_like(states_k[:, 7:], device=self.device), env_ids)

        # Compute masks for all cubes (K channels)
        for k in range(self.cfg.num_goal_cubes):
            pos_k = new_positions[:, k, :3]
            ones = torch.ones((pos_k.shape[0], 1), device=self.device)
            points_3d_hom = torch.cat((pos_k, ones), dim=1).unsqueeze(2)  # [E, 4, 1]
            image_points_hom = torch.bmm(self.projection_matrix[env_ids], points_3d_hom).squeeze(2)
            image_points_2d = image_points_hom[:, :2] / image_points_hom[:, 2:].expand(-1, 2)
            cube_masks_k = torch.flip(
                create_batched_masks(
                    (self.cfg.tiled_camera.width, self.cfg.tiled_camera.height), image_points_2d, MASK_SIZE
                ),
                dims=[2],
            ).to(dtype=torch.float32, device=self.device)
            # assign to channel k
            self.cube_maskas[env_ids, :, :, k] = cube_masks_k

        raw_rgb = (self._tiled_camera.data.output["rgb"].clone()[env_ids] / 255.0).to(dtype=torch.float32)
        mean_rbg = torch.mean(raw_rgb, dim=(1, 2), keepdim=True, dtype=torch.float32)
        self.rgb_image[env_ids] = (raw_rgb - mean_rbg).to(dtype=torch.float32)

        self.last_ee_position = (
            torch.squeeze(self._robot.data.body_com_pos_w[:, self._arm_pos_idx[0]]) - self._robot.data.root_com_pos_w
        )

        tensor_seq = torch.stack(list(self.detection_sequence))  # Convert list of tensors to a single tensor
        tensor_seq[:, env_ids] = 0  # Zero out selected indices across all rows
        self.detection_sequence = deque(list(tensor_seq), maxlen=self.cfg.maxlen)  # Convert back to list if needed

    # @torch.jit.script
    def calculate_contact_reward(self, num_sensors: int) -> torch.Tensor:
        # Contact readings
        contact_readings = []
        for i in range(num_sensors):
            contact_readings.append(self.scene.sensors[f"contact_sensor_{i}"].data.net_forces_w)

        contact_readings_tensor = torch.cat(contact_readings, dim=1)
        self.in_contact = torch.any(contact_readings_tensor != 0, dim=(1, 2))
        return torch.where(
            self.in_contact, torch.tensor(-5.0, device=self.device), torch.tensor(0.0, device=self.device)
        )


@torch.jit.script
def reset_sequence(sequence, env_ids: torch.Tensor):
    for row in sequence:
        row[env_ids] = torch.zeros_like(row[env_ids])


# @torch.jit.script
def create_batched_masks(image_size, cube_pixel_locations, n):
    """
    Generate masks for multiple environments where an n x n block around the pixel location is set to 1,
    and everything else is 0.

    Args:
        image_size (tuple): Size of the image (height, width).
        cube_pixel_locations (torch.Tensor): Tensor of shape (n_envs, 2) containing (x, y) coordinates.
        n (int): Size of the block (n x n).

    Returns:
        torch.Tensor: A binary mask tensor of shape (n_envs, height, width).
    """
    n_envs = cube_pixel_locations.shape[0]
    height, width = image_size

    # Create an empty mask tensor for all environments
    masks = torch.zeros((n_envs, height, width), dtype=torch.float32, device=cube_pixel_locations.device)

    # Extract x and y coordinates from cube_pixel_locations
    pixelx = cube_pixel_locations[:, 0]
    pixely = cube_pixel_locations[:, 1]

    # Calculate the boundaries of the n x n block for each environment
    half_n = n // 2
    x_start = torch.clamp(pixelx - half_n, 0, width - 1).int()
    x_end = torch.clamp(pixelx + half_n + 1, 0, width).int()
    y_start = (torch.clamp(pixely - half_n, 0, height - 1)).int()
    y_end = (torch.clamp(pixely + half_n + 1, 0, height)).int()

    # Set the n x n block to 1 for each environment
    for i in range(n_envs):
        masks[i, y_start[i] : y_end[i], x_start[i] : x_end[i]] = 1.0

    return masks


@torch.jit.script
def calculate_plant_mask(depth_obs):
    plant_mask = ((depth_obs >= 0.1) & (depth_obs <= 0.4)).clone().detach()
    return plant_mask.float()
