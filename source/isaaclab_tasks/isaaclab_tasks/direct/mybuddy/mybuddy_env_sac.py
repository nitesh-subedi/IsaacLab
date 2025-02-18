# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
import omni

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file, ContactSensor, ContactSensorCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg, RenderCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import PhysicsContext
from pxr import UsdPhysics, Gf, PhysxSchema
from omni.physx.scripts import deformableUtils, physicsUtils
from omni.isaac.lab.utils.math import sample_uniform, quat_from_euler_xyz
from omni.isaac.lab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg

# from .model import MultiInputSingleOutputLSTM
# import torchvision.transforms as transforms
from collections import deque
import time
import gymnasium as gym
import numpy as np
from collections import deque


@configclass
class MyBuddyEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 20.0
    action_scale = 0.25
    state_space = 0
    # rerender_on_reset = True

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation,
                                       physx=PhysxCfg(solver_type=1, enable_ccd=True),
                                       gravity=(0.0, 0.0, 9.81),
                                       render=RenderCfg()
                                    )
    # robot
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/mybuddy",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/nitesh/workspace/rosws/mybuddy_ws/src/mybuddy_description/urdf/mybuddy_rotate.usd",
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
                effort_limit=50000.0,
                velocity_limit=20.0,
                stiffness=800.0,
                damping=4.0,
            )
        },
        # init_state=ArticulationCfg.InitialStateCfg(joint_pos={"left_arm_j1": -1.57, 
        #                                                       "left_arm_j2": 0.0, 
        #                                                       "left_arm_j3": 2.0943, 
        #                                                       "left_arm_j4": -2.0943, 
        #                                                       "left_arm_j5": 3.1415}),

    )
    arm_dof_name = ["bc2bl","left_arm_j1", "left_arm_j2", "left_arm_j3", "left_arm_j4", "left_arm_j5"]
    # arm_dof_name = ["left_arm_j1", "left_arm_j2", "left_arm_j3", "left_arm_j4", "left_arm_j5"]
    link_dof_name = ["left_arm_l1", "left_arm_l3", "left_arm_l4", "left_arm_l5", "left_arm_l6"]

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.04, 0.36), rot=(0.0, 0.0, 0.53833, 0.84274), convention="opengl"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.8, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 7.0)
        ),
        width=224,
        height=224,
    )

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(arm_dof_name),))
    observation_space = {"joints": gym.spaces.Box(low=-20.00, high=20.0, shape=(len(arm_dof_name),), dtype=np.float16),
                         "rgb": gym.spaces.Box(low=0, high=255, shape=(tiled_camera.height, tiled_camera.width, 4), dtype=np.uint8),
                         "ee_position": gym.spaces.Box(low=-20.00, high=20.0, shape=(3,), dtype=np.float16)}
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    # reset
    max_y_pos = 0.0

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

        self.image_sequences = deque(maxlen=5)
        self.cube_detected_time = None
        self.root_init_angles = torch.deg2rad(torch.tensor([0, -90, 0, 120, -120, 180], device=self.device)).repeat(self.num_envs, 1).view(-1, len(self.cfg.arm_dof_name))
        self.last_angles = self.root_init_angles.clone()


        self.camera_intrinsics = self._tiled_camera.data.intrinsic_matrices
        camera_rotation = self._tiled_camera.data.quat_w_opengl
        # print(f"Camera Rotation: {camera_rotation}")
        camera_position = self._tiled_camera.data.pos_w
        rotation_matrix = self.quaternion_to_rotation_matrix(camera_rotation)
        translation_vector = -torch.bmm(rotation_matrix, camera_position.unsqueeze(-1))
        self.extrinsic_matrix = torch.cat((rotation_matrix, translation_vector), dim=2)  # [2, 3, 4]
        print(f"Extrinsic Matrix: {self.extrinsic_matrix.shape}")
        self.projection_matrix = torch.bmm(self.camera_intrinsics, self.extrinsic_matrix)
        print(f"Projection Matrix: {self.projection_matrix.shape}")

        self.root_cube_position = self.goal_cube.data.root_com_state_w.clone()
        self.cube_maskas = torch.zeros((self.num_envs, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, 1), device=self.device, dtype=torch.uint8)
        self.rgb_image = torch.zeros((self.num_envs, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, 3), device=self.device, dtype=torch.uint8)
        self.occlusion_time_below_threshold = torch.zeros(self.num_envs, 1, device=self.device)

    
    @staticmethod
    def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of quaternions to rotation matrices.

        Args:
            quat (torch.Tensor): A tensor of shape (N, 4), where each row is (w, x, y, z).

        Returns:
            torch.Tensor: A tensor of shape (N, 3, 3) representing rotation matrices.
        """
        # Normalize the quaternions to ensure valid rotations
        quat = quat / quat.norm(dim=1, keepdim=True)

        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Compute the rotation matrix elements
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rot_mats = torch.stack([
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
        ], dim=1).reshape(-1, 3, 3)

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

    def _setup_scene(self):
        self.stage = stage_utils.get_current_stage()
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())


        euler = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        quat = quat_from_euler_xyz(*euler).cpu().numpy()
        plant_cfg = sim_utils.UsdFileCfg(usd_path="/home/nitesh/isaac_sim_docker_ws/isaac_sim_assets/plant_v21/plant_v21.usd", scale=(0.05, 0.05, 0.05))
        plant_cfg.func("/World/envs/env_.*/Plant1", plant_cfg, translation=(0.06, -0.22, 0.0), orientation=(quat[0], quat[1], quat[2], quat[3]))


        # euler = torch.tensor([0.0, 0.0, 45.0], device=self.device)
        # quat = quat_from_euler_xyz(*euler).cpu().numpy()
        # plant_cfg_2 = sim_utils.UsdFileCfg(usd_path="/home/nitesh/isaac_sim_docker_ws/isaac_sim_assets/plant_v21/plant_v21.usd", scale=(0.05, 0.05, 0.05))
        # plant_cfg_2.func("/World/envs/env_.*/Plant2", plant_cfg_2, translation=(-0.06, -0.22, 0.0), orientation=(quat[0], quat[1], quat[2], quat[3]))
    
        # add light
        light_cfg = sim_utils.SphereLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0), radius=1.0)
        light_cfg.func("/World/envs/env_.*/Light", light_cfg, translation=(0.0, -0.4, 2.0))

        # add background plane
        background_plane_cfg = sim_utils.CuboidCfg(size=(1, 0.01, 1))
        background_plane_cfg.func("/World/envs/env_.*/BackgroundPlane", background_plane_cfg, translation=(0.0, -0.7, 0.0))

        self.cube_size = (0.04, 0.04, 0.04)
        cube_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/GoalCube",
            spawn=sim_utils.CuboidCfg(size=self.cube_size,
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                                      rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                          disable_gravity=True,
                                      ),
                                      mass_props=sim_utils.MassPropertiesCfg(mass=1.0)),
                                      init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.4, 0.22)))
        
        self.goal_cube = RigidObject(cube_cfg)

        contact_sensors_list = []
        for link_names in self.cfg.link_dof_name:
            contact_forces = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/mybuddy/{link_names}",
                update_period=0.0,
                history_length=6,
                debug_vis=True,
                filter_prim_paths_expr=["/World/envs/env_.*/GoalCube"]
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
        self.scene.rigid_objects["goal_cube"] = self.goal_cube
        for i, deformable_object in enumerate(self.deformable_objects_list):
            self.scene.deformable_objects[f"stalk_{i}"] = deformable_object
        
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # self.camera_rotation = self._tiled_camera.data.quat_w_world[0]
        self.detection_sequence = deque(maxlen=10)

        # set all the elements of detection sequence to False
        for i in range(10):
            self.detection_sequence.append(torch.zeros(self.num_envs, device=self.device))



    def attach_cylinder_to_ground(self, prim_dict, prim_name):
        key, value = list(prim_dict.items())[0]
        attachment_path = value.GetPath().AppendElementString(f"attachment_{key}")
        stalk_attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment_path)
        stalk_attachment.GetActor0Rel().SetTargets([value.GetPath()])
        stalk_attachment.GetActor1Rel().SetTargets(["/World/ground/GroundPlane/CollisionPlane"])
        auto_attachment_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(stalk_attachment.GetPrim())
        # Set attributes to reduce initial movement and gap
        auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:deformableVertexOverlapOffset').Set(0.005)
        # auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:rigidSurfaceSamplingDistance').Set(0.01)
        auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:enableDeformableVertexAttachments').Set(True)
        auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:enableRigidSurfaceAttachments').Set(True)
        auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:enableCollisionFiltering').Set(True)
        auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:collisionFilteringOffset').Set(0.01)
        auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:enableDeformableFilteringPairs').Set(True)
    
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
            youngs_modulus=7.5e10,
            poissons_ratio=0.1,
            damping_scale=1.0,
            dynamic_friction=0.5,
            density=10
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
        # Clamp actions and scale
        new_actions = torch.clamp(actions.clone(), -1, 1) * self.dt * 2.0
        # Increment angles based on previous state
        self.last_angles += new_actions
        self.last_angles[:, 1] = torch.clamp(self.last_angles[:, 1], -1.91986, -1.39626)
        self.last_angles[:, 0] = torch.clamp(self.last_angles[:, 0], -0.5, 0.5)

        
        # Update current actions
        self.actions = self.last_angles
        
        # self.actions[:, 1] = torch.clamp(self.actions[:, 1], -0.523599, 0.349066)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.actions, joint_ids=self._arm_dof_idx)
    
    def _get_observations(self) -> dict:
        # Joint readings
        joint_readings = self._robot.data.joint_pos[:, self._arm_dof_idx]

        # RGB
        self.rgb_image = self._tiled_camera.data.output['rgb'].clone()
        self.rgb_image = self.rgb_image.to(dtype=torch.uint8)

        # End Effector Position
        self.ee_position = torch.squeeze(self._robot.data.body_com_pos_w[:, self._arm_pos_idx[0]]) - self._robot.data.root_com_pos_w
        # save_images_to_file(self.rgb_image, "/home/nitesh/IsaacLab/mybuddy_rgb_image.png")
        # self.cube_maskas = torch.zeros_like(self.cube_maskas)
        # self.ee_position = torch.zeros_like(self.ee_position)
        # joint_readings = torch.zeros_like(joint_readings)
        image_obs = torch.cat((self.rgb_image, self.cube_maskas), dim=3)
        # Observations
        obs = {"rgb":image_obs.to(dtype=torch.uint8), "joints": joint_readings.to(dtype=torch.float16), "ee_position": self.ee_position.to(dtype=torch.float16)}
        observations = {"policy": obs}

        return observations


    # def _get_rewards(self) -> torch.Tensor:
    #     depth_image = self._tiled_camera.data.output['depth']
    #     depth_image[depth_image == float("inf")] = 0
    #     fruit_reward, plant_mask = calculate_cube_reward(depth_image.clone(), self.device)

    #     # ee_position_reward
    #     # ee_pos_reward = 0.0#-self.ee_position[:, 1] * 5.0
    #     ee_pos_done_reward = torch.where(self.ee_position[:, 1] > self.cfg.max_y_pos, torch.tensor(-10.0, device=self.device), torch.tensor(0.0, device=self.device)).reshape(self.num_envs)
    #     # print(f"EE Position Reward: {ee_pos_reward}")
    #     ee_position_randomness_reward = torch.norm(self.ee_position - self.last_ee_position, dim=1) * -10.0
    #     self.last_ee_position = self.ee_position

    #     # Collision Reward
    #     contact_reward = self.calculate_contact_reward(len(self.cfg.link_dof_name))

    #     # # Occlusion Reward
    #     occlusion_mask = self.cube_maskas * plant_mask
    #     # save_images_to_file(occlusion_mask * 255, "/home/nitesh/IsaacLab/mybuddy_occlusion_mask.jpg")
    #     occlusion_pixels = torch.sum(occlusion_mask, dim=(1, 2, 3)).reshape(-1, 1)
    #     occlusion_reward = (-occlusion_pixels / (255*255)) * 10.0

    #     full_visibility_reward = torch.where(occlusion_pixels <= 20, torch.tensor(20.0, device=self.device), torch.tensor(0.0, device=self.device)).reshape(self.num_envs)
        
    #     occlusion_reward = occlusion_reward.reshape(self.num_envs)
    #     total_reward = fruit_reward + contact_reward + occlusion_reward + full_visibility_reward + ee_pos_done_reward + ee_position_randomness_reward
    #     # self.extras = {"ee_pos_reward": ee_pos_reward}
    #     return total_reward

    def _get_rewards(self) -> torch.Tensor:
        depth_image = self._tiled_camera.data.output['depth']
        depth_image[depth_image == float("inf")] = 0
        fruit_reward, plant_mask = calculate_cube_reward(depth_image.clone(), self.device)

        # ee_position_reward
        ee_pos_reward = -self.ee_position[:, 1] * 5.0
        ee_pos_done_reward = torch.where(self.ee_position[:, 1] > self.cfg.max_y_pos, 
                                        torch.tensor(-5.0, device=self.device), 
                                        torch.tensor(0.0, device=self.device)).reshape(self.num_envs)
        ee_position_randomness_reward = torch.norm(self.ee_position - self.last_ee_position, dim=1) * -10.0
        self.last_ee_position = self.ee_position

        # Collision Reward
        contact_reward = self.calculate_contact_reward(len(self.cfg.link_dof_name))

        # Occlusion Reward
        # self._compute_cube_masks()
        occlusion_mask = self.cube_maskas * plant_mask
        # save_images_to_file(occlusion_mask * 255, "/home/nitesh/IsaacLab/mybuddy_occlusion_mask.jpg")
        occlusion_pixels = torch.sum(occlusion_mask, dim=(1, 2, 3)).reshape(-1, 1)
        occlusion_reward = (-occlusion_pixels / (255*255)) * 20.0

        full_visibility_reward = torch.where(occlusion_pixels <= 20,
                                            torch.tensor(2.0, device=self.device), 
                                            torch.tensor(0.0, device=self.device)).reshape(self.num_envs)
        
        # Create mask where full visibility is achieved
        full_visibility_mask = (full_visibility_reward > 0)

        self.detection_sequence.append(full_visibility_mask)

        # print(f"Detection Sequence: {self.detection_sequence}")
        visibility_history = torch.stack(list(self.detection_sequence))
        sustained_detection = torch.all(visibility_history, dim=0)
        sustained_reward = sustained_detection.float() * 20.0  # High reward
        self.done = sustained_detection
        
        # Penalize action inputs when visibility is achieved (using action magnitude)
        action_penalty = torch.norm(self.actions, dim=1) * -1.0  # Adjust coefficient as needed
        action_penalty *= full_visibility_mask.float()

        
        occlusion_reward = occlusion_reward.reshape(self.num_envs)
        total_reward = (fruit_reward + contact_reward + occlusion_reward + 
                        full_visibility_reward + ee_pos_done_reward + 
                        ee_position_randomness_reward + action_penalty + ee_pos_reward
                        + sustained_reward)  # Add the penalty here
        
        return total_reward



    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = self.ee_position[:, 1] > torch.ones_like(self.ee_position[:, 1]) * self.cfg.max_y_pos
        return out_of_bounds | self.in_contact | self.done, time_out


    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Handle default environment indices
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids][:, self._arm_dof_idx] + self.root_init_angles[env_ids]
        # Sample random only second joint
        if np.random.uniform() < 0.6:
            joint_pos[:, 2] = sample_uniform(-0.52, -0.17, joint_pos[:, 1].shape, self.device)
        else:
            joint_pos[:, 2] = sample_uniform(0.01, 0.05, joint_pos[:, 1].shape, self.device)

        # Store initial angles reference
        # self.init_angles[env_ids] = joint_pos#joint_init_pos[:, self._arm_dof_idx]
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
            deformable_object : DeformableObject
            nodal_state = deformable_object.data.default_nodal_state_w.clone()
            nodal_state = nodal_state[env_ids, :, :] #+ sample_uniform(-0.05, 0.05, nodal_state[env_ids, :, :].shape, self.device)
            # nodal_state[:, :, 0] += sample_uniform(-0.05, 0.05, nodal_state[:, :, 0].shape, self.device)
            deformable_object.write_nodal_state_to_sim(nodal_state, env_ids)

        cube_position = self.root_cube_position[env_ids]
        cube_position[:, 0] += sample_uniform(-0.15, 0.15, cube_position[:, 0].shape, self.device)
        self.goal_cube.write_root_pose_to_sim(cube_position[:, :7], env_ids)
        self.goal_cube.write_root_velocity_to_sim(torch.zeros_like(cube_position[:, 7:], device=self.device), env_ids)

        # Get Cube Pixel location
        ones = torch.ones(cube_position[:, :3].shape[0], 1, device=self.device)  # [N, 1]
        points_3d_hom = torch.cat((cube_position[:, :3], ones), dim=1).unsqueeze(2)  # [N, 4, 1]
        image_points_hom = torch.bmm(self.projection_matrix[env_ids], points_3d_hom).squeeze(2)
        image_points_2d = image_points_hom[:, :2] / image_points_hom[:, 2:].expand(-1, 2)
        # print(f"Image Points 2D: {image_points_2d}")
        cube_masks = torch.flip(create_batched_masks((self.cfg.tiled_camera.width, self.cfg.tiled_camera.height), image_points_2d, 80), dims=[2]).to(dtype=torch.uint8, device=self.device)
        self.cube_maskas[env_ids] = cube_masks.reshape(-1, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, 1)

        self.rgb_image[env_ids] = (self._tiled_camera.data.output['rgb'].clone()[env_ids]).to(dtype=torch.uint8)
        # save_images_to_file(raw_rgb/255.0, "/home/nitesh/IsaacLab/mybuddy_rgb.jpg")
        # save_images_to_file(self.cube_maskas * self.rgb_image, "/home/nitesh/isaac_omniverse/IsaacLab/mybuddy_cube_masks.jpg")]
        self.last_ee_position = torch.squeeze(self._robot.data.body_com_pos_w[:, self._arm_pos_idx[0]]) - self._robot.data.root_com_pos_w

        for row in self.detection_sequence:
            row[env_ids] = torch.zeros_like(row[env_ids])
        

    # @torch.jit.script
    def calculate_contact_reward(self, num_sensors: int) -> torch.Tensor:
        # Contact readings
        contact_readings = []
        for i in range(num_sensors):
            contact_readings.append(self.scene.sensors[f"contact_sensor_{i}"].data.net_forces_w)
        
        contact_readings_tensor = torch.cat(contact_readings, dim=1)
        self.in_contact = torch.any(contact_readings_tensor != 0, dim=(1, 2))
        return torch.where(self.in_contact, torch.tensor(-5.0, device=self.device), torch.tensor(0.0, device=self.device))

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
    masks = torch.zeros((n_envs, height, width), dtype=torch.uint8, device=cube_pixel_locations.device)
    
    # Extract x and y coordinates from cube_pixel_locations
    pixelx = cube_pixel_locations[:, 0]
    pixely = cube_pixel_locations[:, 1]
    
    # Calculate the boundaries of the n x n block for each environment
    half_n = n // 2
    x_start = torch.clamp(pixelx - half_n, 0, width - 1).int()
    x_end = torch.clamp(pixelx + half_n + 1, 0, width).int()
    y_start = torch.clamp(pixely - half_n, 0, height - 1).int()
    y_end = torch.clamp(pixely + half_n + 1, 0, height).int()

    # Set the n x n block to 1 for each environment
    for i in range(n_envs):
        masks[i, y_start[i]:y_end[i], x_start[i]:x_end[i]] = 255
    
    return masks

# @torch.jit.script
def calculate_cube_reward(depth_obs, device):
    fruit_mask = ((depth_obs >= 0.35) & (depth_obs <= 0.54)).clone().detach().to(dtype=torch.float16, device=device) / 255
    plant_mask = ((depth_obs >= 0.01) & (depth_obs <= 0.35)).clone().detach().to(dtype=torch.float16, device=device) / 255
    fruit_mask = torch.clamp(fruit_mask, 0.0, 1.0)
    plant_mask = torch.clamp(plant_mask, 0.0, 1.0)
    fruit_mask[:, int(1.5 * 256 / 3):, :, :] = 0.0
    # save_images_to_file(fruit_mask * 10, "/home/nitesh/isaac_omniverse/IsaacLab/mybuddy_fruit_mask.png")
    # save_images_to_file(plant_mask * 10, "/home/nitesh/isaac_omniverse/IsaacLab/mybuddy_plant_mask.png")
    # plant_pixels = torch.sum(plant_mask, dim=(1, 2, 3))
    fruit_pixels = torch.sum(fruit_mask, dim=(1, 2, 3))
    fruit_reward = (fruit_pixels / (255*255)) * 5.0
    # plant_reward = -(plant_pixels / (255*255)) * 100.0
    return fruit_reward, plant_mask


# @torch.jit.script
def compute_arm_joint_init_pos(
    env_ids: torch.Tensor,
    default_joint_pos: torch.Tensor,
    arm_dof_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    # Compute ARM_JOINT_INIT_DEGREES as a tensor
    if torch.rand(1, device=device).item() > 0.3:
        arm_joint_degrees = torch.tensor(
            [-90.0, torch.rand(1, device=device).item() * 30 - 50, 120.0, -120],
            device=device,
        )
    else:
        arm_joint_degrees = torch.tensor(
            [-90.0, torch.rand(1, device=device).item() * 20, 120.0, -120],
            device=device,
        )

    # Convert degrees to radians
    arm_joint_radians = torch.deg2rad(arm_joint_degrees)

    # Create base joint configuration with default values
    base_joint_pos = default_joint_pos[env_ids]
    joint_init_pos = torch.zeros_like(base_joint_pos, device=device)

    # Initialize arm joints with strategic default angles
    arm_angles = arm_joint_radians.repeat(len(env_ids), 1)
    joint_init_pos[:, arm_dof_idx] = arm_angles

    return joint_init_pos
