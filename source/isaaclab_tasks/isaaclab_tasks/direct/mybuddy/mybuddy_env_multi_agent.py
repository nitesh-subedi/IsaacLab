# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
import omni
import random

# from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import PhysicsContext
from pxr import UsdPhysics, Gf, PhysxSchema
from omni.physx.scripts import deformableUtils, physicsUtils
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.assets import DeformableObject, DeformableObjectCfg

from .model import MultiInputSingleOutputLSTM
import torchvision.transforms as transforms
from collections import deque
import time
import gymnasium as gym
import numpy as np


@configclass
class MyBuddyEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    max_episode_length = 500
    action_scale = 4.0
    action_space = {"left_arm":gym.spaces.Box(low=-1.0, high=1.0, shape=(6,)), "right_arm":gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))}
    state_space = 0
    # rerender_on_reset = True

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # robot
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/mybuddy",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/nitesh/IsaacLab/mybuddy.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ), 

        actuators={
            ".*": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=10.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            )
        }
    )
    arm_dof_name = ["left_arm_j1", "left_arm_j2", "left_arm_j3", "left_arm_j4", "left_arm_j5", "left_arm_j6"]

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

    observation_space = {"joints": 6, "rgb": [tiled_camera.height, tiled_camera.width, 3], "ee_position": 3}

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=False)

    # reset
    max_y_pos = 0.0

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0


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

        self.model_ = MultiInputSingleOutputLSTM().to(self.device)
        state_dict = torch.load("/home/nitesh/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/mybuddy/model_185.pth", map_location=self.device)

        # Remove the 'module.' prefix from the keys
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.model_.load_state_dict(state_dict)

        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.image_sequences = deque(maxlen=5)
        self.cube_detected_time = None
        self.init_angles = torch.deg2rad(torch.tensor([-90, -10, 120, -120, 180, 0], device=self.device)).repeat(self.num_envs, 1).view(-1, 6)

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _setup_scene(self):
        self.stage = stage_utils.get_current_stage()
        self.physics_context = PhysicsContext()
        self.physics_context.set_solver_type("TGS")
        self.physics_context.set_broadphase_type("GPU")
        self.physics_context.enable_gpu_dynamics(True)
        scene_ = UsdPhysics.Scene.Define(self.stage, "/physicsScene")
        scene_.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, 1.0))
        scene_.CreateGravityMagnitudeAttr().Set(9.81)

        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        plant_cfg = sim_utils.UsdFileCfg(usd_path="/home/nitesh/isaac_sim_docker_ws/isaac_sim_assets/plant_v21/plant_v21.usd", scale=(0.05, 0.05, 0.05))
        plant_cfg.func("/World/envs/env_.*/Plant", plant_cfg, translation=(0.0, -0.2, 0.0))
    
        # add light
        light_cfg = sim_utils.SphereLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0), radius=1.0)
        light_cfg.func("/World/envs/env_.*/Light", light_cfg, translation=(0.0, -0.4, 2.0))

        # add background plane
        background_plane_cfg = sim_utils.CuboidCfg(size=(1, 0.01, 1))
        background_plane_cfg.func("/World/envs/env_.*/BackgroundPlane", background_plane_cfg, translation=(0.0, -0.7, 0.0))

        # add cube
        self.cube_cfg = sim_utils.CuboidCfg(size=(0.04, 0.04, 0.04), visual_material=
                                       sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))
        self.cube_cfg.func("/World/envs/env_.*/GoalCube", self.cube_cfg, translation=(0.0, -0.4, 0.22))

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=[])

        self.deformable_objects_list = []

        deformable_plant_cfg = DeformableObjectCfg(
                prim_path="/World/envs/env_.*/Plant/stalk",
                debug_vis=True,
            )

        plant_object = DeformableObject(deformable_plant_cfg)
        self.deformable_objects_list.append(plant_object)
        
        # Deformable Objects
        for i in range(1, 6):
            deformable_plant_cfg = DeformableObjectCfg(
                prim_path=f"/World/envs/env_.*/Plant/stalk{i}",
                debug_vis=True,
            )

            plant_object = DeformableObject(deformable_plant_cfg)
            self.deformable_objects_list.append(plant_object)

        self.deformable_material_path = omni.usd.get_stage_next_free_path(self.stage, "/plant_material", True)
        for env_path in self.scene.env_prim_paths:
            plant_prim = self.stage.GetPrimAtPath(f"{env_path}/Plant")
            # plant_prim = UsdGeom.Mesh.Define(self.stage, f"{env_path}/Plant/stalk")
            plant_meshes = plant_prim.GetAllChildren()
            plant_meshes = [mesh.GetAllChildren()[0] for mesh in plant_meshes]
            plant_meshes = plant_meshes[1:]
            # print("Plant Prim Children: ", plant_prim.GetAllChildrenNames()[1:])
            prim_dict = dict(zip(plant_prim.GetAllChildrenNames()[1:], plant_meshes))
            self.make_deformable(prim_dict)
            self.attach_cylinder_to_ground(prim_dict, f"{env_path}/Plant")

        # add articulation to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        for i, deformable_object in enumerate(self.deformable_objects_list):
            self.scene.deformable_objects[f"stalk_{i}"] = deformable_object
        
        self.dt = self.cfg.sim.dt * self.cfg.decimation


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
            poissons_ratio=0.3,
            damping_scale=0.5,
            dynamic_friction=0.5,
            density=1000
        )
        deformableUtils.add_physx_deformable_body(
                self.stage,
                value.GetPath(),
                collision_simplification=True,
                simulation_hexahedral_resolution=simulation_resolution,
                self_collision=False,
            )
        physicsUtils.add_physics_material_to_prim(self.stage, value.GetPrim(), self.deformable_material_path)
        
        for key, value in list(prim_dict.items())[1:]:
            deformableUtils.add_physx_deformable_body(
                self.stage,
                value.GetPath(),
                collision_simplification=True,
                simulation_hexahedral_resolution=simulation_resolution,
                self_collision=False,
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        new_actions = torch.clamp(actions.clone(), -1, 1)
        self.actions = self.action_scale * new_actions * self.dt + self.init_angles
        self.init_angles = self.actions

    def _apply_action(self) -> None:
        # print(f"Actions: {self.actions}")
        self._robot.set_joint_position_target(self.actions, joint_ids=self._arm_dof_idx)
    

    def _get_observations(self) -> dict:
        # Joint readings
        joint_readings = self.joint_pos[:, self._arm_dof_idx].clone()

        # RGB
        self.rgb_image = self._tiled_camera.data.output['rgb'] / 255.0
        mean_tensor = torch.mean(self.rgb_image, dim=(1, 2), keepdim=True)
        self.rgb_image -= mean_tensor

        # End Effector Position
        self.ee_position = torch.squeeze(self._robot.data.body_com_pos_w[:, self._arm_pos_idx[0]]) - self._robot.data.root_com_pos_w

        # Observations
        obs = {"rgb":self.rgb_image, "joints": joint_readings, "ee_position": self.ee_position}
        observations = {"policy": obs}

        return observations

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.joint_pos[env_ids, self._arm_dof_idx]
        self._robot.data.body_com_pos_w[env_ids, self._arm_pos_idx[0]] - self._robot.data.root_com_pos_w[env_ids]

    def _get_rewards(self) -> torch.Tensor:
        depth_image = self._tiled_camera.data.output['depth']
        # depth_image = self._rtx_camera.data.output['distance_to_image_plane']
        depth_image[depth_image == float("inf")] = 0
        fruit_reward, plant_reward = self.calculate_cube_reward(depth_image.clone(), self.device)

        # ee_position_reward
        ee_pos_reward = -self.ee_position[:, 1] * 5.0

        # total reward
        total_reward = fruit_reward + ee_pos_reward + plant_reward
        # print(f"Fruit Reward: {fruit_reward}, EE Position Reward: {ee_pos_reward}")
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = self.ee_position[:, 1] > torch.ones_like(self.ee_position[:, 1]) * self.cfg.max_y_pos
        # if time_out.any() or out_of_bounds.any():
        #     print(f"Time Out: {time_out}, \n Out of Bounds: {out_of_bounds}")
        # if self.done.any():
        #     print(f"Done: {self.done}")
        return out_of_bounds, time_out


    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Handle default environment indices
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        print(f"Resetting Environment IDs: {env_ids}")

        # Convert env_ids to a tensor
        env_ids_tensor = torch.tensor(env_ids, device=self.device)

        # Compute joint_init_pos using TorchScript
        joint_init_pos = compute_arm_joint_init_pos(
            env_ids_tensor,
            self._robot.data.default_joint_pos,
            torch.tensor(self._arm_dof_idx, device=self.device),
            self.device,
        )

        # Store initial angles reference
        arm_joint_radians = torch.deg2rad(torch.tensor([-90, 0, 120, -120, 180, 0], device=self.device))
        self.init_angles[env_ids] = arm_joint_radians.repeat(len(env_ids), 1).view(-1, 6)

        # Configure root state with environment offsets
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Update internal state buffers
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self.joint_pos[env_ids] = joint_init_pos
        self.joint_vel[env_ids] = joint_vel

        # Write states to physics simulation
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_init_pos, joint_vel, None, env_ids)

        # Reset the nodal state of the deformable objects
        for deformable_object in self.deformable_objects_list:
            nodal_state = deformable_object.data.default_nodal_state_w.clone()
            nodal_velocity = torch.zeros_like(nodal_state)
            nodal_state = nodal_state[env_ids, :, :]
            nodal_state[:, :, 1] += sample_uniform(-0.05, 0.05, nodal_state[:, :, 1].shape, self.device)
            deformable_object.write_nodal_state_to_sim(nodal_state, env_ids)
            deformable_object.write_nodal_velocity_to_sim(nodal_velocity, env_ids)

        # Reset environment-specific components
        self.image_sequences.clear()

@torch.jit.script
def calculate_cube_reward(self, depth_obs):
    fruit_mask = ((depth_obs >= 0.35) & (depth_obs <= 0.54)).clone().detach().to(dtype=torch.float32, device=self.device)
    plant_mask = ((depth_obs >= 0.01) & (depth_obs <= 0.37)).clone().detach().to(dtype=torch.float32, device=self.device) / 255
    fruit_mask[:, int(1.5 * 256 / 3):, :, :] = 0.0
    # save_images_to_file(fruit_mask, "/home/nitesh/IsaacLab/mybuddy_fruit_mask.png")
    # save_images_to_file(plant_mask * 255, "/home/nitesh/IsaacLab/mybuddy_plant_mask.png")
    plant_pixels = torch.sum(plant_mask, dim=(1, 2, 3))
    fruit_pixels = torch.sum(fruit_mask, dim=(1, 2, 3))
    fruit_reward = (fruit_pixels / (255*255)) * 50.0
    plant_reward = -(plant_pixels / (255*255)) * 500.0
    return fruit_reward, plant_reward

@torch.jit.script
def compute_arm_joint_init_pos(
    env_ids: torch.Tensor,
    default_joint_pos: torch.Tensor,
    arm_dof_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    # Compute ARM_JOINT_INIT_DEGREES
    if torch.rand(1).item() > 0.3:
        arm_joint_degrees = torch.tensor([-90, torch.rand(1).item() * 30 - 40, 120, -120, 180, 0], device=device)
    else:
        arm_joint_degrees = torch.tensor([-90, torch.rand(1).item() * 30, 120, -120, 180, 0], device=device)
    arm_joint_radians = torch.deg2rad(arm_joint_degrees)

    # Create base joint configuration with default values
    base_joint_pos = default_joint_pos[env_ids]
    joint_init_pos = torch.zeros_like(base_joint_pos, device=device)

    # Initialize arm joints with strategic default angles
    arm_angles = arm_joint_radians.repeat(len(env_ids), 1)
    joint_init_pos[:, arm_dof_idx] = arm_angles

    return joint_init_pos