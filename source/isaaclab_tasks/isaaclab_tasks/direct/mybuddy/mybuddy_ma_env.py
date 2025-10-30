from __future__ import annotations

import csv
import gymnasium as gym
import numpy as np

# Standard library imports
import os
import time

# Third-party imports
import torch
from collections import deque
from typing import Any, Dict, Tuple
from collections.abc import Sequence

# Local imports
import omni
import isaacsim.core.utils.stage as stage_utils
# from omni.isaac.core import PhysicsContext
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
    RigidObjectCfg,
)
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg, save_images_to_file
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

# Constants
GRAVITY = (0.0, 0.0, 9.81)
RENDER_DT = 1 / 60
PHYSX_GPU_BUFFER = 2**26
EPISODE_LENGTH = 10.0
ACTION_SCALE = 0.5
PLANT_USD_PATH = "plant_v21.usd"
CUBE_RADIUS = 0.03
CUBE_MASS = 1.0
CUBE_COLOR = (1.0, 0.0, 0.0)
CUBE_INIT_POS = (0.0, -0.4, 0.28)
CAMERA_WIDTH = 256
CAMERA_HEIGHT = 256
CAMERA_FOCAL_LENGTH = 2.87343478
CAMERA_FOCUS_DISTANCE = 0.6
CAMERA_HORIZONTAL_APERTURE = 5.760000228881836
CAMERA_VERTICAL_APERTURE = 3.5999999046325684
CAMERA_CLIPPING_RANGE = (0.076, 10.0)
CAMERA_F_STOP = 240.0
LIGHT_INTENSITY = 1500.0
LIGHT_COLOR = (1.0, 1.0, 1.0)
LIGHT_RADIUS = 1.0
BACKGROUND_INTENSITY = 0.6
BACKGROUND_SIZE = (1.25, 0.01, 1.25)
GROUND_SIZE = (1.25, 0.25, 0.01)
DEFORMABLE_YOUNGS_MODULUS = 7.5e15
DEFORMABLE_POISSONS_RATIO = 0.1
DEFORMABLE_DAMPING_SCALE = 1.0
DEFORMABLE_FRICTION = 0.5
DEFORMABLE_DENSITY = 10
DETECTION_SEQUENCE_MAXLEN = 10
OCCLUSION_REWARD_SCALE = 10.0
FULL_VISIBILITY_REWARD = 3.0
SUSTAINED_REWARD = 20.0
OCCLUSION_PIXEL_THRESHOLD = 160
OCCLUSION_MASK_SIZE = 40
REACH_THRESHOLD = 0.05  # meters — distance under which an end-effector is considered to have reached the fruit
REACH_VIS_REWARD = 5.0  # reward for seeing the fruit and having any arm reach it


@configclass
class MyBuddyEnvCfg(DirectMARLEnvCfg):
    """
    Configuration class for MyBuddyEnv.
    """

    # Environment parameters
    decimation: int = 1
    episode_length_s: float = EPISODE_LENGTH
    action_scale: float = ACTION_SCALE
    possible_agents: Sequence[str] = ("left_arm", "right_arm")
    action_spaces: dict[str, int] = {"left_arm": 6, "right_arm": 5}
    state_space: int = -1  # Placeholder for state space definition

    # Simulation configuration
    sim: SimulationCfg = SimulationCfg(
        dt=RENDER_DT,
        render_interval=decimation,
        physx=PhysxCfg(gpu_temp_buffer_capacity=PHYSX_GPU_BUFFER),
        gravity=GRAVITY,
    )

    # Robot configuration
    robot_cfg: ArticulationCfg = ArticulationCfg(
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

    left_arm_dof_name: Sequence[str] = (
        "bc2bl",
        "left_arm_j1",
        "left_arm_j2",
        "left_arm_j3",
        "left_arm_j4",
        "left_arm_j5",
    )
    right_arm_dof_name: Sequence[str] = ("right_arm_j1", "right_arm_j2", "right_arm_j3", "right_arm_j4", "right_arm_j5")
    left_link_dof_name: Sequence[str] = (#"base_link", 
                                         "left_arm_l1",
                                         "left_arm_l2",
                                         "left_arm_l3",
                                         "left_arm_l4",
                                         "left_arm_l5",
                                         "left_arm_l6"
                                         )
    right_link_dof_name: Sequence[str] = (
        "right_arm_l1",
        "right_arm_l2",
        "right_arm_l3",
        "right_arm_l4",
        "right_arm_l5",
        # "right_arm_l6",
    )
    all_link_dof_name: Sequence[str] = left_link_dof_name + right_link_dof_name

    # Camera configuration
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5), rot=(0.0, 0.0, 0.53833, 0.84274), convention="opengl"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH,
            focus_distance=CAMERA_FOCUS_DISTANCE,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
            vertical_aperture=CAMERA_VERTICAL_APERTURE,
            clipping_range=CAMERA_CLIPPING_RANGE,
            f_stop=CAMERA_F_STOP,
        ),
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )

    observation_spaces: dict[str, dict[str, Any]] = {
        "left_arm": {
            "joints": gym.spaces.Box(low=-20.00, high=20.0, shape=(6,), dtype=np.float32),
            "rgb": gym.spaces.Box(low=-1.0, high=1.0, shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 5), dtype=np.float32),
            "ee_position": gym.spaces.Box(low=-20.00, high=20.0, shape=(3,), dtype=np.float32),
        },
        "right_arm": {
            "joints": gym.spaces.Box(low=-20.00, high=20.0, shape=(5,), dtype=np.float32),
            "rgb": gym.spaces.Box(low=-1.0, high=1.0, shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 5), dtype=np.float32),
            "ee_position": gym.spaces.Box(low=-20.00, high=20.0, shape=(3,), dtype=np.float32),
        },
    }

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    # Reset parameters
    max_y_pos: float = 0.0
    maxlen: int = DETECTION_SEQUENCE_MAXLEN

    # Reward scales
    rew_scale_alive: float = -0.1

    contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/mybuddy/.*", history_length=3, update_period=0.005, track_air_time=True
    )


class MyBuddyEnv(DirectMARLEnv):
    cfg: MyBuddyEnvCfg

    def __init__(self, cfg: MyBuddyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._left_arm_dof_idx, _ = self._robot.find_joints(self.cfg.left_arm_dof_name)
        self._left_arm_pos_idx = self._robot.find_bodies(["left_arm_l6"])  # end effector
        print(f"Arm DOF Index: {self._left_arm_dof_idx}")
        print(f"Arm Position Index: {self._left_arm_pos_idx}")

        self._right_arm_dof_idx, _ = self._robot.find_joints(self.cfg.right_arm_dof_name)
        self._right_arm_pos_idx = self._robot.find_bodies(["right_arm_l6"])
        print(f"Arm DOF Index: {self._right_arm_dof_idx}")
        print(f"Arm Position Index: {self._right_arm_pos_idx}")

        self.collision_links_idx = self._robot.find_bodies(self.cfg.all_link_dof_name)

        self.action_scale = self.cfg.action_scale

        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel

        self.image_sequences = deque(maxlen=20)
        self.cube_detected_time = None  # , 90, 0, -120, -120, 95
        self.left_root_init_angles = (
            torch.deg2rad(torch.tensor([0, -90, 0, 120, -120, 180], device=self.device))
            .repeat(self.num_envs, 1)
            .view(-1, len(self.cfg.left_arm_dof_name))
        )
        self.right_root_init_angles = (
            torch.deg2rad(torch.tensor([90, 0, -120, -120, 95], device=self.device))
            .repeat(self.num_envs, 1)
            .view(-1, len(self.cfg.right_arm_dof_name))
        )

        self.camera_intrinsics = self._tiled_camera.data.intrinsic_matrices
        camera_rotation = self._tiled_camera.data.quat_w_opengl
        camera_position = self._tiled_camera.data.pos_w
        rotation_matrix = self.quaternion_to_rotation_matrix(camera_rotation)
        translation_vector = -torch.bmm(rotation_matrix, camera_position.unsqueeze(-1))
        self.extrinsic_matrix = torch.cat((rotation_matrix, translation_vector), dim=2)  # [2, 3, 4]
        self.projection_matrix = torch.bmm(self.camera_intrinsics, self.extrinsic_matrix)

        self.root_cube_position = self.goal_cube.data.root_com_state_w.clone()
        self.cube_maskas = torch.zeros(
            (self.num_envs, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, 1),
            device=self.device,
            dtype=torch.float32,
        )
        self.rgb_image = torch.zeros(
            (self.num_envs, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self.occlusion_time_below_threshold = torch.zeros(self.num_envs, 1, device=self.device)

        self.left_joint_readings = self._robot.data.joint_pos[:, self._left_arm_dof_idx]
        self.right_joint_readings = self._robot.data.joint_pos[:, self._right_arm_dof_idx]
        self.done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

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

    def _setup_scene(self):
        self.stage = stage_utils.get_current_stage()
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        euler = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        quat = quat_from_euler_xyz(*euler).cpu().numpy()
        plant_cfg = sim_utils.UsdFileCfg(usd_path="plant_v21.usd", scale=(0.05, 0.05, 0.05))
        plant_cfg.func(
            "/World/envs/env_.*/Plant1",
            plant_cfg,
            translation=(0.05, -0.24, 0.0),
            orientation=(quat[0], quat[1], quat[2], quat[3]),
        )

        euler = torch.tensor([0.0, 0.0, 180.0], device=self.device)
        quat = quat_from_euler_xyz(*euler).cpu().numpy()
        plant_cfg_2 = sim_utils.UsdFileCfg(usd_path="plant_v21.usd", scale=(0.05, 0.05, 0.05))
        plant_cfg_2.func(
            "/World/envs/env_.*/Plant2",
            plant_cfg_2,
            translation=(-0.05, -0.24, 0.0),
            orientation=(quat[0], quat[1], quat[2], quat[3]),
        )

        # add light
        light_cfg = sim_utils.SphereLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0), radius=1.0)
        light_cfg.func("/World/envs/env_.*/Light", light_cfg, translation=(0.0, -0.4, 2.0))

        # add background plane
        intensity = 0.6
        background_color = (intensity, intensity, intensity)
        background_plane_cfg = sim_utils.CuboidCfg(
            size=(1.25, 0.01, 1.25), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=background_color)
        )
        background_plane_cfg.func(
            "/World/envs/env_.*/BackgroundPlane", background_plane_cfg, translation=(0.0, -0.45, 0.0)
        )

        # add ground plane
        background_plane_cfg = sim_utils.CuboidCfg(
            size=(1.25, 0.25, 0.01), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=background_color)
        )
        background_plane_cfg.func(
            "/World/envs/env_.*/GroundPlane2", background_plane_cfg, translation=(0.0, -0.35, 0.0)
        )

        self.cube_size = (0.005, 0.005, 0.005)
        cube_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/GoalCube",
            spawn=sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.4, 0.28)),
        )

        self.goal_cube = RigidObject(cube_cfg)

        self.deformable_material_path = omni.usd.get_stage_next_free_path(self.stage, "/plant_material", True)

        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=[])

        self._add_plant(2)

        # add articulation to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # for i, contact_sensor in enumerate(contact_sensors_list):
        #     self.scene.sensors[f"contact_sensor_{i}"] = contact_sensor
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.scene.rigid_objects["goal_cube"] = self.goal_cube
        for i, deformable_object in enumerate(self.deformable_objects_list):
            self.scene.deformable_objects[f"stalk_{i}"] = deformable_object

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # self.camera_rotation = self._tiled_camera.data.quat_w_world[0]
        self.detection_sequence = deque(maxlen=self.cfg.maxlen)

        # set all the elements of detection sequence to False
        for i in range(self.cfg.maxlen):
            self.detection_sequence.append(torch.zeros(self.num_envs, device=self.device))

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "contact_done",
                "distance_to_goal",
            ]
        }

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
            youngs_modulus=7.5e50,
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
            # solver_position_iteration_count=256,
            self_collision=False,
        )
        # physicsUtils.add_physics_material_to_prim(self.stage, value.GetPrim(), self.deformable_material_path)

        for key, value in list(prim_dict.items())[1:]:
            deformableUtils.add_physx_deformable_body(
                self.stage,
                value.GetPath(),
                collision_simplification=True,
                simulation_hexahedral_resolution=simulation_resolution,
                # solver_position_iteration_count=256,
                self_collision=False,
            )

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        # Clamp and scale actions
        self.left_actions = (
            torch.clamp(actions["left_arm"], -1, 1) * self.dt * self.action_scale
            + self._robot.data.joint_pos[:, self._left_arm_dof_idx]
        )
        self.right_actions = (
            torch.clamp(actions["right_arm"], -1, 1) * self.dt * self.action_scale
            + self._robot.data.joint_pos[:, self._right_arm_dof_idx]
        )
        # Apply limits to specific action dimensions
        self.left_actions[:, 0].clamp_(-1.0, 0.7)
        # self.right_actions[:, 0].clamp_(-1.0, 1.0)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.left_actions, joint_ids=self._left_arm_dof_idx)
        self._robot.set_joint_position_target(self.right_actions, joint_ids=self._right_arm_dof_idx)

    def _get_observations(self) -> dict:
        # Joint readings
        left_joint_readings = self._robot.data.joint_pos[:, self._left_arm_dof_idx]
        right_joint_readings = self._robot.data.joint_pos[:, self._right_arm_dof_idx]

        # End Effector Position
        self.left_ee_position = (
            torch.squeeze(self._robot.data.body_com_pos_w[:, self._left_arm_pos_idx[0]])
            - self._robot.data.root_com_pos_w
        )
        self.right_ee_position = (
            torch.squeeze(self._robot.data.body_com_pos_w[:, self._right_arm_pos_idx[0]])
            - self._robot.data.root_com_pos_w
        )

        image_obs = self.get_states()
        left_obs = {
            "rgb": image_obs.to(dtype=torch.float32),
            "joints": left_joint_readings.to(dtype=torch.float32),
            "ee_position": self.left_ee_position.to(dtype=torch.float32),
        }
        right_obs = {
            "rgb": image_obs.to(dtype=torch.float32),
            "joints": right_joint_readings.to(dtype=torch.float32),
            "ee_position": self.right_ee_position.to(dtype=torch.float32),
        }

        observations = {"left_arm": left_obs, "right_arm": right_obs}
        return observations

    def get_states(self) -> torch.Tensor:
        # RGB
        self.rgb_image = self._tiled_camera.data.output["rgb"].clone()
        self.rgb_image = self.rgb_image / 255.0
        # save_images_to_file(self.rgb_image, "rgb_image.png")
        self.rgb_image_raw = self.rgb_image.clone()
        mean_tensor = torch.mean(self.rgb_image, dim=(1, 2), keepdim=True)
        self.rgb_image -= mean_tensor
        self.rgb_image = self.rgb_image.to(dtype=torch.float32)

        self.depth_image = self._tiled_camera.data.output["depth"].clone()
        self.depth_image[self.depth_image == float("inf")] = 0

        image_obs = torch.cat((self.rgb_image, self.cube_maskas, self.depth_image), dim=3)
        image_obs = image_obs.to(dtype=torch.float32)

        return image_obs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        plant_mask = calculate_plant_mask(self.depth_image)
        # print(self.contact_sensor.data.net_forces_w)
        self.in_contact = check_selected_links(self.contact_sensor.data.net_forces_w, self.collision_links_idx[0])
        contact_reward = self.in_contact.float() * -5.0  # Negative reward for contact

        # Logging
        self._episode_sums["contact_done"] += self.in_contact.float()

        # Occlusion Reward
        occlusion_mask = self.cube_maskas * plant_mask
        occlusion_pixels = torch.sum(occlusion_mask, dim=(1, 2, 3)).reshape(-1, 1)
        occlusion_reward = (1 - occlusion_pixels / (40 * 40)) * 10.0

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
        # action_penalty = torch.norm(self.raw_actions.clamp_(-1.0, 1.0), dim=1) * -0.06
        # action_penalty *= full_visibility_mask.float()

        # self.done = torch.logical_or(sustained_detection, self.in_contact).reshape(self.num_envs, )
        self.done = self.in_contact

        occlusion_reward = occlusion_reward.reshape(self.num_envs)

        # Reachability reward: check distance from either end-effector to the cube (world frame)
        try:
            # cube world position: (N, 3)
            cube_pos = self.goal_cube.data.root_com_state_w[:, :3]
            # end-effector world positions: (N, 3)
            left_ee_world = torch.squeeze(self._robot.data.body_com_pos_w[:, self._left_arm_pos_idx[0]])
            right_ee_world = torch.squeeze(self._robot.data.body_com_pos_w[:, self._right_arm_pos_idx[0]])

            # distances (N,)
            d_left = torch.norm(left_ee_world - cube_pos, dim=1)
            d_right = torch.norm(right_ee_world - cube_pos, dim=1)

            self._episode_sums["distance_to_goal"] += torch.minimum(d_left, d_right)

            reach_mask = (d_left <= REACH_THRESHOLD) | (d_right <= REACH_THRESHOLD)
            # Only reward if the cube is visible (full visibility) and reached by any arm
            reach_vis_reward = (reach_mask.float() * REACH_VIS_REWARD) * full_visibility_mask.float()
        except Exception:
            # If any data is missing, fall back to zero reach reward
            reach_vis_reward = torch.zeros(self.num_envs, device=self.device)
        
        # print(type(reach_vis_reward))

        total_reward = full_visibility_reward + sustained_reward + occlusion_reward + reach_vis_reward + contact_reward

        final_reward = {agent: total_reward for agent in self.cfg.possible_agents}

        return final_reward

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {agent: self.done for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int]):
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        left_joint_pos = (
            self._robot.data.default_joint_pos[env_ids][:, self._left_arm_dof_idx] + self.left_root_init_angles[env_ids]
        )
        # Sample random only second joint
        if np.random.uniform() < 0.5:
            left_joint_pos[:, 2] = sample_uniform(-0.52, -0.17, left_joint_pos[:, 2].shape, self.device)
            left_joint_pos[:, 0] = sample_uniform(-0.5, 0.0, left_joint_pos[:, 0].shape, self.device)
        else:
            left_joint_pos[:, 2] = sample_uniform(0.2, 0.3, left_joint_pos[:, 2].shape, self.device)
            left_joint_pos[:, 0] = sample_uniform(0.0, 0.5, left_joint_pos[:, 0].shape, self.device)

        # Configure root state with environment offsets
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Update internal state buffers
        left_joint_vel = torch.zeros_like(left_joint_pos, device=self.device)

        # Write states to physics simulation
        self._robot.set_joint_position_target(left_joint_pos, self._left_arm_dof_idx, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(left_joint_pos, left_joint_vel, self._left_arm_dof_idx, env_ids)

        right_joint_pos = (
            self._robot.data.default_joint_pos[env_ids][:, self._right_arm_dof_idx]
            + self.right_root_init_angles[env_ids]
        )
        right_joint_vel = torch.zeros_like(right_joint_pos, device=self.device)

        # Write states to physics simulation
        self._robot.set_joint_position_target(right_joint_pos, self._right_arm_dof_idx, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(right_joint_pos, right_joint_vel, self._right_arm_dof_idx, env_ids)

        # Reset the nodal state of the deformable objects
        for deformable_object in self.deformable_objects_list:
            deformable_object: DeformableObject
            nodal_state = deformable_object.data.default_nodal_state_w.clone()
            nodal_state = nodal_state[env_ids, :, :]
            deformable_object.write_nodal_state_to_sim(nodal_state, env_ids)

        cube_position = self.root_cube_position[env_ids]
        # if np.random.uniform() < 0.65:
        cube_position[:, 0] += sample_uniform(-0.15, 0.15, cube_position[:, 0].shape, self.device)
        # else:
        #     cube_position[:, 0] += sample_uniform(0.3, 0.6, cube_position[:, 0].shape, self.device)
        self.goal_cube.write_root_pose_to_sim(cube_position[:, :7], env_ids)
        self.goal_cube.write_root_velocity_to_sim(torch.zeros_like(cube_position[:, 7:], device=self.device), env_ids)

        # Get Cube Pixel location
        ones = torch.ones(cube_position[:, :3].shape[0], 1, device=self.device)  # [N, 1]
        points_3d_hom = torch.cat((cube_position[:, :3], ones), dim=1).unsqueeze(2)  # [N, 4, 1]
        image_points_hom = torch.bmm(self.projection_matrix[env_ids], points_3d_hom).squeeze(2)
        image_points_2d = image_points_hom[:, :2] / image_points_hom[:, 2:].expand(-1, 2)
        cube_masks = torch.flip(
            create_batched_masks((self.cfg.tiled_camera.width, self.cfg.tiled_camera.height), image_points_2d, 40),
            dims=[2],
        ).to(dtype=torch.float32, device=self.device)
        self.cube_maskas[env_ids] = cube_masks.reshape(-1, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, 1)

        raw_rgb = (self._tiled_camera.data.output["rgb"].clone()[env_ids] / 255.0).to(dtype=torch.float32)
        mean_rbg = torch.mean(raw_rgb, dim=(1, 2), keepdim=True, dtype=torch.float32)
        self.rgb_image[env_ids] = (raw_rgb - mean_rbg).to(dtype=torch.float32)

        tensor_seq = torch.stack(list(self.detection_sequence))  # Convert list of tensors to a single tensor
        tensor_seq[:, env_ids] = 0  # Zero out selected indices across all rows
        self.detection_sequence = deque(list(tensor_seq), maxlen=self.cfg.maxlen)  # Convert back to list if needed


@torch.jit.script
def check_selected_links(readings: torch.Tensor, selected_links: list[int]) -> torch.Tensor:
    """
    readings: (B, L, 3) tensor of contact readings
    selected_links: list of allowed link indices
    returns: (B,) boolean tensor -> True if any nonzero readings come from selected_links
    """
    B, L, _ = readings.shape

    # mask of nonzero readings
    nonzero_mask = readings.abs().sum(dim=2) > 0  
    # print(f"Nonzero mask: {nonzero_mask}")
    # print(f"Selected links: {selected_links}")

    # mask for allowed links
    allowed_mask = torch.zeros(L, dtype=torch.bool, device=readings.device)
    allowed_mask[selected_links] = True  

    # print(f"Allowed mask: {allowed_mask}")
    # print(f"Nonzero & allowed: {nonzero_mask & allowed_mask}")


    # batch valid if: nonzero implies allowed, and at least one nonzero exists
    valid = ((nonzero_mask & allowed_mask).any(dim=1)) #& nonzero_mask.any(dim=1)
    # print(f"Valid contacts: {valid}")

    return valid



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
