# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the deformable contact sensor to detect
contacts between deformable bodies and rigid bodies.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_deformable_contact_sensor.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the deformable contact sensor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors.deformable_contact_sensor import DeformableContactSensor, DeformableContactSensorCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """Designs the scene with a deformable object and rigid bodies."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create environment origins
    origins = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=origin)

    # Deformable Object (soft cube that will be squeezed)
    deformable_cfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/DeformableCube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.3, 0.3, 0.3),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.01,
                solver_position_iteration_count=16,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.8)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                poissons_ratio=0.45,
                youngs_modulus=5e4,
                damping_scale=0.1,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        debug_vis=False,
    )
    deformable_object = DeformableObject(cfg=deformable_cfg)

    # Rigid Object 1: Box (using MeshCuboidCfg for explicit mesh geometry)
    # Position it to be close/touching the deformable cube (cube is 0.3m, box is 0.2m)
    box1_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box1",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.20, 0.5)),  # Cube half-width (0.15) + box half-width (0.1) = 0.25, use 0.20 for overlap
    )
    box1_object = RigidObject(cfg=box1_cfg)

    # Rigid Object 2: Box (using MeshCuboidCfg for explicit mesh geometry)
    box2_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box2",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.20, 0.5)),
    )
    box2_object = RigidObject(cfg=box2_cfg)

    # Deformable Contact Sensor
    contact_sensor_cfg = DeformableContactSensorCfg(
        prim_path="/World/envs/env_.*/DeformableContactSensor",
        update_period=0.0,  # Update every step
        deformable_body_prim_path="/World/envs/env_.*/DeformableCube",
        rigid_body_prim_paths=[
            "/World/envs/env_.*/Box1",
            "/World/envs/env_.*/Box2",
        ],
        contact_threshold=0.02,  # 2cm contact threshold
        force_stiffness=5000.0,
        force_damping=50.0,
        track_contact_points=True,
        track_deformable_pose=True,
        track_contact_time=True,
        debug_vis=True,
    )
    contact_sensor = DeformableContactSensor(cfg=contact_sensor_cfg)

    # return the scene information
    scene_entities = {
        "deformable_object": deformable_object,
        "box1_object": box1_object,
        "box2_object": box2_object,
        "contact_sensor": contact_sensor,
    }
    return scene_entities, origins


def run_simulator(sim: SimulationContext, entities: dict, origins: list):
    """Runs the simulation loop."""
    # Extract scene entities
    deformable_object = entities["deformable_object"]
    box1_object = entities["box1_object"]
    box2_object = entities["box2_object"]
    contact_sensor = entities["contact_sensor"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Nodal kinematic targets
    nodal_kinematic_target = deformable_object.data.nodal_kinematic_target.clone()

    print("\n" + "=" * 80)
    print("Deformable Contact Sensor Tutorial")
    print("=" * 80)
    print("\nThis tutorial demonstrates contact detection between:")
    print("  - A deformable cube (blue)")
    print("  - A rigid box #1 (red)")
    print("  - A rigid box #2 (green)")
    print("\nThe sensor will detect contacts and estimate forces using mesh proximity.\n")

    # Simulate physics
    while simulation_app.is_running():
        # reset every 500 steps
        if count % 500 == 0:
            # reset counters
            sim_time = 0.0
            count = 0

            # reset the deformable object
            nodal_state = deformable_object.data.default_nodal_state_w.clone()
            origins_tensor = torch.tensor(origins, device=sim.device)
            pos_w = origins_tensor + torch.tensor([0.0, 0.0, 0.5], device=sim.device)
            quat_w = math_utils.random_orientation(deformable_object.num_instances, device=sim.device)
            nodal_state[..., :3] = deformable_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)
            deformable_object.write_nodal_state_to_sim(nodal_state)

            # Free all vertices
            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0
            deformable_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

            # reset rigid objects to squeeze the deformable cube
            identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device).repeat(box1_object.num_instances, 1)

            box1_pos = pos_w + torch.tensor([0.0, 0.20, 0.0], device=sim.device)
            box1_vel = torch.tensor([[0.0, -0.5, 0.0]], device=sim.device).repeat(box1_object.num_instances, 1)
            box1_object.write_root_pose_to_sim(torch.cat([box1_pos, identity_quat], dim=-1))
            box1_object.write_root_velocity_to_sim(torch.cat([box1_vel, torch.zeros_like(box1_vel)], dim=-1))

            box2_pos = pos_w + torch.tensor([0.0, -0.20, 0.0], device=sim.device)
            box2_vel = torch.tensor([[0.0, 0.5, 0.0]], device=sim.device).repeat(box2_object.num_instances, 1)
            box2_object.write_root_pose_to_sim(torch.cat([box2_pos, identity_quat], dim=-1))
            box2_object.write_root_velocity_to_sim(torch.cat([box2_vel, torch.zeros_like(box2_vel)], dim=-1))

            # reset buffers
            deformable_object.reset()
            box1_object.reset()
            box2_object.reset()
            contact_sensor.reset()

            print("\n" + "-" * 80)
            print("[INFO]: Resetting scene...")
            print("-" * 80)

        # write data to simulation
        deformable_object.write_data_to_sim()
        box1_object.write_data_to_sim()
        box2_object.write_data_to_sim()

        # perform step
        sim.step()

        # update sim-time
        sim_time += sim_dt
        count += 1

        # update buffers
        deformable_object.update(sim_dt)
        box1_object.update(sim_dt)
        box2_object.update(sim_dt)
        contact_sensor.update(sim_dt)

        # print contact information every 50 steps
        if count % 50 == 0:
            sensor_data = contact_sensor.data
            # Debug: Check rigid body positions
            box1_pos = box1_object.data.root_pos_w
            box2_pos = box2_object.data.root_pos_w

            print(f"\n[Step {count}] Sim time: {sim_time:.2f}s")
            print(f"Deformable center position: {sensor_data.deformable_pos_w}")
            print(f"Box1 position: {box1_pos}")
            print(f"Box2 position: {box2_pos}")
            print(f"Contact detected (Box1, Box2): {sensor_data.contact_detected}")
            print(f"Number of vertex contacts: {sensor_data.num_contacts}")
            print(f"Net contact forces (N):")
            print(f"  Box1: {sensor_data.net_contact_forces_w[:, 0, :]}")
            print(f"  Box2: {sensor_data.net_contact_forces_w[:, 1, :]}")

            if sensor_data.current_contact_time is not None:
                print(f"Contact time (s): {sensor_data.current_contact_time}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.5])
    # Design scene
    scene_entities, scene_origins = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
