# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Simple test script for deformable contact sensor with a sphere falling onto a ground plane.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/test_deformable_sphere.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test deformable contact sensor with sphere and ground.")
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
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sensors.deformable_contact_sensor import DeformableContactSensor, DeformableContactSensorCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """Designs the scene with a deformable sphere, ground plane, and rigid cube."""
    from isaaclab.assets import RigidObject, RigidObjectCfg

    # Ground plane for PhysX collision
    cfg = sim_utils.GroundPlaneCfg(size=(10.0, 10.0))
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create environment origins
    origins = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=origin)

    # Rigid Cube (positioned on the ground, sphere will fall onto it)
    # Cube is 0.5x0.5x0.3m, positioned so top is at z=0.3 (higher than ground)
    cube_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.5, 0.5, 0.3),  # Wide flat cube
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),  # Center at 0.15, top at 0.15+0.15=0.3
    )
    cube_object = RigidObject(cfg=cube_cfg)

    # Deformable Object (soft sphere that will fall onto the cube)
    deformable_cfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/DeformableSphere",
        spawn=sim_utils.MeshSphereCfg(
            radius=0.15,
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
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        debug_vis=False,
    )
    deformable_object = DeformableObject(cfg=deformable_cfg)

    # Deformable Contact Sensor - monitor contact with the rigid cube
    contact_sensor_cfg = DeformableContactSensorCfg(
        prim_path="/World/envs/env_.*/DeformableContactSensor",
        update_period=0.0,  # Update every step
        deformable_body_prim_path="/World/envs/env_.*/DeformableSphere",
        rigid_body_prim_paths=[
            "/World/envs/env_.*/Cube",  # Monitor the rigid cube
        ],
        contact_threshold=0.02,  # 2cm contact threshold (smaller to catch actual contacts)
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
        "cube_object": cube_object,
        "contact_sensor": contact_sensor,
    }
    return scene_entities, origins


def run_simulator(sim: SimulationContext, entities: dict, origins: list):
    """Runs the simulation loop."""
    # Extract scene entities
    deformable_object = entities["deformable_object"]
    cube_object = entities["cube_object"]
    contact_sensor = entities["contact_sensor"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Nodal kinematic targets
    nodal_kinematic_target = deformable_object.data.nodal_kinematic_target.clone()

    print("\n" + "=" * 80)
    print("Deformable Contact Sensor - Sphere Cube Test")
    print("=" * 80)
    print("\nThis test demonstrates contact detection between:")
    print("  - A deformable sphere (blue) falling from 1m height")
    print("  - A rigid cube (red) resting on the ground")
    print("\nThe sensor monitors only the cube, not the ground plane.\n")

    # Simulate physics
    while simulation_app.is_running():
        # reset every 2000 steps
        if count % 2000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0

            # reset the deformable object
            nodal_state = deformable_object.data.default_nodal_state_w.clone()
            origins_tensor = torch.tensor(origins, device=sim.device)
            pos_w = origins_tensor + torch.tensor([0.0, 0.0, 1.0], device=sim.device)
            quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device).repeat(
                deformable_object.num_instances, 1
            )
            nodal_state[..., :3] = deformable_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)
            deformable_object.write_nodal_state_to_sim(nodal_state)

            # Free all vertices
            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0
            deformable_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

            # reset buffers
            deformable_object.reset()
            cube_object.reset()
            contact_sensor.reset()

            print("\n" + "-" * 80)
            print("[INFO]: Resetting scene - sphere dropping from 1m height...")
            print("-" * 80)

        # write data to simulation
        deformable_object.write_data_to_sim()
        cube_object.write_data_to_sim()

        # perform step
        sim.step()

        # update sim-time
        sim_time += sim_dt
        count += 1

        # update buffers
        deformable_object.update(sim_dt)
        cube_object.update(sim_dt)
        contact_sensor.update(sim_dt)

        # print contact information every 25 steps
        if count % 25 == 0:
            sensor_data = contact_sensor.data
            print(f"\n[Step {count}] Sim time: {sim_time:.2f}s")
            print(f"Sphere center position: {sensor_data.deformable_pos_w}")
            print(f"Contact detected (Cube): {sensor_data.contact_detected}")
            print(f"Number of vertex contacts: {sensor_data.num_contacts}")
            print(f"Net contact forces (N): {sensor_data.net_contact_forces_w[:, 0, :]}")

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
