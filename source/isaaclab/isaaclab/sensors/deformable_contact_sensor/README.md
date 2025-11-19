# Deformable Contact Sensor

A sensor for detecting contacts between deformable bodies and rigid bodies using mesh proximity detection with Warp.

## Overview

The `DeformableContactSensor` extends IsaacLab's sensor capabilities to support contact detection between deformable bodies (soft bodies) and rigid bodies. Unlike the standard `ContactSensor` which relies on PhysX's ContactReporter API (only available for rigid-rigid contacts), this sensor uses GPU-accelerated mesh proximity detection.

## Features

- **Mesh-based proximity detection**: Uses Warp kernels to efficiently check distances between deformable mesh vertices and rigid body collision meshes
- **Contact force estimation**: Estimates contact forces using a spring-damper model based on penetration depth and vertex velocities
- **Multi-body support**: Can monitor contacts between one deformable body and multiple rigid bodies simultaneously
- **Configurable threshold**: Adjustable contact distance threshold for detection sensitivity
- **Optional detailed tracking**: Can track individual contact points, normals, and distances for each vertex
- **Contact time tracking**: Optional tracking of contact/air time durations
- **GPU acceleration**: Leverages Warp for efficient parallel computation

## How It Works

1. **Mesh Access**: Accesses collision mesh vertices of the deformable body from PhysX's `SoftBodyView`
2. **Rigid Mesh Conversion**: Converts rigid body collision meshes to Warp mesh format
3. **Proximity Detection**: Uses Warp's `mesh_query_point` to find closest points on rigid surfaces for each deformable vertex
4. **Contact Classification**: Vertices within the `contact_threshold` distance are considered in contact
5. **Force Estimation**: Applies spring-damper model to estimate contact forces:
   - Spring force: `F_spring = k * penetration_depth`
   - Damping force: `F_damping = c * velocity_normal`

## Usage Example

```python
from isaaclab.sensors.deformable_contact_sensor import DeformableContactSensor, DeformableContactSensorCfg

# Configure the sensor
sensor_cfg = DeformableContactSensorCfg(
    prim_path="/World/envs/env_.*/DeformableContactSensor",
    update_period=0.0,  # Update every physics step
    deformable_body_prim_path="/World/envs/env_.*/SoftObject",
    rigid_body_prim_paths=[
        "/World/envs/env_.*/Robot/gripper_left",
        "/World/envs/env_.*/Robot/gripper_right",
    ],
    contact_threshold=0.01,  # 1cm contact threshold
    force_stiffness=1000.0,
    force_damping=10.0,
    track_contact_points=True,
    track_deformable_pose=True,
    track_contact_time=True,
)

# Create the sensor
sensor = DeformableContactSensor(cfg=sensor_cfg)

# Access sensor data
sensor_data = sensor.data
print(f"Contact detected: {sensor_data.contact_detected}")
print(f"Number of contacts: {sensor_data.num_contacts}")
print(f"Contact forces: {sensor_data.net_contact_forces_w}")
```

## Configuration Parameters

### Required Parameters

- `deformable_body_prim_path` (str): Path to the deformable body prim
- `rigid_body_prim_paths` (list[str]): List of paths to rigid bodies to monitor

### Contact Detection Parameters

- `contact_threshold` (float): Distance threshold for contact detection (default: 0.01m)
- `force_stiffness` (float): Spring stiffness for force estimation (default: 1000.0)
- `force_damping` (float): Damping coefficient for force estimation (default: 10.0)

### Optional Tracking Features

- `track_contact_points` (bool): Track individual vertex contact information (default: False)
  - Warning: Memory intensive for high-resolution meshes
- `track_deformable_pose` (bool): Track deformable body center position (default: True)
- `track_contact_time` (bool): Track contact/air time durations (default: False)

## Output Data

The sensor provides the following data through `sensor.data`:

### Always Available

- `contact_detected` (torch.Tensor): Binary contact flag per rigid body - shape `(N, M)`
- `num_contacts` (torch.Tensor): Number of vertex-level contacts per rigid body - shape `(N, M)`
- `net_contact_forces_w` (torch.Tensor): Net contact forces in world frame - shape `(N, M, 3)`

### Optional (when enabled)

- `contact_points_w` (torch.Tensor): Contact points on rigid surfaces - shape `(N, V, M, 3)`
- `contact_normals_w` (torch.Tensor): Contact normals - shape `(N, V, M, 3)`
- `contact_distances` (torch.Tensor): Signed distances (negative = penetration) - shape `(N, V, M)`
- `deformable_pos_w` (torch.Tensor): Deformable body center position - shape `(N, 3)`
- `current_contact_time` (torch.Tensor): Time in contact - shape `(N, M)`
- `current_air_time` (torch.Tensor): Time not in contact - shape `(N, M)`

Where:
- N = number of environments
- M = number of rigid bodies being monitored
- V = number of vertices in the deformable mesh

## Performance Considerations

- **Mesh Resolution**: Contact detection time scales with the number of deformable mesh vertices
- **Number of Rigid Bodies**: Linear scaling with number of monitored rigid bodies
- **Track Contact Points**: Significantly increases memory usage for detailed tracking
- **Update Period**: Consider setting `update_period > 0` if contact information doesn't need to be updated every step

## Limitations

- Currently supports one deformable body per sensor instance
- Rigid body collision meshes must be triangulated
- Force estimation is approximate (spring-damper model)
- Does not capture tangential/friction forces, only normal forces

## Example Script

See `scripts/tutorials/04_sensors/run_deformable_contact_sensor.py` for a complete working example.

## Implementation Details

### Warp Kernels

The sensor uses three custom Warp kernels defined in `deformable_contact_kernels.py`:

1. `mesh_proximity_detection_kernel`: Single rigid body contact detection
2. `multi_rigid_mesh_proximity_kernel`: Multi-rigid body contact detection
3. `aggregate_contact_forces_kernel`: Force aggregation and estimation

### PhysX Integration

- Deformable mesh vertices are obtained via `SoftBodyView.get_nodal_positions()`
- Only collision mesh vertices are used (not simulation mesh)
- Rigid body transforms are obtained via `RigidBodyView.get_transforms()`
