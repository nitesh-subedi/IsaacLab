# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import CONTACT_SENSOR_MARKER_CFG
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .deformable_contact_sensor import DeformableContactSensor


@configclass
class DeformableContactSensorCfg(SensorBaseCfg):
    """Configuration for the deformable contact sensor.

    This sensor detects contacts between a deformable body and one or more rigid bodies
    using mesh proximity detection with Warp kernels.
    """

    class_type: type = DeformableContactSensor

    deformable_body_prim_path: str = ""
    """Prim path to the deformable body to monitor.

    This should point to a deformable object in the scene. The sensor will track contacts
    between this deformable body and the rigid bodies specified in :attr:`rigid_body_prim_paths`.

    Example: ``/World/envs/env_.*/DeformableObject``
    """

    rigid_body_prim_paths: list[str] = []
    """List of prim paths (or expressions) to rigid bodies to check for contact.

    Each path can be a regex expression. The sensor will detect contacts between the
    deformable body and each of these rigid bodies.

    Example: ``["/World/envs/env_.*/Robot/gripper_left", "/World/envs/env_.*/Robot/gripper_right"]``

    Note:
        The expression can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Robot/gripper`` will be replaced with ``/World/envs/env_.*/Robot/gripper``.
    """

    contact_threshold: float = 0.01
    """Distance threshold (in meters) for contact detection. Defaults to 0.01.

    A vertex of the deformable mesh is considered in contact with a rigid body if it is
    within this distance from the rigid body's collision mesh surface.
    """

    force_stiffness: float = 1000.0
    """Spring stiffness for contact force estimation. Defaults to 1000.0.

    Used in the spring-damper model to estimate contact forces from penetration depth.
    Higher values result in stronger forces for the same penetration.
    """

    force_damping: float = 10.0
    """Damping coefficient for contact force estimation. Defaults to 10.0.

    Used in the spring-damper model to estimate contact forces from vertex velocities.
    Higher values result in more damping of contact oscillations.
    """

    track_contact_points: bool = False
    """Whether to track individual contact points and normals for each vertex. Defaults to False.

    If True, the sensor will store contact information for each vertex of the deformable mesh,
    which can be memory intensive for high-resolution meshes.
    """

    track_deformable_pose: bool = True
    """Whether to track the pose of the deformable body's center. Defaults to True.

    The center is computed as the mean of all vertex positions.
    """

    track_contact_time: bool = False
    """Whether to track the contact/air time for each rigid body. Defaults to False.

    If True, the sensor will track how long each rigid body has been in contact or
    separated from the deformable body.
    """

    visualizer_cfg: VisualizationMarkersCfg = CONTACT_SENSOR_MARKER_CFG.replace(
        prim_path="/Visuals/DeformableContactSensor"
    )
    """The configuration object for the visualization markers. Defaults to CONTACT_SENSOR_MARKER_CFG.

    Note:
        This attribute is only used when debug visualization is enabled.
    """
