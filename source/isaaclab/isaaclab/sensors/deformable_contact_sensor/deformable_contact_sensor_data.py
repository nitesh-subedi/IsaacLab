# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class DeformableContactSensorData:
    """Data container for the deformable contact sensor.

    This class contains the data for contacts between deformable bodies and rigid bodies,
    detected using mesh proximity detection with Warp.
    """

    contact_detected: torch.Tensor | None = None
    """Binary contact detection per rigid body.

    Shape is (N, M), where N is the number of environments and M is the number of rigid bodies
    being monitored. Values are 1 if any vertex of the deformable body is in contact with the
    rigid body, 0 otherwise.
    """

    num_contacts: torch.Tensor | None = None
    """Number of vertex-level contacts per rigid body.

    Shape is (N, M), where N is the number of environments and M is the number of rigid bodies.
    This counts how many vertices of the deformable mesh are in contact with each rigid body.
    """

    net_contact_forces_w: torch.Tensor | None = None
    """Net contact forces on each rigid body from the deformable body in world frame.

    Shape is (N, M, 3), where N is the number of environments and M is the number of rigid bodies.

    Note:
        These forces are estimated using a spring-damper model based on penetration depth
        and vertex velocities. They represent the forces that the deformable body exerts
        on each rigid body.
    """

    contact_points_w: torch.Tensor | None = None
    """Contact points on the rigid body surfaces in world frame.

    Shape is (N, V, M, 3), where N is the number of environments, V is the number of vertices
    in the deformable mesh, and M is the number of rigid bodies.

    For non-contacting vertex-rigid pairs, the values are zero vectors.

    Note:
        If :attr:`DeformableContactSensorCfg.track_contact_points` is False, this is None.
    """

    contact_normals_w: torch.Tensor | None = None
    """Contact normals at contact points in world frame.

    Shape is (N, V, M, 3), where N is the number of environments, V is the number of vertices,
    and M is the number of rigid bodies.

    Normals point away from the rigid body surface (outward normal).

    Note:
        If :attr:`DeformableContactSensorCfg.track_contact_points` is False, this is None.
    """

    contact_distances: torch.Tensor | None = None
    """Signed distances from deformable vertices to rigid body surfaces.

    Shape is (N, V, M), where N is the number of environments, V is the number of vertices,
    and M is the number of rigid bodies.

    Negative values indicate penetration, positive values indicate separation.

    Note:
        If :attr:`DeformableContactSensorCfg.track_contact_points` is False, this is None.
    """

    deformable_pos_w: torch.Tensor | None = None
    """Position of the deformable body's center in world frame.

    Shape is (N, 3), where N is the number of environments.

    This is computed as the mean of all vertex positions.

    Note:
        If :attr:`DeformableContactSensorCfg.track_deformable_pose` is False, this is None.
    """

    last_contact_time: torch.Tensor | None = None
    """Time spent (in s) in contact before the last detach.

    Shape is (N, M), where N is the number of environments and M is the number of rigid bodies.

    Note:
        If :attr:`DeformableContactSensorCfg.track_contact_time` is False, this is None.
    """

    current_contact_time: torch.Tensor | None = None
    """Time spent (in s) in contact since the last contact.

    Shape is (N, M), where N is the number of environments and M is the number of rigid bodies.

    Note:
        If :attr:`DeformableContactSensorCfg.track_contact_time` is False, this is None.
    """

    last_air_time: torch.Tensor | None = None
    """Time spent (in s) not in contact before the last contact.

    Shape is (N, M), where N is the number of environments and M is the number of rigid bodies.

    Note:
        If :attr:`DeformableContactSensorCfg.track_contact_time` is False, this is None.
    """

    current_air_time: torch.Tensor | None = None
    """Time spent (in s) not in contact since the last detach.

    Shape is (N, M), where N is the number of environments and M is the number of rigid bodies.

    Note:
        If :attr:`DeformableContactSensorCfg.track_contact_time` is False, this is None.
    """
