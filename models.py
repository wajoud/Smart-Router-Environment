# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Smart Router Environment.

The smart_router environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SmartRouterAction(Action):
    """Action for the Smart Router environment - just a message to echo."""

    path_selection: int = Field(..., ge=0, le=2)


class SmartRouterObservation(Observation):
    """Observation from the Smart Router environment - the echoed message."""

    latency_ms: float
    packet_loss: float
    is_congested: bool
