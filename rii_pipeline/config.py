"""Configuration constants for the RII Pipeline."""

from __future__ import annotations

import os
from pathlib import Path


WORKSPACE = str(Path(__file__).resolve().parent)
MAP_IN_DIR = f"{WORKSPACE}/src/pcd_package/preclean/map_in"
DEFAULT_PCD_OUT = f"{WORKSPACE}/src/pcd_package/preclean/map_out"
DEFAULT_MAP_SAVE = f"{WORKSPACE}/src/pcd_package/final_2d_represntation"
PRECLEAN_DIR = f"{WORKSPACE}/src/pcd_package/preclean"
PCD_PACKAGE_DIR = f"{WORKSPACE}/src/pcd_package"


def detect_default_point_cloud() -> str:
    """Return a likely default raw point-cloud path inside this workspace."""
    candidates = (
        Path(MAP_IN_DIR) / "GlobalMap.pcd",
        Path(MAP_IN_DIR) / "GlobalMap.ply",
    )
    for path in candidates:
        if path.is_file():
            return str(path)
    return MAP_IN_DIR


DEFAULT_PCD_IN = detect_default_point_cloud()


def detect_ros_distro() -> str:
    """Return the first locally available ROS distro setup script."""
    candidates = []
    env_distro = os.environ.get("ROS_DISTRO")
    if env_distro:
        candidates.append(env_distro)
    candidates.extend(["jazzy", "humble", "iron", "rolling"])

    for distro in candidates:
        if distro and os.path.isfile(f"/opt/ros/{distro}/setup.bash"):
            return distro
    raise FileNotFoundError("Could not find a ROS setup.bash under /opt/ros")


ROS_DISTRO = detect_ros_distro()
ROS_SETUP = f"/opt/ros/{ROS_DISTRO}/setup.bash"
INSTALL_SETUP = f"{WORKSPACE}/install/setup.bash"
SOURCE_CMD = f"source {ROS_SETUP}" + (f" && source {INSTALL_SETUP}" if os.path.isfile(INSTALL_SETUP) else "")
