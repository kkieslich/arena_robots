"""The Nav2 inter planners (behavior trees under config/nav2/interplanners): each config loads its own tree, and
navigate_w_replanning_time_clear is navigate_w_replanning_time plus a global costmap clear when planning fails."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

PKG = Path(__file__).resolve().parents[2] / "arena_robots"
NAV2 = PKG / "config" / "nav2"
INTERPLANNERS = NAV2 / "interplanners"
CLEAR = INTERPLANNERS / "navigate_w_replanning_time_clear"
BASE = INTERPLANNERS / "navigate_w_replanning_time"
GLOBAL_CLEAR_SERVICE = "global_costmap/clear_entirely_global_costmap"


def _config(folder: Path) -> dict:
    return yaml.safe_load((folder / "interplanner_config.yaml").read_text())["bt_navigator"]["ros__parameters"]


def _tree(folder: Path) -> ET.Element:
    root = ET.parse(folder / "interplanner_behavior.xml").getroot()
    assert root.tag == "root" and root.get("BTCPP_format") == "4"
    (main,) = [t for t in root.findall("BehaviorTree") if t.get("ID") == root.get("main_tree_to_execute")]
    (top,) = list(main)
    return top


def _shape(node: ET.Element) -> tuple:
    return (node.tag, tuple(sorted(node.attrib.items())), tuple(_shape(c) for c in node))


def _share(package: str) -> Path | None:
    try:
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    except ImportError:
        return None
    try:
        return Path(get_package_share_directory(package))
    except PackageNotFoundError:
        return None


@pytest.mark.parametrize("folder", sorted(p.name for p in INTERPLANNERS.iterdir() if p.is_dir()))
def test_config_loads_its_own_tree(folder):
    params = _config(INTERPLANNERS / folder)
    paths = {params[key] for key in ("default_nav_to_pose_bt_xml", "default_nav_through_poses_bt_xml")} - {""}
    assert paths == {f"$(find-pkg-share arena_robots)/config/nav2/interplanners/{folder}/interplanner_behavior.xml"}
    assert (INTERPLANNERS / folder / "interplanner_behavior.xml").is_file()


def test_clear_tree_clears_the_global_costmap_once_when_planning_fails():
    top = _tree(CLEAR)
    assert top.tag == "PipelineSequence"
    rate, follow = list(top)
    assert rate.tag == "RateController" and float(rate.get("hz")) == 4.0
    (recovery,) = list(rate)
    assert recovery.tag == "RecoveryNode" and int(recovery.get("number_of_retries")) == 1
    plan, clear = list(recovery)
    assert plan.tag == "ComputePathToPose" and plan.get("planner_id") == "GridBased"
    assert clear.tag == "ClearEntireCostmap" and clear.get("service_name") == GLOBAL_CLEAR_SERVICE
    assert follow.tag == "FollowPath" and not list(follow)


def test_clear_tree_is_the_replanning_tree_plus_the_recovery():
    top = _tree(CLEAR)
    recovery = top.find("RateController/RecoveryNode")
    plan = recovery.find("ComputePathToPose")
    top.find("RateController").remove(recovery)
    top.find("RateController").append(plan)
    assert _shape(top) == _shape(_tree(BASE))


def test_clear_tree_has_no_motion_recovery():
    names = {node.tag for node in _tree(CLEAR).iter()}
    assert not names & {"Spin", "Wait", "BackUp", "DriveOnHeading", "AssistedTeleop", "RoundRobin"}
    clears = [node for node in _tree(CLEAR).iter() if node.tag.startswith("Clear")]
    assert [node.get("service_name") for node in clears] == [GLOBAL_CLEAR_SERVICE]


def test_clear_service_resolves_in_the_robot_namespace():
    # relative: bt_navigator runs in the robot namespace (env_0/jackal_0), the global costmap node is
    # <ns>/global_costmap/global_costmap and serves <ns>/global_costmap/clear_entirely_global_costmap
    assert not GLOBAL_CLEAR_SERVICE.startswith("/")
    params = yaml.safe_load((NAV2 / "nav2.yaml").read_text())
    assert "ros__parameters" in params["global_costmap"]["global_costmap"]
    ns = "/env_0/jackal_0"
    node = f"{ns}/global_costmap/global_costmap"
    assert os.path.normpath(f"{node}/../clear_entirely_global_costmap") == f"{ns}/{GLOBAL_CLEAR_SERVICE}"


def test_clear_config_is_the_replanning_config_with_its_own_tree():
    clear, base = _config(CLEAR), _config(BASE)
    assert {k: v for k, v in clear.items() if k != "default_nav_to_pose_bt_xml"} == {k: v for k, v in base.items() if k != "default_nav_to_pose_bt_xml"}
    assert {"nav2_clear_costmap_service_bt_node", "nav2_recovery_node_bt_node", "nav2_rate_controller_bt_node", "nav2_pipeline_sequence_bt_node"} <= set(clear["plugin_lib_names"])


def test_clear_service_name_is_the_one_of_the_nav2_default_tree():
    share = _share("nav2_bt_navigator")
    if share is None:
        pytest.skip("nav2_bt_navigator not installed")
    default = ET.parse(share / "behavior_trees" / "navigate_to_pose_w_replanning_and_recovery.xml").getroot()
    names = {node.get("name"): node.get("service_name") for node in default.iter("ClearEntireCostmap")}
    assert names["ClearGlobalCostmap-Context"] == GLOBAL_CLEAR_SERVICE


def test_clear_tree_uses_only_known_nodes_and_ports():
    share = _share("nav2_behavior_tree")
    if share is None:
        pytest.skip("nav2_behavior_tree not installed")
    model = ET.parse(share / "nav2_tree_nodes.xml").getroot().find("TreeNodesModel")
    ports = {node.get("ID"): {p.get("name") for p in node} for node in model}
    for node in _tree(CLEAR).iter():
        assert node.tag in ports, node.tag
        assert set(node.attrib) - {"name"} <= ports[node.tag], (node.tag, node.attrib)
