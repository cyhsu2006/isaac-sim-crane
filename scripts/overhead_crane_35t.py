"""
35 噸雙樑天車模擬 — NVIDIA Isaac Sim
雙樑橋式天車 + 左右開合夾具 + 自動路徑規劃

結構：
  - Runway Rails（軌道樑）: 兩側固定軌道，沿 X 軸
  - Bridge（橋樑雙樑）: 沿 X 軸移動
  - Trolley（小車）: 沿 Y 軸移動
  - Hoist（起升機構）: 沿 Z 軸升降
  - Clamp（左右開合夾具）: 可夾取物件

控制：自動路徑規劃（移動 → 下降 → 夾取 → 上升 → 移動 → 下降 → 釋放）
"""

import numpy as np
from isaacsim import SimulationApp

CONFIG = {
    "headless": False,
    "width": 1920,
    "height": 1080,
    "anti_aliasing": 0,
    "renderer": "RayTracedLighting",
    "experience": "/isaac-sim/apps/isaacsim.exp.full.kit",
}
simulation_app = SimulationApp(CONFIG)

import omni.usd
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from omni.isaac.core.prims import XFormPrim
from pxr import UsdPhysics, UsdGeom, Gf, Sdf, PhysxSchema

# ─── 參數設定 ──────────────────────────────────────────────────────────────────
CRANE_CAPACITY = 35000  # kg
RUNWAY_LENGTH = 30.0    # m — 軌道長度 (X 軸)
SPAN = 20.0             # m — 跨距 (Y 軸)
LIFT_HEIGHT = 12.0      # m — 最大揚程 (Z 軸)
RAIL_HEIGHT = 14.0      # m — 軌道樑高度

BRIDGE_BEAM_W = 0.8     # m — 主樑寬度
BRIDGE_BEAM_H = 1.2     # m — 主樑高度
BRIDGE_GAP = 3.0        # m — 雙樑間距

TROLLEY_SIZE = (2.0, 2.5, 0.8)  # m
HOIST_CABLE_R = 0.05    # m — 鋼纜半徑

CLAMP_LENGTH = 2.5      # m — 夾具臂長
CLAMP_WIDTH = 0.3       # m — 夾具臂寬
CLAMP_HEIGHT = 0.8      # m — 夾具臂高
CLAMP_MAX_OPEN = 2.0    # m — 最大開口（單側）
CLAMP_FORCE = 350000    # N — 夾持力

# 速度
BRIDGE_SPEED = 1.5      # m/s — 大車行走速度
TROLLEY_SPEED = 1.0     # m/s — 小車行走速度
HOIST_SPEED = 0.5       # m/s — 起升速度
CLAMP_SPEED = 0.3       # m/s — 夾具開合速度

# 顏色
COLOR_RAIL = (0.4, 0.4, 0.45)
COLOR_BRIDGE = (0.9, 0.6, 0.1)   # 天車黃
COLOR_TROLLEY = (0.3, 0.3, 0.35)
COLOR_CABLE = (0.2, 0.2, 0.2)
COLOR_CLAMP = (0.8, 0.2, 0.1)
COLOR_LOAD = (0.5, 0.5, 0.55)
COLOR_GROUND = (0.25, 0.25, 0.28)


# ─── Helper ───────────────────────────────────────────────────────────────────

def create_box(stage, path, size, position, color, is_static=True):
    """建立方塊幾何體。"""
    xform = UsdGeom.Xform.Define(stage, path)
    cube = UsdGeom.Cube.Define(stage, f"{path}/mesh")
    cube.GetSizeAttr().Set(1.0)

    sx, sy, sz = size
    px, py, pz = position

    xform_prim = stage.GetPrimAtPath(path)
    xform_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(px, py, pz))
    xform_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(sx, sy, sz))

    # 如果沒有 xformOp，手動加入
    xformable = UsdGeom.Xformable(xform_prim)
    if not xformable.GetOrderedXformOps():
        xformable.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
        xformable.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))

    # 顏色
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    return xform_prim


# ─── 自動路徑狀態機 ──────────────────────────────────────────────────────────

class CraneController:
    """天車自動控制狀態機。"""

    STATES = [
        "MOVE_TO_PICKUP",     # 移動到取貨位置上方
        "LOWER_TO_PICKUP",    # 下降到取貨高度
        "CLOSE_CLAMP",        # 夾具閉合
        "LIFT_LOAD",          # 起吊
        "MOVE_TO_DROP",       # 移動到放貨位置上方
        "LOWER_TO_DROP",      # 下降到放貨高度
        "OPEN_CLAMP",         # 夾具打開
        "LIFT_EMPTY",         # 空載上升
        "PAUSE",              # 暫停
    ]

    def __init__(self):
        self.state_idx = 0
        self.state = self.STATES[0]
        self.timer = 0.0

        # 目標位置
        self.bridge_pos = 0.0       # X
        self.trolley_pos = 0.0      # Y
        self.hoist_pos = LIFT_HEIGHT  # Z（鉤高）
        self.clamp_open = CLAMP_MAX_OPEN  # 單側開口

        # 取貨/放貨座標
        self.pickup = {"x": -8.0, "y": 0.0, "z": 2.0}
        self.dropoff = {"x": 8.0, "y": 3.0, "z": 2.0}

        # 路徑點序列
        self.waypoints = [
            # (bridge_x, trolley_y, hoist_z, clamp_open, pause_time)
            (self.pickup["x"], self.pickup["y"], LIFT_HEIGHT, CLAMP_MAX_OPEN, 0),   # 移到取貨上方
            (self.pickup["x"], self.pickup["y"], self.pickup["z"], CLAMP_MAX_OPEN, 0),  # 下降
            (self.pickup["x"], self.pickup["y"], self.pickup["z"], 0.05, 1.0),       # 夾取
            (self.pickup["x"], self.pickup["y"], LIFT_HEIGHT, 0.05, 0),              # 起吊
            (self.dropoff["x"], self.dropoff["y"], LIFT_HEIGHT, 0.05, 0),            # 移到放貨上方
            (self.dropoff["x"], self.dropoff["y"], self.dropoff["z"], 0.05, 0),      # 下降
            (self.dropoff["x"], self.dropoff["y"], self.dropoff["z"], CLAMP_MAX_OPEN, 1.0),  # 釋放
            (self.dropoff["x"], self.dropoff["y"], LIFT_HEIGHT, CLAMP_MAX_OPEN, 0),  # 空載上升
        ]
        self.wp_idx = 0
        self.pause_remaining = 0.0

    def get_target(self):
        return self.waypoints[self.wp_idx]

    def update(self, dt):
        """更新控制器，回傳 (bridge_x, trolley_y, hoist_z, clamp_open)。"""
        if self.pause_remaining > 0:
            self.pause_remaining -= dt
            return self.bridge_pos, self.trolley_pos, self.hoist_pos, self.clamp_open

        target = self.waypoints[self.wp_idx]
        tx, ty, tz, tc, tp = target

        # 逐步逼近目標
        reached = True

        # Bridge (X)
        diff = tx - self.bridge_pos
        if abs(diff) > 0.02:
            self.bridge_pos += np.sign(diff) * min(BRIDGE_SPEED * dt, abs(diff))
            reached = False

        # Trolley (Y)
        diff = ty - self.trolley_pos
        if abs(diff) > 0.02:
            self.trolley_pos += np.sign(diff) * min(TROLLEY_SPEED * dt, abs(diff))
            reached = False

        # Hoist (Z)
        diff = tz - self.hoist_pos
        if abs(diff) > 0.02:
            self.hoist_pos += np.sign(diff) * min(HOIST_SPEED * dt, abs(diff))
            reached = False

        # Clamp
        diff = tc - self.clamp_open
        if abs(diff) > 0.01:
            self.clamp_open += np.sign(diff) * min(CLAMP_SPEED * dt, abs(diff))
            reached = False

        if reached:
            if tp > 0:
                self.pause_remaining = tp
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)

        return self.bridge_pos, self.trolley_pos, self.hoist_pos, self.clamp_open


# ─── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    root_path = "/World/Crane"
    UsdGeom.Xform.Define(stage, root_path)

    # ── 地面 ──
    world.scene.add(FixedCuboid(
        prim_path="/World/Ground",
        name="ground",
        size=1.0,
        scale=np.array([60.0, 40.0, 0.1]),
        position=np.array([0, 0, -0.05]),
        color=np.array(COLOR_GROUND),
    ))

    # ── 軌道支柱 ──
    pillar_positions = []
    for ix, x in enumerate(np.linspace(-RUNWAY_LENGTH/2, RUNWAY_LENGTH/2, 5)):
        for iy, y_sign in enumerate([-1, 1]):
            y = y_sign * (SPAN / 2 + 0.5)
            path = f"{root_path}/Pillar_{ix}_{iy}"
            world.scene.add(FixedCuboid(
                prim_path=path,
                name=f"pillar_{ix}_{iy}",
                size=1.0,
                scale=np.array([0.6, 0.6, RAIL_HEIGHT]),
                position=np.array([x, y, RAIL_HEIGHT / 2]),
                color=np.array(COLOR_RAIL),
            ))

    # ── 軌道樑 ──
    for iy, y_sign in enumerate([-1, 1]):
        y = y_sign * (SPAN / 2 + 0.5)
        path = f"{root_path}/Rail_{iy}"
        world.scene.add(FixedCuboid(
            prim_path=path,
            name=f"rail_{iy}",
            size=1.0,
            scale=np.array([RUNWAY_LENGTH + 1.0, 0.5, 0.6]),
            position=np.array([0, y, RAIL_HEIGHT + 0.3]),
            color=np.array(COLOR_RAIL),
        ))

    # ── 橋樑（雙樑）── 這些會動態移動
    bridge_path = f"{root_path}/Bridge"
    bridge_prim = XFormPrim(
        prim_path=bridge_path,
        name="bridge",
        position=np.array([0, 0, RAIL_HEIGHT + 0.6 + BRIDGE_BEAM_H / 2]),
    )
    world.scene.add(bridge_prim)

    # 主樑 1
    world.scene.add(VisualCuboid(
        prim_path=f"{bridge_path}/Beam_L",
        name="beam_l",
        size=1.0,
        scale=np.array([BRIDGE_BEAM_W, SPAN + 1.5, BRIDGE_BEAM_H]),
        position=np.array([0, 0, RAIL_HEIGHT + 0.6 + BRIDGE_BEAM_H / 2]),
        color=np.array(COLOR_BRIDGE),
    ))

    # 主樑 2
    world.scene.add(VisualCuboid(
        prim_path=f"{bridge_path}/Beam_R",
        name="beam_r",
        size=1.0,
        scale=np.array([BRIDGE_BEAM_W, SPAN + 1.5, BRIDGE_BEAM_H]),
        position=np.array([BRIDGE_GAP, 0, RAIL_HEIGHT + 0.6 + BRIDGE_BEAM_H / 2]),
        color=np.array(COLOR_BRIDGE),
    ))

    # 端樑（連接兩側）
    for iy, y_sign in enumerate([-1, 1]):
        y = y_sign * (SPAN / 2 + 0.5)
        world.scene.add(VisualCuboid(
            prim_path=f"{bridge_path}/EndBeam_{iy}",
            name=f"end_beam_{iy}",
            size=1.0,
            scale=np.array([BRIDGE_GAP + BRIDGE_BEAM_W, 0.5, 0.8]),
            position=np.array([BRIDGE_GAP / 2, y, RAIL_HEIGHT + 0.6 + 0.4]),
            color=np.array(COLOR_BRIDGE),
        ))

    # ── 小車 ──
    trolley_path = f"{bridge_path}/Trolley"
    trolley_prim = XFormPrim(
        prim_path=trolley_path,
        name="trolley",
        position=np.array([BRIDGE_GAP / 2, 0, RAIL_HEIGHT + 0.6 + BRIDGE_BEAM_H + TROLLEY_SIZE[2] / 2]),
    )
    world.scene.add(trolley_prim)

    world.scene.add(VisualCuboid(
        prim_path=f"{trolley_path}/Body",
        name="trolley_body",
        size=1.0,
        scale=np.array(TROLLEY_SIZE),
        position=np.array([BRIDGE_GAP / 2, 0, RAIL_HEIGHT + 0.6 + BRIDGE_BEAM_H + TROLLEY_SIZE[2] / 2]),
        color=np.array(COLOR_TROLLEY),
    ))

    # ── 鋼纜 ──
    cable_path = f"{trolley_path}/Cable"
    cable_prim = XFormPrim(
        prim_path=cable_path,
        name="cable",
    )
    world.scene.add(cable_prim)

    world.scene.add(VisualCuboid(
        prim_path=f"{cable_path}/Wire_L",
        name="wire_l",
        size=1.0,
        scale=np.array([0.08, 0.08, LIFT_HEIGHT]),
        position=np.array([BRIDGE_GAP / 2 - 0.5, 0, RAIL_HEIGHT - LIFT_HEIGHT / 2 + 1.0]),
        color=np.array(COLOR_CABLE),
    ))

    world.scene.add(VisualCuboid(
        prim_path=f"{cable_path}/Wire_R",
        name="wire_r",
        size=1.0,
        scale=np.array([0.08, 0.08, LIFT_HEIGHT]),
        position=np.array([BRIDGE_GAP / 2 + 0.5, 0, RAIL_HEIGHT - LIFT_HEIGHT / 2 + 1.0]),
        color=np.array(COLOR_CABLE),
    ))

    # ── 夾具框架 ──
    clamp_base_path = f"{cable_path}/ClampBase"
    world.scene.add(VisualCuboid(
        prim_path=clamp_base_path,
        name="clamp_base",
        size=1.0,
        scale=np.array([1.5, 3.0, 0.4]),
        position=np.array([BRIDGE_GAP / 2, 0, RAIL_HEIGHT - LIFT_HEIGHT + 0.8]),
        color=np.array(COLOR_TROLLEY),
    ))

    # 夾具左臂
    clamp_l_path = f"{cable_path}/ClampL"
    clamp_l_prim = XFormPrim(prim_path=clamp_l_path, name="clamp_l")
    world.scene.add(clamp_l_prim)
    world.scene.add(VisualCuboid(
        prim_path=f"{clamp_l_path}/Arm",
        name="clamp_l_arm",
        size=1.0,
        scale=np.array([CLAMP_WIDTH, CLAMP_LENGTH, CLAMP_HEIGHT]),
        position=np.array([BRIDGE_GAP / 2, -CLAMP_MAX_OPEN, RAIL_HEIGHT - LIFT_HEIGHT + 0.2]),
        color=np.array(COLOR_CLAMP),
    ))
    # 夾具左臂爪齒
    world.scene.add(VisualCuboid(
        prim_path=f"{clamp_l_path}/Tooth",
        name="clamp_l_tooth",
        size=1.0,
        scale=np.array([CLAMP_WIDTH, 0.15, 0.5]),
        position=np.array([BRIDGE_GAP / 2, -CLAMP_MAX_OPEN + CLAMP_LENGTH / 2 - 0.1, RAIL_HEIGHT - LIFT_HEIGHT - 0.15]),
        color=np.array(COLOR_CLAMP),
    ))

    # 夾具右臂
    clamp_r_path = f"{cable_path}/ClampR"
    clamp_r_prim = XFormPrim(prim_path=clamp_r_path, name="clamp_r")
    world.scene.add(clamp_r_prim)
    world.scene.add(VisualCuboid(
        prim_path=f"{clamp_r_path}/Arm",
        name="clamp_r_arm",
        size=1.0,
        scale=np.array([CLAMP_WIDTH, CLAMP_LENGTH, CLAMP_HEIGHT]),
        position=np.array([BRIDGE_GAP / 2, CLAMP_MAX_OPEN, RAIL_HEIGHT - LIFT_HEIGHT + 0.2]),
        color=np.array(COLOR_CLAMP),
    ))
    world.scene.add(VisualCuboid(
        prim_path=f"{clamp_r_path}/Tooth",
        name="clamp_r_tooth",
        size=1.0,
        scale=np.array([CLAMP_WIDTH, 0.15, 0.5]),
        position=np.array([BRIDGE_GAP / 2, CLAMP_MAX_OPEN - CLAMP_LENGTH / 2 + 0.1, RAIL_HEIGHT - LIFT_HEIGHT - 0.15]),
        color=np.array(COLOR_CLAMP),
    ))

    # ── 待吊物件（鋼胚） ──
    load = world.scene.add(DynamicCuboid(
        prim_path="/World/Load",
        name="steel_billet",
        size=1.0,
        scale=np.array([1.2, 3.0, 1.0]),
        position=np.array([-8.0, 0.0, 0.5]),
        color=np.array(COLOR_LOAD),
        mass=35000.0,
    ))

    # ── 第二個待吊物件 ──
    load2 = world.scene.add(DynamicCuboid(
        prim_path="/World/Load2",
        name="steel_billet_2",
        size=1.0,
        scale=np.array([1.2, 2.5, 0.8]),
        position=np.array([-4.0, -3.0, 0.4]),
        color=np.array((0.6, 0.55, 0.5)),
        mass=28000.0,
    ))

    # ── 控制器 ──
    controller = CraneController()

    # ── 初始位置參考值 ──
    bridge_base_z = RAIL_HEIGHT + 0.6 + BRIDGE_BEAM_H / 2
    trolley_base_z = RAIL_HEIGHT + 0.6 + BRIDGE_BEAM_H + TROLLEY_SIZE[2] / 2
    cable_top_z = RAIL_HEIGHT + 0.6 + BRIDGE_BEAM_H

    # ── 模擬迴圈 ──
    world.reset()
    print("=" * 60)
    print("  35 噸雙樑天車模擬啟動")
    print("  自動路徑：取貨 → 吊運 → 放貨 → 循環")
    print("=" * 60)

    step = 0
    dt = 1.0 / 60.0

    while simulation_app.is_running():
        world.step(render=True)
        step += 1

        if step < 10:
            continue

        # 取得控制器目標
        bx, ty, hz, co = controller.update(dt)

        # 限制範圍
        bx = np.clip(bx, -RUNWAY_LENGTH / 2 + 1, RUNWAY_LENGTH / 2 - 1)
        ty = np.clip(ty, -SPAN / 2 + 1, SPAN / 2 - 1)
        hz = np.clip(hz, 1.0, LIFT_HEIGHT)
        co = np.clip(co, 0.05, CLAMP_MAX_OPEN)

        cable_length = cable_top_z - (cable_top_z - LIFT_HEIGHT + hz)
        hook_z = cable_top_z - cable_length

        # 更新橋樑位置（X 軸移動）
        bridge_prim.set_world_pose(
            position=np.array([bx, 0, bridge_base_z])
        )

        # 更新小車位置（Y 軸移動）
        trolley_prim.set_world_pose(
            position=np.array([bx + BRIDGE_GAP / 2, ty, trolley_base_z])
        )

        # 更新鋼纜與夾具（Z 軸升降）
        cable_center_z = (cable_top_z + hook_z) / 2
        cable_len = cable_top_z - hook_z

        # 鋼纜
        wire_l = world.scene.get_object("wire_l")
        wire_r = world.scene.get_object("wire_r")
        if wire_l and wire_r:
            wire_l.set_world_pose(
                position=np.array([bx + BRIDGE_GAP / 2 - 0.5, ty, cable_center_z])
            )
            wire_l.set_local_scale(np.array([0.08, 0.08, max(cable_len, 0.1)]))
            wire_r.set_world_pose(
                position=np.array([bx + BRIDGE_GAP / 2 + 0.5, ty, cable_center_z])
            )
            wire_r.set_local_scale(np.array([0.08, 0.08, max(cable_len, 0.1)]))

        # 夾具基座
        clamp_base = world.scene.get_object("clamp_base")
        if clamp_base:
            clamp_base.set_world_pose(
                position=np.array([bx + BRIDGE_GAP / 2, ty, hook_z + 0.2])
            )

        # 夾具左臂
        clamp_l_arm = world.scene.get_object("clamp_l_arm")
        clamp_l_tooth = world.scene.get_object("clamp_l_tooth")
        if clamp_l_arm:
            clamp_l_arm.set_world_pose(
                position=np.array([bx + BRIDGE_GAP / 2, ty - co, hook_z - 0.2])
            )
        if clamp_l_tooth:
            clamp_l_tooth.set_world_pose(
                position=np.array([bx + BRIDGE_GAP / 2, ty - co + CLAMP_LENGTH / 2 - 0.1, hook_z - 0.75])
            )

        # 夾具右臂
        clamp_r_arm = world.scene.get_object("clamp_r_arm")
        clamp_r_tooth = world.scene.get_object("clamp_r_tooth")
        if clamp_r_arm:
            clamp_r_arm.set_world_pose(
                position=np.array([bx + BRIDGE_GAP / 2, ty + co, hook_z - 0.2])
            )
        if clamp_r_tooth:
            clamp_r_tooth.set_world_pose(
                position=np.array([bx + BRIDGE_GAP / 2, ty + co - CLAMP_LENGTH / 2 + 0.1, hook_z - 0.75])
            )

        # 狀態顯示（每 120 幀）
        if step % 120 == 0:
            wp = controller.waypoints[controller.wp_idx]
            print(f"[Step {step:>6d}] WP {controller.wp_idx} | "
                  f"Bridge X={bx:+.1f}m  Trolley Y={ty:+.1f}m  "
                  f"Hoist Z={hz:.1f}m  Clamp={co:.2f}m")

    simulation_app.close()


if __name__ == "__main__":
    main()
