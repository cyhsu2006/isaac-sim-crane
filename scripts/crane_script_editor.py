"""
35 噸鋼捲天車 — 循環吊運模擬
路線：靠岸點 → A 點夾取 → B 點放下 → 靠岸點
      靠岸點 → B 點夾取 → A 點放下 → 靠岸點
      無限循環
"""
import omni.usd
import omni.kit.app
from pxr import UsdGeom, UsdPhysics, Gf, Sdf
import numpy as np
import time

# ─── 參數 ─────────────────────────────────────────────────────────
RUNWAY_LENGTH = 40.0
SPAN = 24.0
RAIL_HEIGHT = 16.0
BRIDGE_BEAM_H = 1.2
BRIDGE_GAP = 3.0
LIFT_HEIGHT = 14.0
COIL_INNER_R = 0.35
COIL_OUTER_R = 0.90
COIL_WIDTH = 1.20
COIL_Z_ON_CHOCK = COIL_OUTER_R + 0.3

BRIDGE_SPEED = 2.5
TROLLEY_SPEED = 1.8
HOIST_SPEED = 1.5
CLAMP_SPEED = 0.6

# L 型鉤臂夾具（左右從鋼捲兩端伸入內孔）
HOOK_VERT_D = 0.12      # 垂直段厚度（Y 方向）
HOOK_FOOT_LEN = 0.45    # 水平段長度（Y 方向，伸入長度）
# CLAMP_OPEN：張開，底部水平段尖端要完全超過鋼捲外側
#   水平段尖端 Y = -co + HOOK_FOOT_LEN，需 < -COIL_WIDTH/2
#   所以 co > COIL_WIDTH/2 + HOOK_FOOT_LEN
CLAMP_OPEN = COIL_WIDTH / 2 + HOOK_FOOT_LEN + 0.05   # ≈1.1m，水平段完全在鋼捲外
# CLAMP_CLOSED：收合，垂直段內側面碰到鋼捲端面即停
#   垂直段內側面 Y = -co + HOOK_VERT_D/2，需 = -COIL_WIDTH/2
#   所以 co = COIL_WIDTH/2 + HOOK_VERT_D/2
CLAMP_CLOSED = COIL_WIDTH / 2 + HOOK_VERT_D / 2      # ≈0.66m，剛好碰到鋼捲端面

# 座標定義
PARK_POS = (-18.0, 0.0)     # 靠岸點（軌道最左側）
A_POS = (-8.0, -4.0)        # A 儲位
B_POS = (8.0, -4.0)         # B 儲位

# 顏色
YELLOW = Gf.Vec3f(0.9, 0.6, 0.1)
DARK_YELLOW = Gf.Vec3f(0.7, 0.5, 0.08)
GRAY = Gf.Vec3f(0.4, 0.4, 0.45)
DARK_GRAY = Gf.Vec3f(0.3, 0.3, 0.35)
RED = Gf.Vec3f(0.7, 0.15, 0.1)
ORANGE = Gf.Vec3f(0.85, 0.4, 0.1)
STEEL_COLOR = Gf.Vec3f(0.55, 0.55, 0.58)
CABLE_COLOR = Gf.Vec3f(0.12, 0.12, 0.12)
FLOOR_COLOR = Gf.Vec3f(0.22, 0.22, 0.25)
WALL_COLOR = Gf.Vec3f(0.3, 0.32, 0.35)
MARK_YELLOW = Gf.Vec3f(0.8, 0.7, 0.1)
CHOCK_COLOR = Gf.Vec3f(0.5, 0.35, 0.15)

stage = omni.usd.get_context().get_stage()

if stage.GetPrimAtPath("/World"):
    stage.RemovePrim("/World")

UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/Crane")
UsdGeom.Xform.Define(stage, "/World/Warehouse")
UsdGeom.Xform.Define(stage, "/World/Coils")


def make_box(path, size, pos, color, opacity=1.0):
    UsdGeom.Xform.Define(stage, path)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    cube = UsdGeom.Cube.Define(stage, f"{path}/mesh")
    cube.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(stage.GetPrimAtPath(f"{path}/mesh")).AddScaleOp().Set(Gf.Vec3d(*size))
    cube.GetDisplayColorAttr().Set([color])
    if opacity < 1.0:
        cube.GetDisplayOpacityAttr().Set([opacity])


def make_cylinder(path, radius, height, pos, color, rot=None):
    UsdGeom.Xform.Define(stage, path)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if rot:
        xf.AddRotateXYZOp().Set(Gf.Vec3d(*rot))
    cyl = UsdGeom.Cylinder.Define(stage, f"{path}/mesh")
    cyl.GetRadiusAttr().Set(radius)
    cyl.GetHeightAttr().Set(height)
    cyl.GetDisplayColorAttr().Set([color])


def set_translate(path, pos):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*pos))
            return


def set_scale(path, scale):
    prim = stage.GetPrimAtPath(f"{path}/mesh")
    if not prim:
        return
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            op.Set(Gf.Vec3d(*scale))
            return


# ═══════════════════════════════════════════════════════════════════
# 倉庫場景
# ═══════════════════════════════════════════════════════════════════

make_box("/World/Warehouse/Floor", (50, 32, 0.3), (0, 0, -0.15), FLOOR_COLOR)

for i, x in enumerate(np.linspace(-18, 18, 7)):
    make_box(f"/World/Warehouse/Lane_{i}", (0.15, 28, 0.02), (x, 0, 0.01), MARK_YELLOW)

make_box("/World/Warehouse/Wall_L", (0.3, 32, 10), (-25, 0, 5), WALL_COLOR)
make_box("/World/Warehouse/Wall_R", (0.3, 32, 10), (25, 0, 5), WALL_COLOR)
make_box("/World/Warehouse/Wall_B", (50, 0.3, 10), (0, -16, 5), WALL_COLOR)
make_box("/World/Warehouse/Wall_F", (50, 0.3, 6), (0, 16, 8), WALL_COLOR)

# 半透明屋頂樑
for i, x in enumerate(np.linspace(-20, 20, 9)):
    make_box(f"/World/Warehouse/RoofBeam_{i}", (0.3, 32, 0.4), (x, 0, RAIL_HEIGHT + 3), GRAY, opacity=0.2)

# A 儲位標記
make_box("/World/Warehouse/ZoneA", (6, 6, 0.03), (A_POS[0], A_POS[1], 0.015), Gf.Vec3f(0.15, 0.3, 0.15))
make_box("/World/Warehouse/ZoneA_Text", (1.5, 0.5, 0.02), (A_POS[0], A_POS[1] - 3.5, 0.02), Gf.Vec3f(0.2, 0.5, 0.2))

# B 儲位標記
make_box("/World/Warehouse/ZoneB", (6, 6, 0.03), (B_POS[0], B_POS[1], 0.015), Gf.Vec3f(0.15, 0.15, 0.3))
make_box("/World/Warehouse/ZoneB_Text", (1.5, 0.5, 0.02), (B_POS[0], B_POS[1] - 3.5, 0.02), Gf.Vec3f(0.2, 0.2, 0.5))

# 靠岸點標記
make_box("/World/Warehouse/ParkZone", (4, 4, 0.03), (PARK_POS[0], PARK_POS[1], 0.015), Gf.Vec3f(0.3, 0.3, 0.15))
make_box("/World/Warehouse/ParkText", (1.5, 0.5, 0.02), (PARK_POS[0], PARK_POS[1] - 2.5, 0.02), Gf.Vec3f(0.5, 0.5, 0.2))

# A 點枕木
make_box("/World/Warehouse/ChockA_L", (0.2, COIL_WIDTH + 0.2, 0.3), (A_POS[0] - 0.5, A_POS[1], 0.15), CHOCK_COLOR)
make_box("/World/Warehouse/ChockA_R", (0.2, COIL_WIDTH + 0.2, 0.3), (A_POS[0] + 0.5, A_POS[1], 0.15), CHOCK_COLOR)

# B 點枕木
make_box("/World/Warehouse/ChockB_L", (0.2, COIL_WIDTH + 0.2, 0.3), (B_POS[0] - 0.5, B_POS[1], 0.15), CHOCK_COLOR)
make_box("/World/Warehouse/ChockB_R", (0.2, COIL_WIDTH + 0.2, 0.3), (B_POS[0] + 0.5, B_POS[1], 0.15), CHOCK_COLOR)

# 其他倉庫內鋼捲（背景裝飾，不參與吊運）
bg_coils = [
    (-15, -8), (-12, -8), (-9, -8),
    (-15, 4), (-12, 4),
    (12, -8), (15, -8),
    (12, 4), (15, 4),
]
for i, (cx, cy) in enumerate(bg_coils):
    r = 0.85 + np.random.uniform(-0.05, 0.1)
    w = 1.0 + np.random.uniform(-0.1, 0.2)
    col = Gf.Vec3f(0.45 + np.random.uniform(0, 0.1),
                   0.45 + np.random.uniform(0, 0.1),
                   0.48 + np.random.uniform(0, 0.1))
    make_box(f"/World/Coils/BgChock_{i}_L", (0.2, w + 0.2, 0.3), (cx - 0.5, cy, 0.15), CHOCK_COLOR)
    make_box(f"/World/Coils/BgChock_{i}_R", (0.2, w + 0.2, 0.3), (cx + 0.5, cy, 0.15), CHOCK_COLOR)
    make_cylinder(f"/World/Coils/BgCoil_{i}", r, w, (cx, cy, r + 0.3), col, rot=(90, 0, 0))
    make_cylinder(f"/World/Coils/BgHole_{i}", COIL_INNER_R, w + 0.02, (cx, cy, r + 0.3),
                  Gf.Vec3f(0.08, 0.08, 0.08), rot=(90, 0, 0))

# 頂燈
for i, x in enumerate(np.linspace(-15, 15, 4)):
    for j, y in enumerate(np.linspace(-8, 8, 3)):
        lt = stage.DefinePrim(f"/World/Light_{i}_{j}", "RectLight")
        lt.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(5000.0)
        lt.CreateAttribute("inputs:width", Sdf.ValueTypeNames.Float).Set(3.0)
        lt.CreateAttribute("inputs:height", Sdf.ValueTypeNames.Float).Set(3.0)
        xf = UsdGeom.Xformable(lt)
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, RAIL_HEIGHT + 2.5))
        xf.AddRotateXYZOp().Set(Gf.Vec3d(180, 0, 0))

dome = stage.DefinePrim("/World/DomeLight", "DomeLight")
dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(600.0)

scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)

# ═══════════════════════════════════════════════════════════════════
# 被吊運的鋼捲（開始放在 A 點）
# ═══════════════════════════════════════════════════════════════════

COIL_PATH = "/World/Coils/ActiveCoil"
COIL_HOLE_PATH = "/World/Coils/ActiveCoil_Hole"

make_cylinder(COIL_PATH, COIL_OUTER_R, COIL_WIDTH,
              (A_POS[0], A_POS[1], COIL_Z_ON_CHOCK), STEEL_COLOR, rot=(90, 0, 0))
make_cylinder(COIL_HOLE_PATH, COIL_INNER_R, COIL_WIDTH + 0.02,
              (A_POS[0], A_POS[1], COIL_Z_ON_CHOCK), Gf.Vec3f(0.08, 0.08, 0.08), rot=(90, 0, 0))

# ═══════════════════════════════════════════════════════════════════
# 天車結構
# ═══════════════════════════════════════════════════════════════════

for ix, x in enumerate(np.linspace(-RUNWAY_LENGTH/2, RUNWAY_LENGTH/2, 6)):
    for iy, y_sign in enumerate([-1, 1]):
        y = y_sign * (SPAN / 2 + 0.5)
        make_box(f"/World/Crane/Pillar_{ix}_{iy}", (0.6, 0.6, RAIL_HEIGHT), (x, y, RAIL_HEIGHT / 2), GRAY)

for iy, y_sign in enumerate([-1, 1]):
    y = y_sign * (SPAN / 2 + 0.5)
    make_box(f"/World/Crane/Rail_{iy}", (RUNWAY_LENGTH + 1, 0.5, 0.6), (0, y, RAIL_HEIGHT + 0.3), GRAY)

bridge_z = RAIL_HEIGHT + 0.6 + BRIDGE_BEAM_H / 2
make_box("/World/Crane/Bridge/BeamL", (0.8, SPAN + 1.5, BRIDGE_BEAM_H), (-BRIDGE_GAP/2, 0, bridge_z), YELLOW)
make_box("/World/Crane/Bridge/BeamR", (0.8, SPAN + 1.5, BRIDGE_BEAM_H), (BRIDGE_GAP/2, 0, bridge_z), YELLOW)
for iy, y_sign in enumerate([-1, 1]):
    y = y_sign * (SPAN / 2 + 0.5)
    make_box(f"/World/Crane/Bridge/EndBeam_{iy}", (BRIDGE_GAP + 0.8, 0.5, 0.8), (0, y, RAIL_HEIGHT + 0.6 + 0.4), YELLOW)

trolley_z_base = RAIL_HEIGHT + 0.6 + BRIDGE_BEAM_H + 0.4
make_box("/World/Crane/Bridge/Trolley", (2.5, 3.0, 0.8), (0, 0, trolley_z_base), DARK_YELLOW)
make_box("/World/Crane/Bridge/HoistBox", (2.0, 2.2, 1.2), (0, 0, trolley_z_base + 1.0), DARK_YELLOW)
make_cylinder("/World/Crane/Bridge/Drum", 0.4, 1.5, (0, 0, trolley_z_base + 1.8), DARK_GRAY, rot=(90, 0, 0))

make_box("/World/Crane/Bridge/Cable_L", (0.05, 0.05, 10), (-0.5, 0, 10), CABLE_COLOR)
make_box("/World/Crane/Bridge/Cable_R", (0.05, 0.05, 10), (0.5, 0, 10), CABLE_COLOR)

# ═══════════════════════════════════════════════════════════════════
# L 型鉤臂夾具
# 結構：上橫樑 → 左右 L 型鉤臂（垂直段 + 水平段）
# 動作：左右鉤臂從鋼捲兩端外側向內合攏，水平段伸入內孔
# ═══════════════════════════════════════════════════════════════════

hz_def = 4.0
HOOK_VERT_H = 1.8       # 垂直段高度
HOOK_VERT_W = 0.15      # 垂直段寬（X）
# HOOK_VERT_D 和 HOOK_FOOT_LEN 已在上方定義
HOOK_FOOT_H = 0.15      # 水平段高度（Z）
HOOK_FOOT_W = 0.15      # 水平段寬（X）

# 上橫樑（連接鋼纜）
make_box("/World/Crane/Bridge/Clamp_Top", (1.2, 2.0, 0.3), (0, 0, hz_def), ORANGE)

# 左 L 鉤臂 — 垂直段（在鋼捲左側 Y- 方向）
make_box("/World/Crane/Bridge/HookL_Vert", (HOOK_VERT_W, HOOK_VERT_D, HOOK_VERT_H),
         (0, -CLAMP_OPEN, hz_def - 1.1), ORANGE)
# 左 L 鉤臂 — 水平段（底部向內伸入，+Y 方向）
make_box("/World/Crane/Bridge/HookL_Foot", (HOOK_FOOT_W, HOOK_FOOT_LEN, HOOK_FOOT_H),
         (0, -CLAMP_OPEN + HOOK_FOOT_LEN / 2, hz_def - 2.1), RED)

# 右 L 鉤臂 — 垂直段（在鋼捲右側 Y+ 方向）
make_box("/World/Crane/Bridge/HookR_Vert", (HOOK_VERT_W, HOOK_VERT_D, HOOK_VERT_H),
         (0, CLAMP_OPEN, hz_def - 1.1), ORANGE)
# 右 L 鉤臂 — 水平段（底部向內伸入，-Y 方向）
make_box("/World/Crane/Bridge/HookR_Foot", (HOOK_FOOT_W, HOOK_FOOT_LEN, HOOK_FOOT_H),
         (0, CLAMP_OPEN - HOOK_FOOT_LEN / 2, hz_def - 2.1), RED)


# ═══════════════════════════════════════════════════════════════════
# 循環吊運控制器
# 路線：靠岸 → A夾取 → B放下 → 靠岸 → B夾取 → A放下 → 靠岸 → 循環
# ═══════════════════════════════════════════════════════════════════

class CycleCraneController:
    def __init__(self):
        self.bridge_x = PARK_POS[0]
        self.trolley_y = PARK_POS[1]
        self.hoist_z = LIFT_HEIGHT
        self.clamp = CLAMP_OPEN
        self.holding = False

        # 鋼捲目前位置（開始在 A 點）
        self.coil_at = "A"  # "A" or "B"
        self.coil_x = A_POS[0]
        self.coil_y = A_POS[1]
        self.coil_z = COIL_Z_ON_CHOCK

        self.pickup_hook_z = COIL_Z_ON_CHOCK + 2.5
        self.drop_hook_z = COIL_Z_ON_CHOCK + 2.5

        self.step_idx = 0
        self.pause_timer = 0.0
        self.last_time = time.time()
        self.cycle_count = 0

        self._build_steps()

    def _build_steps(self):
        """建立當前循環的步驟"""
        if self.coil_at == "A":
            pick = A_POS
            drop = B_POS
            self.next_coil_at = "B"
        else:
            pick = B_POS
            drop = A_POS
            self.next_coil_at = "A"

        # 水平段在 hook_z - 2.1，要對準鋼捲中心 COIL_Z_ON_CHOCK
        hz_pick = COIL_Z_ON_CHOCK + 2.1
        hz_drop = COIL_Z_ON_CHOCK + 2.1

        # (name, bridge_x, trolley_y, hoist_z, clamp, holding, pause)
        self.steps = [
            # 從靠岸點出發，移到取貨點上方
            ("移動到取貨點上方",   pick[0], pick[1], LIFT_HEIGHT, CLAMP_OPEN,   False, 0.5),
            # 下降到鋼捲
            ("下降到鋼捲",        pick[0], pick[1], hz_pick,     CLAMP_OPEN,   False, 0.3),
            # 夾具擴張夾取
            ("夾取鋼捲",          pick[0], pick[1], hz_pick,     CLAMP_CLOSED, False, 1.0),
            # 起吊
            ("起吊",              pick[0], pick[1], LIFT_HEIGHT, CLAMP_CLOSED, True,  0.3),
            # 移到放貨點上方
            ("移動到放貨點上方",   drop[0], drop[1], LIFT_HEIGHT, CLAMP_CLOSED, True,  0.5),
            # 下降放置
            ("下降放置",          drop[0], drop[1], hz_drop,     CLAMP_CLOSED, True,  0.3),
            # 夾具收縮釋放
            ("釋放鋼捲",          drop[0], drop[1], hz_drop,     CLAMP_OPEN,   False, 1.0),
            # 空載上升
            ("空載上升",          drop[0], drop[1], LIFT_HEIGHT, CLAMP_OPEN,   False, 0.3),
            # 回到靠岸點
            ("回到靠岸點",  PARK_POS[0], PARK_POS[1], LIFT_HEIGHT, CLAMP_OPEN, False, 1.5),
        ]

    def update(self):
        now = time.time()
        dt = min(now - self.last_time, 0.1)
        self.last_time = now

        if self.pause_timer > 0:
            self.pause_timer -= dt
            self._update_visual()
            return

        step = self.steps[self.step_idx]
        name, tx, ty, tz, tc, hold, pause = step
        reached = True

        d = tx - self.bridge_x
        if abs(d) > 0.05:
            self.bridge_x += np.sign(d) * min(BRIDGE_SPEED * dt, abs(d))
            reached = False

        d = ty - self.trolley_y
        if abs(d) > 0.05:
            self.trolley_y += np.sign(d) * min(TROLLEY_SPEED * dt, abs(d))
            reached = False

        d = tz - self.hoist_z
        if abs(d) > 0.05:
            self.hoist_z += np.sign(d) * min(HOIST_SPEED * dt, abs(d))
            reached = False

        d = tc - self.clamp
        if abs(d) > 0.02:
            self.clamp += np.sign(d) * min(CLAMP_SPEED * dt, abs(d))
            reached = False

        self.holding = hold

        if reached:
            self.pause_timer = pause
            print(f"  [{self.cycle_count + 1}] {name}")

            # 夾取時標記
            if "夾取" in name:
                self.holding = True

            # 釋放時更新鋼捲位置
            if "釋放" in name:
                self.holding = False
                # 放到目標位置
                if self.coil_at == "A":
                    self.coil_x = B_POS[0]
                    self.coil_y = B_POS[1]
                else:
                    self.coil_x = A_POS[0]
                    self.coil_y = A_POS[1]
                self.coil_z = COIL_Z_ON_CHOCK
                set_translate(COIL_PATH, (self.coil_x, self.coil_y, self.coil_z))
                set_translate(COIL_HOLE_PATH, (self.coil_x, self.coil_y, self.coil_z))

            self.step_idx += 1

            # 循環完成，重建步驟
            if self.step_idx >= len(self.steps):
                self.step_idx = 0
                self.coil_at = self.next_coil_at
                self.cycle_count += 1
                self._build_steps()
                print(f"=== 第 {self.cycle_count + 1} 趟開始 "
                      f"({'A→B' if self.coil_at == 'A' else 'B→A'}) ===")

        self._update_visual()

    def _update_visual(self):
        bx = self.bridge_x
        ty_pos = self.trolley_y
        hook_z = self.hoist_z
        co = self.clamp

        cable_length = max(trolley_z_base - hook_z - 0.3, 0.1)
        cable_center = trolley_z_base - cable_length / 2

        # 橋樑
        set_translate("/World/Crane/Bridge/BeamL", (bx - BRIDGE_GAP/2, 0, bridge_z))
        set_translate("/World/Crane/Bridge/BeamR", (bx + BRIDGE_GAP/2, 0, bridge_z))
        for iy, y_sign in enumerate([-1, 1]):
            y = y_sign * (SPAN / 2 + 0.5)
            set_translate(f"/World/Crane/Bridge/EndBeam_{iy}", (bx, y, RAIL_HEIGHT + 0.6 + 0.4))

        # 小車
        set_translate("/World/Crane/Bridge/Trolley", (bx, ty_pos, trolley_z_base))
        set_translate("/World/Crane/Bridge/HoistBox", (bx, ty_pos, trolley_z_base + 1.0))
        set_translate("/World/Crane/Bridge/Drum", (bx, ty_pos, trolley_z_base + 1.8))

        # 鋼纜
        set_translate("/World/Crane/Bridge/Cable_L", (bx - 0.5, ty_pos, cable_center))
        set_translate("/World/Crane/Bridge/Cable_R", (bx + 0.5, ty_pos, cable_center))
        set_scale("/World/Crane/Bridge/Cable_L", (0.05, 0.05, cable_length))
        set_scale("/World/Crane/Bridge/Cable_R", (0.05, 0.05, cable_length))

        # L 型鉤臂夾具
        set_translate("/World/Crane/Bridge/Clamp_Top", (bx, ty_pos, hook_z))

        # 左 L 鉤 — 垂直段在 Y- 側，水平段向 +Y 伸入
        hook_l_y = ty_pos - co
        set_translate("/World/Crane/Bridge/HookL_Vert", (bx, hook_l_y, hook_z - 1.1))
        set_translate("/World/Crane/Bridge/HookL_Foot", (bx, hook_l_y + HOOK_FOOT_LEN / 2, hook_z - 2.1))

        # 右 L 鉤 — 垂直段在 Y+ 側，水平段向 -Y 伸入
        hook_r_y = ty_pos + co
        set_translate("/World/Crane/Bridge/HookR_Vert", (bx, hook_r_y, hook_z - 1.1))
        set_translate("/World/Crane/Bridge/HookR_Foot", (bx, hook_r_y - HOOK_FOOT_LEN / 2, hook_z - 2.1))

        # 鋼捲跟隨（鋼捲中心 = 水平段高度 = hook_z - 2.1）
        if self.holding:
            coil_z = hook_z - 2.1
            set_translate(COIL_PATH, (bx, ty_pos, coil_z))
            set_translate(COIL_HOLE_PATH, (bx, ty_pos, coil_z))


# ─── 啟動 ─────────────────────────────────────────────────────────
controller = CycleCraneController()

def on_update(e):
    controller.update()

update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(on_update)

print("=" * 60)
print("  35 噸鋼捲天車 — 循環吊運模擬")
print(f"  靠岸點: ({PARK_POS[0]}, {PARK_POS[1]})")
print(f"  A 儲位: ({A_POS[0]}, {A_POS[1]})")
print(f"  B 儲位: ({B_POS[0]}, {B_POS[1]})")
print("  路線: 靠岸→A夾取→B放下→靠岸→B夾取→A放下→靠岸→循環")
print("=" * 60)
print("=== 第 1 趟開始 (A→B) ===")
