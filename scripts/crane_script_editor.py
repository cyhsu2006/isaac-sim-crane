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

# 各軸最大速度 (m/s)
BRIDGE_SPEED = 2.5
TROLLEY_SPEED = 1.8
HOIST_SPEED = 1.5
CLAMP_SPEED = 0.6

# 各軸加速度 (m/s²) — 真實天車參數
BRIDGE_ACCEL = 0.4       # 大車加速度（較重，加速慢）
BRIDGE_DECEL = 0.5       # 大車減速度（煞車略快於加速）
TROLLEY_ACCEL = 0.5      # 小車加速度
TROLLEY_DECEL = 0.6      # 小車減速度
HOIST_ACCEL = 0.3        # 起升加速度（載重大，加速最慢）
HOIST_DECEL = 0.4        # 起升減速度
CLAMP_ACCEL = 0.8        # 夾具加速度（輕，反應快）
CLAMP_DECEL = 0.8        # 夾具減速度

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

# 屋頂樑已移除（避免與天車碰撞）

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
# 梯形速度曲線運動控制 + 多速精確定位
# ═══════════════════════════════════════════════════════════════════

# 多速控制參數
CRAWL_DISTANCE = 1.5     # 進入慢速區距離 (m)
CRAWL_SPEED_RATIO = 0.25 # 慢速為最大速度的比例

class AxisMotion:
    """單軸運動控制器 — 梯形速度曲線 + 慢速精確定位"""

    def __init__(self, max_speed, accel, decel):
        self.max_speed = max_speed
        self.accel = accel
        self.decel = decel
        self.velocity = 0.0
        self.prev_velocity = 0.0  # 記錄上一幀速度，用於計算加速度

    def update(self, current, target, dt):
        error = target - current
        distance = abs(error)

        if distance < 0.01:
            self.prev_velocity = self.velocity
            self.velocity = 0.0
            return target, True

        direction = np.sign(error)

        # 多速控制：接近目標時限制最大速度（慢速精確定位）
        if distance < CRAWL_DISTANCE:
            effective_max = self.max_speed * CRAWL_SPEED_RATIO
        else:
            effective_max = self.max_speed

        braking_distance = (self.velocity ** 2) / (2.0 * self.decel) if self.decel > 0 else 0

        self.prev_velocity = self.velocity

        if distance <= braking_distance + 0.02:
            self.velocity = max(self.velocity - self.decel * dt, 0.03)
        elif self.velocity > effective_max:
            # 超過當前區域限速，減速
            self.velocity = max(self.velocity - self.decel * dt, effective_max)
        elif self.velocity < effective_max:
            self.velocity = min(self.velocity + self.accel * dt, effective_max)

        move = min(self.velocity * dt, distance)
        new_pos = current + direction * move

        return new_pos, False

    def get_acceleration(self, dt):
        """回傳當前加速度 (m/s²)，用於擺盪計算（無方向，需外部加方向）"""
        if dt > 0:
            return (self.velocity - self.prev_velocity) / dt
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# 吊重擺盪物理模型（阻尼擺）
# ═══════════════════════════════════════════════════════════════════

GRAVITY = 9.81
SWING_DAMPING = 0.8      # 擺盪阻尼係數（空氣阻力 + 結構阻尼 + 鋼纜摩擦）

class PendulumSwing:
    """
    2D 阻尼擺模型（X/Y 各一軸）
    模擬天車加減速時吊重的擺盪效果
    θ'' = -(g/L)*sin(θ) - damping*θ' - (a_crane/L)
    """

    def __init__(self, damping=SWING_DAMPING):
        self.damping = damping
        # X 方向擺盪（大車加減速引起）
        self.angle_x = 0.0
        self.omega_x = 0.0  # 角速度
        # Y 方向擺盪（小車加減速引起）
        self.angle_y = 0.0
        self.omega_y = 0.0

    def update(self, cable_length, accel_x, accel_y, dt):
        """
        更新擺盪角度
        cable_length: 鋼纜長度 (m)
        accel_x: 大車加速度 (m/s²)，正 = +X 方向加速
        accel_y: 小車加速度 (m/s²)，正 = +Y 方向加速
        """
        if cable_length < 0.5:
            # 鋼纜太短，不擺盪
            self.angle_x *= 0.9
            self.angle_y *= 0.9
            return

        L = cable_length

        # X 軸擺盪（大車運動方向）
        # 天車加速時，吊重因慣性落後 → 擺向反方向
        alpha_x = (-(GRAVITY / L) * np.sin(self.angle_x)
                   - self.damping * self.omega_x
                   - (accel_x / L) * np.cos(self.angle_x))
        self.omega_x += alpha_x * dt
        self.angle_x += self.omega_x * dt
        # 限制最大擺角 ±15°
        self.angle_x = np.clip(self.angle_x, -0.26, 0.26)

        # Y 軸擺盪（小車運動方向）
        alpha_y = (-(GRAVITY / L) * np.sin(self.angle_y)
                   - self.damping * self.omega_y
                   - (accel_y / L) * np.cos(self.angle_y))
        self.omega_y += alpha_y * dt
        self.angle_y += self.omega_y * dt
        self.angle_y = np.clip(self.angle_y, -0.26, 0.26)

    def get_offset(self, cable_length):
        """回傳吊重相對於鉤點的水平偏移 (dx, dy)"""
        dx = cable_length * np.sin(self.angle_x)
        dy = cable_length * np.sin(self.angle_y)
        return dx, dy


# ═══════════════════════════════════════════════════════════════════
# 鋼纜彈性模型（彈簧阻尼器）
# ═══════════════════════════════════════════════════════════════════

CABLE_STIFFNESS = 50.0   # 鋼纜剛度 (N/m 等效，越大越硬)
CABLE_DAMPING = 8.0      # 鋼纜阻尼

class CableElasticity:
    """
    模擬鋼纜彈性：起吊瞬間鋼纜拉緊的延遲 + 微小彈跳
    吊重 Z 位置不會瞬間跟上鉤點，而是有彈簧效果
    """

    def __init__(self):
        self.coil_z_offset = 0.0   # 鋼捲相對於理想位置的偏移
        self.coil_z_vel = 0.0      # 偏移速度

    def update(self, target_offset, dt):
        """
        target_offset: 理想偏移量（通常為 0）
        回傳實際 Z 偏移量
        """
        # 彈簧力 + 阻尼力
        spring_force = -CABLE_STIFFNESS * (self.coil_z_offset - target_offset)
        damping_force = -CABLE_DAMPING * self.coil_z_vel
        accel = spring_force + damping_force

        self.coil_z_vel += accel * dt
        self.coil_z_offset += self.coil_z_vel * dt

        # 限制最大彈性偏移 ±0.3m
        self.coil_z_offset = np.clip(self.coil_z_offset, -0.3, 0.3)

        return self.coil_z_offset


# ═══════════════════════════════════════════════════════════════════
# 循環吊運控制器（含加減速）
# 路線：靠岸 → A夾取 → B放下 → 靠岸 → B夾取 → A放下 → 靠岸 → 循環
# ═══════════════════════════════════════════════════════════════════

class CycleCraneController:
    def __init__(self):
        self.bridge_x = PARK_POS[0]
        self.trolley_y = PARK_POS[1]
        self.hoist_z = LIFT_HEIGHT
        self.clamp = CLAMP_OPEN

        # 各軸獨立運動控制器
        self.bridge_motion = AxisMotion(BRIDGE_SPEED, BRIDGE_ACCEL, BRIDGE_DECEL)
        self.trolley_motion = AxisMotion(TROLLEY_SPEED, TROLLEY_ACCEL, TROLLEY_DECEL)
        self.hoist_motion = AxisMotion(HOIST_SPEED, HOIST_ACCEL, HOIST_DECEL)
        self.clamp_motion = AxisMotion(CLAMP_SPEED, CLAMP_ACCEL, CLAMP_DECEL)

        # 物理系統
        self.pendulum = PendulumSwing()
        self.cable_elastic = CableElasticity()
        self.prev_bridge_x = PARK_POS[0]
        self.prev_trolley_y = PARK_POS[1]
        self.prev_hoist_z = LIFT_HEIGHT
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
            # 暫停中仍需更新擺盪（自然衰減）
            self._update_physics(dt)
            self._update_visual(dt)
            return

        step = self.steps[self.step_idx]
        name, tx, ty, tz, tc, hold, pause = step

        # 各軸使用梯形速度曲線（加速→巡航→減速）
        self.bridge_x, bx_done = self.bridge_motion.update(self.bridge_x, tx, dt)
        self.trolley_y, ty_done = self.trolley_motion.update(self.trolley_y, ty, dt)
        self.hoist_z, hz_done = self.hoist_motion.update(self.hoist_z, tz, dt)
        self.clamp, cl_done = self.clamp_motion.update(self.clamp, tc, dt)

        reached = bx_done and ty_done and hz_done and cl_done

        self.holding = hold

        # 更新物理系統
        self._update_physics(dt)

        if reached:
            # 到達目標，各軸速度歸零
            self.bridge_motion.velocity = 0.0
            self.trolley_motion.velocity = 0.0
            self.hoist_motion.velocity = 0.0
            self.clamp_motion.velocity = 0.0

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

        self._update_visual(dt)

    def _update_physics(self, dt):
        """更新擺盪和鋼纜彈性"""
        if dt <= 0:
            return

        # 用位置差分計算帶方向的速度和加速度
        vel_x = (self.bridge_x - self.prev_bridge_x) / dt
        vel_y = (self.trolley_y - self.prev_trolley_y) / dt

        # 加速度 = 速度變化率（用儲存的前一幀速度）
        if not hasattr(self, '_prev_vel_x'):
            self._prev_vel_x = 0.0
            self._prev_vel_y = 0.0

        accel_x = (vel_x - self._prev_vel_x) / dt
        accel_y = (vel_y - self._prev_vel_y) / dt

        self._prev_vel_x = vel_x
        self._prev_vel_y = vel_y

        # 鋼纜長度
        hook_z = self.hoist_z
        cable_length = max(trolley_z_base - hook_z - 0.3, 0.5)

        # 只在水平移動時才驅動擺盪，純升降不應引起擺盪
        # 限制加速度輸入範圍，避免數值爆炸
        accel_x = np.clip(accel_x, -5.0, 5.0)
        accel_y = np.clip(accel_y, -5.0, 5.0)

        self.pendulum.update(cable_length, accel_x, accel_y, dt)

        # 鋼纜彈性（起吊/下降時的彈跳）
        hoist_vel = (self.hoist_z - self.prev_hoist_z) / dt
        # 起升加速時鋼纜拉伸
        elastic_drive = -hoist_vel * 0.01 if self.holding else 0
        self.cable_elastic.update(elastic_drive, dt)

        self.prev_bridge_x = self.bridge_x
        self.prev_trolley_y = self.trolley_y
        self.prev_hoist_z = self.hoist_z

    def _update_visual(self, dt=0.016):
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

        # 擺盪偏移（作用在整個懸吊體：鋼纜下端 + 夾具 + 鋼捲）
        swing_dx, swing_dy = self.pendulum.get_offset(cable_length)
        elastic_dz = self.cable_elastic.coil_z_offset if self.holding else 0

        # 懸吊點位置（夾具頂部，受擺盪影響）
        hang_x = bx + swing_dx
        hang_y = ty_pos + swing_dy
        hang_z = hook_z + elastic_dz

        # 鋼纜 — 上端固定在小車，下端連接夾具頂部（跟隨擺盪）
        # 左纜：上端 (bx-0.5, ty_pos, trolley_z_base) → 下端 (hang_x-0.5, hang_y, hang_z+0.15)
        # 右纜：上端 (bx+0.5, ty_pos, trolley_z_base) → 下端 (hang_x+0.5, hang_y, hang_z+0.15)
        for side, x_off in [("L", -0.5), ("R", 0.5)]:
            top_x = bx + x_off
            top_y = ty_pos
            bot_x = hang_x + x_off
            bot_y = hang_y
            mid_x = (top_x + bot_x) / 2
            mid_y = (top_y + bot_y) / 2
            set_translate(f"/World/Crane/Bridge/Cable_{side}", (mid_x, mid_y, cable_center))
            set_scale(f"/World/Crane/Bridge/Cable_{side}", (0.05, 0.05, cable_length))

        # L 型鉤臂夾具（整體跟隨擺盪）
        set_translate("/World/Crane/Bridge/Clamp_Top", (hang_x, hang_y, hang_z))

        # 左 L 鉤
        hook_l_y = hang_y - co
        set_translate("/World/Crane/Bridge/HookL_Vert", (hang_x, hook_l_y, hang_z - 1.1))
        set_translate("/World/Crane/Bridge/HookL_Foot", (hang_x, hook_l_y + HOOK_FOOT_LEN / 2, hang_z - 2.1))

        # 右 L 鉤
        hook_r_y = hang_y + co
        set_translate("/World/Crane/Bridge/HookR_Vert", (hang_x, hook_r_y, hang_z - 1.1))
        set_translate("/World/Crane/Bridge/HookR_Foot", (hang_x, hook_r_y - HOOK_FOOT_LEN / 2, hang_z - 2.1))

        # 鋼捲跟隨（與夾具為一體剛體，相同擺盪）
        if self.holding:
            coil_z = hang_z - 2.1
            set_translate(COIL_PATH, (hang_x, hang_y, coil_z))
            set_translate(COIL_HOLE_PATH, (hang_x, hang_y, coil_z))


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
