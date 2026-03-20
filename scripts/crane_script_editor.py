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
# 夾具開合方向：X 軸（平行鋼捲軸心，從前後伸入內孔）
# CLAMP_OPEN：張開，底部水平段尖端完全超過鋼捲外側
CLAMP_OPEN = COIL_WIDTH / 2 + HOOK_FOOT_LEN + 0.05   # ≈1.1m
# CLAMP_CLOSED：收合，垂直段內側面碰到鋼捲端面即停
CLAMP_CLOSED = COIL_WIDTH / 2 + HOOK_VERT_D / 2      # ≈0.66m

# 座標定義
PARK_POS = (-18.0, 0.0)     # 靠岸點（軌道最左側）

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


def set_rotate(path, rot):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            op.Set(Gf.Vec3d(*rot))
            return


def set_scale(path, scale):
    prim = stage.GetPrimAtPath(f"{path}/mesh")
    if not prim:
        return
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            op.Set(Gf.Vec3d(*scale))
            return


def set_cylinder_height(path, height):
    prim = stage.GetPrimAtPath(f"{path}/mesh")
    if not prim:
        return
    cyl = UsdGeom.Cylinder(prim)
    cyl.GetHeightAttr().Set(height)


def make_cable(path, radius, height, pos, color):
    """建立可旋轉的纜繩圓柱體（預設軸 Z，帶 translate + rotate）"""
    UsdGeom.Xform.Define(stage, path)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, 0))
    cyl = UsdGeom.Cylinder.Define(stage, f"{path}/mesh")
    cyl.GetRadiusAttr().Set(radius)
    cyl.GetHeightAttr().Set(height)
    cyl.GetDisplayColorAttr().Set([color])


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

# ═══════════════════════════════════════════════════════════════════
# 儲位系統 — 每個儲位有枕木，大部分放鋼捲，留兩個空位
# ═══════════════════════════════════════════════════════════════════

# 所有儲位座標（4 列 x 4 行 = 16 個儲位）
SLOT_POSITIONS = []
for row, y in enumerate([-8, -4, 0, 4]):
    for col, x in enumerate([-15, -9, 9, 15]):
        SLOT_POSITIONS.append((x, y))

NUM_SLOTS = len(SLOT_POSITIONS)

# 每個儲位建立枕木
for i, (sx, sy) in enumerate(SLOT_POSITIONS):
    make_box(f"/World/Warehouse/Chock_{i}_L", (COIL_WIDTH + 0.2, 0.2, 0.3), (sx, sy - 0.5, 0.15), CHOCK_COLOR)
    make_box(f"/World/Warehouse/Chock_{i}_R", (COIL_WIDTH + 0.2, 0.2, 0.3), (sx, sy + 0.5, 0.15), CHOCK_COLOR)

# 儲位狀態：哪些有鋼捲，哪些是空的
# 隨機選 2 個空位
np.random.seed(42)
empty_slots = set(np.random.choice(NUM_SLOTS, 2, replace=False))

# 鋼捲資料
coil_colors = [
    Gf.Vec3f(0.55, 0.55, 0.58), Gf.Vec3f(0.40, 0.40, 0.43),
    Gf.Vec3f(0.50, 0.50, 0.55), Gf.Vec3f(0.58, 0.55, 0.52),
    Gf.Vec3f(0.45, 0.45, 0.48), Gf.Vec3f(0.52, 0.52, 0.56),
]

class SlotInfo:
    """儲位資訊"""
    def __init__(self, idx, x, y):
        self.idx = idx
        self.x = x
        self.y = y
        self.has_coil = idx not in empty_slots
        self.coil_path = f"/World/Coils/Coil_{idx}"
        self.hole_path = f"/World/Coils/Hole_{idx}"
        self.outer_r = 0.85 + np.random.uniform(-0.05, 0.1)
        self.width = 1.0 + np.random.uniform(-0.1, 0.2)
        self.color = coil_colors[idx % len(coil_colors)]
        self.z = self.outer_r + 0.3

slots = [SlotInfo(i, x, y) for i, (x, y) in enumerate(SLOT_POSITIONS)]

# 建立鋼捲（有鋼捲的儲位才建）
for s in slots:
    if s.has_coil:
        make_cylinder(s.coil_path, s.outer_r, s.width, (s.x, s.y, s.z), s.color, rot=(0, 90, 0))
        make_cylinder(s.hole_path, COIL_INNER_R, s.width + 0.02, (s.x, s.y, s.z),
                      Gf.Vec3f(0.08, 0.08, 0.08), rot=(0, 90, 0))
    else:
        # 空位也建立隱藏的鋼捲（方便後續移入時顯示）
        make_cylinder(s.coil_path, COIL_OUTER_R, COIL_WIDTH, (s.x, s.y, -10), STEEL_COLOR, rot=(0, 90, 0))
        make_cylinder(s.hole_path, COIL_INNER_R, COIL_WIDTH + 0.02, (s.x, s.y, -10),
                      Gf.Vec3f(0.08, 0.08, 0.08), rot=(0, 90, 0))

# 靠岸點標記
make_box("/World/Warehouse/ParkZone", (4, 4, 0.03), (PARK_POS[0], PARK_POS[1], 0.015), Gf.Vec3f(0.3, 0.3, 0.15))

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

# （鋼捲已在儲位系統中建立）

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

# 鋼纜（圓柱體 + 旋轉，可傾斜）
# 一條纜繩穿過動滑輪形成兩段，Y 方向分開（平行夾具開合方向）
CABLE_RADIUS = 0.025     # 纜繩半徑
PULLEY_RADIUS = 0.15     # 動滑輪半徑
CABLE_Y_OFFSET = 0.2     # 兩段纜繩 Y 方向間距（動滑輪寬度的一半）

make_cable("/World/Crane/Bridge/Cable_L", CABLE_RADIUS, 10, (0, -CABLE_Y_OFFSET, 10), CABLE_COLOR)
make_cable("/World/Crane/Bridge/Cable_R", CABLE_RADIUS, 10, (0, CABLE_Y_OFFSET, 10), CABLE_COLOR)

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

# 動滑輪（夾具頂部，X 軸方向旋轉的小圓柱）
make_cylinder("/World/Crane/Bridge/Pulley", PULLEY_RADIUS, 0.5, (0, 0, hz_def + 0.3), DARK_GRAY, rot=(90, 0, 0))

# 上橫樑（連接動滑輪下方）
make_box("/World/Crane/Bridge/Clamp_Top", (1.2, 2.0, 0.3), (0, 0, hz_def), ORANGE)

# 左 L 鉤臂 — 垂直段（在鋼捲前側 X- 方向）
make_box("/World/Crane/Bridge/HookL_Vert", (HOOK_VERT_D, HOOK_VERT_W, HOOK_VERT_H),
         (-CLAMP_OPEN, 0, hz_def - 1.1), ORANGE)
# 左 L 鉤臂 — 水平段（底部向內伸入，+X 方向）
make_box("/World/Crane/Bridge/HookL_Foot", (HOOK_FOOT_LEN, HOOK_FOOT_W, HOOK_FOOT_H),
         (-CLAMP_OPEN + HOOK_FOOT_LEN / 2, 0, hz_def - 2.1), RED)

# 右 L 鉤臂 — 垂直段（在鋼捲後側 X+ 方向）
make_box("/World/Crane/Bridge/HookR_Vert", (HOOK_VERT_D, HOOK_VERT_W, HOOK_VERT_H),
         (CLAMP_OPEN, 0, hz_def - 1.1), ORANGE)
# 右 L 鉤臂 — 水平段（底部向內伸入，-X 方向）
make_box("/World/Crane/Bridge/HookR_Foot", (HOOK_FOOT_LEN, HOOK_FOOT_W, HOOK_FOOT_H),
         (CLAMP_OPEN - HOOK_FOOT_LEN / 2, 0, hz_def - 2.1), RED)


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
SWING_DAMPING = 0.4      # 擺盪阻尼係數（適中，停下後可見 3-5 次擺盪再衰減）

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
# 隨機吊運控制器
# 隨機選一顆鋼捲 → 夾到空位 → 原位變空位 → 無限循環
# ═══════════════════════════════════════════════════════════════════

class RandomCraneController:
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

        # 當前吊運的鋼捲
        self.current_coil = None   # SlotInfo
        self.target_slot = None    # SlotInfo（目標空位）

        self.step_idx = 0
        self.pause_timer = 0.0
        self.last_time = time.time()
        self.cycle_count = 0

        self._pick_next_task()

    def _pick_next_task(self):
        """隨機選一顆鋼捲和一個空位"""
        occupied = [s for s in slots if s.has_coil]
        empty = [s for s in slots if not s.has_coil]

        if not occupied or not empty:
            print("錯誤：沒有可用的鋼捲或空位")
            return

        self.current_coil = np.random.choice(occupied)
        self.target_slot = np.random.choice(empty)

        self.cycle_count += 1
        print(f"=== 第 {self.cycle_count} 趟："
              f"儲位 {self.current_coil.idx}({self.current_coil.x},{self.current_coil.y}) → "
              f"儲位 {self.target_slot.idx}({self.target_slot.x},{self.target_slot.y}) ===")

        self._build_steps()

    def _build_steps(self):
        """建立吊運步驟"""
        pick = self.current_coil
        drop = self.target_slot
        hz_pick = pick.z + 2.1    # 水平段對準鋼捲中心
        hz_drop = COIL_Z_ON_CHOCK + 2.1

        # (name, bridge_x, trolley_y, hoist_z, clamp, holding, pause)
        self.steps = [
            ("移動到取貨點上方",  pick.x, pick.y, LIFT_HEIGHT, CLAMP_OPEN,   False, 3.0),
            ("下降到鋼捲",       pick.x, pick.y, hz_pick,     CLAMP_OPEN,   False, 0.5),
            ("夾取鋼捲",         pick.x, pick.y, hz_pick,     CLAMP_CLOSED, False, 1.0),
            ("起吊",             pick.x, pick.y, LIFT_HEIGHT, CLAMP_CLOSED, True,  1.0),
            ("移動到放貨點上方",  drop.x, drop.y, LIFT_HEIGHT, CLAMP_CLOSED, True,  4.0),
            ("下降放置",         drop.x, drop.y, hz_drop,     CLAMP_CLOSED, True,  0.5),
            ("釋放鋼捲",         drop.x, drop.y, hz_drop,     CLAMP_OPEN,   False, 1.0),
            ("空載上升",         drop.x, drop.y, LIFT_HEIGHT, CLAMP_OPEN,   False, 0.5),
            ("回到靠岸點", PARK_POS[0], PARK_POS[1], LIFT_HEIGHT, CLAMP_OPEN, False, 2.0),
        ]
        self.step_idx = 0

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
                self.current_coil.has_coil = False  # 原位變空

            # 釋放時更新鋼捲位置到目標儲位
            if "釋放" in name:
                self.holding = False
                c = self.current_coil
                t = self.target_slot
                # 鋼捲放到目標位置
                t.has_coil = True
                t.outer_r = c.outer_r
                t.width = c.width
                t.color = c.color
                t.z = c.outer_r + 0.3
                # 移動視覺物件到目標位
                set_translate(c.coil_path, (t.x, t.y, t.z))
                set_translate(c.hole_path, (t.x, t.y, t.z))
                # 隱藏原位鋼捲（已被移走）
                # 交換 path：目標位的隱藏鋼捲移到原位下方，原位鋼捲已到目標位
                set_translate(t.coil_path, (c.x, c.y, -10))  # 隱藏目標位的空殼
                set_translate(t.hole_path, (c.x, c.y, -10))
                # 交換 path 引用
                c.coil_path, t.coil_path = t.coil_path, c.coil_path
                c.hole_path, t.hole_path = t.hole_path, c.hole_path
                print(f"  放下於儲位 {t.idx}")

            self.step_idx += 1

            # 完成一趟，選下一顆
            if self.step_idx >= len(self.steps):
                self._pick_next_task()

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

        # 鋼纜（圓柱體 + 旋轉）
        # 上端固定在小車絞盤，下端固定在動滑輪（跟隨擺盪）
        # 兩段纜繩沿 Y 方向分開（平行夾具開合方向）
        pulley_z = hang_z + 0.3  # 動滑輪位置

        for side, y_off in [("L", -CABLE_Y_OFFSET), ("R", CABLE_Y_OFFSET)]:
            # 上端（小車絞盤）
            top_x = bx
            top_y = ty_pos + y_off
            top_z = trolley_z_base
            # 下端（動滑輪）
            bot_x = hang_x
            bot_y = hang_y + y_off
            bot_z = pulley_z

            # 纜繩中點
            mid_x = (top_x + bot_x) / 2
            mid_y = (top_y + bot_y) / 2
            mid_z = (top_z + bot_z) / 2

            # 實際纜繩長度（含傾斜）
            dx = bot_x - top_x
            dy = bot_y - top_y
            dz = bot_z - top_z
            actual_length = np.sqrt(dx**2 + dy**2 + dz**2)

            # 計算傾斜角度（度）
            # 圓柱預設軸 Z，需旋轉使其對齊 top→bot 方向
            # X 旋轉：Y-Z 平面傾斜（小車 Y 方向移動引起）
            # Y 旋轉：X-Z 平面傾斜（大車 X 方向移動引起）
            rot_x = np.degrees(np.arctan2(dy, -dz)) if abs(dz) > 0.01 else 0
            rot_y = np.degrees(np.arctan2(-dx, -dz)) if abs(dz) > 0.01 else 0

            set_translate(f"/World/Crane/Bridge/Cable_{side}", (mid_x, mid_y, mid_z))
            set_rotate(f"/World/Crane/Bridge/Cable_{side}", (rot_x, rot_y, 0))
            set_cylinder_height(f"/World/Crane/Bridge/Cable_{side}", actual_length)

        # 動滑輪（跟隨夾具擺盪）
        set_translate("/World/Crane/Bridge/Pulley", (hang_x, hang_y, pulley_z))

        # L 型鉤臂夾具（整體跟隨擺盪，開合方向 X）
        set_translate("/World/Crane/Bridge/Clamp_Top", (hang_x, hang_y, hang_z))

        # 左 L 鉤（X- 側，水平段向 +X 伸入）
        hook_l_x = hang_x - co
        set_translate("/World/Crane/Bridge/HookL_Vert", (hook_l_x, hang_y, hang_z - 1.1))
        set_translate("/World/Crane/Bridge/HookL_Foot", (hook_l_x + HOOK_FOOT_LEN / 2, hang_y, hang_z - 2.1))

        # 右 L 鉤（X+ 側，水平段向 -X 伸入）
        hook_r_x = hang_x + co
        set_translate("/World/Crane/Bridge/HookR_Vert", (hook_r_x, hang_y, hang_z - 1.1))
        set_translate("/World/Crane/Bridge/HookR_Foot", (hook_r_x - HOOK_FOOT_LEN / 2, hang_y, hang_z - 2.1))

        # 鋼捲跟隨（與夾具為一體剛體，相同擺盪）
        if self.holding and self.current_coil:
            coil_z = hang_z - 2.1
            set_translate(self.current_coil.coil_path, (hang_x, hang_y, coil_z))
            set_translate(self.current_coil.hole_path, (hang_x, hang_y, coil_z))


# ─── 啟動 ─────────────────────────────────────────────────────────
controller = RandomCraneController()

def on_update(e):
    controller.update()

update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(on_update)

occupied_count = sum(1 for s in slots if s.has_coil)
empty_count = sum(1 for s in slots if not s.has_coil)
print("=" * 60)
print("  35 噸鋼捲天車 — 隨機吊運模擬")
print(f"  儲位數量: {NUM_SLOTS}")
print(f"  鋼捲: {occupied_count} 顆 | 空位: {empty_count} 個")
print(f"  靠岸點: ({PARK_POS[0]}, {PARK_POS[1]})")
print("  邏輯: 隨機選鋼捲 → 夾到空位 → 原位變空 → 循環")
print("=" * 60)
