import time
import math
import collections
import collections.abc

# --- 【修復關鍵】這段一定要放在最上面 ---
try:
    if not hasattr(collections, 'MutableMapping'):
        collections.MutableMapping = collections.abc.MutableMapping
    if not hasattr(collections, 'Iterable'):
        collections.Iterable = collections.abc.Iterable
except Exception:
    pass
# --------------------------------------

from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

# --- 解決 Python 3.10+ 相容性問題 ---
try:
    collections.MutableMapping = collections.abc.MutableMapping
except AttributeError:
    pass

# --- 參數設定 ---
CONNECTION_STRING = 'udp:127.0.0.1:14551' # 預設連線 SITL
INITIAL_TARGET_ALTITUDE = 30 # 初始起飛高度
OUTER_RADIUS = 50   
INNER_RADIUS = 28   
FLIGHT_SPEED = 8    
APPROACH_SPEED = 15 

# 多層六芒星參數
NUM_LAYERS = 3                 # 六芒星層數
ALTITUDE_DECREMENT_PER_LAYER = 5 # 每層高度遞減值
RADIUS_DECREMENT_FACTOR_PER_LAYER = 0.8 # 每層半徑遞減係數 (例如 0.8 表示下一層半徑是當前層的 80%)

print(f"正在連接至無人機 ({CONNECTION_STRING})...")
vehicle = connect(CONNECTION_STRING, wait_ready=True)

def arm_and_takeoff(target_alt):
    """
    解鎖並起飛至指定高度
    """
    print(">>> 執行起飛程序...")
    while not vehicle.is_armable:
        print(" 等待無人機初始化 (GPS/磁力計)...")
        time.sleep(1)
        
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    
    while not vehicle.armed:
        print(" 等待解鎖...")
        time.sleep(1)
        
    print(f">>> 正在起飛至 {target_alt}m...")
    vehicle.simple_takeoff(target_alt)
    
    while True:
        alt = vehicle.location.global_relative_frame.alt
        if alt >= target_alt * 0.95:
            print(" 已到達目標高度")
            break
        time.sleep(1)

def set_yaw(heading, relative=False):
    """
    控制無人機機頭轉向 (MAV_CMD_CONDITION_YAW)
    """
    if relative:
        is_relative = 1 # 相對角度
    else:
        is_relative = 0 # 絕對角度 (0 = 北)
    
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, # command
        0,       # confirmation
        heading, # param 1, angle in degrees
        0,       # param 2, speed degree/s
        1,       # param 3, direction -1 ccw, 1 cw
        is_relative, # param 4, relative offset 1, absolute 0
        0, 0, 0) # param 5 ~ 7 not used
    vehicle.send_mavlink(msg)

def fly_to_local(target_n, target_e, target_alt, speed=None):
    """
    使用 NED 座標系導航
    """
    # 1. 如果有設定速度，發送速度指令
    if speed:
        msg = vehicle.message_factory.command_long_encode(
            0, 0, mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 0, 0, speed, -1, 0, 0, 0, 0)
        vehicle.send_mavlink(msg)

    # 2. 計算目標方位角，讓機頭轉向目標
    curr_n = vehicle.location.local_frame.north or 0
    curr_e = vehicle.location.local_frame.east or 0
    angle = math.degrees(math.atan2(target_e - curr_e, target_n - curr_n))
    if angle < 0: angle += 360
    set_yaw(angle)

    # 3. 發送位置目標 (SET_POSITION_TARGET_LOCAL_NED)
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, 
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, 
        0b0000111111111000, # 控制位置
        target_n, target_e, -target_alt, 
        0, 0, 0, 0, 0, 0, 0, 0)
    vehicle.send_mavlink(msg)
    
    # 4. 抵達判定
    while True:
        n = vehicle.location.local_frame.north or 0
        e = vehicle.location.local_frame.east or 0
        alt = vehicle.location.global_relative_frame.alt
        dist_horizontal = math.sqrt((target_n - n)**2 + (target_e - e)**2)
        dist_vertical = abs(target_alt - alt)
        if dist_horizontal < 1.5 and dist_vertical < 1.0: # 抵達半徑 1.5m, 垂直誤差 1m
            break 
        time.sleep(0.5)

def generate_star_points(outer_r, inner_r):
    """
    生成六芒星的頂點
    """
    points = []
    for i in range(13): 
        angle_deg = i * 30 
        rad = math.radians(angle_deg)
        r = outer_r if i % 2 == 0 else inner_r
        points.append((r * math.cos(rad), r * math.sin(rad)))
    return points

# --- 主程序執行 ---
try:
    # 執行流程
    arm_and_takeoff(INITIAL_TARGET_ALTITUDE)
    
    current_alt = INITIAL_TARGET_ALTITUDE
    current_outer_radius = OUTER_RADIUS
    current_inner_radius = INNER_RADIUS

    for layer in range(NUM_LAYERS):
        print(f"\n>>> 開始繪製第 {layer + 1} 層六芒星 (高度: {current_alt:.1f}m, 外半徑: {current_outer_radius:.1f}m)...")
        points = generate_star_points(current_outer_radius, current_inner_radius)

        # 前往第一個點 (每層的第一個點都用 APPROACH_SPEED)
        start_n, start_e = points[0]
        print(f" -> 前往起始點 (N:{start_n:.1f}, E:{start_e:.1f})")
        fly_to_local(start_n, start_e, current_alt, speed=APPROACH_SPEED)

        # 清除軌跡提示 (只在第一層繪製前提示一次)
        if layer == 0:
            print("\n" + "!"*40)
            print(" 請在 QGC/Mission Planner 清除目前軌跡線 (Clear Track)")
            print("!"*40)
            for i in range(10, 0, -1):
                print(f" 繪製倒數 {i} 秒...", end="\r")
                time.sleep(1)
        
        # 開始繪圖
        for idx, (n, e) in enumerate(points[1:]): 
            print(f" -> 飛往頂點 {idx+2}/13 (N:{n:.1f}, E:{e:.1f})")
            fly_to_local(n, e, current_alt, speed=FLIGHT_SPEED)

        # 更新下一層的高度和半徑
        current_alt = max(5, current_alt - ALTITUDE_DECREMENT_PER_LAYER) # 避免高度過低
        current_outer_radius *= RADIUS_DECREMENT_FACTOR_PER_LAYER
        current_inner_radius *= RADIUS_DECREMENT_FACTOR_PER_LAYER

    print("\n>>> 所有六芒星繪製完成！執行原地降落...")
    vehicle.mode = VehicleMode("LAND")
    
    while vehicle.location.global_relative_frame.alt > 0.3:
        print(f" 降落高度: {vehicle.location.global_relative_frame.alt:.1f}m", end="\r")
        time.sleep(1)
    
    print("\n任務成功。")

except KeyboardInterrupt:
    print("\n使用者手動中斷，緊急切換至 LAND")
    vehicle.mode = VehicleMode("LAND")
finally:
    vehicle.close()
    print("通訊連線已安全關閉")