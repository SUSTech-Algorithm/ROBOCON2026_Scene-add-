import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import cv2
from pathlib import Path

# --- 1. 路径自动解析 (保持不变，确保能找到文件) ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent 
SCENE_XML_PATH = PROJECT_ROOT / "models" / "mjcf" / "scene_costume_R2.xml"

# --- 2. 配置参数 (使用您提供的参数) ---
MAX_SPEED = 50.0       
ROTATION_SPEED = 3.0   
RAIL_MIN = 0.0
RAIL_MAX = 0.25        
RAIL_STEP = 0.02       

# 摄像头名称
CAMERA_NAME = "rgb_camera"

# Offset
OFFSETS = {
    'front_left':   -np.pi/4, 
    'front_right':  +np.pi/4,
    'rear_left':    -3*np.pi/4,
    'rear_right':   +3*np.pi/4
}

# 映射关系
WHEEL_MAP_CONFIG = {
    'front_left':   'RR', 
    'front_right':  'LR',  
    'rear_left':    'RF',  
    'rear_right':   'LF',  
}

# 几何坐标 (X=右, Y=前)
WHEEL_GEOMETRY = {
    'front_left':   (-1.0,  1.0), 
    'front_right':  ( 1.0,  1.0), 
    'rear_left':    (-1.0, -1.0), 
    'rear_right':   ( 1.0, -1.0), 
}

class ChassisController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        self.actuators = {}
        # 查找所有电机
        for name in ['LF', 'RF', 'LR', 'RR']:
            self.actuators[f"{name}_steer"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_steer")
            self.actuators[f"{name}_drive"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_drive")
            
            # 悬挂电机检查
            rail_name = f"{name}_rail"
            rail_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, rail_name)
            if rail_id != -1:
                self.actuators[rail_name] = rail_id
            else:
                print(f"[严重警告] XML中没找到 {rail_name}！按升降键将无效。")

        self.wheels = {}
        for logic_name, xml_prefix in WHEEL_MAP_CONFIG.items():
            # 这里添加 try-except 防止电机ID没找到报错
            try:
                steer_id = self.actuators[f"{xml_prefix}_steer"]
                drive_id = self.actuators[f"{xml_prefix}_drive"]
                
                if steer_id == -1 or drive_id == -1: continue

                wheel_data = {
                    'steer_id': steer_id,
                    'drive_id': drive_id,
                    'pos': WHEEL_GEOMETRY[logic_name]
                }
                if f"{xml_prefix}_rail" in self.actuators:
                    wheel_data['rail_id'] = self.actuators[f"{xml_prefix}_rail"]
                
                self.wheels[logic_name] = wheel_data
            except KeyError:
                pass

        self.vx = 0.0 
        self.vy = 0.0 
        self.w  = 0.0
        self.rail_height = 0.0

    def key_callback(self, keycode):
        # --- 按键侦测器 (完全使用您的逻辑) ---
        print(f"Debug: Key Pressed Code = {keycode}") 

        self.vx = 0.0
        self.vy = 0.0
        self.w = 0.0
        
        # 移动 (上下左右)
        if keycode == 265 or keycode == 87:   # Up / W
            self.vy = 1.0
        elif keycode == 264 or keycode == 83: # Down / S
            self.vy = -1.0
        elif keycode == 263 or keycode == 65: # Left / A
            self.vx = -1.0
        elif keycode == 262 or keycode == 68: # Right / D
            self.vx = 1.0
        
        # 旋转 (注意：这里严格按照您提供的：Q=-1, E=1)
        elif keycode == 81:  # Q
            self.w = -1.0  
        elif keycode == 69:  # E
            self.w = 1.0   

        # --- 悬挂控制 ---
        elif keycode == 61: # + 键 -> 设为最大
            self.rail_height = RAIL_MAX 
            print(f"悬挂: 升至最高 ({RAIL_MAX})")
            
        elif keycode == 45: # - 键 -> 设为最小
            self.rail_height = RAIL_MIN 
            print(f"悬挂: 降至最低 ({RAIL_MIN})")

        elif keycode == 32:  # Space
            self.vy = self.vx = self.w = 0.0

        # 限制范围
        self.rail_height = np.clip(self.rail_height, RAIL_MIN, RAIL_MAX)

    def update(self):
        for name, wheel in self.wheels.items():
            # 1. 悬挂控制
            if 'rail_id' in wheel:
                self.data.ctrl[wheel['rail_id']] = self.rail_height

            # 2. 运动学
            rx, ry = wheel['pos'] 
            wheel_vx = self.vx - (self.w * ROTATION_SPEED) * ry
            wheel_vy = self.vy + (self.w * ROTATION_SPEED) * rx
            
            target_speed = np.sqrt(wheel_vx**2 + wheel_vy**2)
            
            if target_speed < 0.1:
                self.data.ctrl[wheel['drive_id']] = 0.0
                continue

            target_angle = np.arctan2(wheel_vy, wheel_vx)
            final_angle = target_angle + OFFSETS[name]
            final_angle = (final_angle + np.pi) % (2 * np.pi) - np.pi
            
            self.data.ctrl[wheel['steer_id']] = final_angle
            self.data.ctrl[wheel['drive_id']] = target_speed * MAX_SPEED

def main():
    # 路径安全检查
    xml_path = SCENE_XML_PATH
    if not xml_path.exists():
        fallback_path = CURRENT_DIR / "scene_costume_R2.xml"
        if fallback_path.exists():
            xml_path = fallback_path
        else:
            print(f"[错误] 找不到 XML 文件: {xml_path}")
            return

    print(f"[-] 加载模型: {xml_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"[加载失败] {e}")
        return

    controller = ChassisController(model, data)

    # --- 摄像头设置 ---
    renderer = None
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id != -1:
        renderer = mujoco.Renderer(model, height=480, width=640)
        print(f"[-] 摄像头 '{CAMERA_NAME}' 就绪")
    else:
        print(f"[提示] 未找到摄像头 '{CAMERA_NAME}'，仅显示主窗口")

    dt = model.opt.timestep
    fps_target = 60.0
    steps_per_frame = int((1.0 / fps_target) / dt)

    with mujoco.viewer.launch_passive(model, data, key_callback=controller.key_callback) as viewer:
        mujoco.mj_resetData(model, data)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE 
        last_cam_time = 0

        print("=== 系统启动 ===")
        print("移动: 方向键 或 WASD")
        print("旋转: Q / E")
        print("悬挂: [ (降低) / ] (升高)")
        
        while viewer.is_running():
            step_start = time.time()
            
            # 更新控制器
            controller.update()
            
            # 物理步进
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
                
            viewer.sync()

            # OpenCV 渲染 (如果摄像头存在)
            if renderer and viewer.is_running():
                now = time.time()
                if now - last_cam_time > 0.05: 
                    try:
                        renderer.update_scene(data, camera=CAMERA_NAME)
                        img = renderer.render()
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Robot Cam", img_bgr)
                        if cv2.waitKey(1) == 27: break
                        last_cam_time = now
                    except Exception: pass

            elapsed = time.time() - step_start
            if elapsed < 1.0/fps_target:
                time.sleep(1.0/fps_target - elapsed)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()