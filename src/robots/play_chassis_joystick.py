import cv2
import mujoco
import mujoco.viewer
import numpy as np
import os
import signal
import sys
import time
import pygame

# ================= 配置区域 =================
XML_PATH = "models/mjcf/scene_costume_R2.xml"

RAIL_MIN = -0.50
RAIL_MAX = 0.25   
RAIL_SPEED = 0.001

NORMAL_SPEED = 20.0    
TURBO_SPEED  = 200.0   
ROTATION_SPEED = 5.0   

# 按键配置
KEY_CONFIG = {
    'FORWARD':  pygame.K_UP,
    'BACKWARD': pygame.K_DOWN,
    'LEFT':     pygame.K_LEFT,
    'RIGHT':    pygame.K_RIGHT,
    'TURN_L':   pygame.K_q,
    'TURN_R':   pygame.K_e,
    'F_UP':     pygame.K_EQUALS, # =
    'F_DOWN':   pygame.K_MINUS,  # -
    'R_UP':     pygame.K_LSHIFT,
    'R_DOWN':   pygame.K_RETURN, # Enter
    'TURBO':    pygame.K_SPACE,
    'QUIT':     pygame.K_ESCAPE
}

OFFSETS = {
    'front_left': -np.pi/4, 'front_right': +np.pi/4,
    'rear_left': -3*np.pi/4, 'rear_right': +3*np.pi/4
}
WHEEL_MAP_CONFIG = {
    'front_left': 'RR', 'front_right': 'LR',  
    'rear_left': 'RF', 'rear_right': 'LF',  
}
WHEEL_GEOMETRY = {
    'front_left': (-1.0, 1.0), 'front_right': (1.0, 1.0), 
    'rear_left': (-1.0, -1.0), 'rear_right': (1.0, -1.0), 
}
CAMERA_NAME = "rgb_camera"

class ChassisController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.init_input_system()
        
        self.is_auto_running = False  # 是否正在执行自动动作
        self.auto_start_time = 0.0    # 记录开始时间
        self.auto_phase = 0

        self.vx, self.vy, self.w = 0.0, 0.0, 0.0
        self.rail_pos_front = 0.0
        self.rail_pos_rear = 0.0
        
        self.rail_targets = {k: 0.0 for k in ['front_left', 'front_right', 'rear_left', 'rear_right']}
        self.current_max_speed = NORMAL_SPEED 

        self.actuators = {}
        self.wheels = {}
        
        for name in ['LF', 'RF', 'LR', 'RR']:
            s_n, d_n, r_n = f"{name}_steer", f"{name}_drive", f"{name}_rail"
            s_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, s_n)
            d_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, d_n)
            r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, r_n)
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_yaw_joint")
            if j_id == -1: j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_steer_joint")
            q_adr = model.jnt_qposadr[j_id] if j_id != -1 else None
            self.actuators[f"{name}_data"] = {'s': s_id, 'd': d_id, 'r': r_id, 'q': q_adr}

        for logic, prefix in WHEEL_MAP_CONFIG.items():
            d = self.actuators[f"{prefix}_data"]
            self.wheels[logic] = {'steer': d['s'], 'drive': d['d'], 'rail': d['r'], 'q': d['q'], 'pos': WHEEL_GEOMETRY[logic]}

    def init_input_system(self):
        pygame.init()
        # 🔥🔥🔥 关键修改：必须创建一个窗口才能接收键盘 🔥🔥🔥
        pygame.display.set_caption("点击这个窗口来控制机器人")
        self.screen = pygame.display.set_mode((400, 100))
        
        # 在窗口上写字提示
        font = pygame.font.SysFont("Arial", 24)
        text = font.render("Click HERE to control robot!", True, (255, 255, 255))
        self.screen.blit(text, (20, 30))
        pygame.display.flip()
        
        print("\n✅ 控制窗口已创建 - 请确保你点中了那个黑色小窗口！")

    def process_input(self):
        # 处理事件循环
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        keys = pygame.key.get_pressed()
        self.vx, self.vy, self.w = 0.0, 0.0, 0.0
        
        # 调试打印：如果你按键时这里没反应，说明窗口没聚焦点
        # if keys[KEY_CONFIG['FORWARD']]: print("DEBUG: 前进") 

        if keys[KEY_CONFIG['FORWARD']]:  self.vy = 1.0
        if keys[KEY_CONFIG['BACKWARD']]: self.vy = -1.0
        if keys[KEY_CONFIG['LEFT']]:     self.vx = -1.0
        if keys[KEY_CONFIG['RIGHT']]:    self.vx = 1.0
        if keys[KEY_CONFIG['TURN_L']]:   self.w = 1.0
        if keys[KEY_CONFIG['TURN_R']]:   self.w = -1.0

        if keys[KEY_CONFIG['TURBO']]:
            self.current_max_speed = TURBO_SPEED
        else:
            self.current_max_speed = NORMAL_SPEED

        if keys[KEY_CONFIG['F_UP']]:   self.rail_pos_front -= RAIL_SPEED
        if keys[KEY_CONFIG['F_DOWN']]: self.rail_pos_front += RAIL_SPEED
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: self.rail_pos_rear -= RAIL_SPEED
        if keys[KEY_CONFIG['R_DOWN']]: self.rail_pos_rear += RAIL_SPEED

        self.rail_pos_front = float(np.clip(self.rail_pos_front, RAIL_MIN, RAIL_MAX))
        self.rail_pos_rear  = float(np.clip(self.rail_pos_rear,  RAIL_MIN, RAIL_MAX))

        self.rail_targets['front_left']  = self.rail_pos_front
        self.rail_targets['front_right'] = self.rail_pos_front
        self.rail_targets['rear_left']   = self.rail_pos_rear
        self.rail_targets['rear_right']  = self.rail_pos_rear

    def optimize_module(self, current_angle, target_angle, target_speed):
        error = target_angle - current_angle
        error = np.arctan2(np.sin(error), np.cos(error))
        if abs(error) > (np.pi / 2):
            target_angle += np.pi
            target_speed = -target_speed
            error = target_angle - current_angle
            error = np.arctan2(np.sin(error), np.cos(error))
        scale_factor = np.cos(error)
        if scale_factor < 0.1: scale_factor = 0.0
        return np.arctan2(np.sin(target_angle), np.cos(target_angle)), target_speed * scale_factor

    def update(self):
        self.process_input()
        for name, wheel in self.wheels.items():
            if wheel['rail'] != -1:
                self.data.ctrl[wheel['rail']] = self.rail_targets[name]
            rx, ry = wheel['pos'] 
            wheel_vx = self.vx - (self.w * ROTATION_SPEED) * ry
            wheel_vy = self.vy + (self.w * ROTATION_SPEED) * rx
            raw_target_speed = np.sqrt(wheel_vx**2 + wheel_vy**2)
            if raw_target_speed < 0.05:
                self.data.ctrl[wheel['drive']] = 0.0
                continue
            raw_target_angle = np.arctan2(wheel_vy, wheel_vx) + OFFSETS[name]
            current_angle = 0.0
            if wheel['q'] is not None:
                raw_q = self.data.qpos[wheel['q']]
                current_angle = np.arctan2(np.sin(raw_q), np.cos(raw_q))
            opt_angle, opt_speed_factor = self.optimize_module(current_angle, raw_target_angle, raw_target_speed)
            self.data.ctrl[wheel['steer']] = opt_angle
            self.data.ctrl[wheel['drive']] = opt_speed_factor * self.current_max_speed
    def process_input(self):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            
            # ✅ 检测空格键：启动自动程序
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not self.is_auto_running:
                    print("🚀 [Space] 启动平滑连招...")
                    self.is_auto_running = True
                    self.auto_start_time = time.time()

        keys = pygame.key.get_pressed()
        
      
        if self.is_auto_running:
            # 1. 计算经过的时间
            elapsed = time.time() - self.auto_start_time
            
            # 2. 定义动作速度 (数值越小越慢)
            # 假设你的循环是60帧/秒，0.005 * 60 = 0.3 (即1秒内移动0.3的距离)
            AUTO_RAIL_SPEED = 0.0008  

            # --- 动作 A: 始终保持向前运动 ---
            self.vx = 0.0
            self.vy = 1.0   # 强制向前
            self.w  = 0.0

            # --- 阶段 0: 0秒 ~ 1.0秒 (缓慢上升/伸长) ---
            if elapsed < 1.0:
                # 前后腿同时慢慢伸长
                self.rail_pos_front += AUTO_RAIL_SPEED
                self.rail_pos_rear  += AUTO_RAIL_SPEED
            
            # --- 阶段 1: 1.0秒 ~ 2.0秒 (缓慢收 # ==========================================
        # 🎮 手动模式逻辑
        # ==========================================前腿) ---
            elif 1.02 <= elapsed < 2.04:
                # 慢慢收起前腿
                self.rail_pos_front -= AUTO_RAIL_SPEED

            # --- 阶段 2: 2.0秒 ~ 3.0秒 (缓慢收后腿) ---
            elif 6.0 <= elapsed < 7.36:
                # 慢慢收起后腿
                self.rail_pos_rear -= AUTO_RAIL_SPEED

            # --- 结束: 超过 3.0秒 ---
            elif elapsed >= 8.05:
                print("✅ 连招结束，切回手动模式")
                self.is_auto_running = False 
                # 如果希望结束后立刻停止移动，取消下面这行的注释
                # self.vx, self.vy, self.w = 0.0, 0.0, 0.0

       
        else:
            self.vx, self.vy, self.w = 0.0, 0.0, 0.0
            
            # 运动控制
            if keys[KEY_CONFIG['FORWARD']]:  self.vy = 1.0
            if keys[KEY_CONFIG['BACKWARD']]: self.vy = -1.0
            if keys[KEY_CONFIG['LEFT']]:     self.vx = -1.0
            if keys[KEY_CONFIG['RIGHT']]:    self.vx = 1.0
            if keys[KEY_CONFIG['TURN_L']]:   self.w = 1.0
            if keys[KEY_CONFIG['TURN_R']]:   self.w = -1.0

            # 导轨控制 (手动)
            if keys[KEY_CONFIG['F_UP']]:   self.rail_pos_front -= RAIL_SPEED
            if keys[KEY_CONFIG['F_DOWN']]: self.rail_pos_front += RAIL_SPEED
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: self.rail_pos_rear -= RAIL_SPEED
            if keys[KEY_CONFIG['R_DOWN']]: self.rail_pos_rear += RAIL_SPEED

        
        
        # 1. 速度限制 (Turbo模式)
        if keys[KEY_CONFIG['TURBO']]:
            self.current_max_speed = TURBO_SPEED
        else:
            self.current_max_speed = NORMAL_SPEED

        # 2. 物理限位 (防止超出导轨行程)
        # 这一步非常重要，它保证了自动模式下即使一直在加，也不会超出 max
        self.rail_pos_front = float(np.clip(self.rail_pos_front, RAIL_MIN, RAIL_MAX))
        self.rail_pos_rear  = float(np.clip(self.rail_pos_rear,  RAIL_MIN, RAIL_MAX))

        # 3. 将计算好的位置应用到电机目标
        self.rail_targets['front_left']  = self.rail_pos_front
        self.rail_targets['front_right'] = self.rail_pos_front
        self.rail_targets['rear_left']   = self.rail_pos_rear
        self.rail_targets['rear_right']  = self.rail_pos_rear

def main():
    stop_requested = False
    viewer_ref = {'viewer': None}
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    abs_xml_path = os.path.join(current_dir, "../../", XML_PATH)
    if not os.path.exists(abs_xml_path):
        if os.path.exists(XML_PATH): abs_xml_path = XML_PATH
        else: return

    def _request_stop(_signum=None, _frame=None):
        nonlocal stop_requested
        stop_requested = True
        if viewer_ref['viewer']: 
            try: viewer_ref['viewer'].close()
            except: pass
        try: cv2.destroyAllWindows()
        except: pass
        try: pygame.quit()
        except: pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        model = mujoco.MjModel.from_xml_path(abs_xml_path)
        data = mujoco.MjData(model)
        controller = ChassisController(model, data)
    except Exception as e:
        print(f"Error: {e}")
        return

    renderer = None
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id != -1: renderer = mujoco.Renderer(model, height=480, width=640)

    print("\n🎮 === 启动步骤 ===")
    print("1. 程序会弹出一个写着 'Click HERE' 的黑色小窗口。")
    print("2. ⚠️ 必须用鼠标点击那个黑色小窗口！⚠️")
    print("3. 然后按 ↑ ↓ ← → 控制移动，= - 升降前腿。")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer_ref['viewer'] = viewer
        mujoco.mj_resetData(model, data)
        # 🔥 这里我把初始高度设高了一点，防止还没开始动就陷进地里
        data.qpos[2] = 0.5 
        mujoco.mj_forward(model, data)
        start_time = time.time()
        last_cam_time = 0.0

        while viewer.is_running():
            if stop_requested: break
            step_start = time.time()
            controller.update()
            mujoco.mj_step(model, data)
            viewer.sync()

            if renderer and (time.time() - last_cam_time > 0.05):
                renderer.update_scene(data, camera=CAMERA_NAME)
                img = renderer.render()
                cv2.imshow("Robot Cam", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) == 27: break
                last_cam_time = time.time()

            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0: time.sleep(time_until_next)

if __name__ == "__main__":
    main()