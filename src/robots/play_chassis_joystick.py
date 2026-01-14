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
XML_PATH = "../../models/mjcf/scene_costume_R2.xml"
WHEEL_RADIUS = 0.03

# 速度配置
NORMAL_SPEED = 80.0    
TURBO_SPEED  = 200.0   
ROTATION_SPEED = 5.0   
RAIL_MIN = 0.00
RAIL_MAX = 0.25        

OFFSETS = {
    'front_left':   -np.pi/4, 'front_right':  +np.pi/4,
    'rear_left':    -3*np.pi/4, 'rear_right':   +3*np.pi/4
}

WHEEL_MAP_CONFIG = {
    'front_left':   'RR', 'front_right':  'LR',  
    'rear_left':    'RF', 'rear_right':   'LF',  
}

WHEEL_GEOMETRY = {
    'front_left':   (-1.0,  1.0), 'front_right':  ( 1.0,  1.0), 
    'rear_left':    (-1.0, -1.0), 'rear_right':   ( 1.0, -1.0), 
}

# 摄像头名称
CAMERA_NAME = "rgb_camera"

# ================= 滤波器类 (DLPF) =================
class ImuProcessor:
    def __init__(self, sample_rate=500.0, cutoff_freq=10.0):
        dt = 1.0 / sample_rate
        rc = 1.0 / (2 * np.pi * cutoff_freq)
        self.alpha = dt / (dt + rc)
        
        self.last_acc = None
        self.last_gyro = None

        # 模拟零偏 (Bias)
        self.acc_bias = np.array([0.02, -0.02, 0.05]) 
        self.gyro_bias = np.array([0.001, 0.001, -0.001])

    def process(self, raw_acc, raw_gyro):
        # 1. 加 Bias
        curr_acc = raw_acc + self.acc_bias
        curr_gyro = raw_gyro + self.gyro_bias
        
        # 2. 初始化
        if self.last_acc is None:
            self.last_acc = curr_acc
            self.last_gyro = curr_gyro
            return curr_acc, curr_gyro
        
        # 3. 滤波
        filt_acc = self.alpha * curr_acc + (1.0 - self.alpha) * self.last_acc
        filt_gyro = self.alpha * curr_gyro + (1.0 - self.alpha) * self.last_gyro
        
        # 更新
        self.last_acc = filt_acc
        self.last_gyro = filt_gyro
        
        return filt_acc, filt_gyro

# ================= 控制器类 (保持不变) =================
class ChassisController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.init_joystick()
        
        self.vx, self.vy, self.w = 0.0, 0.0, 0.0
        self._btn_prev = {'Y': False, 'A': False}
        self.front_raised = False
        self.rear_raised = False
        self.rail_targets = {
            'front_left': 0.0,
            'front_right': 0.0,
            'rear_left': 0.0,
            'rear_right': 0.0,
        }
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

    def init_joystick(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"✅ 手柄已连接: {self.joystick.get_name()}")
        else:
            print("❌ 未检测到手柄，请检查连接")

    def process_input(self):
        events = pygame.event.get()
        self.vx, self.vy, self.w = 0.0, 0.0, 0.0

        def _toggle_front():
            self.front_raised = not self.front_raised
            target = RAIL_MAX if self.front_raised else RAIL_MIN
            self.rail_targets['front_left'] = target
            self.rail_targets['front_right'] = target
            print(f"[rail] front -> {target:.3f}")

        def _toggle_rear():
            self.rear_raised = not self.rear_raised
            target = RAIL_MAX if self.rear_raised else RAIL_MIN
            self.rail_targets['rear_left'] = target
            self.rail_targets['rear_right'] = target
            print(f"[rail] rear  -> {target:.3f}")

        for ev in events:
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_y:
                    _toggle_front()
                elif ev.key == pygame.K_a:
                    _toggle_rear()
        
        if self.joystick:
            val_lx = self.joystick.get_axis(0) 
            val_ly = self.joystick.get_axis(1) 
            val_rx = self.joystick.get_axis(3) 

            if abs(val_lx) < 0.1: val_lx = 0
            if abs(val_ly) < 0.1: val_ly = 0
            if abs(val_rx) < 0.1: val_rx = 0
            
            self.vx = val_lx
            self.vy = -val_ly
            self.w  = -val_rx

            if self.joystick.get_button(5):
                self.current_max_speed = TURBO_SPEED
            else:
                self.current_max_speed = NORMAL_SPEED

            # 升降：前轮组=Y(3)，后轮组=A(0)，单键在 上/下 两个状态间切换
            y_now = bool(self.joystick.get_button(3))
            a_now = bool(self.joystick.get_button(0))
            if y_now and not self._btn_prev['Y']:
                _toggle_front()
            if a_now and not self._btn_prev['A']:
                _toggle_rear()
            self._btn_prev['Y'] = y_now
            self._btn_prev['A'] = a_now

        for k in self.rail_targets:
            self.rail_targets[k] = float(np.clip(self.rail_targets[k], RAIL_MIN, RAIL_MAX))

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

# ================= 主程序 =================
def main():
    stop_requested = False
    viewer_ref = {'viewer': None}

    def _request_stop(_signum=None, _frame=None):
        nonlocal stop_requested
        stop_requested = True
        print("\n[Ctrl+C] 收到停止信号，正在退出...", flush=True)

        v = viewer_ref.get('viewer')
        if v is not None:
            try:
                v.close()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            pygame.quit()
        except Exception:
            pass

        # 让退出尽可能确定（即使部分 GUI 调用正在阻塞）
        raise SystemExit(0)

    # 让 Ctrl+C/SIGTERM 更稳定地触发退出（部分 GUI 循环会让 KeyboardInterrupt 不及时）
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    # 尝试中断阻塞系统调用（sleep / I/O 等）
    try:
        signal.siginterrupt(signal.SIGINT, True)
        signal.siginterrupt(signal.SIGTERM, True)
    except Exception:
        pass

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    controller = ChassisController(model, data)

    # --- 摄像头设置 ---
    renderer = None
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id != -1:
        renderer = mujoco.Renderer(model, height=480, width=640)
        print(f"[-] 摄像头 '{CAMERA_NAME}' 就绪")
    else:
        print(f"[提示] 未找到摄像头 '{CAMERA_NAME}'，仅显示主窗口")
    
    # 滤波器 (10Hz)
    imu_filter = ImuProcessor(sample_rate=500.0, cutoff_freq=10.0)

    # 获取传感器地址
    try:
        acc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")
        gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        acc_adr = model.sensor_adr[acc_id]
        gyro_adr = model.sensor_adr[gyro_id]
    except:
        print("错误: XML 中找不到 imu_acc 或 imu_gyro")
        return

    # 数据记录
    history = {
        'time': [], 
        'acc_x': [], 'acc_y': [], 'acc_z': [], 
        'gyro_x': [], 'gyro_y': [], 'gyro_z': [], # 记录三轴陀螺仪
        'truth_w_z': [] # 记录真值 Z轴角速度
    }

    print("\n=== 开始仿真：请使用手柄控制 ===")
    print("👉 请尝试原地旋转、急转弯，观察陀螺仪数据")
    print("按手柄 [Start] 键或键盘 ESC 退出并查看图表")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer_ref['viewer'] = viewer
            mujoco.mj_resetData(model, data)
            data.qpos[2] = 0.17
            mujoco.mj_forward(model, data)

            start_time = time.time()
            last_cam_time = 0.0

            while viewer.is_running():
                if stop_requested:
                    print("收到停止信号，结束记录...")
                    break

                step_start = time.time()
                sim_time = step_start - start_time

                # 1. 控制更新
                controller.update()
                mujoco.mj_step(model, data)
                viewer.sync()

                # OpenCV 渲染 (如果摄像头存在)
                if renderer and viewer.is_running():
                    wall_time = time.time()
                    if wall_time - last_cam_time > 0.05:
                        renderer.update_scene(data, camera=CAMERA_NAME)
                        img = renderer.render()
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        img_bgr = cv2.resize(img_bgr, (1280, 960))
                        cv2.imshow("Robot Cam", img_bgr)
                        if cv2.waitKey(1) == 27:
                            print("ESC 退出")
                            break
                        last_cam_time = wall_time

                # 2. 数据采集
                raw_acc = data.sensordata[acc_adr:acc_adr+3].copy()
                raw_gyro = data.sensordata[gyro_adr:gyro_adr+3].copy()

                filt_acc, filt_gyro = imu_filter.process(raw_acc, raw_gyro)
                truth_w_z = data.qvel[5]

                history['time'].append(sim_time)
                history['acc_x'].append(filt_acc[0])
                history['acc_y'].append(filt_acc[1])
                history['acc_z'].append(filt_acc[2])

                history['gyro_x'].append(filt_gyro[0])
                history['gyro_y'].append(filt_gyro[1])
                history['gyro_z'].append(filt_gyro[2])
                history['truth_w_z'].append(truth_w_z)

                # 退出检测
                if controller.joystick and controller.joystick.get_button(7):
                    print("检测到 Start 键，结束记录...")
                    break

                time_until_next = model.opt.timestep - (time.time() - step_start)
                if stop_requested:
                    break
                if time_until_next > 0:
                    time.sleep(time_until_next)
    finally:
        viewer_ref['viewer'] = None
        cv2.destroyAllWindows()
        try:
            pygame.quit()
        except Exception:
            pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt 退出", flush=True)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass
        sys.exit(0)