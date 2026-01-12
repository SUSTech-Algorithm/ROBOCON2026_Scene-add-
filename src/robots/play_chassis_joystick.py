import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import cv2
from pathlib import Path
from pynput import keyboard 

# ================= 1. 路径配置 =================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent 
SCENE_XML_PATH = PROJECT_ROOT / "models" / "mjcf" / "scene_costume_R2.xml"

# ================= 2. 运动学解算 =================
class SwerveDriveKinematics:
    def compute(self, vx, vy, wz):
        speed = np.sqrt(vx**2 + vy**2)
        angle = np.arctan2(vy, vx)
        
        if abs(wz) > 0.05:
            angles = np.array([np.pi/4, 3*np.pi/4, np.pi/4, 3*np.pi/4])
            speeds = np.array([-wz, -wz, -wz, -wz]) * 100.0
        else:
            angles = np.array([angle, angle, angle, angle])
            speeds = np.array([speed, speed, speed, speed]) * 100.0
        return angles, speeds

# ================= 3. 键盘控制器 =================
class KeyboardController:
    def __init__(self):
        self.kinematics = SwerveDriveKinematics()
        self.pressed_keys = set()
        
        # 初始状态
        self.rail_target = 0.0
        
        # 速度参数
        self.MOVE_SPEED = 2.0
        self.TURN_SPEED = 3.0
        self.RAIL_SPEED = 0.002
        
        # 暂停状态
        self.paused = True  # 初始设置为暂停

        # 启动监听
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.daemon = True
        self.listener.start()

    def on_press(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                char = key.char.upper()
                if char == 'P':  # 按P键也可暂停/继续
                    self.paused = not self.paused
                else:
                    self.pressed_keys.add(char)
        except AttributeError:
            if key == keyboard.Key.space:  # 空格键暂停/继续
                self.paused = not self.paused

    def on_release(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                char = key.char.upper()
                if char in self.pressed_keys:
                    self.pressed_keys.remove(char)
        except AttributeError:
            pass

    def get_control(self, model, data):
        vx, vy, wz = 0.0, 0.0, 0.0

        if 'W' in self.pressed_keys: vx += self.MOVE_SPEED
        if 'S' in self.pressed_keys: vx -= self.MOVE_SPEED
        if 'A' in self.pressed_keys: vy += self.MOVE_SPEED
        if 'D' in self.pressed_keys: vy -= self.MOVE_SPEED
        if 'Q' in self.pressed_keys: wz += self.TURN_SPEED
        if 'E' in self.pressed_keys: wz -= self.TURN_SPEED
        if 'R' in self.pressed_keys: self.rail_target += self.RAIL_SPEED
        if 'F' in self.pressed_keys: self.rail_target -= self.RAIL_SPEED
        
        self.rail_target = np.clip(self.rail_target, 0.0, 0.25)
        wheel_angles, wheel_speeds = self.kinematics.compute(vx, vy, wz)
        
        data.ctrl[0:4] = self.rail_target
        data.ctrl[4:8] = wheel_angles
        data.ctrl[8:12] = wheel_speeds

# ================= 4. 主程序 =================
def main():
    # --- A. 路径检查 ---
    if not SCENE_XML_PATH.exists():
        local_path = CURRENT_DIR / "costume_R2.xml"
        if local_path.exists():
            xml_path = str(local_path)
        else:
            print(f"[错误] 找不到文件: {SCENE_XML_PATH}")
            return
    else:
        xml_path = str(SCENE_XML_PATH)

    print(f"[-] 加载模型: {xml_path}")
    
    # --- B. 加载模型 ---
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"[加载失败] {e}")
        return

    model.opt.gravity[:] = [0, 0, -9.81]

    # --- C. 初始化控制器 ---
    controller = KeyboardController()
    
    # --- D. 重要步骤：在开始仿真前计算初始位置 ---
    # 这确保模型被完全放置在初始位置
    mujoco.mj_forward(model, data)
    print("模型已加载到初始位置")
    print(f"机器人位置: {data.qpos[0:3]}")
    print(f"机器人姿态: {data.qpos[3:7]}")
    
    # 显示初始暂停信息
    print("=== 仿真初始暂停中 ===")
    print("按 [空格键] 或 [P键] 开始/暂停仿真")
    
    # --- E. 初始化摄像头渲染器 ---
    camera_name = "head_camera"
    renderer = None
    try:
        model.camera(camera_name)
        renderer = mujoco.Renderer(model, height=480, width=640)
        print(f"[-] 摄像头 '{camera_name}' 初始化成功")
    except Exception as e:
        print(f"[警告] 摄像头初始化失败: {e}")

    # 创建OpenCV窗口
    cv2.namedWindow("Robot Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Camera", 640, 480)
    
    # 添加一个简单的键盘状态检查，不使用mujoco的回调
    # 因为mujoco的key_callback在暂停时可能不响应
    last_pause_check = 0
    last_control_update = 0

    # --- F. 启动被动查看器循环 ---
    with mujoco.viewer.launch_passive(model, data, show_right_ui=False) as viewer:
        print("=== 仿真就绪 ===")
        print(" [控制] 空格键/P键: 暂停/继续")
        print(" [移动] W/A/S/D: 移动 | Q/E: 旋转 | R/F: 升降")
        
        last_render_time = 0
        
        try:
            while viewer.is_running():
                step_start = time.time()
                
                # 1. 检查键盘输入（包括空格键）
                # 使用cv2的waitKey检查空格键
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') or key == ord('p') or key == ord('P'):
                    # 简单的防抖处理
                    if time.time() - last_pause_check > 0.2:
                        controller.paused = not controller.paused
                        if controller.paused:
                            print("=== 仿真已暂停 ===")
                        else:
                            print("=== 仿真运行中 ===")
                        last_pause_check = time.time()
                
                # 2. 更新控制指令（无论是否暂停）
                # 这样可以确保暂停时也能设置目标位置
                if time.time() - last_control_update > 0.01:  # 100Hz控制更新
                    controller.get_control(model, data)
                    last_control_update = time.time()
                
                # 3. 物理步进（仅在非暂停状态）
                if not controller.paused:
                    mujoco.mj_step(model, data)
                else:
                    # 暂停时，我们可以执行一些其他操作
                    # 比如更新viewer但不进行物理计算
                    pass
                
                # 4. 同步 MuJoCo 窗口
                viewer.sync()
                
                # 5. 处理摄像头画面
                if renderer and (time.time() - last_render_time > 0.033):
                    try:
                        renderer.update_scene(data, camera=camera_name)
                        img = renderer.render()
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        
                        # 在画面上添加暂停状态
                        if controller.paused:
                            cv2.putText(img_bgr, "PAUSED", (20, 40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        cv2.imshow("Robot Camera", img_bgr)
                        last_render_time = time.time()
                    except Exception as e:
                        pass
                
                # 6. 时间同步
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
        except KeyboardInterrupt:
            print("\n[!] 用户中断")
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()