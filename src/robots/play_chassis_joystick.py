import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import cv2  # 用于显示摄像头画面
from pathlib import Path
from pynput import keyboard 

# ================= 1. 路径配置 =================
# 获取当前脚本所在目录
CURRENT_DIR = Path(__file__).resolve().parent

# 根据你的目录结构，回退两层找到项目根目录
# 假设结构: ROBOCON2026_Scene/src/robots/此脚本.py
PROJECT_ROOT = CURRENT_DIR.parent.parent 

# 指向你的 XML 文件 (请确认文件名是否正确)
# 这里指向我们刚才做好的 "costume_R2.xml" (包含光源、地面、摄像头)
SCENE_XML_PATH = PROJECT_ROOT / "models" / "mjcf" / "scene_costume_R2.xml"

# ================= 2. 运动学解算 (Swerve Drive) =================
class SwerveDriveKinematics:
    def compute(self, vx, vy, wz):
        speed = np.sqrt(vx**2 + vy**2)
        angle = np.arctan2(vy, vx)
        
        if abs(wz) > 0.05:
            # 自旋模式
            angles = np.array([np.pi/4, 3*np.pi/4, np.pi/4, 3*np.pi/4])
            # 轮子速度指令 (对应 XML 里的 <velocity>)
            speeds = np.array([-wz, -wz, -wz, -wz]) * 100.0
        else:
            # 平移模式
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
        self.MOVE_SPEED = 2.0   # 移动速度
        self.TURN_SPEED = 3.0   # 旋转速度
        self.RAIL_SPEED = 0.002 # 升降速度

        # 启动监听 (设置为守护线程，随主程序退出)
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.daemon = True
        self.listener.start()

    def on_press(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.pressed_keys.add(key.char.upper())
        except AttributeError:
            pass

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

        # 解析移动 WASD
        if 'W' in self.pressed_keys: vx += self.MOVE_SPEED
        if 'S' in self.pressed_keys: vx -= self.MOVE_SPEED
        if 'A' in self.pressed_keys: vy += self.MOVE_SPEED
        if 'D' in self.pressed_keys: vy -= self.MOVE_SPEED

        # 解析旋转 QE
        if 'Q' in self.pressed_keys: wz += self.TURN_SPEED
        if 'E' in self.pressed_keys: wz -= self.TURN_SPEED

        # 解析升降 RF
        if 'R' in self.pressed_keys: self.rail_target += self.RAIL_SPEED
        if 'F' in self.pressed_keys: self.rail_target -= self.RAIL_SPEED
        
        # 限制高度 0 ~ 0.25米
        self.rail_target = np.clip(self.rail_target, 0.0, 0.25)

        # 运动学解算
        wheel_angles, wheel_speeds = self.kinematics.compute(vx, vy, wz)

        # 下发指令
        data.ctrl[0:4] = self.rail_target   # 升降
        data.ctrl[4:8] = wheel_angles       # 转向
        data.ctrl[8:12] = wheel_speeds      # 驱动

# ================= 4. 主程序 (融合了控制和画面显示) =================
def main():
    # --- A. 路径检查 ---
    if not SCENE_XML_PATH.exists():
        # 如果找不到，尝试在当前目录找
        local_path = CURRENT_DIR / "costume_R2.xml"
        if local_path.exists():
            xml_path = str(local_path)
        else:
            print(f"[错误] 找不到文件: {SCENE_XML_PATH}")
            print(f"       也没有找到: {local_path}")
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
    mujoco.set_mjcb_control(controller.get_control)

    # --- D. 初始化摄像头渲染器 (用于 OpenCV 显示) ---
    camera_name = "head_camera"
    renderer = None
    try:
        # 检查摄像头是否存在
        model.camera(camera_name)
        # 创建渲染器: 640x480 分辨率
        renderer = mujoco.Renderer(model, height=480, width=640)
        print(f"[-] 摄像头 '{camera_name}' 初始化成功")
    except KeyError:
        print(f"[警告] XML中找不到摄像头 '{camera_name}'，将无法显示第一人称画面。")

    # --- E. 启动被动查看器循环 ---
    # 使用 launch_passive 才能自定义循环，实现 OpenCV 和 MuJoCo 并存
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("=== 仿真已启动 ===")
        print(" [键盘] W/A/S/D: 移动 | Q/E: 旋转 | R/F: 升降")
        
        last_render_time = 0
        
        while viewer.is_running():
            step_start = time.time()

            # 1. 物理步进 (让机器人动起来)
            mujoco.mj_step(model, data)

            # 2. 同步 MuJoCo 窗口 (显示光源和场景)
            viewer.sync()

            # 3. 处理摄像头画面 (OpenCV)
            if renderer:
                # 限制刷新率 30FPS，防止卡顿
                if time.time() - last_render_time > 0.033:
                    try:
                        renderer.update_scene(data, camera=camera_name)
                        img = renderer.render()
                        # RGB 转 BGR (OpenCV格式)
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Robot Camera", img_bgr)
                        cv2.waitKey(1)
                        last_render_time = time.time()
                    except Exception as e:
                        pass # 忽略渲染错误，防止程序崩溃

            # 4. 时间同步
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()