import sys
import threading
import numpy as np
import mujoco
import mujoco.viewer as viewer
from pathlib import Path
from pynput import keyboard 

# ================= 路径配置 =================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
# 确保指向刚才修改好的 XML 文件
SCENE_XML_PATH = PROJECT_ROOT / "models" / "mjcf" / "scene_costume_R2.xml"

# ================= 运动学解算 =================
class SwerveDriveKinematics:
    def compute(self, vx, vy, wz):
        speed = np.sqrt(vx**2 + vy**2)
        angle = np.arctan2(vy, vx)
        
        if abs(wz) > 0.05:
            # 自旋模式
            angles = np.array([np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4])
            # 轮子速度指令 (对应 XML 里的 <velocity>)
            speeds = np.array([-wz, -wz, -wz, -wz]) * 100.0
        else:
            # 平移模式
            angles = np.array([angle, angle, angle, angle])
            speeds = np.array([speed, speed, speed, speed]) * 100.0
        return angles, speeds

# ================= 键盘控制器 =================
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

        # 启动监听
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
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

        # 下发指令 (与 XML actuator 顺序一致)
        # [0-3] Rail (Position控制)
        data.ctrl[0:4] = self.rail_target
        # [4-7] Yaw (Position控制)
        data.ctrl[4:8] = wheel_angles
        # [8-11] Wheel (Velocity控制)
        data.ctrl[8:12] = wheel_speeds

# ================= 主程序 =================
def load_callback(model=None, data=None):
    if not SCENE_XML_PATH.exists():
        print(f"[错误] 找不到文件: {SCENE_XML_PATH}")
        sys.exit(1)

    print(f"[-] 加载模型: {SCENE_XML_PATH.name}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = [0, 0, -9.81]

    controller = KeyboardController()
    mujoco.set_mjcb_control(controller.get_control)
    return model, data

if __name__ == "__main__":
    print("=== 键盘控制已启动 ===")
    print(" W/S: 前后 | A/D: 左右 | Q/E: 旋转")
    print(" R: 升高 | F: 降低")
    viewer.launch(loader=load_callback)