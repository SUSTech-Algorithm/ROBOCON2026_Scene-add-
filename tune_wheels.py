import sys
import threading
import numpy as np
import mujoco
import mujoco.viewer as viewer
from pathlib import Path
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

# ================= 路径配置 =================
# 使用当前目录作为基准 (修复了之前的路径错误)
CURRENT_DIR = Path(__file__).resolve().parent
SCENE_XML_PATH = CURRENT_DIR / "models" / "mjcf" / "scene_costume_R2.xml"

# ================= 调参配置 =================
WHEEL_GEOMS = [
    "LF_wheel_geom_visual",
    "LR_wheel_geom_visual",
    "RF_wheel_geom_visual",
    "RR_wheel_geom_visual"
]

class WheelTuner:
    def __init__(self):
        # 存储偏移量
        self.rot_offsets = {name: np.array([0.0, 0.0, 0.0]) for name in WHEEL_GEOMS} # Euler (Rad)
        self.pos_offsets = {name: np.array([0.0, 0.0, 0.0]) for name in WHEEL_GEOMS} # Position (m)
        
        # 原始位置备份 (在 load_callback 中初始化)
        self.default_pos = {} 

        # 状态控制
        self.current_wheel_idx = 0
        self.current_axis_idx = 1   # 0:X, 1:Y, 2:Z
        self.mode = "ROTATION"      # "ROTATION" or "TRANSLATION"
        
        # 步长设置
        self.step_rot = 0.01        # 旋转步长 (约0.5度)
        self.step_pos = 0.001       # 平移步长 (1mm)
        
        self.model = None

        # 启动键盘监听
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            # === 1. 模式切换 (Tab) ===
            if key == keyboard.Key.tab:
                if self.mode == "ROTATION":
                    self.mode = "TRANSLATION"
                else:
                    self.mode = "ROTATION"
                print(f"\n[切换模式]: 当前为 === {self.mode} (平移/旋转) ===")

            # === 2. 保存代码 (Enter) ===
            elif key == keyboard.Key.enter:
                self.save_to_file()

            # === 3. 选择轮子 (1-4) ===
            elif hasattr(key, 'char') and key.char in ['1', '2', '3', '4']:
                self.current_wheel_idx = int(key.char) - 1
                name = WHEEL_GEOMS[self.current_wheel_idx]
                print(f"\n[选中轮子]: {name}")
            
            # === 4. 选择轴 (Z/X/C) ===
            elif hasattr(key, 'char'):
                char = key.char.lower()
                if char == 'z':
                    self.current_axis_idx = 0
                    print(f"[选中轴]: X 轴 (红)")
                elif char == 'x':
                    self.current_axis_idx = 1
                    print(f"[选中轴]: Y 轴 (绿)")
                elif char == 'c':
                    self.current_axis_idx = 2
                    print(f"[选中轴]: Z 轴 (蓝)")
                elif char == 'p': # 依然保留打印功能
                    self.print_xml_code()

            # === 5. 调整数值 (Up/Down) ===
            if key == keyboard.Key.up:
                self.adjust_value(1)
            elif key == keyboard.Key.down:
                self.adjust_value(-1)

        except AttributeError:
            pass

    def adjust_value(self, direction):
        target = WHEEL_GEOMS[self.current_wheel_idx]
        
        if self.mode == "ROTATION":
            self.rot_offsets[target][self.current_axis_idx] += direction * self.step_rot
            vals = self.rot_offsets[target]
            print(f"旋转 {target} -> XYZ: [{vals[0]:.3f}, {vals[1]:.3f}, {vals[2]:.3f}]")
        else:
            self.pos_offsets[target][self.current_axis_idx] += direction * self.step_pos
            vals = self.pos_offsets[target]
            print(f"平移 {target} -> XYZ: [{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}]")
        
        self.apply_offsets()

    def apply_offsets(self):
        if self.model is None: return
        
        for name in WHEEL_GEOMS:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id == -1: continue

            # 1. 应用旋转 (修改 geom_quat)
            euler = self.rot_offsets[name]
            r = R.from_euler('xyz', euler)
            quat_scipy = r.as_quat() # [x, y, z, w]
            # MuJoCo format: [w, x, y, z]
            self.model.geom_quat[geom_id] = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])

            # 2. 应用平移 (修改 geom_pos)
            # 注意：这是基于原始位置的累加
            if name in self.default_pos:
                original = self.default_pos[name]
                offset = self.pos_offsets[name]
                self.model.geom_pos[geom_id] = original + offset

    def save_to_file(self):
        filename = "adjusted_params.xml"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n")
                f.write(self.generate_xml_string())
            print(f"\n✅ 保存成功！已写入文件: {CURRENT_DIR / filename}")
            print("请打开该文件，复制内容到你的 scene_costume_R2.xml 中。")
        except Exception as e:
            print(f"❌ 保存失败: {e}")

    def print_xml_code(self):
        print("\n" + "="*50)
        print(self.generate_xml_string())
        print("="*50 + "\n")

    def generate_xml_string(self):
        lines = []
        for name in WHEEL_GEOMS:
            rot = self.rot_offsets[name]
            pos = self.pos_offsets[name] # 这是偏移量，不是绝对位置，但 XML 里的 pos 是相对 body 的
            
            # 注意：XML 里的 pos 是覆盖式的。
            # 如果原始 XML 里 visual geom 有 pos 参数，你需要手动加上这个偏移量。
            # 如果原始 XML 里 visual geom 没有 pos (默认为0)，则直接用这个值。
            
            rot_str = f"{rot[0]:.4f} {rot[1]:.4f} {rot[2]:.4f}"
            pos_str = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" # 这里假设 XML 原始 pos 为 0 0 0
            
            lines.append(f'')
            # 生成完整的 geom 标签建议
            lines.append(f'<geom class="visual" mesh="{name.replace("_geom_visual", "_link_vis")}" material="wheel_material"')
            lines.append(f'      euler="{rot_str}"')
            lines.append(f'      pos="{pos_str}" />\n')
        return "\n".join(lines)

# ================= 主程序 =================
tuner = WheelTuner()

def load_callback(model=None, data=None):
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))
        data = mujoco.MjData(model)
    
    tuner.model = model
    
    # === 初始化：备份原始位置 ===
    if not tuner.default_pos:
        for name in WHEEL_GEOMS:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id != -1:
                # 深拷贝当前位置作为基准
                tuner.default_pos[name] = model.geom_pos[geom_id].copy()
                print(f"加载基准位置 {name}: {tuner.default_pos[name]}")

    tuner.apply_offsets()
    return model, data

if __name__ == "__main__":
    print("\n🎮 终极轮子调参工具 V2")
    print("-----------------------------------------")
    print(" [Tab]         -> 切换模式 (旋转 <-> 平移)")
    print(" [1-4]         -> 选择轮子")
    print(" [Z / X / C]   -> 选择轴 (X / Y / Z)")
    print(" [↑ / ↓]       -> 增减数值")
    print(" [Enter]       -> 保存代码到文件")
    print("-----------------------------------------")
    
    viewer.launch(loader=load_callback)