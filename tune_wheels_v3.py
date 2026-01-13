import sys
import threading
import numpy as np
import mujoco
import mujoco.viewer as viewer
from pathlib import Path
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

# ================= 路径配置 =================
# 自动定位 XML 文件位置
CURRENT_DIR = Path(__file__).resolve().parent
SCENE_XML_PATH = CURRENT_DIR / "models" / "mjcf" / "scene_costume_R2.xml"

# ================= 轮子配置 =================
# 这里对应 XML 中 <geom> 的名字
WHEEL_GEOMS = [
    "LF_wheel_geom_visual",  # 1: 左前
    "LR_wheel_geom_visual",  # 2: 左后
    "RF_wheel_geom_visual",  # 3: 右前
    "RR_wheel_geom_visual"   # 4: 右后
]

class WheelTuner:
    def __init__(self):
        # 存储调整参数
        # 初始旋转设为 0 1.57 0 (90度)，这是大多数轮子的修正基准
        self.rot_offsets = {name: np.array([0.0, 1.57, 0.0]) for name in WHEEL_GEOMS} 
        self.pos_offsets = {name: np.array([0.0, 0.0, 0.0]) for name in WHEEL_GEOMS}
        
        # 状态控制
        self.current_wheel_idx = 0
        self.current_axis_idx = 1   # 默认选中 Y轴 (绿色)
        self.mode = "ROTATION"      # 模式: "ROTATION" 或 "TRANSLATION"
        
        # 步长 (按住 Shift 可以微调吗? 目前直接由代码定死)
        self.step_rot = 0.01        # 旋转每次 0.01 弧度 (约0.5度)
        self.step_pos = 0.0005      # 平移每次 0.5 毫米
        
        self.model = None
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            # === 1. 模式切换 (Tab) ===
            if key == keyboard.Key.tab:
                self.mode = "TRANSLATION" if self.mode == "ROTATION" else "ROTATION"
                print(f"\n[切换模式] === {self.mode} (平移/旋转) ===")

            # === 2. 保存 (Enter) ===
            elif key == keyboard.Key.enter:
                self.save_to_file()

            # === 3. 选择轮子 (1-4) ===
            elif hasattr(key, 'char') and key.char in ['1', '2', '3', '4']:
                self.current_wheel_idx = int(key.char) - 1
                print(f"\n[选中轮子]: {WHEEL_GEOMS[self.current_wheel_idx]}")
            
            # === 4. 选择轴 (Z=X轴, X=Y轴, C=Z轴) ===
            elif hasattr(key, 'char'):
                char = key.char.lower()
                if char == 'z': # 键盘左下角
                    self.current_axis_idx = 0
                    print(f"[选中轴]: X 轴 (红 - Roll/前后)")
                elif char == 'x': 
                    self.current_axis_idx = 1
                    print(f"[选中轴]: Y 轴 (绿 - Pitch/侧倾)")
                elif char == 'c': 
                    self.current_axis_idx = 2
                    print(f"[选中轴]: Z 轴 (蓝 - Yaw/转向)")

            # === 5. 调整数值 (上下箭头) ===
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
            print(f"旋转调试 ({target}) -> Euler: [{vals[0]:.3f}, {vals[1]:.3f}, {vals[2]:.3f}]")
        else:
            self.pos_offsets[target][self.current_axis_idx] += direction * self.step_pos
            vals = self.pos_offsets[target]
            print(f"平移调试 ({target}) -> Pos:   [{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}]")
        
        self.apply_offsets()

    def apply_offsets(self):
        if self.model is None: return
        
        for name in WHEEL_GEOMS:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id == -1: continue

            # 1. 应用旋转
            euler = self.rot_offsets[name]
            r = R.from_euler('xyz', euler)
            quat_scipy = r.as_quat() # [x, y, z, w]
            # MuJoCo Quat 顺序是 [w, x, y, z]
            self.model.geom_quat[geom_id] = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])

            # 2. 应用平移 (由于 geom_pos 是相对 body 的，直接覆盖即可)
            # 注意：如果 XML 原本有 pos 值，这里会覆盖它。建议 XML 初始设为 0 0 0
            self.model.geom_pos[geom_id] = self.pos_offsets[name]

    def save_to_file(self):
        filename = "final_wheel_params.xml"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n")
                f.write(self.generate_xml_string())
            print(f"\n✅ 保存成功！文件已生成: {CURRENT_DIR / filename}")
            print("快去复制粘贴吧！")
        except Exception as e:
            print(f"❌ 保存失败: {e}")

    def generate_xml_string(self):
        lines = []
        for name in WHEEL_GEOMS:
            rot = self.rot_offsets[name]
            pos = self.pos_offsets[name]
            
            rot_str = f"{rot[0]:.4f} {rot[1]:.4f} {rot[2]:.4f}"
            pos_str = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
            
            # 自动推断 mesh 名字
            mesh_name = name.replace("geom_visual", "link_vis")
            
            lines.append(f'')
            lines.append(f'<geom class="visual" mesh="{mesh_name}" material="wheel_material"')
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
    # 启动时立刻应用初始值
    tuner.apply_offsets()
    return model, data

if __name__ == "__main__":
    print("\n🎮 轮子手动校准工具 V3.0")
    print("-----------------------------------------")
    print(" 1. 按 [1/2/3/4] 选择轮子")
    print(" 2. 默认是 [旋转模式]，修复夹角")
    print("    按 [X] 键选中绿轴 (Pitch)，按 [↑/↓] 调整，直到轮毂变平")
    print(" 3. 按 [Tab] 切换到 [平移模式]，修复偏心")
    print("    按 [Z/X] 键选轴，按 [↑/↓] 移动，直到轴线穿过中心")
    print(" 4. 调好后按 [Enter] 保存代码")
    print("-----------------------------------------")
    
    viewer.launch(loader=load_callback)
    print("\n🎉 退出程序，感谢使用！")