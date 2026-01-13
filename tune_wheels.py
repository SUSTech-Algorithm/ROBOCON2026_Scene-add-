import sys
import shutil
import re
import numpy as np
import mujoco
import mujoco.viewer as viewer
from pathlib import Path
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

# ================= 路径配置 (关键修改) =================
# 根目录: /home/yxm/ROBOCON2026_Scene
CURRENT_DIR = Path(__file__).resolve().parent

# 之前的错误路径: .../models/mjcf/scene_costume_R2.xml
# 现在的修正路径: 指向你提供的 /models/robots/costume_R2/ 文件夹
# 请确认你的 XML 文件名！通常是 costume_R2.xml 或 scene_costume_R2.xml
# 这里我尝试找该目录下的 .xml 文件
ROBOT_DIR = Path("/home/yxm/ROBOCON2026_Scene/models/robots/costume_R2")
xml_files = list(ROBOT_DIR.glob("*.xml"))

if not xml_files:
    print(f"❌ 错误: 在 {ROBOT_DIR} 里没找到任何 .xml 文件！")
    print("请确认文件名，并在脚本第 22 行手动指定。")
    sys.exit(1)

# 默认取第一个找到的 XML (通常就是你要改的那个)
SCENE_XML_PATH = xml_files[0]
print(f"✅ 锁定目标文件: {SCENE_XML_PATH}")

# ================= 轮子配置 =================
WHEEL_MESH_NAMES = {
    "LF (左前)": "LF_wheel_link_vis",
    "LR (左后)": "LR_wheel_link_vis",
    "RF (右前)": "RF_wheel_link_vis",
    "RR (右后)": "RR_wheel_link_vis"
}
WHEEL_KEYS = list(WHEEL_MESH_NAMES.keys())

class WheelTuner:
    def __init__(self):
        # 默认给一个 90度 修正
        self.rot_offsets = {k: np.array([0.0, 1.57, 0.0]) for k in WHEEL_KEYS} 
        self.pos_offsets = {k: np.array([0.0, 0.0, 0.0]) for k in WHEEL_KEYS}
        
        self.current_idx = 0
        self.current_axis = 1 
        self.mode = "ROTATION"      
        self.step_rot = 0.01        
        self.step_pos = 0.0005      
        
        self.model = None
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key == keyboard.Key.tab:
                self.mode = "TRANSLATION" if self.mode == "ROTATION" else "ROTATION"
                print(f"[模式]: {self.mode}")
            elif key == keyboard.Key.enter:
                self.save_with_regex()
            elif hasattr(key, 'char') and key.char in ['1', '2', '3', '4']:
                self.current_idx = int(key.char) - 1
                print(f"[选中]: {WHEEL_KEYS[self.current_idx]}")
            elif hasattr(key, 'char'):
                c = key.char.lower()
                if c == 'z': self.current_axis = 0; print("[轴]: X (红)")
                elif c == 'x': self.current_axis = 1; print("[轴]: Y (绿)")
                elif c == 'c': self.current_axis = 2; print("[轴]: Z (蓝)")
            if key == keyboard.Key.up: self.adjust_value(1)
            elif key == keyboard.Key.down: self.adjust_value(-1)
        except AttributeError: pass

    def adjust_value(self, direction):
        target = WHEEL_KEYS[self.current_idx]
        if self.mode == "ROTATION":
            self.rot_offsets[target][self.current_axis] += direction * self.step_rot
            print(f"旋转: {self.rot_offsets[target]}")
        else:
            self.pos_offsets[target][self.current_axis] += direction * self.step_pos
            print(f"平移: {self.pos_offsets[target]}")
        self.apply_offsets_visual()

    def apply_offsets_visual(self):
        if self.model is None: return
        for key in WHEEL_KEYS:
            mesh_name = WHEEL_MESH_NAMES[key]
            # 尝试推断 geom name
            geom_name = mesh_name.replace("link_vis", "geom_visual") 
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            
            # 如果按名字找不到，尝试按 mesh 名字找 (防止 geom 名字不一样)
            if geom_id == -1:
                 # 遍历所有 geom 找匹配的 mesh ID
                 target_mesh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
                 if target_mesh_id != -1:
                     for i in range(self.model.ngeom):
                         if self.model.geom_dataid[i] == target_mesh_id:
                             geom_id = i
                             break

            if geom_id != -1:
                euler = self.rot_offsets[key]
                r = R.from_euler('xyz', euler)
                quat = r.as_quat()
                self.model.geom_quat[geom_id] = np.array([quat[3], quat[0], quat[1], quat[2]])
                self.model.geom_pos[geom_id] = self.pos_offsets[key]

    def save_with_regex(self):
        print("\n正在修改文件...")
        print(f"目标: {SCENE_XML_PATH}")
        
        with open(SCENE_XML_PATH, 'r', encoding='utf-8') as f:
            content = f.read()

        modified_count = 0
        for key in WHEEL_KEYS:
            mesh_name = WHEEL_MESH_NAMES[key]
            rot = self.rot_offsets[key]
            pos = self.pos_offsets[key]
            
            rot_str = f'{rot[0]:.4f} {rot[1]:.4f} {rot[2]:.4f}'
            pos_str = f'{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}'

            # 正则替换：寻找 mesh="..." 的标签
            # 兼容 geom 有没有名字的情况
            pattern = re.compile(f'(<geom[^>]*mesh="{mesh_name}"[^>]*>)')
            
            match = pattern.search(content)
            if match:
                original_tag = match.group(1)
                new_tag = original_tag
                
                # 替换或添加 euler
                if 'euler="' in new_tag:
                    new_tag = re.sub(r'euler="[^"]*"', f'euler="{rot_str}"', new_tag)
                else:
                    new_tag = new_tag.replace('<geom', f'<geom euler="{rot_str}"')
                
                # 替换或添加 pos
                if 'pos="' in new_tag:
                    new_tag = re.sub(r'pos="[^"]*"', f'pos="{pos_str}"', new_tag)
                else:
                    new_tag = new_tag.replace('<geom', f'<geom pos="{pos_str}"')
                
                content = content.replace(original_tag, new_tag)
                print(f"✅ 已修改: {mesh_name}")
                modified_count += 1
            else:
                print(f"⚠️ 找不到 Mesh 为 {mesh_name} 的标签")

        if modified_count > 0:
            shutil.copy(SCENE_XML_PATH, SCENE_XML_PATH.with_suffix(".xml.bak"))
            with open(SCENE_XML_PATH, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"🎉 成功保存！请重启 main.py")
        else:
            print("❌ 没有任何修改。")

# ================= 主程序 =================
tuner = WheelTuner()

def load_callback(model=None, data=None):
    if model is None:
        try:
            model = mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))
            data = mujoco.MjData(model)
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            sys.exit(1)
    tuner.model = model
    tuner.apply_offsets_visual()
    return model, data

if __name__ == "__main__":
    print("\n🎮 路径修正版调参工具")
    print("-----------------------------------------")
    print(f"正在编辑目录: {ROBOT_DIR}")
    if 'SCENE_XML_PATH' in locals():
        print(f"正在编辑文件: {SCENE_XML_PATH.name}")
    print("-----------------------------------------")
    viewer.launch(loader=load_callback)