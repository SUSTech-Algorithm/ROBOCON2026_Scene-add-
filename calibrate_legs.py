import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import sys
import select
import tty
import termios
import time
import math
import re

# ================= 配置 =================
XML_PATH = Path("models/robots/costume_R2/costume_R2.xml")
STEP_POS = 0.005   # 平移步长 (5mm)
STEP_ROT = 0.05    # 旋转步长 (约2.8度)

if not XML_PATH.exists():
    print(f"❌ 找不到文件: {XML_PATH}")
    sys.exit(1)

# 加载模型
try:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# 获取腿部 ID
leg_names = ["LF_rail_link", "LR_rail_link", "RF_rail_link", "RR_rail_link"]
leg_ids = []
for name in leg_names:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if idx == -1:
        print(f"⚠️ 警告: 找不到 Body '{name}'")
    else:
        leg_ids.append(idx)

# 初始状态记录
current_pos = {i: model.body_pos[i].copy() for i in leg_ids}
current_quat = {i: model.body_quat[i].copy() for i in leg_ids}

# --- 数学工具 ---
def quat_mult(q, r):
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def rotate_axis(q_orig, angle, axis):
    axis = np.array(axis) / np.linalg.norm(axis)
    half = angle / 2.0
    sin_half = np.sin(half)
    q_rot = np.array([np.cos(half), axis[0]*sin_half, axis[1]*sin_half, axis[2]*sin_half])
    return quat_mult(q_orig, q_rot)

def clear_screen():
    print("\033[H\033[J", end="")

def print_status():
    clear_screen()
    print("===================== 🦾 机器人参数校准 (终极版) =====================")
    print(f"{'Body Name':<15} | {'POS (X Y Z)':<28} | {'QUAT (W X Y Z)':<30}")
    print("-" * 78)
    
    for i, name in zip(leg_ids, leg_names):
        p = current_pos[i]
        q = current_quat[i]
        p_str = f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
        q_str = f"{q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f}"
        print(f"{name:<15} | {p_str:<28} | {q_str:<30}")
        
    print("=====================================================================")
    print(" 🎮 操作指南:")
    print(" [J] / [L] : ↔️  左右伸缩 (调节轮距宽窄) <--- 保留功能")
    print(" [A] / [D] : ⬅️  整体 X 轴平移 (整体左移/右移)")
    print(" [W] / [S] : ⬆️  整体 Y 轴平移 (整体前移/后移)")
    print(" [Q] / [E] : ⏫  整体 Z 轴平移 (整体高度升降)")
    print(" --------------------------------------------")
    print(" [R] / [F] : 🔄  Roll 旋转 (绕X轴 - 扶正倒下的腿)")
    print(" [T] / [G] : 🔄  Pitch 旋转 (绕Y轴)")
    print(" [Y] / [H] : 🔄  Yaw 旋转 (绕Z轴)")
    print(" --------------------------------------------")
    print(" [Enter]   : 💾  保存到 XML")
    print(" [Esc]     : ❌  退出")

def save_changes():
    print("\n💾 正在保存 XML...")
    try:
        with open(XML_PATH, "r", encoding="utf-8") as f:
            content = f.readlines()
        
        new_content = []
        for line in content:
            for i, name in zip(leg_ids, leg_names):
                if f'name="{name}"' in line:
                    x, y, z = current_pos[i]
                    w, qx, qy, qz = current_quat[i]
                    # 替换 POS
                    if 'pos="' in line:
                        line = re.sub(r'pos="[^"]+"', f'pos="{x:.5f} {y:.5f} {z:.5f}"', line)
                    # 替换 QUAT
                    if 'quat="' in line:
                        line = re.sub(r'quat="[^"]+"', f'quat="{w:.5f} {qx:.5f} {qy:.5f} {qz:.5f}"', line)
                    break 
            new_content.append(line)
                
        with open(XML_PATH, "w", encoding="utf-8") as f:
            f.writelines(new_content)
        print(f"✅ 保存成功!")
        time.sleep(1)
    except Exception as e:
        print(f"❌ 保存失败: {e}")

def is_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

# ================= 主程序 =================
with mujoco.viewer.launch_passive(model, data) as viewer:
    old_settings = termios.tcgetattr(sys.stdin)
    print_status() 
    
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        while viewer.is_running():
            viewer.sync()
            
            if is_data():
                c = sys.stdin.read(1).lower()
                updated = False
                
                # --- 伸缩功能 (Original J/L) ---
                if c == 'l': # 变宽 (Expand)
                    for i in leg_ids:
                        if current_pos[i][0] > 0: current_pos[i][0] += STEP_POS
                        else: current_pos[i][0] -= STEP_POS
                    updated = True
                elif c == 'j': # 变窄 (Contract)
                    for i in leg_ids:
                        if current_pos[i][0] > 0: current_pos[i][0] -= STEP_POS
                        else: current_pos[i][0] += STEP_POS
                    updated = True

                # --- 整体平移 (Translation) ---
                elif c == 'a':    # X - (整体左移)
                    for i in leg_ids: current_pos[i][0] -= STEP_POS
                    updated = True
                elif c == 'd':  # X + (整体右移)
                    for i in leg_ids: current_pos[i][0] += STEP_POS
                    updated = True
                elif c == 's':  # Y -
                    for i in leg_ids: current_pos[i][1] -= STEP_POS
                    updated = True
                elif c == 'w':  # Y +
                    for i in leg_ids: current_pos[i][1] += STEP_POS
                    updated = True
                elif c == 'q':  # Z -
                    for i in leg_ids: current_pos[i][2] -= STEP_POS
                    updated = True
                elif c == 'e':  # Z +
                    for i in leg_ids: current_pos[i][2] += STEP_POS
                    updated = True

                # --- 旋转 (Rotation) ---
                elif c == 'r': # Roll +
                    for i in leg_ids: current_quat[i] = rotate_axis(current_quat[i], STEP_ROT, [1, 0, 0])
                    updated = True
                elif c == 'f': # Roll -
                    for i in leg_ids: current_quat[i] = rotate_axis(current_quat[i], -STEP_ROT, [1, 0, 0])
                    updated = True
                elif c == 't': # Pitch +
                    for i in leg_ids: current_quat[i] = rotate_axis(current_quat[i], STEP_ROT, [0, 1, 0])
                    updated = True
                elif c == 'g': # Pitch -
                    for i in leg_ids: current_quat[i] = rotate_axis(current_quat[i], -STEP_ROT, [0, 1, 0])
                    updated = True
                elif c == 'y': # Yaw +
                    for i in leg_ids: current_quat[i] = rotate_axis(current_quat[i], STEP_ROT, [0, 0, 1])
                    updated = True
                elif c == 'h': # Yaw -
                    for i in leg_ids: current_quat[i] = rotate_axis(current_quat[i], -STEP_ROT, [0, 0, 1])
                    updated = True
                
                # --- 系统 ---
                elif c == '\n' or c == '\r':
                    save_changes()
                    print_status()
                elif c == '\x1b': # ESC
                    break

                if updated:
                    for i in leg_ids:
                        model.body_pos[i] = current_pos[i]
                        model.body_quat[i] = current_quat[i]
                    mujoco.mj_forward(model, data)
                    print_status()

            time.sleep(0.01)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\n程序已退出。")