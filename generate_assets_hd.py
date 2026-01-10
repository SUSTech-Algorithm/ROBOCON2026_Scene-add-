import open3d as o3d
import os
import copy
from pathlib import Path

# ================= 配置区域 =================
# 你的模型文件夹路径 (根据报错图片中的路径设置)
# 如果脚本放在 meshes 同级目录，可以直接写 Path("meshes")
TARGET_DIR = Path("/home/yxm/ROBOCON2026_Scene/models/robots/costume_R2/meshes")

# 输出文件夹名称
VISUAL_FOLDER = "visual"      # 高画质模型文件夹
COLLISION_FOLDER = "collision" # 碰撞模型文件夹

# 面数限制 (MuJoCo/Gazebo 的限制通常是 200k 左右)
MAX_VISUAL_FACES = 150000     # 视觉模型上限 (留有余量，避免踩 200k 红线)
MAX_COLLISION_FACES = 2000    # 碰撞模型上限 (越低仿真越流畅)

# 是否将视觉模型转为 OBJ 格式？
# True: 生成 .obj 文件 (推荐，材质兼容性好)
# False: 生成 .stl 文件 (如果你不想改 URDF 的后缀，选这个)
EXPORT_VISUAL_AS_OBJ = False 
# ===========================================

def process_meshes():
    # 检查路径是否存在
    if not TARGET_DIR.exists():
        print(f"❌ 错误: 找不到路径 {TARGET_DIR}")
        print("请修改脚本中的 TARGET_DIR 为你实际的 meshes 文件夹路径。")
        return

    # 创建输出目录
    vis_dir = TARGET_DIR / VISUAL_FOLDER
    col_dir = TARGET_DIR / COLLISION_FOLDER
    vis_dir.mkdir(parents=True, exist_ok=True)
    col_dir.mkdir(parents=True, exist_ok=True)

    print(f"[-] 开始处理目录: {TARGET_DIR}")
    print(f"[-] 视觉模型面数上限: {MAX_VISUAL_FACES}")
    print(f"[-] 碰撞模型面数上限: {MAX_COLLISION_FACES}")
    print("-" * 50)

    # 遍历所有 STL 文件
    stl_files = list(TARGET_DIR.glob("*.STL")) + list(TARGET_DIR.glob("*.stl"))
    
    if not stl_files:
        print("❌ 未找到任何 STL 文件！请检查路径。")
        return

    for file_path in stl_files:
        # 跳过以 ._ 开头的隐藏文件（Mac/Linux 常见垃圾文件）
        if file_path.name.startswith("._"): 
            continue
        
        # 跳过我们刚刚生成的文件夹
        if VISUAL_FOLDER in str(file_path) or COLLISION_FOLDER in str(file_path):
            continue

        print(f"正在处理: {file_path.name} ...")

        try:
            # 1. 读取原始模型
            mesh = o3d.io.read_triangle_mesh(str(file_path))
            
            if not mesh.has_triangles():
                print(f"  ⚠️ 跳过: 文件无法读取或为空")
                continue

            original_faces = len(mesh.triangles)
            print(f"  -> 原始面数: {original_faces}")

            # ==========================================
            # 2. 生成视觉模型 (Visual Mesh)
            # ==========================================
            # 使用深拷贝，防止后续操作影响
            vis_mesh = copy.deepcopy(mesh)

            # 如果面数超标，进行降采样
            if original_faces > MAX_VISUAL_FACES:
                print(f"  -> [视觉] 面数过高，优化中 ({original_faces} -> {MAX_VISUAL_FACES})...")
                vis_mesh = vis_mesh.simplify_quadric_decimation(target_number_of_triangles=MAX_VISUAL_FACES)
            else:
                print(f"  -> [视觉] 面数合规，仅转换格式 (ASCII->Binary)")

            # 重新计算法线以保证光照正确
            vis_mesh.compute_vertex_normals()

            # 保存视觉模型
            if EXPORT_VISUAL_AS_OBJ:
                vis_save_path = vis_dir / (file_path.stem + ".obj")
                o3d.io.write_triangle_mesh(str(vis_save_path), vis_mesh)
            else:
                vis_save_path = vis_dir / file_path.name
                # write_triangle_mesh 默认保存为 Binary STL，直接解决 ASCII 报错
                o3d.io.write_triangle_mesh(str(vis_save_path), vis_mesh)
            
            print(f"  ✅ 视觉模型已保存: {vis_save_path.name}")

            # ==========================================
            # 3. 生成碰撞模型 (Collision Mesh)
            # ==========================================
            col_mesh = copy.deepcopy(mesh)
            
            # 碰撞模型必须简化，否则仿真器会卡死
            if original_faces > MAX_COLLISION_FACES:
                col_mesh = col_mesh.simplify_quadric_decimation(target_number_of_triangles=MAX_COLLISION_FACES)
            
            col_save_path = col_dir / file_path.name
            o3d.io.write_triangle_mesh(str(col_save_path), col_mesh)
            print(f"  ✅ 碰撞模型已保存: {col_save_path.name} (面数: {len(col_mesh.triangles)})")

        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
        
        print("-" * 30)

    print("\n🎉 所有处理完成！")
    print(f"请记得修改你的 URDF 文件，将路径指向 '{VISUAL_FOLDER}' 和 '{COLLISION_FOLDER}' 文件夹。")

if __name__ == "__main__":
    process_meshes()