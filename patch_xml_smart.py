import xml.etree.ElementTree as ET
from pathlib import Path
import copy

# 目标文件
XML_PATH = Path("models/robots/costume_R2/costume_R2.xml")

def run_smart_patch():
    if not XML_PATH.exists():
        print(f"❌ 找不到文件: {XML_PATH}")
        return

    print(f"[-] 读取文件: {XML_PATH}")
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    # 1. 修正编译器设置 (移除 meshdir，使用显式路径)
    compiler = root.find("compiler")
    if compiler is not None:
        if "meshdir" in compiler.attrib:
            print("  - 移除 compiler meshdir (改用显式路径)")
            del compiler.attrib["meshdir"]
    else:
        # 如果没有 compiler，创建一个
        compiler = ET.SubElement(root, "compiler", angle="radian", autolimits="true")

    # 2. 升级 Asset (网格资源)
    assets = root.find("asset")
    if assets is None:
        print("❌ 错误: XML 中没有 <asset> 标签")
        return

    # 收集旧网格名字，以便后续替换 Geom
    mesh_map = {} # old_name -> filename_stem

    # 找到所有的 mesh 标签
    original_meshes = assets.findall("mesh")
    for mesh in original_meshes:
        name = mesh.get("name")
        file = mesh.get("file")
        if not name or not file: continue
        
        # 提取文件名主体 (例如 "base_link.STL" -> "base_link")
        stem = Path(file).stem
        mesh_map[name] = stem
        
        # 移除旧标签
        assets.remove(mesh)

    # 添加新标签 (Visual 和 Collision)
    print(f"  - 升级了 {len(mesh_map)} 个网格资源")
    for name, stem in mesh_map.items():
        # Visual (OBJ)
        ET.SubElement(assets, "mesh", 
                      name=f"{name}_vis", 
                      file=f"meshes/visual/{stem}.obj")
        # Collision (STL)
        ET.SubElement(assets, "mesh", 
                      name=f"{name}_col", 
                      file=f"meshes/collision/{stem}.STL")

    # 3. 升级 Worldbody (几何体引用)
    # 递归查找所有 body 下的 geom
    fixed_geoms = 0
    for body in root.iter("body"):
        # 找到该 body 下所有的 geom
        geoms = body.findall("geom")
        for geom in geoms:
            mesh_name = geom.get("mesh")
            
            # 只处理引用了我们已知网格的 geom
            if mesh_name in mesh_map:
                # 1. 修改当前 geom 为 Visual
                geom.set("mesh", f"{mesh_name}_vis")
                geom.set("group", "1") # 视觉组
                geom.set("contype", "0") # 不碰撞
                geom.set("conaffinity", "0")
                geom.set("class", "visual") # 尝试继承 visual 类
                
                # 2. 创建一个新的 Collision geom
                col_geom = copy.deepcopy(geom)
                col_geom.set("mesh", f"{mesh_name}_col")
                col_geom.set("group", "3") # 碰撞组(隐藏)
                col_geom.set("contype", "1")
                col_geom.set("conaffinity", "1")
                col_geom.set("class", "collision")
                # 移除材质属性 (碰撞体不需要材质)
                if "material" in col_geom.attrib:
                    del col_geom.attrib["material"]
                
                # 将碰撞体加入 body (插在视觉体后面)
                # ElementTree 插入比较麻烦，这里直接 append
                body.append(col_geom)
                fixed_geoms += 1

    print(f"  - 替换了 {fixed_geoms} 个几何体为 (Visual + Collision) 对")

    # 4. 确保 base_link 抬高 (防止卡地)
    # 找到第一个名为 base_link 的 body
    for body in root.findall(".//body"):
        if body.get("name") == "base_link":
            current_pos = body.get("pos", "0 0 0")
            print(f"  - 检查 Base Link 位置: {current_pos}")
            # 如果是 0 0 0，强制改为 0 0 0.2
            if current_pos.strip() == "0 0 0":
                body.set("pos", "0 0 0.2")
                print("    -> 已自动抬高至 0 0 0.2 (防止穿模)")
            break

    # 保存
    tree.write(XML_PATH, encoding="utf-8", xml_declaration=True)
    print(f"✅ 修复完成！已保存至: {XML_PATH}")

if __name__ == "__main__":
    run_smart_patch()