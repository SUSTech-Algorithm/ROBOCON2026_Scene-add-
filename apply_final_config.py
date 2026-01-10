import os
from pathlib import Path

# 目标文件
xml_path = Path("models/robots/costume_R2/costume_R2.xml")

# V8 最终配置内容 (无 meshdir, 显式路径, 视觉物理分离)
correct_content = r"""<mujoco model="costume_R2">
  <compiler angle="radian" autolimits="true"/>

  <default>
    <default class="costume_R2">
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="1 1 1 1"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" condim="3" contype="1" conaffinity="1" friction="0.6 0.005 0.0001"/>
      </default>
      <joint damping="0.1" frictionloss="0.01" armature="0.01"/>
    </default>
  </default>

  <asset>
    <mesh name="base_link_vis" file="meshes/visual/base_link.obj" />
    <mesh name="LF_rail_link_vis" file="meshes/visual/LF_rail_link.obj" />
    <mesh name="LF_yaw_link_vis" file="meshes/visual/LF_yaw_link.obj" />
    <mesh name="LF_wheel_link_vis" file="meshes/visual/LF_wheel_link.obj" />
    <mesh name="LR_rail_link_vis" file="meshes/visual/LR_rail_link.obj" />
    <mesh name="LR_yaw_link_vis" file="meshes/visual/LR_yaw_link.obj" />
    <mesh name="LR_wheel_link_vis" file="meshes/visual/LR_wheel_link.obj" />
    <mesh name="RF_rail_link_vis" file="meshes/visual/RF_rail_link.obj" />
    <mesh name="RF_yaw_link_vis" file="meshes/visual/RF_yaw_link.obj" />
    <mesh name="RF_wheel_link_vis" file="meshes/visual/RF_wheel_link.obj" />
    <mesh name="RR_rail_link_vis" file="meshes/visual/RR_rail_link.obj" />
    <mesh name="RR_yaw_link_vis" file="meshes/visual/RR_yaw_link.obj" />
    <mesh name="RR_wheel_link_vis" file="meshes/visual/RR_wheel_link.obj" />

    <mesh name="base_link_col" file="meshes/collision/base_link.STL" />
    <mesh name="LF_rail_link_col" file="meshes/collision/LF_rail_link.STL" />
    <mesh name="LF_yaw_link_col" file="meshes/collision/LF_yaw_link.STL" />
    <mesh name="LF_wheel_link_col" file="meshes/collision/LF_wheel_link.STL" />
    <mesh name="LR_rail_link_col" file="meshes/collision/LR_rail_link.STL" />
    <mesh name="LR_yaw_link_col" file="meshes/collision/LR_yaw_link.STL" />
    <mesh name="LR_wheel_link_col" file="meshes/collision/LR_wheel_link.STL" />
    <mesh name="RF_rail_link_col" file="meshes/collision/RF_rail_link.STL" />
    <mesh name="RF_yaw_link_col" file="meshes/collision/RF_yaw_link.STL" />
    <mesh name="RF_wheel_link_col" file="meshes/collision/RF_wheel_link.STL" />
    <mesh name="RR_rail_link_col" file="meshes/collision/RR_rail_link.STL" />
    <mesh name="RR_yaw_link_col" file="meshes/collision/RR_yaw_link.STL" />
    <mesh name="RR_wheel_link_col" file="meshes/collision/RR_wheel_link.STL" />

    <material name="base_material" rgba="0.898 0.918 0.929 1" />
    <material name="rail_material" rgba="0.647 0.620 0.588 1" />
    <material name="yaw_material" rgba="0.898 0.918 0.929 1" />
    <material name="wheel_material" rgba="0.298 0.298 0.298 1" />
  </asset>

  <worldbody>
    <body name="base_link" childclass="costume_R2" pos="0 0 0.2">
      <inertial pos="0.001206 -0.353411 0.016987" mass="15.0" diaginertia="0.1 0.1 0.2" />
      <freejoint/>
      
      <geom class="visual" mesh="base_link_vis" material="base_material" />
      <geom class="collision" mesh="base_link_col" />

      <site name="imu_site" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>

      <body name="LF_rail_link" pos="0.30193 -0.654 0.33384" quat="0 -0.7071 0 -0.7071">
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01" />
        <joint name="LF_rail_joint" type="slide" axis="0 0 -1" limited="true" range="0 0.25" armature="0.01" damping="10" />
        <geom class="visual" mesh="LF_rail_link_vis" material="rail_material" />
        <geom class="collision" mesh="LF_rail_link_col" />

        <body name="LF_yaw_link" pos="0 0 -0.3778">
          <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001" />
          <joint name="LF_yaw_joint" type="hinge" axis="0 0 1" limited="true" range="-3.14 3.14" armature="0.001" damping="0.1" />
          <geom class="visual" mesh="LF_yaw_link_vis" material="yaw_material" />
          <geom class="collision" mesh="LF_yaw_link_col" />

          <body name="LF_wheel_link" pos="-0.0098995 -0.0098995 -0.0465" quat="0.7071 0 0 -0.7071">
            <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001" />
            <joint name="LF_wheel_joint" type="hinge" axis="0 0 -1" armature="0.001" damping="0.05" />
            <geom class="visual" mesh="LF_wheel_link_vis" material="wheel_material" />
            <geom class="collision" mesh="LF_wheel_link_col" />
          </body>
        </body>
      </body>

      <body name="LR_rail_link" pos="0.30193 -0.054 0.33384" quat="0 -0.7071 0 -0.7071">
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01" />
        <joint name="LR_rail_joint" type="slide" axis="0 0 -1" limited="true" range="0 0.25" armature="0.01" damping="10" />
        <geom class="visual" mesh="LR_rail_link_vis" material="rail_material" />
        <geom class="collision" mesh="LR_rail_link_col" />

        <body name="LR_yaw_link" pos="0 0 -0.3778">
          <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001" />
          <joint name="LR_yaw_joint" type="hinge" axis="0 0 1" limited="true" range="-3.14 3.14" armature="0.001" damping="0.1" />
          <geom class="visual" mesh="LR_yaw_link_vis" material="yaw_material" />
          <geom class="collision" mesh="LR_yaw_link_col" />

          <body name="LR_wheel_link" pos="0.0098995 -0.0098995 -0.0465" quat="0.7071 0 0 0.7071">
            <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001" />
            <joint name="LR_wheel_joint" type="hinge" axis="0 0 -1" armature="0.001" damping="0.05" />
            <geom class="visual" mesh="LR_wheel_link_vis" material="wheel_material" />
            <geom class="collision" mesh="LR_wheel_link_col" />
          </body>
        </body>
      </body>

      <body name="RF_rail_link" pos="-0.29807 -0.654 0.33384" quat="0 -0.7071 0 -0.7071">
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01" />
        <joint name="RF_rail_joint" type="slide" axis="0 0 -1" limited="true" range="0 0.25" armature="0.01" damping="10" />
        <geom class="visual" mesh="RF_rail_link_vis" material="rail_material" />
        <geom class="collision" mesh="RF_rail_link_col" />

        <body name="RF_yaw_link" pos="0 0 -0.3778">
          <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001" />
          <joint name="RF_yaw_joint" type="hinge" axis="0 0 1" limited="true" range="-3.14 3.14" armature="0.001" damping="0.1" />
          <geom class="visual" mesh="RF_yaw_link_vis" material="yaw_material" />
          <geom class="collision" mesh="RF_yaw_link_col" />

          <body name="RF_wheel_link" pos="-0.0098995 0.0098995 -0.0465" quat="0.7071 0 0 -0.7071">
            <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001" />
            <joint name="RF_wheel_joint" type="hinge" axis="0 0 -1" armature="0.001" damping="0.05" />
            <geom class="visual" mesh="RF_wheel_link_vis" material="wheel_material" />
            <geom class="collision" mesh="RF_wheel_link_col" />
          </body>
        </body>
      </body>

      <body name="RR_rail_link" pos="-0.29807 -0.054 0.33384" quat="0 -0.7071 0 -0.7071">
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01" />
        <joint name="RR_rail_joint" type="slide" axis="0 0 -1" limited="true" range="0 0.25" armature="0.01" damping="10" />
        <geom class="visual" mesh="RR_rail_link_vis" material="rail_material" />
        <geom class="collision" mesh="RR_rail_link_col" />

        <body name="RR_yaw_link" pos="0 0 -0.3778">
          <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001" />
          <joint name="RR_yaw_joint" type="hinge" axis="0 0 1" limited="true" range="-3.14 3.14" armature="0.001" damping="0.1" />
          <geom class="visual" mesh="RR_yaw_link_vis" material="yaw_material" />
          <geom class="collision" mesh="RR_yaw_link_col" />

          <body name="RR_wheel_link" pos="0.0098995 0.0098995 -0.0465" quat="0.7071 0 0 0.7071">
            <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001" />
            <joint name="RR_wheel_joint" type="hinge" axis="0 0 -1" armature="0.001" damping="0.05" />
            <geom class="visual" mesh="RR_wheel_link_vis" material="wheel_material" />
            <geom class="collision" mesh="RR_wheel_link_col" />
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="LF_rail_motor" joint="LF_rail_joint" gear="1" ctrlrange="-40 40" ctrllimited="true" />
    <motor name="LR_rail_motor" joint="LR_rail_joint" gear="1" ctrlrange="-40 40" ctrllimited="true" />
    <motor name="RF_rail_motor" joint="RF_rail_joint" gear="1" ctrlrange="-40 40" ctrllimited="true" />
    <motor name="RR_rail_motor" joint="RR_rail_joint" gear="1" ctrlrange="-40 40" ctrllimited="true" />
    <motor name="LF_yaw_motor" joint="LF_yaw_joint" gear="1" ctrlrange="-1 1" ctrllimited="true" />
    <motor name="LR_yaw_motor" joint="LR_yaw_joint" gear="1" ctrlrange="-1 1" ctrllimited="true" />
    <motor name="RF_yaw_motor" joint="RF_yaw_joint" gear="1" ctrlrange="-1 1" ctrllimited="true" />
    <motor name="RR_yaw_motor" joint="RR_yaw_joint" gear="1" ctrlrange="-1 1" ctrllimited="true" />
    <motor name="LF_wheel_motor" joint="LF_wheel_joint" gear="1" ctrlrange="-10 10" ctrllimited="true" />
    <motor name="LR_wheel_motor" joint="LR_wheel_joint" gear="1" ctrlrange="-10 10" ctrllimited="true" />
    <motor name="RF_wheel_motor" joint="RF_wheel_joint" gear="1" ctrlrange="-10 10" ctrllimited="true" />
    <motor name="RR_wheel_motor" joint="RR_wheel_joint" gear="1" ctrlrange="-10 10" ctrllimited="true" />
  </actuator>

  <contact>
    <exclude body1="base_link" body2="LF_rail_link"/>
    <exclude body1="base_link" body2="LR_rail_link"/>
    <exclude body1="base_link" body2="RF_rail_link"/>
    <exclude body1="base_link" body2="RR_rail_link"/>
    <exclude body1="LF_rail_link" body2="LF_yaw_link" />
    <exclude body1="LF_yaw_link" body2="LF_wheel_link" />
    <exclude body1="LR_rail_link" body2="LR_yaw_link" />
    <exclude body1="LR_yaw_link" body2="LR_wheel_link" />
    <exclude body1="RF_rail_link" body2="RF_yaw_link" />
    <exclude body1="RF_yaw_link" body2="RF_wheel_link" />
    <exclude body1="RR_rail_link" body2="RR_yaw_link" />
    <exclude body1="RR_yaw_link" body2="RR_wheel_link" />
  </contact>

  <sensor>
    <jointpos name="LF_rail_pos" joint="LF_rail_joint" />
    <jointpos name="LR_rail_pos" joint="LR_rail_joint" />
    <jointpos name="RF_rail_pos" joint="RF_rail_joint" />
    <jointpos name="RR_rail_pos" joint="RR_rail_joint" />
    <jointpos name="LF_yaw_pos" joint="LF_yaw_joint" />
    <jointpos name="LR_yaw_pos" joint="LR_yaw_joint" />
    <jointpos name="RF_yaw_pos" joint="RF_yaw_joint" />
    <jointpos name="RR_yaw_pos" joint="RR_yaw_joint" />
    <jointpos name="LF_wheel_pos" joint="LF_wheel_joint" />
    <jointpos name="LR_wheel_pos" joint="LR_wheel_joint" />
    <jointpos name="RF_wheel_pos" joint="RF_wheel_joint" />
    <jointpos name="RR_wheel_pos" joint="RR_wheel_joint" />
    <accelerometer name="base_accel" site="imu_site" />
    <gyro name="base_gyro" site="imu_site" />
  </sensor>
</mujoco>
"""

# 强制写入
if xml_path.parent.exists():
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(correct_content)
    print(f"✅ 文件已强制写入 (Size: {len(correct_content)} bytes)")
    
    # 验证写入结果
    with open(xml_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        second_line = f.readline()
        print("\n[验证文件头]")
        print(f"Line 1: {first_line.strip()}")
        print(f"Line 2: {second_line.strip()}")
        if 'meshdir' not in second_line:
            print("✨ 验证成功！旧的 meshdir 属性已移除。")
        else:
            print("❌ 警告：写入似乎失败，文件仍包含旧属性！")
else:
    print(f"❌ 目录不存在: {xml_path.parent}")