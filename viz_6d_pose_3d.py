#!/usr/bin/env python3
"""Visualize 3D Object 6D Pose and Hand Interaction"""

import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# 添加路径 (请根据你的项目结构调整)
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "lib" / "models" / "common" / "networks"))
sys.path.insert(0, str(Path(__file__).parent / "lib" / "models" / "common"))

from lib.datasets import create_dataset
from lib.utils.config import get_config

def set_axes_equal(ax, pts):
    """设置 3D 坐标轴等比例，防止点云被拉伸变形"""
    if len(pts) == 0:
        return
    x_limits = [np.min(pts[:, 0]), np.max(pts[:, 0])]
    y_limits = [np.min(pts[:, 1]), np.max(pts[:, 1])]
    z_limits = [np.min(pts[:, 2]), np.max(pts[:, 2])]
    
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    
    max_range = max([x_range, y_range, z_range]) / 2.0
    
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def viz_obj_pose_3d():
    """Visualize 3D Hand and Object Pose Verification"""
    cfg_file = "config/release/HOR_DexYCBMV.yaml"
    cfg = get_config(config_file=cfg_file, arg=None, merge=True)

    # 1. 创建数据集
    train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)

    # 2. 获取第一个样本 (此时已经经过了数据增强、裁切、旋转、甚至翻转)
    sample = train_data[0]

    print("=" * 80)
    print("Visualizing 6D Pose Transformation on Augmented Data")
    print("=" * 80)

    # =======================================================
    # 3. 提取核心数据 (兼容多视角与单视角)
    # =======================================================
    # 我们固定取多视角序列的第 0 个视角进行 3D 几何验证
    view_idx = 0  

    # --- 提取手部 3D 关节点 ---
    hand_joints_mv = sample.get('target_joints_3d', sample.get('joints_3d'))  
    hand_joints_cam = hand_joints_mv[view_idx] if hand_joints_mv.ndim == 3 else hand_joints_mv
    
    # 提取手腕 Root 节点 (你设定为 9 号节点)
    hand_root = hand_joints_cam[9].reshape(1, 3) 

    # --- 提取物体模板 (Rest Space) ---
    pc_rest_mv = sample.get('target_obj_pc_sparse_rest', sample.get('obj_pc_sparse_rest'))
    # 模板是物理绝对形状，没有多视角概念，强制剥离 View 维度变为 (2048, 3)
    pc_rest = pc_rest_mv[0] if pc_rest_mv.ndim == 3 else pc_rest_mv

    # --- 提取 6D 位姿标签 (R 和 t) ---
    R_label_mv = sample.get('target_R_label', sample.get('R_label'))
    R_label = R_label_mv[view_idx] if R_label_mv.ndim == 3 else R_label_mv  # (3, 3)

    t_label_rel_mv = sample.get('target_t_label_rel', sample.get('t_label_rel'))
    t_label_rel = t_label_rel_mv[view_idx].reshape(1, 3) if t_label_rel_mv.ndim == 2 else t_label_rel_mv.reshape(1, 3)
    
    # --- 提取绝对坐标系下真实的物体点云 (仅供比对与计算 MSE) ---
    pc_cam_gt_mv = sample.get('target_obj_pc_sparse', sample.get('obj_pc_sparse'))
    if pc_cam_gt_mv is not None:
        pc_cam_gt = pc_cam_gt_mv[view_idx] if pc_cam_gt_mv.ndim == 3 else pc_cam_gt_mv
    else:
        pc_cam_gt = None

    print(f"Hand joints shape: {hand_joints_cam.shape}")
    print(f"Template points shape: {pc_rest.shape}")
    print(f"R_label shape: {R_label.shape}")
    print(f"t_label_rel shape: {t_label_rel.shape}")

    # =======================================================
    # 4. 执行数学变换验证
    # =======================================================
    # (A) 模拟网络前向：通过标签的 R_label 和 t_label_rel，将标准模板变换到手的局部空间
    # 矩阵乘法: (2048, 3) @ (3, 3).T + (1, 3) -> (2048, 3)
    pc_local = (R_label @ pc_rest.T).T + t_label_rel
    
    # (B) 模拟网络后处理：将局部空间的点云，加上手腕绝对坐标，恢复回相机绝对空间
    pc_cam_recovered = pc_local + hand_root

    # =======================================================
    # 5. 绘制 3D 灵魂三图
    # =======================================================
    output_dir = Path("tmp/obj_pose_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(24, 8))

    # [子图 1]：标准空间 (Rest Space) - 纯净的 CAD 模板
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(pc_rest[:, 0], pc_rest[:, 1], pc_rest[:, 2], c='cornflowerblue', s=2, alpha=0.6)
    ax1.scatter(0, 0, 0, c='black', marker='X', s=150, label='Origin (0,0,0)')
    ax1.set_title("1. Standard Rest Space\n(CAD Template)", fontsize=14, pad=20)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend()
    set_axes_equal(ax1, pc_rest)

    # [子图 2]：手部局部空间 (Hand-Centric Local Space) - 送入网络的真实样子
    ax2 = fig.add_subplot(132, projection='3d')
    # 把手也挪到局部空间（减去手腕 9 号点）
    hand_local = hand_joints_cam - hand_root 
    ax2.scatter(pc_local[:, 0], pc_local[:, 1], pc_local[:, 2], c='darkorange', s=2, alpha=0.6, label='Obj (Local)')
    ax2.plot(hand_local[:, 0], hand_local[:, 1], hand_local[:, 2], c='crimson', marker='o', markersize=5, linestyle='None', label='Hand Joints')
    ax2.scatter(0, 0, 0, c='black', marker='X', s=150, label='Hand Root (Origin)')
    ax2.set_title("2. Hand-Centric Local Space\n(Network Input / Transform Target)", fontsize=14, pad=20)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.legend()
    set_axes_equal(ax2, np.vstack((pc_local, hand_local)))

    # [子图 3]：验证恢复精度 (Absolute Camera Space)
    ax3 = fig.add_subplot(133, projection='3d')
    if pc_cam_gt is not None:
        ax3.scatter(pc_cam_gt[:, 0], pc_cam_gt[:, 1], pc_cam_gt[:, 2], c='mediumseagreen', s=8, alpha=0.8, label='GT Absolute PC')
    
    ax3.scatter(pc_cam_recovered[:, 0], pc_cam_recovered[:, 1], pc_cam_recovered[:, 2], c='magenta', s=2, alpha=0.5, label='Recovered PC')
    ax3.plot(hand_joints_cam[:, 0], hand_joints_cam[:, 1], hand_joints_cam[:, 2], c='crimson', marker='o', markersize=5, linestyle='None', label='Hand Joints')
    
    # =======================================================
    # 🔍 终极诊断：揪出平移偏差
    # =======================================================
    print("\n--- Diagnostic Info ---")
    gt_center = pc_cam_gt.mean(axis=0)
    rec_center = pc_cam_recovered.mean(axis=0)
    print(f"1. GT Object Center: {gt_center}")
    print(f"2. Recovered Center: {rec_center}")
    print(f"3. The Difference:   {gt_center - rec_center}")
    print(f"4. Hand Root Used:   {hand_root.flatten()}")
    print(f"5. t_label_rel Used: {t_label_rel.flatten()}")
    print("-----------------------\n")
    
    # 核心审判：计算恢复的点云和真实点云的 MSE
    mse_error = np.mean((pc_cam_gt - pc_cam_recovered)**2) if pc_cam_gt is not None else 0.0
    title_color = 'green' if mse_error < 1e-10 else 'red'
    
    ax3.set_title(f"3. Absolute Space Recovery\n[MSE: {mse_error:.2e}]", fontsize=14, pad=20, color=title_color, fontweight='bold')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.legend()
    set_axes_equal(ax3, np.vstack((pc_cam_gt if pc_cam_gt is not None else pc_cam_recovered, hand_joints_cam)))

    plt.tight_layout()
    output_path = output_dir / "6d_pose_verification.png"
    plt.savefig(str(output_path), dpi=200, bbox_inches='tight')
    print(f"\n✅ 成功！3D 验证图已保存至: {output_path}")
    
    if mse_error < 1e-10:
        print("🎉 完美闭环！MSE 接近于 0，说明你的 DataLoader 和 Transform 处理非常正确。")
    else:
        print("🚨 警告：MSE 误差过大！请务必检查 DexYCB 类或 Transform 中，R 和 t 是否有写反、未转置或少做了操作。")

if __name__ == "__main__":
    # 为了避免 DataLoader 多进程 worker 可能导致的 RuntimeError
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    viz_obj_pose_3d()