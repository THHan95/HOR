import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

# ==========================================================
# 🌟 可视化配置 & 辅助函数
# ==========================================================
MANO_SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 20],  # 大拇指
    [0, 4], [4, 5], [5, 6], [6, 7],   # 食指
    [0, 8], [8, 9], [9, 10], [10, 11], # 中指
    [0, 12], [12, 13], [13, 14], [14, 15], # 无名指
    [0, 16], [16, 17], [17, 18], [18, 19]  # 小指
]

# ⚠️ 如果你的数据配置里 MEAN 和 STD 不是这组，请修改！
MEAN = np.array([0.485, 0.456, 0.406])  
STD = np.array([0.229, 0.224, 0.225])   

def denormalize_image(image_tensor):
    """还原归一化的 Tensor 到 RGB 图像"""
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    img = (img * 255.0).clip(0, 255).astype(np.uint8) 
    return img

def project_3d_to_2d_correct(points_3d, intr):
    """
    🌟 真正的修复在这里：必须使用完整的矩阵乘法！
    因为 POEM 数据增强后的 intr 包含了 2D 的仿射旋转和平移，
    绝对不能只取 fx, fy, cx, cy。
    """
    # 矩阵乘法: (3x3) @ (3xN) -> (3xN) -> 转置回 (Nx3)
    pts_2d = np.matmul(intr, points_3d.T).T
    
    # 提取深度 Z，防止除以 0
    Z = pts_2d[:, 2:3].copy()
    Z[np.abs(Z) < 1e-6] = 1e-6
    
    # 透视除法，得到像素坐标 (u, v)
    pts_2d = (pts_2d / Z)[:, :2]
    return pts_2d

def visualize_poem_multiview_sample(batch, batch_index=0, save_name='poem_augmented_viz.png'):
    """
    画出指定 Batch 样本的 8 个视角裁剪图，并同步投影物体点云、手部网格和关键点。
    """
    print(f"🕵️‍♂️ 正在执行 Batch 索引 [{batch_index}] 的多视角可视化诊断...")
    
    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    axes = axes.flatten()
    
    # 假设 POEM 的 MultiView DataLoader 将 8 个视角压在了第二个维度: (B, 8, ...)
    for view_idx in range(8):
        ax = axes[view_idx]
        
        # 1. 提取当前视角的数据
        img_tensor = batch['image'][batch_index, view_idx]
        joints_3d = batch['target_joints_3d'][batch_index, view_idx].cpu().numpy()
        verts_3d = batch['target_verts_3d'][batch_index, view_idx].cpu().numpy()
        intr = batch['target_cam_intr'][batch_index, view_idx].cpu().numpy()
        
        # 尝试提取稀疏和密集点云 (为了兼容性，使用 .get() 和抛异常机制)
        try:
            obj_pc_sparse = batch['target_obj_pc_sparse'][batch_index, view_idx].cpu().numpy()
        except KeyError:
            obj_pc_sparse = None
            
        try:
            obj_pc_dense = batch['target_obj_pc_dense'][batch_index, view_idx].cpu().numpy()
        except KeyError:
            obj_pc_dense = None

        # 2. 还原图像
        img_rgb = denormalize_image(img_tensor)
        ax.imshow(img_rgb)
        
        # ==========================================================
        # 🌟 3. 执行修正后的正确投影
        # ==========================================================
        
        # a) 投影密集点云 (紫色，点更细，作为底层背景) - 如果有的话
        if obj_pc_dense is not None:
            dense_2d = project_3d_to_2d_correct(obj_pc_dense, intr)
            ax.scatter(dense_2d[:, 0], dense_2d[:, 1], s=1, c='purple', alpha=0.3, marker='.')

        # b) 投影稀疏点云 (蓝色点)
        if obj_pc_sparse is not None:
            sparse_2d = project_3d_to_2d_correct(obj_pc_sparse, intr)
            ax.scatter(sparse_2d[:, 0], sparse_2d[:, 1], s=4, c='b', alpha=0.7, marker='o')
        
        # c) 投影手部网格顶点 (绿色，带透明度)
        verts_2d = project_3d_to_2d_correct(verts_3d, intr)
        ax.scatter(verts_2d[:, 0], verts_2d[:, 1], s=1, c='g', alpha=0.2, marker='.')
        
        # d) 投影手部关键点和骨架 (红色)
        joints_2d = project_3d_to_2d_correct(joints_3d, intr)
        ax.scatter(joints_2d[:, 0], joints_2d[:, 1], s=30, c='r', marker='o', edgecolors='k')
        
        for connection in MANO_SKELETON:
            pt1 = joints_2d[connection[0]]
            pt2 = joints_2d[connection[1]]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], c='r', linewidth=2)
            
        # 设置标题和关闭坐标轴
        ax.set_title(f"View {view_idx}", fontsize=16)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.close(fig)
    print(f"✅ 增强结果可视化已保存至: {save_name}")