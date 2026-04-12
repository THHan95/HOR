import numpy as np
import torch


def _triangulation_work_dtype(*tensors):
    for tensor in tensors:
        if torch.is_tensor(tensor) and tensor.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
    for tensor in tensors:
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            return tensor.dtype
    return torch.float32


def _solve_dlt_inhomogeneous(A, reg=1e-4):
    """Solve DLT with x4 fixed to 1 via regularized least squares.

    The classic homogeneous SVD solution is numerically fragile in backward when the
    smallest singular values become close, which is common early in training. For hand
    reconstruction we only need finite 3D points in front of the camera, so the
    inhomogeneous least-squares form is a better fit and yields a stable gradient.
    """
    lhs = A[..., :3]
    rhs = -A[..., 3:]
    solve_dtype = lhs.dtype
    lhs_t = lhs.transpose(-1, -2)
    normal = torch.matmul(lhs_t, lhs).to(dtype=solve_dtype)
    rhs_normal = torch.matmul(lhs_t, rhs).to(dtype=solve_dtype)

    eye = torch.eye(3, device=A.device, dtype=solve_dtype).unsqueeze(0).expand(normal.shape[0], -1, -1)
    scale = normal.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    reg_term = eye * (reg * scale.clamp_min(1.0))
    solution = torch.linalg.solve(normal + reg_term, rhs_normal)
    return solution.squeeze(-1)


def batch_triangulate_dlt_torch_sigma(kp2ds, Ks, Extrs, sigmas=None):
    """torch: Triangulate multiple 2D points from multiple sets of multiviews using the DLT algorithm.
    NOTE: Expend to Batch and nJoints dimension. Supports Weighted Triangulation based on sigmas.

    see Hartley & Zisserman section 12.2 (p.312) for info on SVD,
    see Hartley & Zisserman (2003) p. 593 (see also p. 587).

    Args:
        kp2ds (torch.Tensor): Shape: (B, N, J, 2).
        Ks (torch.Tensor): Shape: (B, N, 3, 3).
        Extrs (torch.Tensor): Shape: (B, N, 4, 4).
        sigmas (torch.Tensor, optional): Shape (B, N, J, 2). Uncertainties from RLE.
                                         Higher sigma means lower confidence. Defaults to None.

    Returns:
        torch.Tensor: Shape: (B, J, 3).
    """
    work_dtype = _triangulation_work_dtype(kp2ds, Ks, Extrs, sigmas)
    kp2ds = kp2ds.to(dtype=work_dtype)
    Ks = Ks.to(dtype=work_dtype)
    Extrs = Extrs.to(dtype=work_dtype)
    if sigmas is not None:
        sigmas = sigmas.to(dtype=work_dtype)

    nJoints = kp2ds.shape[-2]
    batch_size = kp2ds.shape[0]
    nCams = kp2ds.shape[1]

    # 1. 构造投影矩阵 P = K [R|t]
    Pmat = Extrs[..., :3, :]  # (B, N, 3, 4)
    Mmat = torch.matmul(Ks, Pmat)  # (B, N, 3, 4)

    # 扩展以匹配关节维度 (B, J, N, 3, 4)
    Mmat = Mmat.unsqueeze(1).repeat(1, nJoints, 1, 1, 1)
    Mmat = Mmat.reshape(batch_size * nJoints, nCams, *Pmat.shape[-2:])  # (BxJ, N, 3, 4)
    M_row2 = Mmat[..., 2:3, :]  # (BxJ, N, 1, 4)

    # 2. 构造 DLT 方程矩阵 A
    # kp2ds: (B, N, J, 2) -> (B, J, N, 2) -> (BxJ, N, 2) -> (BxJ, N, 2, 1)
    kp2ds_reshaped = kp2ds.permute(0, 2, 1, 3).reshape(batch_size * nJoints, nCams, 2).unsqueeze(3)

    # A = [u * P_3 - P_1, v * P_3 - P_2]
    A = kp2ds_reshaped * M_row2  # (BxJ, N, 2, 4)
    A = A - Mmat[..., :2, :]  # (BxJ, N, 2, 4)

    # =========================================================
    # 🌟 核心：引入 RLE 不确定性进行方程加权 (Weighted DLT)
    # =========================================================
    if sigmas is not None:
        # 应用 sigma 权重 (置信度是不确定性 sigma 的倒数)
        weights = 1.0 / (sigmas + 1e-5)  # (B, N, J, 2)

        # 归一化权重：使每个关节在 N 个视角上的权重和等于 N
        # 这样可以保持矩阵 A 的数值量级，防止 SVD 出现数值不稳定
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-5) * nCams

        # 维度对齐：(B, N, J, 2) -> (B, J, N, 2) -> (BxJ, N, 2, 1)
        weights = weights.permute(0, 2, 1, 3).reshape(batch_size * nJoints, nCams, 2).unsqueeze(3)

        # 将权重广播应用到 A 的每一行
        A = A * weights
    # =========================================================

    A = A.reshape(batch_size * nJoints, -1, 4)  # (BxJ, 2xN, 4)

    X = _solve_dlt_inhomogeneous(A)
    X = X.reshape(batch_size, nJoints, 3)  # (B, J, 3)

    return X


def batch_triangulate_dlt_torch_confidence(kp2ds, Ks, Extrs, confidences=None):
    """torch: Triangulate multiple 2D points from multiple sets of multiviews using the DLT algorithm.
    NOTE: Expend to Batch and nJoints dimension. Supports Weighted Triangulation based on confidences.

    see Hartley & Zisserman section 12.2 (p.312) for info on SVD,
    see Hartley & Zisserman (2003) p. 593 (see also p. 587).

    Args:
        kp2ds (torch.Tensor): Shape: (B, N, J, 2).
        Ks (torch.Tensor): Shape: (B, N, 3, 3).
        Extrs (torch.Tensor): Shape: (B, N, 4, 4).
        confidences (torch.Tensor, optional): Shape (B, N, J) or (B, N, J, 2). 
                                              Higher value means higher confidence. Defaults to None.

    Returns:
        torch.Tensor: Shape: (B, J, 3).
    """
    work_dtype = _triangulation_work_dtype(kp2ds, Ks, Extrs, confidences)
    kp2ds = kp2ds.to(dtype=work_dtype)
    Ks = Ks.to(dtype=work_dtype)
    Extrs = Extrs.to(dtype=work_dtype)
    if confidences is not None:
        confidences = confidences.to(dtype=work_dtype)

    nJoints = kp2ds.shape[-2]
    batch_size = kp2ds.shape[0]
    nCams = kp2ds.shape[1]

    # 1. 构造投影矩阵 P = K [R|t]
    Pmat = Extrs[..., :3, :]  # (B, N, 3, 4)
    Mmat = torch.matmul(Ks, Pmat)  # (B, N, 3, 4)

    # 扩展以匹配关节维度 (B, J, N, 3, 4)
    Mmat = Mmat.unsqueeze(1).repeat(1, nJoints, 1, 1, 1)
    Mmat = Mmat.reshape(batch_size * nJoints, nCams, *Pmat.shape[-2:])  # (BxJ, N, 3, 4)
    M_row2 = Mmat[..., 2:3, :]  # (BxJ, N, 1, 4)

    # 2. 构造 DLT 方程矩阵 A
    # kp2ds: (B, N, J, 2) -> (B, J, N, 2) -> (BxJ, N, 2) -> (BxJ, N, 2, 1)
    kp2ds_reshaped = kp2ds.permute(0, 2, 1, 3).reshape(batch_size * nJoints, nCams, 2).unsqueeze(3)

    # A = [u * P_3 - P_1, v * P_3 - P_2]
    A = kp2ds_reshaped * M_row2  # (BxJ, N, 2, 4)
    A = A - Mmat[..., :2, :]  # (BxJ, N, 2, 4)

    # =========================================================
    # 🌟 核心：引入置信度进行方程加权 (Weighted DLT)
    # =========================================================
    if confidences is not None:
        # 兼容处理：如果输入没有最后一维 2 (即 x 和 y 共享一个置信度)，则扩维以触发广播机制
        if confidences.dim() == 3:
            weights = confidences.unsqueeze(-1)  # (B, N, J, 1)
        else:
            weights = confidences.clone()        # (B, N, J, 2)

        # 归一化权重：使每个关节在 N 个视角上的权重和等于 N
        # 这一步非常关键！它保持了矩阵 A 的整体能量尺度，防止因权重极小导致 SVD 分解崩溃
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-5) * nCams

        # 维度对齐：(B, N, J, x) -> (B, J, N, x) -> (BxJ, N, x, 1)
        dim_last = weights.shape[-1]
        weights = weights.permute(0, 2, 1, 3).reshape(batch_size * nJoints, nCams, dim_last).unsqueeze(3)

        # 将权重广播应用到 A 的方程行中
        A = A * weights
    # =========================================================

    A = A.reshape(batch_size * nJoints, -1, 4)  # (BxJ, 2xN, 4)

    X = _solve_dlt_inhomogeneous(A)
    X = X.reshape(batch_size, nJoints, 3)  # (B, J, 3)

    return X


def batch_triangulate_dlt_torch(kp2ds, Ks, Extrs):
    """torch: Triangulate multiple 2D points from multiple sets of multiviews using the DLT algorithm.
    NOTE: Expend to Batch and nJoints dimension.

    see Hartley & Zisserman section 12.2 (p.312) for info on SVD, 
    see Hartley & Zisserman (2003) p. 593 (see also p. 587).

    Args:
        kp2ds (torch.Tensor): Shape: (B, N, J, 2).
        Ks (torch.Tensor): Shape: (B, N, 3, 3).
        Extrs (torch.Tensor): Shape: (B, N, 4, 4).

    Returns:
        torch.Tensor: Shape: (B, J, 3).
    """
    # assert kp2ds.shape[0] == Ks.shape[0] == Extrs.shape[0], "batch shape mismatch"
    # assert kp2ds.shape[1] == Ks.shape[1] == Extrs.shape[1], "nCams shape mismatch"
    # assert kp2ds.shape[-1] == 2, "keypoints must be 2D"
    # assert Ks.shape[-2:] == (3, 3), "K must be 3x3"
    # assert Extrs.shape[-2:] == (4, 4), "Extr must be 4x4"

    work_dtype = _triangulation_work_dtype(kp2ds, Ks, Extrs)
    kp2ds = kp2ds.to(dtype=work_dtype)
    Ks = Ks.to(dtype=work_dtype)
    Extrs = Extrs.to(dtype=work_dtype)

    nJoints = kp2ds.shape[-2]
    batch_size = kp2ds.shape[0]
    nCams = kp2ds.shape[1]

    Pmat = Extrs[..., :3, :]  # (B, N, 3, 4)
    Mmat = torch.matmul(Ks, Pmat)  # (B, N, 3, 4)
    Mmat = Mmat.unsqueeze(1).repeat(1, nJoints, 1, 1, 1)  # (B, J, N, 3, 4)
    Mmat = Mmat.reshape(batch_size * nJoints, nCams, *Pmat.shape[-2:])  # (BxJ, N, 3, 4)
    M_row2 = Mmat[..., 2:3, :]  # (BxJ, N, 1, 4)

    # kp2ds: (B, N, J, 2) -> (B, J, N, 2) -> (BxJ, N, 2) -> (BxJ, N, 2, 1)
    kp2ds = kp2ds.permute(0, 2, 1, 3).reshape(batch_size * nJoints, nCams, 2).unsqueeze(3)  # (BxJ, N, 2, 1)
    A = kp2ds * M_row2  # (BxJ, N, 2, 4)
    A = A - Mmat[..., :2, :]  # (BxJ, N, 2, 4)
    A = A.reshape(batch_size * nJoints, -1, 4)  # (BxJ, 2xN, 4)

    X = _solve_dlt_inhomogeneous(A)
    X = X.reshape(batch_size, nJoints, 3)  # (B, J, 3)
    return X


def triangulate_dlt_torch(kp2ds, Ks, Extrs):
    """torch: Triangulate multiple 2D points from one set of multiviews using the DLT algorithm.
    NOTE: Expend to nJoints dimension.

    see Hartley & Zisserman section 12.2 (p.312) for info on SVD, 
    see Hartley & Zisserman (2003) p. 593 (see also p. 587).

    Args:
        kp2ds (torch.Tensor): Shape: (N, J, 2).
        Ks (torch.Tensor): Shape: (N, 3, 3).
        Extrs (torch.Tensor): Shape: (N, 4, 4).

    Returns:
        torch.Tensor: Shape: (J, 3).
    """
    work_dtype = _triangulation_work_dtype(kp2ds, Ks, Extrs)
    kp2ds = kp2ds.to(dtype=work_dtype)
    Ks = Ks.to(dtype=work_dtype)
    Extrs = Extrs.to(dtype=work_dtype)

    nJoints = kp2ds.shape[-2]

    Pmat = Extrs[:, :3, :]  # (N, 3, 4)
    Mmat = torch.matmul(Ks, Pmat)  # (N, 3, 4)
    Mmat = Mmat.unsqueeze(0).repeat(nJoints, 1, 1, 1)  # (J, N, 3, 4)
    M_row2 = Mmat[..., 2:3, :]  # (J, N, 1, 4)

    kp2ds = kp2ds.permute(1, 0, 2).unsqueeze(3)  # (J, N, 2, 1)
    A = kp2ds * M_row2  # (J, N, 2, 4)
    A = A - Mmat[..., :2, :]  # (J, N, 2, 4)
    A = A.reshape(nJoints, -1, 4)  # (J, 2xN, 4)

    X = _solve_dlt_inhomogeneous(A)
    return X


def triangulate_one_point_dlt(points_2d_set, Ks, Extrs):
    """Triangulate one point from one set of multiviews using the DLT algorithm.
    Implements a linear triangulation method to find a 3D
    point. For example, see Hartley & Zisserman section 12.2 (p.312).
    for info on SVD, see Hartley & Zisserman (2003) p. 593 (see also p. 587)

    Args:
        points_2d_set (set): first element is the camera index, second element is the 2d point, shape: (2,)
        Ks (np.ndarray): Camera intrinsics. Shape: (N, 3, 3). 
        Extrs (np.ndarray): Camera extrinsics. Shape: (N, 4, 4).

    Returns:
        np.ndarray: Triangulated 3D point. Shape: (3,).
    """
    A = []
    for n, pt2d in points_2d_set:
        K = Ks[int(n)]  #  (3, 3)
        Extr = Extrs[int(n)]  # (4, 4)
        P = Extr[:3, :]  # (3, 4)
        M = K @ P  # (3, 4)
        row_2 = M[2, :]
        x, y = pt2d[0], pt2d[1]
        A.append(x * row_2 - M[0, :])
        A.append(y * row_2 - M[1, :])
    # Calculate best point
    A = np.array(A)
    u, d, vt = np.linalg.svd(A)
    X = vt[-1, 0:3] / vt[-1, 3]  # normalize
    return X


def triangulate_dlt(pts, confis, Ks, Extrs, confi_thres=0.5):
    """Triangulate multiple 2D points from one set of multiviews using the DLT algorithm.
    Args:
        pts (np.ndarray): 2D points in the image plane. Shape: (N, J, 2).
        confis (np.ndarray): Confidence scores of the points. Shape: (N, J,).
        Ks (np.ndarray): Camera intrinsics. Shape: (N, 3, 3).
        Extrs (np.ndarray): Camera extrinsics. Shape: (N, 4, 4).
        confi_thres (float): Threshold of confidence score.
    Returns:
        np.ndarray: Triangulated 3D points. Shape: (N, J, 3).
    """

    assert pts.ndim == 3 and pts.shape[-1] == 2
    assert confis.ndim == 2 and confis.shape[0] == pts.shape[0]
    assert Ks.ndim == 3 and Ks.shape[1:] == (3, 3)
    assert Extrs.ndim == 3 and Extrs.shape[1:] == (4, 4)
    assert Ks.shape[0] == Extrs.shape[0] == pts.shape[0]

    dtype = pts.dtype
    nJoints = pts.shape[1]
    p3D = np.zeros((nJoints, 3), dtype=dtype)

    for j, conf in enumerate(confis.T):
        while True:
            sel_cam_idx = np.where(conf > confi_thres)[0]
            if confi_thres <= 0:
                break
            if len(sel_cam_idx) <= 1:
                confi_thres -= 0.05
                # print('confi threshold too high, decrease to', confi_thres)
            else:
                break
        points_2d_set = []
        for n in sel_cam_idx:
            points_2d = pts[n, j, :]
            points_2d_set.append((str(n), points_2d))
        p3D[j, :] = triangulate_one_point_dlt(points_2d_set, Ks, Extrs)
    return p3D


def test_batch_triangulate_dlt_torch():
    from scripts.viz_multiview_dataset import DEXYCB_3D_CONFIG
    from lib.utils.config import CN
    from lib.datasets.dexycb import DexYCBMultiView

    cfg = CN(DEXYCB_3D_CONFIG)
    cfg.MASTER_SYSTEM = "as_constant_camera"
    dataset = DexYCBMultiView(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, sample in enumerate(dataloader):
        joints_2d = sample['target_joints_2d']  # (B, N, J, 2)
        Ks = sample['target_cam_intr']  # (B, N, 3, 3)
        Extrs = torch.linalg.inv(sample['target_cam_extr'])  # (B, N, 4, 4)
        P3d = batch_triangulate_dlt_torch(joints_2d, Ks, Extrs)
        master_joints_3d = sample["master_joints_3d"]  # (B, J, 3)

        diff = P3d - master_joints_3d
        diff_norm = torch.norm(diff, dim=-1)  # meter
        print(diff_norm)
        # assert False


def test_triangulate_dlt():
    from scripts.viz_multiview_dataset import DEXYCB_3D_CONFIG
    from lib.utils.config import CN
    from lib.datasets.dexycb import DexYCBMultiView

    cfg = CN(DEXYCB_3D_CONFIG)
    cfg.MASTER_SYSTEM = "as_constant_camera"
    dataset = DexYCBMultiView(cfg)

    for i in range(len(dataset)):
        sample = dataset[i]
        joints_2d = sample['target_joints_2d']  # (N, J, 2)
        print(joints_2d.shape)
        Ks = sample['target_cam_intr']  # (N, 3, 3)
        Extrs = np.linalg.inv(sample['target_cam_extr'])  # (N, 4, 4)
        confis = np.ones((joints_2d.shape[0], joints_2d.shape[1]))  # (N, J)

        P3d = triangulate_dlt(joints_2d, confis, Ks, Extrs)
        master_joints_3d = sample["master_joints_3d"]
        print(P3d - master_joints_3d)
        # assert False


if __name__ == "__main__":
    test_batch_triangulate_dlt_torch()
