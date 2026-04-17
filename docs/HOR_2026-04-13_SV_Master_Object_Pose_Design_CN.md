# HOR 单视角/主视角物体位姿设计

日期：2026-04-13  
当前提交：`39c635c`

## 目标

本文档给出当前 HOR 物体位姿分支的一套更合理设计，目标是：

- 让 `SV` 物体分支输出真正有用的位姿，而不是只输出一个旋转残片
- 让 `SV` 和 `master` 的位姿定义在语义上统一
- 直接利用现有 DexYCB 真值标签监督
- 让 `master -> SV` 的自监督在旋转和平移两部分都成立
- 让预测结果能够直接用于可视化、重建和 stage2 精化

## 当前问题

当前实现只形成了半闭环：

- `SV` 预测每个视角的物体旋转 `obj_view_rot6d_cam`
- `SV` 没有显式预测每个视角的物体平移
- 当前物体平移/物体中心主要来自：
  - 2D 物体中心预测
  - 多视角三角化得到的 `ref_obj`
- `master` 预测的是：
  - `obj_rot6d`：主视角相机系下的物体旋转
  - `obj_trans`：相对主视角手根节点的物体平移

这会带来几个结构性问题：

1. `SV` 位姿不完整
- 旋转来自单视角特征
- 平移不是由单视角特征直接学习
- 所以 `SV` 不是一个真正完整的位姿分支

2. `SV` 和 `master` 的位姿表达不统一
- `SV` 目前只有相机系旋转
- `master` 是相机系旋转加相对手根平移
- 这会削弱闭环蒸馏

3. `master -> SV` 现在只做了旋转自监督
- 当前代码只把 `master` 旋转变到各视角，再监督 `SV` 旋转
- 平移没有蒸馏
- 整个位姿闭环不完整

4. stage2 过度依赖三角化得到的物体中心
- 多视角 2D 中心当然有用，但只要视角质量不一致就容易被污染
- 如果 `SV` 自己从来不学相对平移，那初始化就始终脆弱

## 设计原则

`SV` 和 `master` 应该采用同一种位姿语义定义。

推荐统一定义为：

- 旋转始终定义在当前相机坐标轴下
- 平移始终定义为相对当前手根节点的位移

也就是说：

- `SV pose` 对于第 `v` 个视角：
  - `R_v`：第 `v` 个视角相机系下的物体旋转
  - `d_v`：物体中心相对第 `v` 个视角手根节点的位移
- `master pose`：
  - `R_m`：主视角相机系下的物体旋转
  - `d_m`：物体中心相对主视角手根节点的位移

这样两个分支预测的是同一种物理量，只是所在视角不同。

## 为什么这样更合理

### 1. 现有标签就能直接监督

每个视角当前已经有：

- `target_rot6d_label`
- `target_t_label_rel`

主视角当前已经有：

- `master_obj_rot6d_label`
- `master_obj_t_label_rel`

因此这次改造不需要额外构造伪标签。

### 2. 单视角其实是可以学相对平移的

单张图里对物体相对手根位置是有信息的，例如：

- 手部姿态
- 接触几何关系
- 物体在手中的摆放方式
- 可见尺度
- 遮挡模式

虽然它不如多视角三角化稳定，但它是可学习、可监督、而且对最终任务有用的。

因此 `SV` 只学旋转，等于浪费了一部分可以直接利用的监督。

### 3. master 到各视角的蒸馏会变得很干净

因为 `SV` 和 `master` 都使用“相对手根”的平移定义，所以从 `master` 变到视角 `v` 时公式非常简单。

如果：

- `d_m = c_obj^m - c_hand^m`

那么在 `master -> v` 的旋转 `R_{v<-m}` 下：

- `d_v = R_{v<-m} d_m`

这里不需要额外平移项，因为：

- 物体中心和手根都会一起从主视角坐标系变到当前视角坐标系
- 两者做差后，相机平移项自然抵消

这正是这套设计最大的结构优势。

## 推荐的位姿定义

### SV 分支

对每个视角 `v`，预测：

- `sv_obj_rot6d_cam[v] = R_v`
- `sv_obj_trel_cam[v] = d_v`

其中：

- `R_v` 位于当前视角相机系
- `d_v = obj_center_v - hand_root_v`

可进一步构造：

- `sv_obj_center_cam[v] = sv_hand_root_cam[v] + d_v`
- `sv_obj_points_cam[v] = R_v * X_rest + sv_obj_center_cam[v]`

### master 分支

预测：

- `obj_rot6d = R_m`
- `obj_trans = d_m`

其中：

- `R_m` 位于主视角相机系
- `d_m = obj_center_m - hand_root_m`

进一步可构造：

- `obj_center_master = hand_root_master + d_m`
- `obj_points_master = R_m * X_rest + obj_center_master`

## 监督方式

### Stage1

`SV` 应该被完整监督：

- `loss_sv_rot`
  - `sv_obj_rot6d_cam` 对齐 `target_rot6d_label`
- `loss_sv_trans`
  - `sv_obj_trel_cam` 对齐 `target_t_label_rel`
- 可选 `loss_sv_points`
  - 由姿态构造出的 `sv_obj_points_cam` 对齐 `target_obj_pc_sparse` 或更稠密点云
- 可选 `loss_sv_center`
  - `sv_obj_center_cam` 对齐相机系下 GT 物体中心

这样 stage1 得到的是完整单视角物体位姿分支，而不是一个“只给 master 初始化旋转”的辅助头。

### Stage2

`master` 保持直接 GT 监督：

- `loss_master_rot`
  - `obj_rot6d` 对齐 `master_obj_rot6d_label`
- `loss_master_trans`
  - `obj_trans` 对齐 `master_obj_t_label_rel`
- `loss_master_points`
  - 构造出的主视角物体点云对齐 `master_obj_sparse`

同时 stage2 中 `SV` 仍然应该继续吃 GT 监督：

- 不要被自监督完全替换
- 自监督应该是后期开启的附加项

## master -> SV 自监督

自监督应该同时覆盖旋转和平移，而不是只覆盖旋转。

### 旋转蒸馏

给定主视角最终预测：

- `R_m`

和主视角到当前视角的旋转：

- `R_{v<-m}`

则：

- `R_v^distill = R_{v<-m} R_m`

这一部分和当前实现比较接近。

### 平移蒸馏

给定主视角相对平移：

- `d_m`

则蒸馏到当前视角的教师平移为：

- `d_v^distill = R_{v<-m} d_m`

这之所以成立，是因为 `d` 本身就是从手根到物体中心的相对向量。

### 自监督 loss

teacher 必须 `detach`：

- `R_m.detach()`
- `d_m.detach()`

可加：

- `loss_sv_rot_self`
  - `sv_obj_rot6d_cam` 对齐 `R_v^distill`
- `loss_sv_trans_self`
  - `sv_obj_trel_cam` 对齐 `d_v^distill`

注意：

- `SV` 的 GT 监督必须保留
- 自监督只能在后期按 epoch 或置信度门控开启

## 训练节奏建议

### Stage1

训练：

- 单视角手部分支
- 物体 `SV` 旋转分支
- 物体 `SV` 平移分支
- 必要时保留手的 keypoint/master 精化

此阶段不要开启自监督。

### Stage2 前期

训练：

- `SV` 继续吃 GT
- `master` 继续吃 GT
- `master` 初始化来自 `SV`

但先不要用 `master -> SV` 蒸馏。

原因：

- stage2 刚开始时，`master` 不一定比 `SV` 好
- 太早蒸馏会反而把 `SV` 拉坏

### Stage2 后期

再开启带门控的自监督：

- `master -> SV rotation`
- `master -> SV translation`

最简单的门控方式仍然是配置里的 epoch gate。

## 2D 物体中心分支的作用变化

如果 `SV` 升级成预测完整位姿 `(R_v, d_v)`，那么 2D 物体中心分支就不应该再承担主路径作用。

它最多只应作为：

- 早期视觉定位先验
- 可见性/置信度线索
- 辅助几何正则
- 可视化和 debug 的补充信息

它不应该继续承担：

- `ref_obj` 的主要初始化来源
- `master` 初始化主来源
- `SV` 平移的替代品

## 多物体歧义：为什么 2D 物体中心应从主路径彻底移除

当前 2D 物体中心分支存在结构性歧义。

与手不同：

- 手的拓扑固定
- 手关节语义稳定
- 网络通常不会在若干不相关类别之间混淆“目标手”

但物体中心分支面对的是：

- 物体类别多样
- 图像中可能同时出现多个物体
- 当前 2D center head 未必是“目标物体条件化”的
- 图像中最显眼的物体，并不一定是当前样本要预测的那个物体

因此会出现一种危险模式：

- 网络高置信度地预测了“别的物体”的中心
- 三角化出的 `ref_obj` 会直接偏掉
- stage2 初始化会被错误物体牵着走

这在验证集和测试集上尤其危险，因为这不是普通噪声，而是“目标关联错误”。

一旦错误物体的中心进入主路径：

- 它不是轻微偏差
- 而是语义层面的错误初始化
- 后续所有位姿预测都会被系统性带偏

因此，如果下一步准备让 `SV` 直接预测完整 pose `(R_v, d_v)`，那么推荐决策就是：

- 将 2D 物体中心分支从主训练路径和主推理路径中彻底移除

也就是说：

- 不再用它做 `ref_obj` 主初始化
- 不再用它做 `master` 位姿初始化
- 不再用它替代 `SV` 平移
- 不再让它主导最终物体位姿的置信度或加权

除非以后给它补上明确的目标条件，例如：

- 目标物体 identity conditioning
- 模板条件
- 物体专属 ROI
- 显式目标选择机制

否则它在多物体场景下始终存在天然歧义。

## 推荐的最终判断

更强的推荐是：

- `SV` 预测完整 pose `(R_v, d_v)`
- `master` 预测完整 pose `(R_m, d_m)`
- 2D 物体中心分支从主物体位姿流水线中移除

在这个设计下：

- `SV -> master` 初始化来自 `SV pose`
- `master -> SV` 自监督同时覆盖旋转和平移
- 主路径不再依赖一个有多物体歧义的 2D center 检测器

这套方案在几何上更统一，在语义上也更安全。

## 建议新增输出

### SV 分支

推荐新增：

- `obj_view_rot6d_cam` `(B, N, 6)`
- `obj_view_trel_cam` `(B, N, 3)`
- `obj_view_center_cam` `(B, N, 1, 3)`
- `obj_view_xyz_cam` `(B, N, P, 3)`

可选：

- `obj_view_pose_conf`
- `obj_view_trans_valid`

### master 分支

继续保留：

- `obj_rot6d` `(L, B, 6)`
- `obj_trans` `(L, B, 3)`
- `obj_xyz_master` `(L, B, P, 3)`
- `obj_center_xyz_master` `(L, B, 1, 3)`

## 初始化路径建议

当前初始化路径应该从：

- `SV rotation -> master init rotation`

改成：

- `SV pose -> master init pose`

保守第一版建议：

1. 每个视角先预测 `(R_v, d_v)`
2. 转到主视角：
   - `R_m^{(v)} = R_{m<-v} R_v`
   - `d_m^{(v)} = R_{m<-v} d_v`
3. 第一版可直接使用主视角的 `SV pose` 作为 `master init pose`

这样最稳，也最接近你当前已有代码结构。

## 最小落地改造步骤

### Step 1

在以下文件中增加 `SV` 平移头：

- [`lib/models/HOR.py`](/media/hl/data/code/han/POEM/lib/models/HOR.py)
- [`lib/models/HOR_heatmap.py`](/media/hl/data/code/han/POEM/lib/models/HOR_heatmap.py)

第一版输入可先与当前 `SV rotation` 分支保持一致：

- `global_feat`
- 以后再考虑是否融合 `hand_obj_fused_flat`

### Step 2

输出每个视角：

- `obj_view_trel_cam`

并构造：

- `obj_view_center_cam`
- `obj_view_xyz_cam`

### Step 3

增加 GT loss：

- `obj_view_trel_cam` 对齐 `target_t_label_rel`

### Step 4

用主视角 `SV pose` 作为 stage2 初始化：

- `obj_init_rot6d`
- `obj_init_trans`

### Step 5

在后期开启：

- `master rot -> sv rot`
- `master trans -> sv trans`

## 为什么这是当前代码库里最合理的方案

与其他选择相比：

### 方案 A：SV 预测相机系绝对平移

不推荐。

原因：

- 绝对相机平移更依赖视角本身
- 它与当前 `master obj_trans` 语义不统一
- `master -> SV` 的变换也不再干净

### 方案 B：SV 只预测旋转

这就是当前设计，不理想。

原因：

- 没把现有监督用满
- `SV` 位姿不完整
- stage2 太依赖三角化物体中心
- 自监督只覆盖了一半位姿

### 方案 C：SV 和 master 都使用“相机系旋转 + 相对手根平移”

推荐。

原因：

- 现有标签可直接监督
- 两个分支位姿语义统一
- `master -> SV` 旋转和平移蒸馏都成立
- 预测结果可直接用于可视化、重建和验证

## 最终建议

下一版物体位姿设计应当明确改成：

- `SV` 预测 `(R_v, d_v)`
- `master` 预测 `(R_m, d_m)`
- 2D 物体中心分支从主物体位姿路径中移除

其中：

- `R`：当前相机系下的旋转
- `d`：当前手根到物体中心的相对位移

这是当前最干净、最稳定、也最充分利用现有 GT 标签的方案。

## 相关代码位置

当前相关文件：

- [`lib/models/HOR.py`](/media/hl/data/code/han/POEM/lib/models/HOR.py)
- [`lib/models/HOR_heatmap.py`](/media/hl/data/code/han/POEM/lib/models/HOR_heatmap.py)
- [`lib/models/heads/ptHOR_head.py`](/media/hl/data/code/han/POEM/lib/models/heads/ptHOR_head.py)
- [`lib/datasets/dexycb.py`](/media/hl/data/code/han/POEM/lib/datasets/dexycb.py)
- [`lib/datasets/hdata.py`](/media/hl/data/code/han/POEM/lib/datasets/hdata.py)
- [`lib/utils/transform.py`](/media/hl/data/code/han/POEM/lib/utils/transform.py)

## 当前实现状态

本轮代码已经完成的改动：

- `HOR.py`
  - `SV` 物体分支已经改为预测每个视角的完整位姿：
    - `obj_view_rot6d_cam`
    - `obj_view_trans`
  - `SV` 物体特征改为使用 `global_feat + hand_obj_fused_flat`
  - `master` 初始化已改为直接使用主视角的 `SV pose`：
    - `obj_init_rot6d`
    - `obj_init_trans`
  - stage1/stage2 对 `SV` 的监督已经同时覆盖旋转和平移
  - `SV` 重建指标现在基于 `SV` 相对位姿重建物体点云，不再依赖 `ref_obj`

- `HOR_heatmap.py`
  - 已与 `HOR.py` 做同语义同步
  - heatmap 的物体中心分支不再作为主物体位姿路径的初始化来源

- `ptHOR_head.py`
  - 已支持接收 `obj_init_trans`
  - 物体初始化中心改为 `hand_center + obj_init_trans`
  - stage2 物体分支不再把中心三角化结果作为主平移来源

当前保留的简化：

- 代码里 `master -> SV` 自监督目前仍只保留旋转部分
- 平移自监督建议等这版 `SV full pose` 监督稳定后再接入
