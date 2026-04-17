# HOR 伪标签弱监督训练方案

日期：2026-04-13

## 1. 方案定位

当前可行的方向不是严格意义上的“纯自监督”，而是：

- 利用离线教师模型生成伪标签
- 结合多视角几何一致性进行 student 训练
- 用手部弱监督和物体伪监督共同驱动手物联合学习

更准确地说，这是一套：

**伪标签驱动的多视角弱监督手物联合训练方案**

如果用英文表述，建议采用类似说法：

- `Pseudo-Label Guided Multi-View Hand-Object Pose Learning`
- `Weakly Supervised Multi-View HO Pose Estimation with Offline Teachers`
- `Teacher-Student Multi-View HO Learning with Pseudo Labels`

不建议直接写成：

- `self-supervised hand-object pose estimation`

因为只要物体 6D 位姿来自外部教师模型，它在定义上就不是纯自监督。

## 2. 核心设想

希望摆脱人工 3D 标注依赖，但保留以下可获得信息：

- 多视角 RGB 图像
- 多视角相机内参、外参
- 物体模板点云或 canonical 点云
- 手部伪标签：每个视角的 2D 关键点
- 物体伪标签：每个视角的 6D 位姿

其中伪标签来源可以是：

- 手部教师模型：如 `WiLoR`
- 物体教师模型：如 `HORT`

然后训练当前 HOR student 网络，使其通过：

- 图像特征
- 多视角几何一致性
- 模板点云刚性变换
- 手部运动学先验

逐步学会比离线教师更稳定、更适合多视角融合的结果。

## 3. 为什么这个方向合理

## 3.1 手部分支本身就适合弱监督

手部分支可以主要依靠：

- 2D 重投影监督
- 多视角三角化一致性
- MANO 先验
- 多视角 mesh/joint 一致性

如果你之前的工作已经证明“仅用手部自监督/弱监督可行”，那么当前方法天然可以继承这部分经验。

## 3.2 物体天然更适合伪标签驱动

对于已知模板物体，外部模型给出每视角 6D pose 后，可以直接构造：

- 各视角物体旋转标签
- 各视角物体平移标签
- 由模板点云变换得到的目标点云
- 统一到 master/root 空间后的 master 标签

这比直接预测“物体中心点”更稳定，也更适合当前 HOR 的刚性模板设定。

## 3.3 多视角 student 有机会超过单视角 teacher

即便伪标签来自单视角或弱多视角教师，student 仍有机会更强，因为 student 使用了：

- 多视角同步输入
- 手物联合表示
- master 空间融合
- 模板点云约束
- 跨视角一致性约束

因此 student 不只是“模仿 teacher”，而是在 teacher 伪标签基础上做几何纠偏和多视角整合。

## 4. 当前 HOR 框架中可直接沿用的部分

当前 HOR 已具备以下结构，适合承接伪标签训练：

- `SV hand branch`
- `SV object pose branch`
- `multi-view master fusion`
- `object template rigid transform`
- `master init pose + final residual pose`
- `master -> sv consistency`

也就是说，从框架角度看，不需要重写整套方法，只需要把监督来源从“人工 GT”为主，切换到“伪标签 + 几何一致性”为主。

## 5. 最小可训练输入集合

如果按伪标签弱监督路线推进，建议的最小输入字段如下。

### 5.1 必须有

- `image`
- `target_cam_intr`
- `target_cam_extr`
- `master_id`
- `master_obj_sparse_rest`

### 5.2 手部分支需要

- `target_joints_uvd[..., :2]` 或等价的 2D 手关键点伪标签
- `target_joints_vis` 或等价可见性置信度

### 5.3 物体分支需要

- `target_rot6d_label`
- `target_t_label_rel`

这两者可以由外部物体教师模型输出。

### 5.4 可由伪标签和相机参数再生成

- `master_obj_rot6d_label`
- `master_obj_t_label_rel`
- `target_obj_pc_sparse`
- `master_obj_sparse`

其中：

- `target_obj_pc_sparse` 可由 `master_obj_sparse_rest + target_rot6d_label + target_t_label_rel` 生成
- `master_obj_rot6d_label / master_obj_t_label_rel` 可由 per-view 伪标签换算到统一 master/root 空间后得到

## 6. 推荐的监督划分

## 6.1 手部分支

建议保留：

- 2D hand reprojection loss
- multi-view triangulation consistency
- MANO pose prior
- MANO shape prior
- hand cross-view consistency

建议弱化或移除：

- 人工 3D hand GT 直接监督
- 人工 3D mesh GT 直接监督

如果没有人工 3D，当前代码中的一些损失需要改为：

- 由伪 2D + 相机参数三角化出的弱标签
- 或者完全替换为 consistency 型损失

## 6.2 SV 物体分支

建议保留：

- `obj_view_rot6d_cam` 对 `target_rot6d_label`
- `obj_view_trans` 对 `target_t_label_rel`
- 模板点云变换后的 `SV object points` 对 `target_obj_pc_sparse`

也就是说，SV 分支依然使用教师提供的 per-view 伪标签进行直接监督。

## 6.3 Master 初始位姿

当前已经改成：

- 由所有视角的 `SV pose` 融合得到 `obj_init_rot6d / obj_init_trans`

这部分建议继续保留监督：

- `obj_init_rot6d` 对 `master_obj_rot6d_label`
- `obj_init_trans` 对 `master_obj_t_label_rel`

这样 fused init 本身就会被训练成一个可靠的多视角初值，而不是纯中间变量。

## 6.4 Master 最终位姿

建议保留：

- `obj_rot6d` 对 `master_obj_rot6d_label`
- `obj_trans` 对 `master_obj_t_label_rel`
- 模板点云刚性变换后的 `obj_xyz_master` 对 `master_obj_sparse`

这样可以保证 master 分支真正学习到统一空间下的物体姿态，而不是只是在初始化周围漂移。

## 7. 训练逻辑建议

## 7.1 Stage1

目标：

- 先学稳 hand branch
- 先学稳 SV object pose branch

推荐监督：

- 手部 2D reprojection
- 手部 consistency / prior
- SV object rot/trans direct supervision
- SV object point reconstruction

这一阶段不要太早启用 master 对 SV 的自蒸馏。

原因很简单：

- 早期 master 还不如 per-view SV 稳定
- 过早蒸馏会把劣质 master 误差反向灌给 SV

## 7.2 Stage2

目标：

- 用多视角 SV pose 融合得到更稳定的 `obj_init`
- 在此基础上用 hand-object interaction 预测最终 master pose

推荐监督：

- `obj_init` 对 master pose
- `final master pose` 对 master pose
- `master object points` 对 master sparse points

## 7.3 自蒸馏策略

建议继续保留“延后开启”：

- `SV_SELF_DISTILL_ENABLED = true`
- `SV_SELF_DISTILL_START_EPOCH` 设为较晚 epoch

因为你当前的逻辑判断是成立的：

- stage1 后期，SV 往往比刚进入 stage2 的 master 更稳
- 只有当 master 真正学得比 SV 更好时，`master -> sv` 的蒸馏才有正作用

## 8. 这种路线的核心优势

## 8.1 降低人工 3D 标注依赖

这是最大价值。

如果手部 3D GT 难获取，伪标签弱监督可以显著降低数据准备成本。

## 8.2 多视角 student 具备纠偏能力

teacher 可能是：

- 单视角
- 遮挡敏感
- 容易漂移

但 student 有多视角融合和模板约束，理论上能在一些场景下超过 teacher。

## 8.3 物体模板约束很强

相比自由形变物体，已知模板物体的 pose 学习更适合做伪标签蒸馏，因为：

- 刚性变换明确
- 统一空间容易定义
- 点云重建误差可解释

## 9. 主要风险

## 9.1 Teacher bias

student 容易学到 teacher 的系统性偏差，例如：

- WiLoR 在遮挡下手 2D 漂移
- HORT 在相似物体或边缘区域的 pose 偏差

如果只拟合伪标签，不加几何一致性，student 只是 teacher mimicry。

## 9.2 伪标签质量不均匀

不同视角的质量差异会很大：

- 有的视角看不见手
- 有的视角物体严重遮挡
- 有的视角预测到邻近干扰物

因此必须设计伪标签筛选或置信度过滤机制。

## 9.3 方法表述风险

如果物体伪标签来自 `HORT`，而论文主对比对象又是 `HORT`，需要非常谨慎：

- 必须明确说明 teacher 来源
- 必须强调 student 利用的是多视角一致性与模板约束
- 必须做消融证明不是单纯 teacher copying

否则很容易被质疑：

- “你只是用竞争方法给自己打标签”

## 10. 建议的伪标签质量控制

建议至少加入以下过滤策略。

### 10.1 手部伪标签过滤

- 2D 关键点置信度阈值
- 低置信关键点 masking
- 多视角三角化后重投影误差筛选

### 10.2 物体伪标签过滤

- 位姿置信度阈值
- 模板点云投影后的 reprojection 误差筛选
- 多视角 pose 一致性筛选
- 异常平移幅值或异常旋转跳变剔除

### 10.3 样本级过滤

- 过滤大面积遮挡样本
- 过滤 teacher 明显失败样本
- 按置信度分阶段纳入训练

## 11. 推荐实验路线

建议按下面顺序推进，而不是一步到位。

### 实验 1：物体仍用伪 6D，手只用伪 2D

目标：

- 验证当前框架在无人工 hand 3D GT 下是否仍可稳定训练

这是最关键的一步。

### 实验 2：加入伪标签质量过滤

目标：

- 比较不过滤和过滤伪标签的效果差异

重点观察：

- `SVRot / SVTr`
- `Master Rot / Tr`
- `FS@5 / FS@10 / CD`

### 实验 3：加入多视角 consistency 强化

目标：

- 验证 student 是否真正超过 teacher

建议比较：

- 只拟合伪标签
- 伪标签 + 几何一致性
- 伪标签 + 几何一致性 + master fusion

### 实验 4：逐步弱化物体 6D 监督

这是更后面的方向。

如果前面已经稳定，再尝试把物体监督从“直接 6D GT”逐步替换为：

- 多视角模板投影一致性
- silhouette/mask 对齐
- master-sv cycle consistency

这时才真正开始逼近“更自监督”的设定。

## 12. 对当前 HOR 的改造建议

如果未来正式切到这条线，建议改造优先级如下。

### 优先级 1

- 保留当前整体网络框架
- 把手部 3D GT 直接监督切换为伪 2D + consistency
- 保留物体 per-view 6D 伪监督
- 保留物体 master pose 监督

这是最稳的过渡路线。

### 优先级 2

- 给伪标签加质量权重
- 对低质量视角做 masking
- 训练时按置信度 curriculum 学习

### 优先级 3

- 逐步减少对外部物体 6D teacher 的依赖
- 强化模板投影一致性和跨视角 consistency

## 13. 最终判断

这条路线是合理的，而且有研究价值。

它的本质不是：

- 从完全无标注数据中自发学出手物姿态

而是：

- 用离线 teacher 产生可扩展伪标签
- 再利用多视角几何和模板先验，把 single-view teacher 的能力提升为 multi-view student 的能力

一句话总结：

**这不是纯自监督，而是“伪标签弱监督 + 多视角一致性增强”的手物联合学习方案。**

如果设计和实验做扎实，它比单纯继续调当前全监督 loss 更像一条真正可写成方法的研究路线。
