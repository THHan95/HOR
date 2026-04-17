# HOR SV/Master Object Pose Design

Date: 2026-04-13
Current Commit: `39c635c`

## Goal

This note summarizes a more coherent object-pose design for the current HOR pipeline.

The target is:

- make the `SV` object branch predict a useful pose, not just a partial rotation signal
- keep `SV` and `master` pose definitions aligned in semantics
- use the existing DexYCB labels directly without inventing weak pseudo-labels
- make `master -> SV` self-distillation mathematically valid for both rotation and translation
- keep the pose outputs directly usable for visualization, reconstruction, and later stage2 refinement

## Current Problem

The current implementation is only half-closed:

- `SV` predicts per-view object rotation `obj_view_rot6d_cam`
- `SV` does not predict per-view object translation explicitly
- object translation / object center is mainly inherited from:
  - 2D object-center prediction
  - triangulated `ref_obj`
- `master` predicts:
  - `obj_rot6d`: object rotation in master camera frame
  - `obj_trans`: object translation relative to master hand root

This causes several structural issues:

1. `SV` pose is incomplete
- rotation is learned from single-view features
- translation is not learned from single-view features
- therefore `SV` is not a true pose branch

2. `SV` and `master` are not expressed in the same pose parameterization
- `SV` currently uses camera-frame rotation only
- `master` uses camera-frame rotation plus hand-root-relative translation
- this weakens closed-loop distillation

3. `master -> SV` self-distillation is only applied on rotation
- current self-distill converts `master` rotation into each view and supervises `SV` rotation
- translation is not distilled
- the pose loop is therefore incomplete

4. stage2 depends too much on triangulated object center quality
- object center from multiview 2D is useful, but noisy views corrupt the result
- if `SV` never learns its own relative translation, the initialization remains fragile

## Design Principle

Use the same pose semantics for `SV` and `master`.

Recommended unified definition:

- rotation is always defined in the current camera axis
- translation is always defined relative to the current hand root

That means:

- `SV pose` for view `v`:
  - `R_v`: object rotation in view-`v` camera frame
  - `d_v`: object center relative to view-`v` hand root
- `master pose`:
  - `R_m`: object rotation in master camera frame
  - `d_m`: object center relative to master hand root

So both branches are predicting the same type of quantity, only in different views.

## Why This Is Better

### 1. Existing GT labels already match this definition

For each view, the current data pipeline already provides:

- `target_rot6d_label`
- `target_t_label_rel`

For the master view, the current data pipeline already provides:

- `master_obj_rot6d_label`
- `master_obj_t_label_rel`

So this redesign does not require constructing new supervision.

### 2. SV can learn useful single-view geometry

A single view does contain information for object translation relative to hand root:

- hand pose
- contact layout
- visible object scale
- occlusion pattern
- grasp type

This signal is weaker than multiview triangulation, but it is still learnable and useful.

Therefore, using `SV` only for rotation wastes available supervision and weakens stage1.

### 3. Master-to-view distillation becomes clean

Because both `SV` and `master` use the same translation semantics relative to hand root, the conversion is simple.

If:

- `d_m = c_obj^m - c_hand^m`

then under master-to-view rigid transform with rotation `R_{v<-m}`:

- `d_v = R_{v<-m} d_m`

No extra translational offset is needed because both centers are transformed together and the camera translation term cancels out.

This is the most important structural advantage of this design.

## Recommended Pose Definition

### SV branch

Per view `v`, predict:

- `sv_obj_rot6d_cam[v] = R_v`
- `sv_obj_trel_cam[v] = d_v`

where:

- `R_v` is in camera frame `v`
- `d_v = obj_center_v - hand_root_v`

Derived quantities:

- `sv_obj_center_cam[v] = sv_hand_root_cam[v] + d_v`
- `sv_obj_points_cam[v] = R_v * X_rest + sv_obj_center_cam[v]`

### Master branch

Predict:

- `obj_rot6d = R_m`
- `obj_trans = d_m`

where:

- `R_m` is in master camera frame
- `d_m = obj_center_m - hand_root_m`

Derived quantities:

- `obj_center_master = hand_root_master + d_m`
- `obj_points_master = R_m * X_rest + obj_center_master`

## Supervision

### Stage1 supervision

`SV` branch should be fully supervised:

- `loss_sv_rot`
  - supervise `sv_obj_rot6d_cam` with `target_rot6d_label`
- `loss_sv_trans`
  - supervise `sv_obj_trel_cam` with `target_t_label_rel`
- optional `loss_sv_points`
  - compare pose-built `sv_obj_points_cam` with `target_obj_pc_sparse` or dense eval points
- optional `loss_sv_center`
  - compare `sv_obj_center_cam` with GT object center in camera frame

At stage1, this gives a real single-view object pose branch rather than just a rotation helper.

### Stage2 supervision

`master` branch keeps direct GT supervision:

- `loss_master_rot`
  - supervise `obj_rot6d` with `master_obj_rot6d_label`
- `loss_master_trans`
  - supervise `obj_trans` with `master_obj_t_label_rel`
- `loss_master_points`
  - supervise built master object point cloud with `master_obj_sparse`

`SV` branch should still keep its own GT supervision in stage2:

- do not replace GT with self-distill
- self-distill should be additive and gated later

## Master -> SV Self-Distillation

Self-distillation should be applied to both rotation and translation, not rotation only.

### Rotation

Given final master prediction:

- `R_m`

and master-to-view extrinsic rotation:

- `R_{v<-m}`

then:

- `R_v^distill = R_{v<-m} R_m`

This is already close to the current implementation.

### Translation

Given master relative translation:

- `d_m`

then the view-relative translation teacher is:

- `d_v^distill = R_{v<-m} d_m`

This is valid because `d` is a relative vector from hand root to object center.

### Distill losses

Use detached teacher:

- `R_m.detach()`
- `d_m.detach()`

Suggested losses:

- `loss_sv_rot_self`
  - compare `sv_obj_rot6d_cam` with `R_v^distill`
- `loss_sv_trans_self`
  - compare `sv_obj_trel_cam` with `d_v^distill`

Important:

- keep GT supervision active for `SV`
- self-distill should only start after a configured epoch or confidence condition

## Recommended Training Schedule

### Stage1

Train:

- hand single-view branch
- object center branch
- `SV` object rotation branch
- `SV` object translation branch
- keypoint/master hand refinement branch if needed

Do not use self-distillation here.

### Stage2 early phase

Train:

- `SV` still supervised by GT
- `master` supervised by GT
- `master` initialized from `SV`

Do not yet distill `master -> SV`.

Reason:

- early stage2 master is not necessarily better than `SV`
- forcing it back onto `SV` can destabilize `SV`

### Stage2 later phase

Enable gated self-distillation:

- `master -> SV rotation`
- `master -> SV translation`

Use one of the following gates:

- epoch gate
- confidence gate
- metric gate

The simplest option is still an epoch gate in config.

## Suggested Outputs

### Add new SV tensors

Recommended new outputs:

- `obj_view_rot6d_cam` `(B, N, 6)`
- `obj_view_trel_cam` `(B, N, 3)`
- `obj_view_center_cam` `(B, N, 1, 3)`
- `obj_view_xyz_cam` `(B, N, P, 3)`

Optional:

- `obj_view_pose_conf`
- `obj_view_trans_valid`

### Keep current master outputs

Keep:

- `obj_rot6d` `(L, B, 6)`
- `obj_trans` `(L, B, 3)`
- `obj_xyz_master` `(L, B, P, 3)`
- `obj_center_xyz_master` `(L, B, 1, 3)`

## Initialization Path

The current initialization path should evolve from:

- `SV rotation -> master init rotation`

into:

- `SV pose -> master init pose`

Recommended approach:

1. predict `(R_v, d_v)` for each view
2. convert each view pose to master frame:
- `R_m^{(v)} = R_{m<-v} R_v`
- `d_m^{(v)} = R_{m<-v} d_v`
3. pick one of:
- master-view pose only
- confidence-weighted fusion over views

The conservative first implementation is:

- use master-view `SV pose` directly as `master init pose`

This matches the current `master init rot` logic and is easy to stabilize.

## Role of the Object Center Branch

The object center branch is still useful, but its role should change.

It should serve as:

- 2D localization prior
- triangulation prior
- visibility / confidence cue
- auxiliary geometric regularizer

It should not remain the only source of translation information for `SV` object pose.

## Multi-Object Ambiguity and Why 2D Object Center Should Be Removed from the Main Path

There is a structural ambiguity in the current 2D object-center branch.

Unlike the hand branch:

- hand topology is fixed
- hand keypoints have stable semantics
- the network is not usually confused by several unrelated hand categories

the object branch faces a different problem:

- objects are category-diverse
- the current image may contain multiple objects
- a plain 2D center head is not necessarily target-object-conditioned
- the most visually salient object center is not always the labeled target object center

This creates a failure mode:

- the network predicts the center of another visible object
- the confidence can still be high
- the triangulated 3D object reference becomes wrong but over-confident
- stage2 object initialization is then pulled toward the wrong target

This risk is especially harmful on validation and test:

- the error is not just noisy localization
- it is a semantically wrong object association
- once it enters `ref_obj` or object initialization, the whole downstream pose branch can be biased

Because of this, the current 2D object-center branch should not remain on the primary object-pose path unless it becomes explicitly target-conditioned.

## Recommended Decision About the 2D Object Center Branch

If the next refactor upgrades `SV` to predict full pose `(R_v, d_v)`, then the recommended decision is:

- remove the 2D object-center branch from the primary training and inference path

That means:

- do not use it for `ref_obj` primary initialization
- do not use it for `master` object pose initialization
- do not use it as the substitute for `SV` translation
- do not let it dominate confidence weighting for the final object pose path

This branch may still be kept only if one of the following is added:

- target object identity conditioning
- target object template conditioning
- object-specific ROI conditioning
- an explicit target-selection mechanism

Without such conditioning, the branch is structurally ambiguous in multi-object scenes.

## Updated Recommendation

The stronger recommendation for the next design iteration is:

- `SV` predicts full pose `(R_v, d_v)`
- `master` predicts full pose `(R_m, d_m)`
- 2D object center is removed from the main object-pose pipeline

Under this design:

- `SV -> master` initialization comes from `SV` pose, not from triangulated object center
- `master -> SV` self-distillation operates on both rotation and translation
- the main object pose path no longer depends on an ambiguous 2D center detector

This is more coherent geometrically and safer semantically in multi-object scenes.

## Recommended Loss Structure

### SV branch

- `loss_obj_view_rot_geo`
- `loss_obj_view_rot_l1`
- `loss_obj_view_trans_l1`
- optional `loss_obj_view_points`
- optional `loss_obj_view_center`
- later: `loss_obj_view_rot_self`
- later: `loss_obj_view_trans_self`

### Master branch

- `loss_obj_init_rot`
- `loss_obj_init_trans` optional
- `loss_obj_rot_geo`
- `loss_obj_rot_l1`
- `loss_obj_trans`
- `loss_obj_points`
- optional dense reconstruction metrics only for val/test

## Minimal Migration Plan

### Step 1

Add a new `SV` translation head in:

- `lib/models/HOR.py`
- `lib/models/HOR_heatmap.py`

Input can stay aligned with the current `SV` rotation branch:

- `global_feat`
- optionally later add `hand_obj_fused_flat`

### Step 2

Produce per-view:

- `obj_view_trel_cam`

and build:

- `obj_view_center_cam`
- `obj_view_xyz_cam`

### Step 3

Add GT loss:

- `obj_view_trel_cam` vs `target_t_label_rel`

### Step 4

Use the master view `SV pose` as stage2 initial pose:

- `obj_init_rot6d`
- `obj_init_trans`

### Step 5

Add late-stage self-distillation:

- `master rot -> sv rot`
- `master trans -> sv trans`

## Why This Design Is the Most Reasonable for the Current Codebase

Compared with other alternatives:

### Alternative A: SV predicts camera-frame absolute translation

Not recommended.

Reason:

- absolute camera translation is more viewpoint-dependent
- it does not align naturally with current master `obj_trans`
- `master -> SV` self-distill becomes less clean because absolute centers require full rigid point transformation rather than simple vector rotation

### Alternative B: SV predicts only rotation

This is the current design and is not ideal.

Reason:

- supervision is under-used
- `SV` pose is incomplete
- stage2 relies too much on triangulated object center quality
- self-distillation only covers half the pose

### Alternative C: SV and master both use camera-frame rotation + hand-root-relative translation

Recommended.

Reason:

- uses existing labels directly
- provides full pose for both branches
- clean master-to-view conversion
- clean self-distillation
- useful for visualization and geometry checks

## Final Recommendation

The next refactor should move the object branch to this unified pose design:

- `SV`: predict `(R_v, d_v)`
- `master`: predict `(R_m, d_m)`

with:

- `R`: rotation in current camera frame
- `d`: object-center translation relative to current hand root

This is the cleanest way to:

- use current GT labels
- make `SV` predictions actually useful
- make `master` and `SV` live in one coherent pose family
- enable valid `master -> SV` self-distillation for both rotation and translation

## Related Current Code Paths

Current relevant files:

- [`lib/models/HOR.py`](/media/hl/data/code/han/POEM/lib/models/HOR.py)
- [`lib/models/HOR_heatmap.py`](/media/hl/data/code/han/POEM/lib/models/HOR_heatmap.py)
- [`lib/models/heads/ptHOR_head.py`](/media/hl/data/code/han/POEM/lib/models/heads/ptHOR_head.py)
- [`lib/datasets/dexycb.py`](/media/hl/data/code/han/POEM/lib/datasets/dexycb.py)
- [`lib/datasets/hdata.py`](/media/hl/data/code/han/POEM/lib/datasets/hdata.py)
- [`lib/utils/transform.py`](/media/hl/data/code/han/POEM/lib/utils/transform.py)

## Implementation Status

Implemented in code in this iteration:

- `HOR.py`
  - `SV` object branch now predicts full per-view pose:
    - `obj_view_rot6d_cam`
    - `obj_view_trans`
  - `SV` object feature uses `global_feat + hand_obj_fused_flat`
  - master initialization now uses `SV` master-view pose:
    - `obj_init_rot6d`
    - `obj_init_trans`
  - stage1/stage2 `SV` supervision now covers both rotation and translation
  - `SV` reconstruction metrics now rebuild object point clouds from `SV` relative pose instead of `ref_obj`

- `HOR_heatmap.py`
  - same semantic changes as `HOR.py`
  - heatmap object-center branch is no longer used as the main object-pose initialization path

- `ptHOR_head.py`
  - accepts `obj_init_trans`
  - object initialization center is derived from `hand_center + obj_init_trans`
  - stage2 object branch no longer depends on center-triangulation as the primary translation source

Current simplification kept intentionally:

- `master -> SV` self-distillation is still rotation-only in code for now
- translation self-distillation can be added later after the new supervised `SV` full-pose branch is verified stable
