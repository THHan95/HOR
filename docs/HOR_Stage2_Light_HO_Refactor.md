# Stage2 Lightweight HO Refactor

## Goal

This refactor simplifies the final hand-object interaction path in `stage2` while keeping the hand branch stable.

The intended behavior is:

- keep the hand refinement branch unchanged
- use the predicted KP hand mesh as a fixed anchor in `stage2`
- let the object point cloud interact with the fixed hand mesh only once
- predict the final master object pose in a single HO pass
- avoid the previous heavy multi-layer HO refinement loop

This matches the current design target:

- the hand mesh should not be repeatedly distorted by later HO interaction
- the object branch should adapt to the hand geometry, not re-drive the hand branch
- the final HO module should stay lightweight but still preserve a transformer-style point interaction structure

## Main Changes

### 1. Added a transformer-style lightweight HO module

Files:

- [`lib/models/layers/ptEmb_transformer.py`](/media/hl/data/code/han/POEM/lib/models/layers/ptEmb_transformer.py)
- [`lib/models/bricks/pt_metro_transformer.py`](/media/hl/data/code/han/POEM/lib/models/bricks/pt_metro_transformer.py)

New module:

- `HORTR_HO_Light`
- `point_METRO_block_HO_Light`
- `point_METRO_layer_obj_pose`
- `pointer_layer_obj_pose`

Current structure:

- object points are used as query points
- KP hand mesh points are used as key/value context points
- a single object self-attention is applied first
- a single hand-object cross-attention is applied next
- pose regression is done inside the point transformer path
- rotation and translation are predicted by two separate branches:
  - `rot_branch`
  - `trans_branch`

Important detail:

- the `rot / trans` prediction is no longer described as an extra external MLP head after HO interaction
- it is now produced inside `self.vec_attn` of the HO pose layer, which is closer to the hand branch style already used in the repo

### 2. One-pass HO update in stage2

File:

- [`lib/models/heads/ptHOR_head.py`](/media/hl/data/code/han/POEM/lib/models/heads/ptHOR_head.py)

Current stage2 HO flow:

1. build the initial object point cloud from the initialized SV pose
2. sample multiview object point features once
3. use the predicted KP hand mesh as the fixed hand anchor
4. run one HO-light pass:
   - object self-attn
   - hand-object interaction
   - direct master pose update
5. output one-layer object results only

Compared with the older heavy HO branch:

- no repeated multi-layer HO iteration
- no repeated object pose update loop across several transformer blocks
- no need to mirror the hand-only branch depth in the HO branch

### 3. Rot / trans deltas are bounded inside the HO-light block

Files:

- [`lib/models/heads/ptHOR_head.py`](/media/hl/data/code/han/POEM/lib/models/heads/ptHOR_head.py)
- [`lib/models/layers/ptEmb_transformer.py`](/media/hl/data/code/han/POEM/lib/models/layers/ptEmb_transformer.py)

Current behavior:

- `OBJ_POSE_ROT_DELTA_ABS_MAX`
- `OBJ_POSE_TRANS_DELTA_ABS_MAX`

are passed into `HORTR_HO_Light`, then into `pointer_layer_obj_pose`, where:

- `delta_rot` is constrained by `tanh(...) * rot_delta_abs_max`
- `delta_trans` is constrained by `tanh(...) * trans_delta_abs_max`

This keeps the one-pass update stable and prevents the lightweight block from making overly aggressive corrections in early stage2.

### 4. Old external object delta modules are not part of the active HO path

Files:

- [`lib/models/heads/ptHOR_head.py`](/media/hl/data/code/han/POEM/lib/models/heads/ptHOR_head.py)
- [`lib/models/HOR_heatmap.py`](/media/hl/data/code/han/POEM/lib/models/HOR_heatmap.py)
- [`lib/models/HOR.py`](/media/hl/data/code/han/POEM/lib/models/HOR.py)

Notes:

- `obj_pose_delta_regressor`
- `obj_feat_fuser`

still exist in the head because other code paths and compatibility logic still reference them.

However, for the active lightweight HO route:

- they are not the main stage2 HO prediction path anymore
- they are explicitly frozen in stage setup to avoid interference and DDP-related issues

### 5. Stage-freeze logic was updated for the new HO module

Files:

- [`lib/models/HOR_heatmap.py`](/media/hl/data/code/han/POEM/lib/models/HOR_heatmap.py)
- [`lib/models/HOR.py`](/media/hl/data/code/han/POEM/lib/models/HOR.py)

Changes:

- freeze logic no longer assumes the HO branch must expose the old multi-block iteration interface
- the new HO-light module can be enabled in `stage2` without startup issues

## Files Changed

- [`docs/HOR_Stage2_Light_HO_Refactor.md`](/media/hl/data/code/han/POEM/docs/HOR_Stage2_Light_HO_Refactor.md)
- [`lib/models/bricks/pt_metro_transformer.py`](/media/hl/data/code/han/POEM/lib/models/bricks/pt_metro_transformer.py)
- [`lib/models/layers/ptEmb_transformer.py`](/media/hl/data/code/han/POEM/lib/models/layers/ptEmb_transformer.py)
- [`lib/models/heads/ptHOR_head.py`](/media/hl/data/code/han/POEM/lib/models/heads/ptHOR_head.py)
- [`lib/models/HOR_heatmap.py`](/media/hl/data/code/han/POEM/lib/models/HOR_heatmap.py)
- [`lib/models/HOR.py`](/media/hl/data/code/han/POEM/lib/models/HOR.py)

## Runtime Verification

A short `stage2-only` GPU debug run was executed with:

- `config/release/HOR_DexYCBMV_stage2_only_debug.yaml`

Observed results for the current implementation:

- `HORTR_HO_Light` parameter count: `9.81M`
- `HOR_Projective_SelfAggregation_Head` parameter count: `117.57M`
- full `POEM_Heatmap` parameter count: `239.02M`

The runtime trace confirmed:

- forward pass works
- backward pass works
- optimizer step works
- object outputs remain single-layer:
  - `obj_states=(1, B, 2048, 3)`
  - `obj_rot6d=(1, B, 6)`
  - `obj_trans=(1, B, 3)`

## Current Interpretation

The current HO branch is no longer the earlier ultra-light MLP-style pose update.

It is now a compromise design:

- still only one HO interaction pass
- still much lighter than the original heavy HO branch
- but structurally closer to the hand branch because it keeps a point-transformer style interaction path

This is why the parameter count increased from the temporary `2.76M` version to about `9.81M`.

That increase is expected and comes from:

- adding object self-attention
- restoring a more transformer-like HO interaction block
- splitting pose prediction into dedicated rotation / translation branches inside the point interaction module

## Remaining Issue

This refactor improves the HO branch design and keeps the stage2 path simpler, but it does not by itself resolve the `SVRot` optimization problem.

Current debugging conclusion:

- the HO branch structure was one factor worth simplifying
- but `SVRot` behavior in `stage2` is still affected by optimization competition and/or loss balance outside this HO structural change alone
