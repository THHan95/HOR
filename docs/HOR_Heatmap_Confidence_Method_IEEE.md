# Heatmap-Derived Confidence Estimation for Multi-View Hand-Object Perception

## Abstract-Style Summary

We estimate a bounded confidence score directly from the predicted 2D heatmap distribution. The objective is to assign high confidence to sharp, unimodal, and spatially concentrated heatmaps, while suppressing uncertain predictions caused by occlusion, ambiguity, or diffuse responses. Unlike a raw peak score or an unbounded inverse-variance score, the proposed formulation combines directional spread, peak saliency, and entropy in a normalized 0-1 range, which makes it numerically stable for triangulation and view-feature aggregation.

## Notation

For a joint or object center, let H(u,v) in R^(h x w) denote the normalized heatmap probability mass function over the discrete heatmap grid, where sum_u sum_v H(u,v) = 1. Let N = h w be the total number of spatial bins. We define the horizontal and vertical marginals as

p_u(u) = sum_v H(u,v),

p_v(v) = sum_u H(u,v).

The horizontal and vertical coordinates are normalized to [0, 1] as

x_u = u / (w - 1),    y_v = v / (h - 1).

## Directional Uncertainty

We first estimate the directional mean and variance of the heatmap distribution:

mu_u = sum_u p_u(u) x_u,

mu_v = sum_v p_v(v) y_v,

sigma_u^2 = sum_u p_u(u) (x_u - mu_u)^2,

sigma_v^2 = sum_v p_v(v) (y_v - mu_v)^2.

To map uncertainty into a bounded concentration score, we compare each standard deviation against the standard deviation of a uniform distribution on [0, 1], namely sigma_uni = 1 / sqrt(12). The directional spread scores are

s_u = 1 - clip( sqrt(sigma_u^2) / sigma_uni, 0, 1),

s_v = 1 - clip( sqrt(sigma_v^2) / sigma_uni, 0, 1).

Therefore, s_u and s_v approach 1 when the heatmap is spatially concentrated, and approach 0 when the distribution becomes diffuse.

## Peak Saliency and Entropy

A good heatmap should also exhibit a salient dominant mode. Let

p_max = max_{u,v} H(u,v).

Instead of directly using p_max, which may be overly sensitive to local activation spikes, we compute a log-compressed peak score:

r_peak = clip( N p_max, 1, +inf ),

s_peak = clip( log(r_peak) / log(N), 0, 1).

In parallel, we evaluate the global ambiguity of the full heatmap using Shannon entropy:

E = - sum_u sum_v H(u,v) log(H(u,v) + epsilon),

s_ent = clip( 1 - E / log(N), 0, 1).

Here, s_peak rewards strong local evidence, whereas s_ent penalizes multi-modal or highly uncertain distributions that often occur under occlusion or severe truncation.

## Final Confidence Formulation

The global structural quality of the heatmap is defined as the average of the peak and entropy terms:

s_g = 0.5 (s_peak + s_ent).

The final axis-wise confidence is then computed as

c = ( sqrt( [s_u, s_v] * s_g ) )^gamma,

where [s_u, s_v] denotes channel-wise multiplication with the scalar s_g, and gamma > 1 is a mild compression factor. In the current implementation, gamma = 1.2. The final confidence is clipped into [epsilon, 1].

This yields two bounded confidence values, c_u and c_v, corresponding to the horizontal and vertical localization quality. When a scalar confidence is required, one may use c_bar = (c_u + c_v) / 2.

## Properties and Motivation

The formulation has the following desirable properties.

(1) Error-consistent trend: if the predicted heatmap becomes sharper and more localized around the correct position, then the directional variances decrease, the peak score increases, and the entropy decreases, leading to a higher confidence.

(2) Occlusion awareness: partial occlusion or feature ambiguity often produces broad or multi-modal heatmaps, which simultaneously reduce the spread score and the entropy-based quality term.

(3) Bounded scale: unlike inverse-standard-deviation confidence, the proposed score is bounded in [0, 1], which is easier to use in weighted triangulation, feature fusion, and qualitative visualization.

(4) Anti-saturation behavior: compared with a plain maximum-activation score, the log-compressed peak term and the gamma correction reduce the tendency of confidence values to collapse near 0.9-1.0 for moderately good predictions.

## Implementation Notes

In the current HOR implementation, the confidence is computed from the normalized heatmap probability after integral decoding. The resulting confidence is subsequently used in at least two places: (1) confidence-weighted multi-view triangulation, and (2) view-aware feature aggregation in the downstream hand-object head. Because the triangulation module renormalizes view weights across cameras, the absolute scale of confidence mainly affects relative reliability ranking rather than causing uncontrolled numerical growth.

## Suggested Paper Paragraph

We derive a confidence estimate directly from the predicted heatmap distribution instead of regressing an additional uncertainty head. Given a normalized heatmap probability map, we compute its horizontal and vertical marginal variances, a log-compressed peak saliency term, and a normalized entropy score. The directional variances measure localization spread, while the peak and entropy terms characterize the global sharpness and ambiguity of the heatmap. The final confidence is obtained by combining directional concentration with global structural quality, followed by a mild gamma compression. This design yields a bounded confidence score in [0, 1] that increases for sharp unimodal responses and decreases for diffuse or multi-modal responses typically caused by occlusion, truncation, or view ambiguity. The confidence is then used to weight multi-view triangulation and confidence-aware view fusion.

## Compact Equation Block

p_u(u) = sum_v H(u,v),    p_v(v) = sum_u H(u,v)

sigma_u^2 = sum_u p_u(u) (x_u - mu_u)^2,    sigma_v^2 = sum_v p_v(v) (y_v - mu_v)^2

s_u = 1 - clip( sigma_u / sigma_uni, 0, 1),    s_v = 1 - clip( sigma_v / sigma_uni, 0, 1)

s_peak = clip( log( max( N max(H), 1 ) ) / log(N), 0, 1)

s_ent = clip( 1 - [ - sum H log(H + epsilon) ] / log(N), 0, 1)

s_g = 0.5 (s_peak + s_ent)

c = ( sqrt( [s_u, s_v] * s_g ) )^gamma,    gamma = 1.2
