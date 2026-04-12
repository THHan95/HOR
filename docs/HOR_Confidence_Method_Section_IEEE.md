# Method Draft: Heatmap-Derived Confidence Estimation

## C. Heatmap-Derived Confidence Estimation

To improve the reliability of multiview triangulation and confidence-aware feature aggregation, we estimate a per-view confidence score directly from the predicted 2D heatmap distribution. The design principle is that a reliable prediction should produce a heatmap that is spatially concentrated, locally salient, and globally low-entropy. In contrast, occlusion, truncation, or severe ambiguity usually lead to diffuse or multi-modal heatmaps, which should be assigned lower confidence.

Let H_k in R^(h x w) denote the normalized heatmap probability map of the k-th keypoint or object center, where

sum_{u=1}^{w} sum_{v=1}^{h} H_k(u,v) = 1.

We first compute the horizontal and vertical marginal distributions

p_k^u(u) = sum_v H_k(u,v),    p_k^v(v) = sum_u H_k(u,v),

and normalize the discrete coordinates to [0,1] as x_u = u/(w-1) and y_v = v/(h-1). The corresponding means and variances are given by

mu_k^u = sum_u p_k^u(u) x_u,      mu_k^v = sum_v p_k^v(v) y_v,

sigma_k^{u2} = sum_u p_k^u(u) (x_u - mu_k^u)^2,

sigma_k^{v2} = sum_v p_k^v(v) (y_v - mu_k^v)^2.

These variances measure directional uncertainty. To obtain bounded concentration scores, we normalize them by the standard deviation of a uniform distribution on [0,1], i.e., sigma_uni = 1/sqrt(12), and define

s_k^u = 1 - clip( sqrt(sigma_k^{u2}) / sigma_uni, 0, 1),

s_k^v = 1 - clip( sqrt(sigma_k^{v2}) / sigma_uni, 0, 1).

In addition to directional spread, we characterize the global quality of the heatmap using peak saliency and entropy. Let N = h w be the number of spatial bins and let p_k^max = max_{u,v} H_k(u,v). We define a log-compressed peak score as

r_k^peak = max(N p_k^max, 1),

s_k^peak = clip( log(r_k^peak) / log(N), 0, 1).

The normalized entropy score is defined as

E_k = - sum_u sum_v H_k(u,v) log(H_k(u,v) + epsilon),

s_k^ent = clip( 1 - E_k / log(N), 0, 1).

The peak term favors a dominant local mode, whereas the entropy term penalizes ambiguous or multi-modal responses. We combine them into a global structural quality score

s_k^g = 0.5 (s_k^peak + s_k^ent).

Finally, the axis-wise confidence is obtained by coupling directional concentration with the global quality term:

c_k = ( sqrt( [s_k^u, s_k^v] * s_k^g ) )^gamma,

where [s_k^u, s_k^v] denotes channel-wise multiplication with the scalar s_k^g, and gamma = 1.2 in our implementation.
