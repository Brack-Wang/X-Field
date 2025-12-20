/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 *
 * Re-implemented with ray-Gaussian intersection method
 */
#include <stdio.h>
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Structure to hold full 3x3 NDC covariance result
struct CovarianceResult {
    float hata;   // cov[0][0]
    float hatb;   // cov[0][1]
    float hatc;   // cov[0][2]
    float hatd;   // cov[1][1]
    float hate;   // cov[1][2]
    float hatf;   // cov[2][2]
};

// Function to compute the inverse of a 3x3 matrix
__device__ glm::mat3 inverse3x3(const glm::mat3& mat) {
    float det = glm::determinant(mat);
    if (fabs(det) < 1e-6f) {
        return glm::mat3(0.0f);
    }
    return glm::inverse(mat);
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
// We do not need SH for X-ray imaging.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
// Returns full 3x3 NDC covariance matrix for ray-Gaussian intersection
__device__ CovarianceResult computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix, const int mode)
{
	// Transform point from world to view space
	float3 t = transformPoint4x3(mean, viewmatrix);
    glm::mat3 J;

	if (mode == 0) {  // parallel beam
		const float limx = 1.3f;
		const float limy = 1.3f;
		t.x = min(limx, max(-limx, t.x));
		t.y = min(limx, max(-limx, t.y));

		J = glm::mat3(
			focal_x, 0.0f, 0.0f,
			0.0f, focal_y, 0.0f,
			0.0f, 0.0f, 1.0f);
	}
	else {  // cone beam
		const float limx = 1.3f * tan_fovx;
		const float limy = 1.3f * tan_fovy;
		const float txtz = t.x / t.z;
		const float tytz = t.y / t.z;
		t.x = min(limx, max(-limx, txtz)) * t.z;
		t.y = min(limy, max(-limy, tytz)) * t.z;

		const float l = sqrt(t.x * t.x + t.y * t.y + t.z * t.z);
		J = glm::mat3(
			focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
			0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
			t.x / l, t.y / l, t.z / l);
	}

	// View matrix rotation part
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 M = W * J;

	// 3D covariance matrix
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// Transform covariance to NDC space: J W Sigma W^T J^T
	glm::mat3 cov = glm::transpose(M) * glm::transpose(Vrk) * M;

	// Add small regularization for numerical stability (matching original behavior)
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;

	CovarianceResult result;
    result.hata = cov[0][0];
    result.hatb = cov[0][1];
    result.hatc = cov[0][2];
    result.hatd = cov[1][1];
    result.hate = cov[1][2];
    result.hatf = cov[2][2];

	return result;
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float3* ndcz,
	float* cov3Ds,
	float4* conic_opacity,
	float3* cov3D1,
	float3* cov3D2,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const int mode
	)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// Compute or use precomputed 3D covariance
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute full 3x3 NDC covariance matrix
	CovarianceResult cov_result = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix, mode);

	// Extract 2D covariance for screen-space computations
	float4 cov = make_float4(cov_result.hata, cov_result.hatb, cov_result.hatd, 0.0f);

	// Invert 2D covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Store helper data
	depths[idx] = p_view.z;
	ndcz[idx] = {p_proj.x, p_proj.y, p_proj.z};
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };

	// Store full 3x3 NDC covariance for ray-Gaussian intersection
	cov3D1[idx] = {cov_result.hata, cov_result.hatb, cov_result.hatc};
	cov3D2[idx] = {cov_result.hatd, cov_result.hate, cov_result.hatf};

	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method with ray-Gaussian intersection
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float3* __restrict__ cov3D1,
	const float3* __restrict__ cov3D2,
	const float3* __restrict__ ndcz,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	// Load start/end range of IDs to process
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Shared memory for batch fetching
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float3 collected_cov3D1[BLOCK_SIZE];
	__shared__ float3 collected_cov3D2[BLOCK_SIZE];
	__shared__ float3 collected_ndcz[BLOCK_SIZE];

	// Initialize helper variables
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Algorithm 1: Segment length correction state
	// Tracks the far intersection point of previous Gaussian for overlap detection
	float prev_z_far = -1e10f;

	// Iterate over batches
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_cov3D1[block.thread_rank()] = cov3D1[coll_id];
			collected_cov3D2[block.thread_rank()] = cov3D2[coll_id];
			collected_ndcz[block.thread_rank()] = ndcz[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor++;

			// Load full 3x3 NDC covariance and compute inverse
			float3 cov3Dtop = collected_cov3D1[j];
			float3 cov3Ddow = collected_cov3D2[j];
			glm::mat3 c3D;
			c3D[0][0] = cov3Dtop.x;  // hata
			c3D[0][1] = cov3Dtop.y;  // hatb
			c3D[0][2] = cov3Dtop.z;  // hatc
			c3D[1][0] = cov3Dtop.y;  // hatb (symmetric)
			c3D[1][1] = cov3Ddow.x;  // hatd
			c3D[1][2] = cov3Ddow.y;  // hate
			c3D[2][0] = cov3Dtop.z;  // hatc (symmetric)
			c3D[2][1] = cov3Ddow.y;  // hate (symmetric)
			c3D[2][2] = cov3Ddow.z;  // hatf
			glm::mat3 cov_inv = inverse3x3(c3D);

			// Load Gaussian center position in NDC
			float2 xy = collected_xy[j];
			float3 ndc = collected_ndcz[j];
			float3 P = {pix2Ndc(xy.x, W), pix2Ndc(xy.y, H), ndc.z};

			// Current pixel in NDC
			float3 u = {pix2Ndc(pixf.x, W), pix2Ndc(pixf.y, H), 0};

			// Vector from Gaussian center to pixel (ray origin)
			float3 a = make_float3(u.x - P.x, u.y - P.y, u.z - P.z);

			// Ray-Gaussian intersection calculation
			// A = d^T * Σ^{-1} * d where d = (0,0,1) is ray direction
			float A_value = cov_inv[2][2];

			// B = a^T * Σ^{-1} * d
			float3 v;
			v.x = cov_inv[0][2];
			v.y = cov_inv[1][2];
			v.z = cov_inv[2][2];
			float B_value = a.x * v.x + a.y * v.y + a.z * v.z;

			// C = a^T * Σ^{-1} * a
			float3 c_pro;
			c_pro.x = cov_inv[0][0] * a.x + cov_inv[0][1] * a.y + cov_inv[0][2] * a.z;
			c_pro.y = cov_inv[1][0] * a.x + cov_inv[1][1] * a.y + cov_inv[1][2] * a.z;
			c_pro.z = cov_inv[2][0] * a.x + cov_inv[2][1] * a.y + cov_inv[2][2] * a.z;
			float C_value = a.x * c_pro.x + a.y * c_pro.y + a.z * c_pro.z;

			// Calculate intersection length: l = 2 * sqrt(D) / A
			// where D = B^2 - A*(C-1)
			float D_value = (B_value * B_value) - A_value * (C_value - 1);
			float l;
			const float epsilon = 1e-6f;
			if (D_value >= 0.0f && fabs(A_value) > epsilon) {
				// Compute sqrt(D) once and reuse
				float sqrt_D = sqrt(D_value);
				float half_l = sqrt_D / A_value;
				l = 2.0f * half_l;  // l = 2 * sqrt(D) / A

				// Algorithm 1: Segment length correction for overlapping ellipsoids
				// Compute intersection interval [z_near, z_far] along the ray
				float z_center = -B_value / A_value;
				float z_near = z_center - half_l;
				float z_far = z_center + half_l;

				// Check for overlap with previous Gaussian and apply mild correction
				if (z_near < prev_z_far) {
					// Overlap detected - compute corrected segment length
					float l_corrected = fmaxf(z_far - prev_z_far, 0.0f);
					float overlap = prev_z_far - z_near;

					// Only apply correction when overlap is small (< 30% of original length)
					// This prevents aggressive truncation that hurts training
					if (overlap < 0.3f * l) {
						l = l_corrected;
					}
					// else: keep original l for large overlaps (training stability)
				}

				// Update state for next Gaussian
				prev_z_far = fmaxf(prev_z_far, z_far);
			} else {
				l = 0.0f;
			}

			// Apply sigmoid transformation to l
			float k = 10.0f;
			float sigmoid_l = 1.0f / (1.0f + expf(-k * (l - 0.5f)));

			// Load opacity and 2D conic
			float4 con_o = collected_conic_opacity[j];
			float opacity = con_o.w;

			// Compute 2D Gaussian weight (for screen-space falloff)
			float2 xyp = collected_xy[j];
			float2 dp = { xyp.x - pixf.x, xyp.y - pixf.y };
			float power = -0.5f * (con_o.x * dp.x * dp.x + con_o.z * dp.y * dp.y) - con_o.y * dp.x * dp.y;
			if (power > 0.0f)
				continue;

			// Final alpha computation
			const float alpha = opacity * sigmoid_l * exp(power);
			if (alpha < 0.00001f)
				continue;

			// Accumulate for X-ray imaging
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += alpha;

			last_contributor = contributor;
		}
	}

	// Write out final rendering data
	if (inside)
	{
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch];
	}
}


void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float4* conic_opacity,
	const float3* cov3D1,
	const float3* cov3D2,
	const float3* ndcz,
	uint32_t* n_contrib,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		conic_opacity,
		cov3D1,
		cov3D2,
		ndcz,
		n_contrib,
		out_color);
}


void FORWARD::preprocess(int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float3* ndcz,
	float* cov3Ds,
	float4* conic_opacity,
	float3* cov3D1,
	float3* cov3D2,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const int mode
	)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		cov3D_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		ndcz,
		cov3Ds,
		conic_opacity,
		cov3D1,
		cov3D2,
		grid,
		tiles_touched,
		prefiltered,
		mode
		);
}
