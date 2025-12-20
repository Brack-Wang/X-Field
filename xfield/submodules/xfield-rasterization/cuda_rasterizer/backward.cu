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
 * Re-implemented with ray-Gaussian intersection method gradients
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for each Gaussian.
// Not used for X-ray imaging.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (placeholder - gradients computed in renderCUDA)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	const float* dL_dmus,
	float3* dL_dmeans,
	float* dL_dcov,
	const int mode)
{
	// Gradients are computed in renderCUDA for the ray-Gaussian intersection method
	// This kernel is kept for compatibility but does minimal work
}

// Backward pass for the conversion of scale and rotation to a
// 3D covariance matrix for each Gaussian.
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
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

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}

// Backward pass of the preprocessing steps
template<int C>
__global__ void preprocessCUDA(
	int P,
	const float3* means,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	dL_dmeans[idx] += dL_dmean;

	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure with ray-Gaussian intersection gradients
template <uint32_t C>
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
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dmean3D,
	float* __restrict__ dL_dcov,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dmus,
	const float* __restrict__ depths,
	const float3* __restrict__ means,
	const float* __restrict__ cov3Ds,
	int P,
	const int* __restrict__ radii,
	const float* view_matrix,
	const float h_x,
	const float h_y,
	const float tan_fovx,
	const float tan_fovy)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float3 collected_cov3D1[BLOCK_SIZE];
	__shared__ float3 collected_cov3D2[BLOCK_SIZE];
	__shared__ float3 collected_ndcz[BLOCK_SIZE];

	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_cov3D1[block.thread_rank()] = cov3D1[coll_id];
			collected_cov3D2[block.thread_rank()] = cov3D2[coll_id];
			collected_ndcz[block.thread_rank()] = ndcz[coll_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Load full 3x3 NDC covariance and compute inverse
			float3 cov3Dtop = collected_cov3D1[j];
			float3 cov3Ddow = collected_cov3D2[j];
			glm::mat3 c3D;
			c3D[0][0] = cov3Dtop.x;
			c3D[0][1] = cov3Dtop.y;
			c3D[0][2] = cov3Dtop.z;
			c3D[1][0] = cov3Dtop.y;
			c3D[1][1] = cov3Ddow.x;
			c3D[1][2] = cov3Ddow.y;
			c3D[2][0] = cov3Dtop.z;
			c3D[2][1] = cov3Ddow.y;
			c3D[2][2] = cov3Ddow.z;

			// Regularized matrix inversion to avoid NaN gradients
			float det = glm::determinant(c3D);
			glm::mat3 cov_inv;
			if (fabs(det) < 1e-6f) {
				// Add small regularization to make matrix invertible
				// instead of returning zero matrix which causes NaN propagation
				float reg = 0.01f;
				c3D[0][0] += reg;
				c3D[1][1] += reg;
				c3D[2][2] += reg;
				cov_inv = glm::inverse(c3D);
			} else {
				cov_inv = glm::inverse(c3D);
			}

			// Load positions
			float2 xy_b = collected_xy[j];
			float3 ndc = collected_ndcz[j];
			float z_b = ndc.z;
			float3 P_b = { pix2Ndc(xy_b.x, W), pix2Ndc(xy_b.y, H) , z_b};
			float3 u_b = { pix2Ndc(pixf.x, W), pix2Ndc(pixf.y, H), 0};
			float3 a_b = make_float3(u_b.x - P_b.x, u_b.y - P_b.y, u_b.z - P_b.z);

			const float2 d = { xy_b.x - pixf.x, xy_b.y - pixf.y };

			// A = d^T * Sigma^{-1} * d
			float A_value = cov_inv[2][2];

			// B = a^T * Sigma^{-1} * d
			float3 b_pro;
			b_pro.x = cov_inv[0][2];
			b_pro.y = cov_inv[1][2];
			b_pro.z = cov_inv[2][2];
			float B_value = a_b.x * b_pro.x + a_b.y * b_pro.y + a_b.z * b_pro.z;

			// C = a^T * Sigma^{-1} * a
			float3 c_pro;
			c_pro.x = cov_inv[0][0] * a_b.x + cov_inv[0][1] * a_b.y + cov_inv[0][2] * a_b.z;
			c_pro.y = cov_inv[1][0] * a_b.x + cov_inv[1][1] * a_b.y + cov_inv[1][2] * a_b.z;
			c_pro.z = cov_inv[2][0] * a_b.x + cov_inv[2][1] * a_b.y + cov_inv[2][2] * a_b.z;
			float C_value = a_b.x * c_pro.x + a_b.y * c_pro.y + a_b.z * c_pro.z;

			// Calculate l
			float D_value = (B_value * B_value) - A_value * (C_value - 1);
			float l_b;
			const float epsilon = 1e-6f;
			if (D_value >= 0.0f && fabs(A_value) > epsilon) {
				l_b = 2 * sqrt(D_value) / A_value;
			} else {
				l_b = 0.0f;
			}

			// Sigmoid
			float k = 10.0f;
			float sigmoid_lb = 1.0f / (1.0f + expf(-k * (l_b - 0.5f)));

			// Load opacity
			const float4 con_o = collected_conic_opacity[j];

			// 2D Gaussian weight
			float2 xyp = collected_xy[j];
			float2 dp = { xyp.x - pixf.x, xyp.y - pixf.y };
			float power = -0.5f * (con_o.x * dp.x * dp.x + con_o.z * dp.y * dp.y) - con_o.y * dp.x * dp.y;
			if (power > 0.0f)
				continue;
			const float G = exp(power);

			// Final alpha
			const float alpha = con_o.w * sigmoid_lb * G;
			if (alpha < 0.00001f)
				continue;

			// Gradient calculation
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += 1.f * dL_dchannel;
			}

			// dL_dG = dL_dalpha * dalpha_dG = dL_dalpha * opacity * sigmoid_lb
			const float dL_dG = con_o.w * sigmoid_lb * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -1.0f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity
			atomicAdd(&(dL_dopacity[global_id]), sigmoid_lb * G * dL_dalpha);

			// Gradient w.r.t. l through sigmoid
			float d_sigmoid_lb = sigmoid_lb * (1.0f - sigmoid_lb) * k;
			float dL_dl_b = con_o.w * G * dL_dalpha * d_sigmoid_lb;
			atomicAdd(&(dL_dmus[global_id]), dL_dl_b);

			// Gradient calculation for covariance matrix
			if (global_id >= P || !(radii[global_id] > 0))
				continue;

			const float* cov3D = cov3Ds + 6 * global_id;
			float3 mean = means[global_id];
			float3 dL_dconic = { dL_dconic2D[global_id].x, dL_dconic2D[global_id].y, dL_dconic2D[global_id].w };
			float dL_dmu = dL_dmus[global_id];
			float3 t = transformPoint4x3(mean, view_matrix);

			glm::mat3 J;
			float x_grad_mul, y_grad_mul;

			const float limx = 1.3f * tan_fovx;
			const float limy = 1.3f * tan_fovy;
			const float txtz = t.x / t.z;
			const float tytz = t.y / t.z;
			t.x = min(limx, max(-limx, txtz)) * t.z;
			t.y = min(limy, max(-limy, tytz)) * t.z;

			x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
			y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

			const float l = sqrt(t.x * t.x + t.y * t.y + t.z * t.z);
			J = glm::mat3(
				h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
				0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
				t.x / l, t.y / l, t.z / l);

			glm::mat3 W = glm::mat3(
				view_matrix[0], view_matrix[4], view_matrix[8],
				view_matrix[1], view_matrix[5], view_matrix[9],
				view_matrix[2], view_matrix[6], view_matrix[10]);

			glm::mat3 Vrk_tmp = glm::mat3(
				cov3D[0], cov3D[1], cov3D[2],
				cov3D[1], cov3D[3], cov3D[4],
				cov3D[2], cov3D[4], cov3D[5]);

			glm::mat3 M = W * J;
			glm::mat3 cov_tmp = glm::transpose(M) * glm::transpose(Vrk_tmp) * M;

			cov_tmp[0][0] += 0.3f;
			cov_tmp[1][1] += 0.3f;
			float hata = cov_tmp[0][0];
			float hatb = cov_tmp[0][1];
			float hatc = cov_tmp[0][2];
			float hatd = cov_tmp[1][1];
			float hate = cov_tmp[1][2];
			float hatf = cov_tmp[2][2];

			float dL_dhata = 0, dL_dhatb = 0, dL_dhatc = 0, dL_dhatd = 0, dL_dhate = 0, dL_dhatf = 0;
			float denom = hata * hatd - hatb * hatb;
			float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

			if (denom2inv != 0.0f && l_b != 0.0f) {
				// exp(*) part gradient
				dL_dhata = denom2inv * (-hatd * hatd * dL_dconic.x + hatb * hatd * dL_dconic.y + (denom - hata * hatd) * dL_dconic.z);
				dL_dhatd = denom2inv * (-hata * hata * dL_dconic.z + hata * hatb * dL_dconic.y + (denom - hata * hatd) * dL_dconic.x);
				dL_dhatb = denom2inv * (2 * hatb * hatd * dL_dconic.x - (denom + 2 * hatb * hatb) * dL_dconic.y + 2 * hata * hatb * dL_dconic.z);

				// Gradient through ray-Gaussian intersection (matching ori's numerical behavior)
				float sqrt_D = sqrt(D_value);
				float dlb_dA = -(C_value - 1) / (A_value * sqrt_D) - 2 * sqrt_D / (A_value * A_value);
				float dlb_dB = 2 * B_value / (A_value * sqrt_D);
				float dlb_dC = -1 / sqrt_D;

				glm::vec3 a_b_vec(a_b.x, a_b.y, a_b.z);
				glm::vec3 inv_S = cov_inv * a_b_vec;
				float inv_S_x = inv_S.x;
				float inv_S_y = inv_S.y;
				float inv_S_z = inv_S.z;

				float dlb_dhata = -1 * (dlb_dA * cov_inv[2][0] * cov_inv[0][2] + dlb_dB * inv_S_x * cov_inv[0][2] + dlb_dC * inv_S_x * inv_S_x);
				float dlb_dhatb = -2 * (dlb_dA * cov_inv[2][0] * cov_inv[1][2] + dlb_dB * inv_S_x * cov_inv[1][2] + dlb_dC * inv_S_x * inv_S_y);
				float dlb_dhatc = -2 * (dlb_dA * cov_inv[2][0] * cov_inv[2][2] + dlb_dB * inv_S_x * cov_inv[2][2] + dlb_dC * inv_S_x * inv_S_z);
				float dlb_dhatd = -1 * (dlb_dA * cov_inv[2][1] * cov_inv[1][2] + dlb_dB * inv_S_y * cov_inv[1][2] + dlb_dC * inv_S_y * inv_S_y);
				float dlb_dhate = -2 * (dlb_dA * cov_inv[2][1] * cov_inv[2][2] + dlb_dB * inv_S_y * cov_inv[2][2] + dlb_dC * inv_S_y * inv_S_z);
				float dlb_dhatf = -1 * (dlb_dA * cov_inv[2][2] * cov_inv[2][2] + dlb_dB * inv_S_z * cov_inv[2][2] + dlb_dC * inv_S_z * inv_S_z);

				dL_dhata += dlb_dhata * dL_dmu;
				dL_dhatb += dlb_dhatb * dL_dmu;
				dL_dhatc += dlb_dhatc * dL_dmu;
				dL_dhatd += dlb_dhatd * dL_dmu;
				dL_dhate += dlb_dhate * dL_dmu;
				dL_dhatf += dlb_dhatf * dL_dmu;

				// Gradients w.r.t. 3D covariance
				atomicAdd(&dL_dcov[6 * global_id + 0], M[0][0]*M[0][0]*dL_dhata + M[0][0]*M[1][0]*dL_dhatb + M[0][0]*M[2][0]*dL_dhatc + M[1][0]*M[1][0]*dL_dhatd + M[1][0]*M[2][0]*dL_dhate + M[2][0]*M[2][0]*dL_dhatf);
				atomicAdd(&dL_dcov[6 * global_id + 3], M[0][1]*M[0][1]*dL_dhata + M[0][1]*M[1][1]*dL_dhatb + M[0][1]*M[2][1]*dL_dhatc + M[1][1]*M[1][1]*dL_dhatd + M[1][1]*M[2][1]*dL_dhate + M[2][1]*M[2][1]*dL_dhatf);
				atomicAdd(&dL_dcov[6 * global_id + 5], M[0][2]*M[0][2]*dL_dhata + M[0][2]*M[1][2]*dL_dhatb + M[0][2]*M[2][2]*dL_dhatc + M[1][2]*M[1][2]*dL_dhatd + M[1][2]*M[2][2]*dL_dhate + M[2][2]*M[2][2]*dL_dhatf);
				atomicAdd(&dL_dcov[6 * global_id + 1], 2*M[0][0]*M[0][1]*dL_dhata + (M[0][1]*M[1][0]+M[0][0]*M[1][1])*dL_dhatb + (M[0][1]*M[2][0]+M[0][0]*M[2][1])*dL_dhatc + 2*M[1][0]*M[1][1]*dL_dhatd + (M[1][1]*M[2][0]+M[1][0]*M[2][1])*dL_dhate + 2*M[2][0]*M[2][1]*dL_dhatf);
				atomicAdd(&dL_dcov[6 * global_id + 2], 2*M[0][0]*M[0][2]*dL_dhata + (M[0][2]*M[1][0]+M[0][0]*M[1][2])*dL_dhatb + (M[0][2]*M[2][0]+M[0][0]*M[2][2])*dL_dhatc + 2*M[1][0]*M[1][2]*dL_dhatd + (M[1][2]*M[2][0]+M[1][0]*M[2][2])*dL_dhate + 2*M[2][0]*M[2][2]*dL_dhatf);
				atomicAdd(&dL_dcov[6 * global_id + 4], 2*M[0][1]*M[0][2]*dL_dhata + (M[0][2]*M[1][1]+M[0][1]*M[1][2])*dL_dhatb + (M[0][2]*M[2][1]+M[0][1]*M[2][2])*dL_dhatc + 2*M[1][1]*M[1][2]*dL_dhatd + (M[1][2]*M[2][1]+M[1][1]*M[2][2])*dL_dhate + 2*M[2][1]*M[2][2]*dL_dhatf);

				// Gradient w.r.t. means through Jacobian
				float aa = cov3D[0];
				float bb = cov3D[1];
				float cc = cov3D[2];
				float dd = cov3D[3];
				float ee = cov3D[4];
				float ff = cov3D[5];

				float dL_dM00 = 2*(M[0][0]*aa + M[0][1]*bb + M[0][2]*cc)*dL_dhata + (M[1][0]*aa + M[1][1]*bb + M[1][2]*cc)*dL_dhatb + (M[2][0]*aa + M[2][1]*bb + M[2][2]*cc)*dL_dhatc;
				float dL_dM01 = 2*(M[0][0]*bb + M[0][1]*dd + M[0][2]*ee)*dL_dhata + (M[1][0]*bb + M[1][1]*dd + M[1][2]*ee)*dL_dhatb + (M[2][0]*bb + M[2][1]*dd + M[2][2]*ee)*dL_dhatc;
				float dL_dM02 = 2*(M[0][0]*cc + M[0][1]*ee + M[0][2]*ff)*dL_dhata + (M[1][0]*cc + M[1][1]*ee + M[1][2]*ff)*dL_dhatb + (M[2][0]*cc + M[2][1]*ee + M[2][2]*ff)*dL_dhatc;

				float dL_dM10 = (M[0][0]*aa + M[0][1]*bb + M[0][2]*cc)*dL_dhatb + 2*(M[1][0]*aa + M[1][1]*bb + M[1][2]*cc)*dL_dhatd + (M[2][0]*aa + M[2][1]*bb + M[2][2]*cc)*dL_dhate;
				float dL_dM11 = (M[0][0]*bb + M[0][1]*dd + M[0][2]*ee)*dL_dhatb + 2*(M[1][0]*bb + M[1][1]*dd + M[1][2]*ee)*dL_dhatd + (M[2][0]*bb + M[2][1]*dd + M[2][2]*ee)*dL_dhate;
				float dL_dM12 = (M[0][0]*cc + M[0][1]*ee + M[0][2]*ff)*dL_dhatb + 2*(M[1][0]*cc + M[1][1]*ee + M[1][2]*ff)*dL_dhatd + (M[2][0]*cc + M[2][1]*ee + M[2][2]*ff)*dL_dhate;

				float dL_dM20 = (M[0][0]*aa + M[0][1]*bb + M[0][2]*cc)*dL_dhatc + (M[1][0]*aa + M[1][1]*bb + M[1][2]*cc)*dL_dhate + 2*(M[2][0]*aa + M[2][1]*bb + M[2][2]*cc)*dL_dhatf;
				float dL_dM21 = (M[0][0]*bb + M[0][1]*dd + M[0][2]*ee)*dL_dhatc + (M[1][0]*bb + M[1][1]*dd + M[1][2]*ee)*dL_dhate + 2*(M[2][0]*bb + M[2][1]*dd + M[2][2]*ee)*dL_dhatf;
				float dL_dM22 = (M[0][0]*cc + M[0][1]*ee + M[0][2]*ff)*dL_dhatc + (M[1][0]*cc + M[1][1]*ee + M[1][2]*ff)*dL_dhate + 2*(M[2][0]*cc + M[2][1]*ee + M[2][2]*ff)*dL_dhatf;

				float dL_dJ00 = W[0][0]*dL_dM00 + W[0][1]*dL_dM01 + W[0][2]*dL_dM02;
				float dL_dJ02 = W[2][0]*dL_dM00 + W[2][1]*dL_dM01 + W[2][2]*dL_dM02;
				float dL_dJ11 = W[1][0]*dL_dM10 + W[1][1]*dL_dM11 + W[1][2]*dL_dM12;
				float dL_dJ12 = W[2][0]*dL_dM10 + W[2][1]*dL_dM11 + W[2][2]*dL_dM12;
				float dL_dJ20 = W[0][0]*dL_dM20 + W[0][1]*dL_dM21 + W[0][2]*dL_dM22;
				float dL_dJ21 = W[1][0]*dL_dM20 + W[1][1]*dL_dM21 + W[1][2]*dL_dM22;
				float dL_dJ22 = W[2][0]*dL_dM20 + W[2][1]*dL_dM21 + W[2][2]*dL_dM22;

				float tx = t.x;
				float ty = t.y;
				float tz = t.z;
				float inv_tz = 1.f / tz;
				float inv_tz2 = inv_tz * inv_tz;
				float inv_tz3 = inv_tz2 * inv_tz;
				float circledcirc = sqrt(tx * tx + ty * ty + tz * tz);
				float inv_circledcirc3 = 1 / (circledcirc * circledcirc * circledcirc);
				float dL_dtx = x_grad_mul * (-h_x*inv_tz2*dL_dJ02 + (1/circledcirc - tx*tx*inv_circledcirc3)*dL_dJ20 - tx*ty*inv_circledcirc3*dL_dJ21 - tx*tz*inv_circledcirc3*dL_dJ22);
				float dL_dty = y_grad_mul * (-h_y*inv_tz2*dL_dJ12 - tx*ty*inv_circledcirc3*dL_dJ20 + (1/circledcirc - ty*ty*inv_circledcirc3)*dL_dJ21 - ty*tz*inv_circledcirc3*dL_dJ22);
				float dL_dtz = -h_x*inv_tz2*dL_dJ00 + 2*h_x*tx*inv_tz3*dL_dJ02 - h_y*inv_tz2*dL_dJ11 + 2*h_y*ty*inv_tz3*dL_dJ12 - tx*tz*inv_circledcirc3*dL_dJ20 - ty*tz*inv_circledcirc3*dL_dJ21 + (1/circledcirc-tz*tz*inv_circledcirc3)*dL_dJ22;

				float3 dL_dmean_result = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

				atomicAdd(&dL_dmean3D[global_id].x, dL_dmean_result.x);
				atomicAdd(&dL_dmean3D[global_id].y, dL_dmean_result.y);
				atomicAdd(&dL_dmean3D[global_id].z, dL_dmean_result.z);
			}
		}
	}
}


void BACKWARD::preprocess(
	int P,
	const float3* means3D,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	const float* dL_dmu,
	glm::vec3* dL_dmean3D,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	const int mode)
{
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		dL_dmu,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		mode);

	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P,
		(float3*)means3D,
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcov3D,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float4* conic_opacity,
	const float3* cov3D1,
	const float3* cov3D2,
	const float3* ndcz,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float3* dL_dmean3D,
	float* dL_dcov3D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dmu,
	const float* depths,
	const float3* means3D,
	const float* cov3Ds,
	int P,
	const int* radii,
	const float* viewmatrix,
	const float focal_x,
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		means2D,
		conic_opacity,
		cov3D1,
		cov3D2,
		ndcz,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dmean3D,
		dL_dcov3D,
		dL_dconic2D,
		dL_dopacity,
		dL_dmu,
		depths,
		means3D,
		cov3Ds,
		P,
		radii,
		viewmatrix,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy);
}
