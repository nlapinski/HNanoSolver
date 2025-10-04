#pragma once

#include "../Utils/Stencils.hpp"
#include "Kernels.cuh"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"


__device__ __inline__ uint32_t hash_u32(uint32_t x, uint32_t y, uint32_t z) {
	uint32_t h = x * 374761393u + y * 668265263u + z * 2246822519u;
	h ^= h >> 13;
	h *= 1274126177u;
	h ^= h >> 16;
	return h;
}

__device__ __inline__ float u32_to_uniform01(uint32_t h) { return (h & 0x00FFFFFFu) * (1.0f / 16777216.0f); }

__device__ __inline__ int floordiv_int(int a, int b) {
	int q = a / b;
	int r = a - q * b;
	return (r != 0 && ((r > 0) != (b > 0))) ? (q - 1) : q;
}

__device__ __inline__ float saturate(float x) { return fminf(1.0f, fmaxf(0.0f, x)); }

__device__ __inline__ void gradientP_centered(const IndexSampler<float, 0>& pSampler, const nanovdb::Coord& c, float inv_dx, float& gx,
                                          float& gy, float& gz) {
	const float p_xp = pSampler(c + nanovdb::Coord(1, 0, 0));
	const float p_xm = pSampler(c - nanovdb::Coord(1, 0, 0));
	const float p_yp = pSampler(c + nanovdb::Coord(0, 1, 0));
	const float p_ym = pSampler(c - nanovdb::Coord(0, 1, 0));
	const float p_zp = pSampler(c + nanovdb::Coord(0, 0, 1));
	const float p_zm = pSampler(c - nanovdb::Coord(0, 0, 1));
	const float s = 0.5f * inv_dx;
	gx = (p_xp - p_xm) * s;
	gy = (p_yp - p_ym) * s;
	gz = (p_zp - p_zm) * s;
}

__device__ __inline__ void gradient_centered_density(const IndexSampler<float, 0>& dSampler, const nanovdb::Coord& c, const float inv_dx,
                                                 float& gx, float& gy, float& gz) {
	gx = (dSampler(c + nanovdb::Coord(1, 0, 0)) - dSampler(c - nanovdb::Coord(1, 0, 0))) * 0.5f * inv_dx;
	gy = (dSampler(c + nanovdb::Coord(0, 1, 0)) - dSampler(c - nanovdb::Coord(0, 1, 0))) * 0.5f * inv_dx;
	gz = (dSampler(c + nanovdb::Coord(0, 0, 1)) - dSampler(c - nanovdb::Coord(0, 0, 1))) * 0.5f * inv_dx;
}


// =====================
// Activity mask builder
// =====================
__global__ void build_activity_mask(const float* density, const nanovdb::Vec3f* vel, const float* src_density,
                                    const nanovdb::Vec3f* src_vel_v3, const float* src_vx, const float* src_vy, const float* src_vz,
                                    unsigned char* active, int N, float dens_eps, float vel_eps) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	bool on = false;
	if (density) {
		on |= fabsf(density[i]) > dens_eps;
	}
	if (vel) {
		const nanovdb::Vec3f v = vel[i];
		on |= (fabsf(v[0]) + fabsf(v[1]) + fabsf(v[2])) > vel_eps;
	}
	if (src_density) {
		on |= fabsf(src_density[i]) > 0.f;
	}
	if (src_vel_v3) {
		const nanovdb::Vec3f s = src_vel_v3[i];
		on |= (s[0] != 0.f || s[1] != 0.f || s[2] != 0.f);
	} else if (src_vx && src_vy && src_vz) {
		on |= (src_vx[i] != 0.f || src_vy[i] != 0.f || src_vz[i] != 0.f);
	}
	//active[i] = on ? 1u : 0u;
	active[i] = 1u;
}

// =====================================================
// Common helpers: SDF sampling, normals, boundary logic
// =====================================================
template <typename Vec3T>
__device__ float sampleSDF(const float* sdfData, const Vec3T& coord, const IndexSampler<float, 1>& sampler) {
	if (!sdfData) return 1.0f;
	return sampler(coord);
}

template <typename Vec3T>
__device__ nanovdb::Vec3f gradientSDF(const float* sdfData, const Vec3T& coord, const IndexSampler<float, 1>& sampler,
                                      const float inv_voxelSize) {
	if (!sdfData) return nanovdb::Vec3f(0);

	const float right = sampler(coord + Vec3T(1, 0, 0));
	const float left = sampler(coord + Vec3T(-1, 0, 0));
	const float top = sampler(coord + Vec3T(0, 1, 0));
	const float bottom = sampler(coord + Vec3T(0, -1, 0));
	const float front = sampler(coord + Vec3T(0, 0, 1));
	const float back = sampler(coord + Vec3T(0, 0, -1));

	return nanovdb::Vec3f(right - left, top - bottom, front - back) * (0.5f * inv_voxelSize);
}

template <typename Vec3T>
__device__ bool isInCollision(const float* sdfData, const Vec3T& pos, const IndexSampler<float, 1>& sampler, float threshold) {
	if (!sdfData) return false;
	const float sdfValue = sampleSDF(sdfData, pos, sampler);
	return sdfValue < threshold;
}

template <typename Vec3T>
__device__ nanovdb::Vec3f getSDFNormal(const float* sdfData, Vec3T& pos, const IndexSampler<float, 1>& sampler, float epsilon) {
	if (!sdfData) return nanovdb::Vec3f(0.0f);
	const nanovdb::Vec3f g = gradientSDF(sdfData, pos, sampler, epsilon);
	const float len = g.length();
	return len > 1e-6f ? g / len : nanovdb::Vec3f(0.0f);
}

__device__ nanovdb::Vec3f applySpecularReflection(const nanovdb::Vec3f& v, const nanovdb::Vec3f& n) {
	const float vdotn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
	return v - n * (2.0f * vdotn);
}

__device__ nanovdb::Vec3f applyNoSlipBoundary(const nanovdb::Vec3f& velocity, const nanovdb::Vec3f& normal) {
	const float vdotn = velocity[0] * normal[0] + velocity[1] * normal[1] + velocity[2] * normal[2];
	const nanovdb::Vec3f v_normal = normal * vdotn;
	const nanovdb::Vec3f v_tangent = velocity - v_normal;
	return v_tangent;  // strict no-slip would return (0,0,0)
}

// ================================
// Collision boundary enforcement
// ================================
__global__ void enforceCollisionBoundaries(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                           const nanovdb::Coord* __restrict__ coords, nanovdb::Vec3f* __restrict__ velocityData,
                                           const float* __restrict__ collisionSDF, const float voxelSize, size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;
	if (!collisionSDF) return;

	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const IndexSampler<float, 1> sdfSampler(idxSampler, collisionSDF);
	const nanovdb::Coord coord = coords[idx];

	const float sdf_value = sampleSDF(collisionSDF, coord, sdfSampler);

	if (sdf_value < 0.0f) {
		velocityData[idx] = nanovdb::Vec3f(0.0f);
		return;
	}

	const float collisionMargin = 0.1f;
	if (sdf_value < collisionMargin) {
		const nanovdb::Vec3f normal = getSDFNormal(collisionSDF, coord, sdfSampler, 1.0f / voxelSize);
		const float blend = 1.0f - (sdf_value / collisionMargin);
		const nanovdb::Vec3f velocity = velocityData[idx];
		const nanovdb::Vec3f modified_velocity = applyNoSlipBoundary(velocity, normal);
		velocityData[idx] = velocity * (1.0f - blend) + modified_velocity * blend;
	}
}

// =======================
// Advection (multi-field)
// =======================
__global__ void advect_scalars(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                               const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                               float** __restrict__ inDataArrays, float** __restrict__ outDataArrays, const int numScalars,
                               const float* __restrict__ collisionSDF, const bool hasCollision, const unsigned char* __restrict__ active,
                               const size_t totalVoxels, const float dt, const float inv_voxelSize) {

	IndexOffsetSampler<0> s_idxSampler(domainGrid);
	const IndexSampler<float, 1> sdfSampler(s_idxSampler, collisionSDF);

	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx >= totalVoxels || (active && !active[idx])) return;
	// advect_scalars(...) – add copy-through for inactive
	if (idx >= totalVoxels) return;
	if (active && !active[idx]) {
		// copy-through each scalar
		for (int s = 0; s < numScalars; ++s) outDataArrays[s][idx] = inDataArrays[s][idx];
		return;
	}



	const float scaled_dt = dt * inv_voxelSize;

	const nanovdb::Coord coord = coords[idx];
	uint64_t origIndex = s_idxSampler.offset(coord);
	origIndex = origIndex == 0 ? 0 : origIndex - 1;

	const nanovdb::Vec3f posCell = coord.asVec3s();
	const nanovdb::Vec3f velCenter = velocityData[origIndex];

	// backtrace
	nanovdb::Vec3f backPos = posCell - velCenter * scaled_dt;
	if (hasCollision && collisionSDF) {
		if (isInCollision(collisionSDF, backPos, sdfSampler, 0.0f)) backPos = posCell;
	}

	struct OptimizedInterpData {
		uint64_t indices[8];
		float weights[8];
	};

	auto setupInterpolation = [&](const nanovdb::Vec3f& pos) -> OptimizedInterpData {
		const float x = pos[0], y = pos[1], z = pos[2];
		const int i0 = __float2int_rd(x), i1 = i0 + 1;
		const int j0 = __float2int_rd(y), j1 = j0 + 1;
		const int k0 = __float2int_rd(z), k1 = k0 + 1;

		const float tx = x - i0, ty = y - j0, tz = z - k0;
		const float itx = 1.0f - tx, ity = 1.0f - ty, itz = 1.0f - tz;

		const float w00 = itx * ity, w10 = tx * ity, w01 = itx * ty, w11 = tx * ty;

		OptimizedInterpData data{};
		data.weights[0] = w00 * itz;
		data.weights[1] = w10 * itz;
		data.weights[2] = w01 * itz;
		data.weights[3] = w11 * itz;
		data.weights[4] = w00 * tz;
		data.weights[5] = w10 * tz;
		data.weights[6] = w01 * tz;
		data.weights[7] = w11 * tz;

		const nanovdb::Coord c[8] = {{i0, j0, k0}, {i1, j0, k0}, {i0, j1, k0}, {i1, j1, k0},
		                             {i0, j0, k1}, {i1, j0, k1}, {i0, j1, k1}, {i1, j1, k1}};
#pragma unroll
		for (int i = 0; i < 8; ++i) {
			const uint64_t off = s_idxSampler.offset(c[i]);
			data.indices[i] = off == 0 ? 0 : off - 1;
		}
		return data;
	};

	OptimizedInterpData backPosData = setupInterpolation(backPos);

	nanovdb::Vec3f velF(0.0f);
#pragma unroll
	for (int j = 0; j < 8; ++j) {
		const nanovdb::Vec3f v = velocityData[backPosData.indices[j]];
		velF = velF + v * backPosData.weights[j];
	}

	nanovdb::Vec3f fwdPos2 = backPos + velF * scaled_dt;
	if (hasCollision && collisionSDF) {
		bool fwdInCollision = isInCollision(collisionSDF, fwdPos2, sdfSampler, 0.0f);
		fwdPos2 = fwdInCollision ? backPos : fwdPos2;
	}

	OptimizedInterpData fwdPos2Data = setupInterpolation(fwdPos2);

	static __device__ const int3 offs[6] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
	uint32_t nbrIdx[6];
#pragma unroll
	for (int n = 0; n < 6; ++n) {
		uint32_t offset = s_idxSampler.offset(coord[0] + offs[n].x, coord[1] + offs[n].y, coord[2] + offs[n].z);
		nbrIdx[n] = offset == 0 ? 0 : offset - 1;
	}

	for (int s = 0; s < numScalars; ++s) {
		const float* __restrict__ inData = inDataArrays[s];
		float* __restrict__ outData = outDataArrays[s];

		const float phiOrig = inData[origIndex];

		float phiForward = 0.0f;
		float phiBackward = 0.0f;
#pragma unroll
		for (int j = 0; j < 8; ++j) {
			phiForward = __fmaf_rn(inData[backPosData.indices[j]], backPosData.weights[j], phiForward);
			phiBackward = __fmaf_rn(inData[fwdPos2Data.indices[j]], fwdPos2Data.weights[j], phiBackward);
		}

		const float error = phiOrig - phiBackward;
		float phiCorr = __fmaf_rn(0.5f, error, phiForward);

		float minVal = phiOrig, maxVal = phiOrig;
#pragma unroll
		for (int n = 0; n < 6; ++n) {
			const float val = inData[nbrIdx[n]];
			minVal = fminf(minVal, val);
			maxVal = fmaxf(maxVal, val);
		}
		minVal = fminf(minVal, phiForward);
		maxVal = fmaxf(maxVal, phiForward);

		outData[idx] = fmaxf(minVal, fminf(phiCorr, maxVal));
	}
}

// ==================
// Advection (scalar)
// ==================
__global__ void advect_scalar(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                              const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                              const float* __restrict__ inData, float* __restrict__ outData, const float* __restrict__ collisionSDF,
                              const bool hasCollision, const size_t totalVoxels, const float dt, const float inv_voxelSize,
                              const unsigned char* __restrict__ active) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels || (active && !active[idx])) return;

	const float scaled_dt = dt * inv_voxelSize;

	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto sdfSampler = IndexSampler<float, 1>(idxSampler, collisionSDF);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);
	const auto dataSampler = IndexSampler<float, 1>(idxSampler, inData);

	const nanovdb::Coord coord = coords[idx];
	const nanovdb::Vec3f posCell = coord.asVec3s();
	const float phiOrig = dataSampler(coord);

	const nanovdb::Vec3f velCenter = velocitySampler(coord);

	nanovdb::Vec3f backPos = posCell - velCenter * scaled_dt;
	if (hasCollision && collisionSDF) {
		if (isInCollision(collisionSDF, backPos, sdfSampler, 0.0f)) backPos = posCell;
	}
	const float phiForward = dataSampler(backPos);

	const nanovdb::Vec3f velF = velocitySampler(backPos);
	nanovdb::Vec3f fwdPos2 = backPos + velF * scaled_dt;
	if (hasCollision && collisionSDF) {
		if (isInCollision(collisionSDF, fwdPos2, sdfSampler, 0.0f)) fwdPos2 = backPos;
	}
	const float phiBackward = dataSampler(fwdPos2);

	const float error = phiOrig - phiBackward;
	float phiCorr = phiForward + 0.5f * error;

	float minVal = phiOrig, maxVal = phiOrig;
	for (int dim = 0; dim < 3; ++dim) {
		for (int off = -1; off <= 1; off += 2) {
			nanovdb::Coord n = coord;
			n[dim] += off;
			const float v = dataSampler(n);
			minVal = fminf(minVal, v);
			maxVal = fmaxf(maxVal, v);
		}
	}
	minVal = fminf(minVal, phiForward);
	maxVal = fmaxf(maxVal, phiForward);
	outData[idx] = fmaxf(minVal, fminf(phiCorr, maxVal));
}

// =================
// Advection (vec3)
// =================
__global__ void advect_vector(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                              const nanovdb::Coord* __restrict__ coords, const nanovdb::Vec3f* __restrict__ velocityData,
                              nanovdb::Vec3f* __restrict__ outVelocity, const float* __restrict__ collisionSDF, const bool hasCollision,
                              const unsigned char* __restrict__ active, const size_t totalVoxels, const float dt,
                              const float inv_voxelSize) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx >= totalVoxels || (active && !active[idx])) return;
	if (idx >= totalVoxels) return;
	if (active && !active[idx]) {
		outVelocity[idx] = velocityData[idx];
		return;
	}

	const float scaled_dt = dt * inv_voxelSize;

	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 1>(idxSampler, velocityData);
	const auto sdfSampler = IndexSampler<float, 1>(idxSampler, collisionSDF);

	const nanovdb::Coord coord = coords[idx];
	const nanovdb::Vec3f pos = coord.asVec3s();

	const nanovdb::Vec3f velOrig = velocitySampler(coord);

	nanovdb::Vec3f backPos = pos - velOrig * scaled_dt;
	if (hasCollision && collisionSDF) {
		if (isInCollision(collisionSDF, backPos, sdfSampler, 0.0f)) backPos = pos;
	}

	const nanovdb::Vec3f velForward = velocitySampler(backPos);
	nanovdb::Vec3f fwdPos2 = backPos + velForward * scaled_dt;
	if (hasCollision && collisionSDF) {
		if (isInCollision(collisionSDF, fwdPos2, sdfSampler, 0.0f)) fwdPos2 = backPos;
	}

	const nanovdb::Vec3f velBackward = velocitySampler(fwdPos2);

	const nanovdb::Vec3f errorVec = velOrig - velBackward;
	nanovdb::Vec3f velCorr = velForward + 0.5f * errorVec;

	nanovdb::Vec3f minVel, maxVel;
	for (int c = 0; c < 3; ++c) {
		minVel[c] = velOrig[c];
		maxVel[c] = velOrig[c];
	}

	for (int dim = 0; dim < 3; ++dim) {
		for (int off = -1; off <= 1; off += 2) {
			nanovdb::Coord n = coord;
			n[dim] += off;
			const nanovdb::Vec3f vn = velocitySampler(n);
			for (int c = 0; c < 3; ++c) {
				minVel[c] = fminf(minVel[c], vn[c]);
				maxVel[c] = fmaxf(maxVel[c], vn[c]);
			}
		}
	}

	for (int c = 0; c < 3; ++c) {
		minVel[c] = fminf(minVel[c], velForward[c]);
		maxVel[c] = fmaxf(maxVel[c], velForward[c]);
		velCorr[c] = fmaxf(minVel[c], fminf(velCorr[c], maxVel[c]));
	}

	if (hasCollision && collisionSDF) {
		const float sdf_value = sampleSDF(collisionSDF, coord, sdfSampler);
		if (sdf_value < 0.0f) {
			velCorr = nanovdb::Vec3f(0.0f);
		} else if (sdf_value < 0.1f) {
			const nanovdb::Vec3f normal = getSDFNormal(collisionSDF, coord, sdfSampler, inv_voxelSize);
			const float blend = 1.0f - (sdf_value / 1.5f);
			const nanovdb::Vec3f no_slip = applyNoSlipBoundary(velCorr, normal);
			velCorr = velCorr * (1.0f - blend) + no_slip * blend;
		}
	}

	outVelocity[idx] = velCorr;
}


// ====================
// Divergence (masked)
// ====================
__global__ void divergence(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                           const nanovdb::Coord* __restrict__ d_coord, const nanovdb::Vec3f* __restrict__ velocityData,
                           float* __restrict__ outDivergence, const float inv_dx, const unsigned char* __restrict__ active,
                           const size_t totalVoxels) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels || (active && !active[tid])) return;

	const nanovdb::Vec3f center = velocityData[tid];

	const nanovdb::Coord c = d_coord[tid];
	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 0>(idxSampler, velocityData);

	const float xp = (center + velocitySampler(c + nanovdb::Coord(1, 0, 0)))[0] * 0.5f;
	const float xm = (center + velocitySampler(c - nanovdb::Coord(1, 0, 0)))[0] * 0.5f;
	const float yp = (center + velocitySampler(c + nanovdb::Coord(0, 1, 0)))[1] * 0.5f;
	const float ym = (center + velocitySampler(c - nanovdb::Coord(0, 1, 0)))[1] * 0.5f;
	const float zp = (center + velocitySampler(c + nanovdb::Coord(0, 0, 1)))[2] * 0.5f;
	const float zm = (center + velocitySampler(c - nanovdb::Coord(0, 0, 1)))[2] * 0.5f;

	outDivergence[tid] = (xp - xm + yp - ym + zp - zm) * inv_dx;
}


// ==============================
// Red-Black Gauss-Seidel (mask)
// ==============================
__global__ void redBlackGaussSeidelUpdate(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                          const nanovdb::Coord* __restrict__ d_coord, const float* __restrict__ divergence,
                                          float* __restrict__ pressure, const float dx, const size_t totalVoxels, const int color,
                                          const float omega, const unsigned char* __restrict__ active) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels || (active && !active[tid])) return;

	const nanovdb::Coord c = d_coord[tid];
	const int i = c.x(), j = c.y(), k = c.z();

	if (((i + j + k) & 1) != color) return;

	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto pSampler = IndexSampler<float, 0>(idxSampler, pressure);

	const float dx2 = dx * dx;
	constexpr float inv6 = 0.166666667f;

	const float pxp1 = pSampler(nanovdb::Coord(i + 1, j, k));
	const float pxm1 = pSampler(nanovdb::Coord(i - 1, j, k));
	const float pyp1 = pSampler(nanovdb::Coord(i, j + 1, k));
	const float pym1 = pSampler(nanovdb::Coord(i, j - 1, k));
	const float pzp1 = pSampler(nanovdb::Coord(i, j, k + 1));
	const float pzm1 = pSampler(nanovdb::Coord(i, j, k - 1));

	const float divVal = divergence[tid];
	const float pOld = pressure[tid];

	const float pGS = ((pxp1 + pxm1 + pyp1 + pym1 + pzp1 + pzm1) - divVal * dx2) * inv6;
	pressure[tid] = pOld + omega * (pGS - pOld);
}


// ==========================================
// Subtract Pressure Gradient (masked + SDF)
// ==========================================
__global__ void subtractPressureGradient(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coords,
                                         const size_t totalVoxels, const nanovdb::Vec3f* velocity, const float* pressure,
                                         nanovdb::Vec3f* out, const float* __restrict__ collisionSDF, const bool hasCollision,
                                         const float inv_voxelSize, const unsigned char* __restrict__ active) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	//if (tid >= totalVoxels || (active && !active[tid])) return;
	//  subtractPressureGradient(...) – add copy-through for inactive
	if (tid >= totalVoxels) return;
	if (active && !active[tid]) {
		out[tid] = velocity[tid];
		return;
	}


	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto pressureSampler = IndexSampler<float, 0>(idxSampler, pressure);
	const auto sdfSampler = IndexSampler<float, 1>(idxSampler, collisionSDF);

	const nanovdb::Coord c = d_coords[tid];

	const nanovdb::Vec3f u_star_c = velocity[tid];

	const float p_xp = pressureSampler(c + nanovdb::Coord(1, 0, 0));
	const float p_xm = pressureSampler(c - nanovdb::Coord(1, 0, 0));
	const float p_yp = pressureSampler(c + nanovdb::Coord(0, 1, 0));
	const float p_ym = pressureSampler(c - nanovdb::Coord(0, 1, 0));
	const float p_zp = pressureSampler(c + nanovdb::Coord(0, 0, 1));
	const float p_zm = pressureSampler(c - nanovdb::Coord(0, 0, 1));

	const nanovdb::Vec3f gradP_c = nanovdb::Vec3f(p_xp - p_xm, p_yp - p_ym, p_zp - p_zm) * (0.5f * inv_voxelSize);

	nanovdb::Vec3f u_final_c = u_star_c - gradP_c;

	if (hasCollision && collisionSDF) {
		const float sdf_value = sampleSDF(collisionSDF, c, sdfSampler);
		if (sdf_value < 0.0f) {
			u_final_c = nanovdb::Vec3f(0.0f);
		} else if (sdf_value < 0.1f) {
			const nanovdb::Vec3f normal = getSDFNormal(collisionSDF, c, sdfSampler, inv_voxelSize);
			const float blend = 1.0f - (sdf_value / 0.1f);
			const nanovdb::Vec3f no_slip = applyNoSlipBoundary(u_final_c, normal);
			u_final_c = u_final_c * (1.0f - blend) + no_slip * blend;
		}
	}

	out[tid] = u_final_c;
}

// ======================
// Temperature buoyancy
// ======================
__global__ void temperature_buoyancy(const nanovdb::Vec3f* __restrict__ velocityData, const float* __restrict__ tempData,
                                     nanovdb::Vec3f* __restrict__ outVel, const float dt, const float ambient_temp,
                                     const float buoyancy_strength, const unsigned char* __restrict__ active, const size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels || (active && !active[idx])) return;

	const nanovdb::Vec3f vel = velocityData[idx];
	const float temp = tempData[idx];
	if (temp <= ambient_temp) {
		outVel[idx] = vel;
		return;
	}

	const float tempDiff = temp - ambient_temp;
	const nanovdb::Vec3f buoyancyForce(0.0f, fmaxf(0.0f, tempDiff * buoyancy_strength), 0.0f);

	outVel[idx] = vel + buoyancyForce * dt;
}

// ================
// Combustion step
// ================
__global__ void combustion(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ d_coords,
                           const float* __restrict__ fuelData, const float* __restrict__ tempData, float* __restrict__ outFuel,
                           float* __restrict__ outTemp, const float dt, float ignition_temp, float combustion_rate, float heat_release,
                           size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const float fuel = fuelData[idx];
	const float temp = tempData[idx];
	float newFuel = fuel;
	float newTemp = temp;

	if (fuel > 0.0f && temp >= ignition_temp) {
		const float fuelBurned = fminf(fuel, combustion_rate * dt);
		newFuel -= fuelBurned;
		newTemp += fuelBurned * heat_release;
	}

	outFuel[idx] = newFuel;
	outTemp[idx] = newTemp;
}

// ================
// Diffusion step
// ================
__global__ void diffusion(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* __restrict__ d_coords,
                          const float* tempData, const float* fuelData, float* outTemp, float* outFuel, const float dt, float temp_diff,
                          float fuel_diff, float ambient_temp, size_t totalVoxels) {
	const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalVoxels) return;

	const IndexOffsetSampler<0> sampler(domainGrid);
	const auto tempSampler = IndexSampler<float, 0>(sampler, tempData);
	const auto fuelSampler = IndexSampler<float, 0>(sampler, fuelData);

	const nanovdb::Coord coord = d_coords[idx];
	const float centerTemp = tempSampler(coord);
	const float centerFuel = fuelSampler(coord);

	float tempLaplacian = 0.0f;
	float fuelLaplacian = 0.0f;
	int neighbors = 0;

	nanovdb::Coord offsets[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

	for (auto offset : offsets) {
		const nanovdb::Coord n = coord + offset;

		const float tN = tempSampler(n);
		if (tN == 0.0f) continue;

		const float fN = fuelSampler(n);
		if (fN == 0.0f) continue;

		tempLaplacian += (tN - centerTemp);
		fuelLaplacian += (fN - centerFuel);
		neighbors++;
	}

	if (neighbors > 0) {
		outTemp[idx] = centerTemp + temp_diff * dt * tempLaplacian;
		outFuel[idx] = centerFuel + fuel_diff * dt * fuelLaplacian;
	} else {
		outTemp[idx] = centerTemp;
		outFuel[idx] = centerFuel;
	}

	outTemp[idx] += (ambient_temp - outTemp[idx]) * dt * 0.1f;
}

// =======================
// Combustion + oxygening
// =======================
__global__ void combustion_oxygen(const float* __restrict__ fuelData, const float* __restrict__ wasteData,
                                  const float* __restrict__ temperatureData, float* __restrict__ divergenceData,
                                  const float* __restrict__ flameData, float* __restrict__ outFuel, float* __restrict__ outWaste,
                                  float* __restrict__ outTemperature, float* __restrict__ outFlame, const float temp_gain,
                                  const float expansion, const unsigned char* __restrict__ active, int N) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N || (active && !active[idx])) return;

	float fuel = fuelData[idx];
	float waste = wasteData[idx];
	float temperature = temperatureData[idx];
	float flame = flameData[idx];

	if (fuel < 0.001f) fuel = 0.0f;

	float oxygen = 1.0f - fuel - waste;
	if (oxygen < 0.0f) {
		outFuel[idx] = fuel;
		outWaste[idx] = waste;
		outTemperature[idx] = temperature;
		outFlame[idx] = flame;
		return;
	}

	float burn = fminf(oxygen, fuel);

	float newFuel = fuel - burn;
	float newWaste = waste + burn * 2.0f;
	float newFlame = fmaxf(flame, fminf(1.0f, burn * 10.0f));
	float newTemperature = temperature + burn * temp_gain;

	outFuel[idx] = newFuel;
	outWaste[idx] = newWaste;
	outTemperature[idx] = newTemperature;
	divergenceData[idx] += burn * expansion;
	outFlame[idx] = newFlame;
}

// ========================
// Vorticity confinement
// ========================
__global__ void vorticityConfinement(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
                                     const nanovdb::Coord* __restrict__ d_coord, const nanovdb::Vec3f* __restrict__ velocityData,
                                     nanovdb::Vec3f* __restrict__ outForce, const float dt, const float inv_dx,
                                     const float confinementScale, const float factorScale, const unsigned char* __restrict__ active,
                                     const size_t totalVoxels) {
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= totalVoxels || (active && !active[tid])) return;

	const nanovdb::Coord c = d_coord[tid];

	const IndexOffsetSampler<0> idxSampler(domainGrid);
	const auto velocitySampler = IndexSampler<nanovdb::Vec3f, 0>(idxSampler, velocityData);
	const float factor = 0.5f * inv_dx;

	const nanovdb::Vec3f u_pX = velocitySampler(c + nanovdb::Coord(1, 0, 0));
	const nanovdb::Vec3f u_mX = velocitySampler(c - nanovdb::Coord(1, 0, 0));
	const nanovdb::Vec3f u_pY = velocitySampler(c + nanovdb::Coord(0, 1, 0));
	const nanovdb::Vec3f u_mY = velocitySampler(c - nanovdb::Coord(0, 1, 0));
	const nanovdb::Vec3f u_pZ = velocitySampler(c + nanovdb::Coord(0, 0, 1));
	const nanovdb::Vec3f u_mZ = velocitySampler(c - nanovdb::Coord(0, 0, 1));

	const float omega_x = ((u_pY[2] - u_mY[2]) - (u_pZ[1] - u_mZ[1])) * factor;
	const float omega_y = ((u_pZ[0] - u_mZ[0]) - (u_pX[2] - u_mX[2])) * factor;
	const float omega_z = ((u_pX[1] - u_mX[1]) - (u_pY[0] - u_mY[0])) * factor;

	const float vortMag_pX = computeVorticityMag(velocitySampler, c + nanovdb::Coord(factorScale, 0, 0), factor);
	const float vortMag_mX = computeVorticityMag(velocitySampler, c - nanovdb::Coord(factorScale, 0, 0), factor);
	const float vortMag_pY = computeVorticityMag(velocitySampler, c + nanovdb::Coord(0, factorScale, 0), factor);
	const float vortMag_mY = computeVorticityMag(velocitySampler, c - nanovdb::Coord(0, factorScale, 0), factor);
	const float vortMag_pZ = computeVorticityMag(velocitySampler, c + nanovdb::Coord(0, 0, factorScale), factor);
	const float vortMag_mZ = computeVorticityMag(velocitySampler, c - nanovdb::Coord(0, 0, factorScale), factor);

	const float grad_x = (vortMag_pX - vortMag_mX) * 0.5f * inv_dx;
	const float grad_y = (vortMag_pY - vortMag_mY) * 0.5f * inv_dx;
	const float grad_z = (vortMag_pZ - vortMag_mZ) * 0.5f * inv_dx;

	const float gradLen = sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z) + 1e-5f;
	const float Nx = grad_x / gradLen;
	const float Ny = grad_y / gradLen;
	const float Nz = grad_z / gradLen;

	outForce[tid] = velocityData[tid] + nanovdb::Vec3f{confinementScale * (Ny * omega_z - Nz * omega_y),
	                                                   confinementScale * (Nz * omega_x - Nx * omega_z),
	                                                   confinementScale * (Nx * omega_y - Ny * omega_x)} *
	                                        dt;
}

// =======================
// Source compositing ops
// =======================
__global__ void add_scalar_source(float* __restrict__ field, const float* __restrict__ src, float dt,
                                  const unsigned char* __restrict__ active, int N) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || (active && !active[i])) return;
	field[i] += dt * src[i];
}

__global__ void add_velocity_source(nanovdb::Vec3f* __restrict__ vel, const nanovdb::Vec3f* __restrict__ src, float dt,
                                    const unsigned char* __restrict__ active, int N) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || (active && !active[i])) return;
	vel[i] += src[i] * dt;
}

__global__ void add_velocity_source_xyz(nanovdb::Vec3f* __restrict__ vel, const float* __restrict__ sx, const float* __restrict__ sy,
                                        const float* __restrict__ sz, float dt, const unsigned char* __restrict__ active, int N) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || (active && !active[i])) return;
	nanovdb::Vec3f v = vel[i];
	v[0] += dt * sx[i];
	v[1] += dt * sy[i];
	v[2] += dt * sz[i];
	vel[i] = v;
}

// ==============
// Disturbances
// ==============

//___global__ void inject_edge_disturbance_vel_from_density(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
//                                                          const nanovdb::Coord* __restrict__ d_coords, const float* __restrict__ density,
//                                                          const float densMin, const float densMax, const float* __restrict__ collisionSDF,
//                                                          const bool hasCollision, nanovdb::Vec3f* __restrict__ inoutVel,
//                                                          const size_t totalVoxels, const float dt, const float inv_dx,
//                                                          const float gradThresh, const float strength, const float swirl,
//                                                          const int patternBlock, const unsigned char* __restrict__ active) {
//	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//	if (tid >= totalVoxels || (active && !active[tid])) return;
//
//	const IndexOffsetSampler<0> idxSampler(domainGrid);
//	const auto dSampler = IndexSampler<float, 0>(idxSampler, density);
//	const auto sdfSampler = IndexSampler<float, 1>(idxSampler, collisionSDF);
//
//	const nanovdb::Coord c = d_coords[tid];
//
//	if (hasCollision && collisionSDF) {
//		if (sdfSampler(c) < 0.0f) return;
//	}
//
//	const float dens = dSampler(c);
//	const float mask = (densMax > densMin) ? saturate((dens - densMin) / (densMax - densMin)) : (dens > densMin ? 1.0f : 0.0f);
//	if (mask <= 0.0f) return;
//
//	float gx, gy, gz;
//	gradient_centered_density(dSampler, c, inv_dx, gx, gy, gz);
//
//	const float g2 = gx * gx + gy * gy + gz * gz;
//	if (g2 < gradThresh * gradThresh) return;
//
//	const float g = sqrtf(g2) + 1e-8f;
//	float nx = gx / g, ny = gy / g, nz = gz / g;
//
//	const int bs = max(1, patternBlock);
//	const int bx = floordiv_int(c.x(), bs);
//	const int by = floordiv_int(c.y(), bs);
//	const int bz = floordiv_int(c.z(), bs);
//
//	const uint32_t h = hash_u32((uint32_t)bx, (uint32_t)by, (uint32_t)bz);
//
//	float rx = u32_to_uniform01(h) * 2.0f - 1.0f;
//	float ry = u32_to_uniform01(h * 1664525u + 1013904223u) * 2.0f - 1.0f;
//	float rz = u32_to_uniform01(h * 22695477u + 1u) * 2.0f - 1.0f;
//
//	const float rdotn = rx * nx + ry * ny + rz * nz;
//	rx -= rdotn * nx;
//	ry -= rdotn * ny;
//	rz -= rdotn * nz;
//	const float rl = sqrtf(rx * rx + ry * ry + rz * rz) + 1e-8f;
//	rx /= rl;
//	ry /= rl;
//	rz /= rl;
//
//	float dirx = nx + swirl * (rx - nx);
//	float diry = ny + swirl * (ry - ny);
//	float dirz = nz + swirl * (rz - nz);
//	const float dl = sqrtf(dirx * dirx + diry * diry + dirz * dirz) + 1e-8f;
//	dirx /= dl;
//	diry /= dl;
//	dirz /= dl;
//
//	const float s = (strength * dt) * mask;
//	nanovdb::Vec3f v = inoutVel[tid];
//	v[0] += dirx * s;
//	v[1] += diry * s;
//	v[2] += dirz * s;
//	inoutVel[tid] = v;
//}
//

// ==============
// Gravity (mask)
// ==============
__global__ void add_gravity(nanovdb::Vec3f* __restrict__ vel, float g, float dt, const unsigned char* __restrict__ active, int N) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || (active && !active[i])) return;
	vel[i][1] -= g * dt;
}
