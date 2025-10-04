#include <openvdb/Types.h>

#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"
#include "Kernels.cuh"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"
#include "nanovdb/util/GridHandle.h"

#include <unordered_map>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <utility>

static inline bool is_source(const std::string& n) { return n.rfind("src_", 0) == 0; }

struct StreamOwner {
	cudaStream_t stream = nullptr;
	~StreamOwner() {
		if (stream) {
			(void)cudaStreamDestroy(stream);
		}
	}
};

struct SimContext {
	StreamOwner so;
	size_t N = 0;
	float voxelSize = 0.f;
	const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* d_grid = nullptr;

	DeviceMemory<nanovdb::Coord> d_coords;
	DeviceMemory<nanovdb::Vec3f> d_velocity;
	DeviceMemory<nanovdb::Vec3f> d_advectedVel;
	DeviceMemory<float> d_divergence;
	DeviceMemory<float> d_pressure;
	DeviceMemory<float> d_collisionSDF;

	DeviceMemory<nanovdb::Vec3f> d_src_vel_vec3;
	DeviceMemory<float> d_src_vel_x, d_src_vel_y, d_src_vel_z;

	std::unordered_map<std::string, DeviceMemory<float>> d_src;
	std::unordered_map<std::string, DeviceMemory<float>> d_state;
};


extern "C" void HNS_DestroyContext(void** ctx) {
	if (!ctx || !*ctx) return;
	auto* C = reinterpret_cast<SimContext*>(*ctx);
	delete C;
	*ctx = nullptr;
}

extern "C" void HNS_EnsureContext(void** ctx, HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle,
                                  float voxelSize, bool hasCollision) {
	const size_t N = data.size();
	if (!N) throw std::runtime_error("HNS_EnsureContext: N==0");

	auto* C = reinterpret_cast<SimContext*>(*ctx);
	const bool needNew = (!C) || (C->N != N) || (C->voxelSize != voxelSize);

	// Always derive the device grid pointer from the CACHE-OWNED handle
	const auto* d_grid = handle.template deviceGrid<nanovdb::ValueOnIndex>();
	if (!d_grid) throw std::runtime_error("HNS_EnsureContext: deviceGrid() returned null");

	if (needNew) {
		if (C) {
			delete C;
			C = nullptr;
		}
		C = new SimContext();
		*ctx = C;

		C->N = N;
		C->voxelSize = voxelSize;
		C->d_grid = d_grid;

		CUDA_CHECK(cudaStreamCreate(&C->so.stream));

		C->d_coords = DeviceMemory<nanovdb::Coord>(N, C->so.stream);
		C->d_velocity = DeviceMemory<nanovdb::Vec3f>(N, C->so.stream);
		C->d_advectedVel = DeviceMemory<nanovdb::Vec3f>(N, C->so.stream);
		C->d_divergence = DeviceMemory<float>(N, C->so.stream);
		C->d_pressure = DeviceMemory<float>(N, C->so.stream);

		C->d_state.clear();

		// coords
		CUDA_CHECK(cudaMemcpyAsync(C->d_coords.get(), data.pCoords(), N * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice, C->so.stream));

		// velocity
		{
			const auto vecNames = data.getBlocksOfType<openvdb::Vec3f>();
			if (vecNames.empty()) throw std::runtime_error("No Vec3f blocks in feedback");
			std::string velName = vecNames[0];
			for (const auto& n : vecNames)
				if (n == "velocity") {
					velName = n;
					break;
				}
			const auto* h_vel = reinterpret_cast<const nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(velName));
			CUDA_CHECK(cudaMemcpyAsync(C->d_velocity.get(), h_vel, N * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, C->so.stream));
		}

		// persistent scalars (exclude src_* and collision_sdf)
		for (const auto& name : data.getBlocksOfType<float>()) {
			if (name == "collision_sdf" || is_source(name)) continue;
			const float* h = data.pValues<float>(name);
			DeviceMemory<float> d(N, C->so.stream);
			CUDA_CHECK(cudaMemcpyAsync(d.get(), h, N * sizeof(float), cudaMemcpyHostToDevice, C->so.stream));
			C->d_state.emplace(name, std::move(d));
		}

		// collision sdf (optional)
		if (hasCollision) {
			const auto& floats = data.getBlocksOfType<float>();
			if (std::find(floats.begin(), floats.end(), "collision_sdf") != floats.end()) {
				C->d_collisionSDF = DeviceMemory<float>(N, C->so.stream);
				CUDA_CHECK(cudaMemcpyAsync(C->d_collisionSDF.get(), data.pValues<float>("collision_sdf"), N * sizeof(float),
				                           cudaMemcpyHostToDevice, C->so.stream));
			} else {
				C->d_collisionSDF = DeviceMemory<float>();  // empty
			}
		} else {
			C->d_collisionSDF = DeviceMemory<float>();  // empty
		}

		CUDA_CHECK(cudaStreamSynchronize(C->so.stream));
		return;
	}

	// Reuse existing allocations; just refresh grid pointer and (optionally) collision field.
	C->d_grid = d_grid;

	if (hasCollision) {
		const auto& floats = data.getBlocksOfType<float>();
		if (std::find(floats.begin(), floats.end(), "collision_sdf") != floats.end()) {
			if (!C->d_collisionSDF.get()) C->d_collisionSDF = DeviceMemory<float>(N, C->so.stream);
			CUDA_CHECK(cudaMemcpyAsync(C->d_collisionSDF.get(), data.pValues<float>("collision_sdf"), N * sizeof(float),
			                           cudaMemcpyHostToDevice, C->so.stream));
			CUDA_CHECK(cudaStreamSynchronize(C->so.stream));
		} else {
			C->d_collisionSDF = DeviceMemory<float>();  // drop if not provided this cook
		}
	} else {
		C->d_collisionSDF = DeviceMemory<float>();
	}
}


extern "C" void HNS_UploadSources(void* ctx, HNS::GridIndexedData& data) {
	auto* C = reinterpret_cast<SimContext*>(ctx);
	if (!C || C->N == 0) return;

	const size_t N = C->N;

	// Reset per-cook sources
	{
		// avoid operator[] on mapped_type with deleted copy-assign
		std::unordered_map<std::string, DeviceMemory<float>> empty;
		C->d_src.swap(empty);
	}
	C->d_src_vel_vec3 = DeviceMemory<nanovdb::Vec3f>();
	C->d_src_vel_x = DeviceMemory<float>();
	C->d_src_vel_y = DeviceMemory<float>();
	C->d_src_vel_z = DeviceMemory<float>();

	// Velocity sources
	bool haveVec3 = false, haveX = false, haveY = false, haveZ = false;
	{
		const auto v3 = data.getBlocksOfType<openvdb::Vec3f>();
		for (const auto& n : v3)
			if (n == "src_vel") {
				haveVec3 = true;
				break;
			}
		const auto flds = data.getBlocksOfType<float>();
		haveX = std::find(flds.begin(), flds.end(), "src_vel_x") != flds.end();
		haveY = std::find(flds.begin(), flds.end(), "src_vel_y") != flds.end();
		haveZ = std::find(flds.begin(), flds.end(), "src_vel_z") != flds.end();
	}

	if (haveVec3) {
		const auto* h = reinterpret_cast<const nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>("src_vel"));
		if (h) {
			C->d_src_vel_vec3 = DeviceMemory<nanovdb::Vec3f>(N, C->so.stream);
			CUDA_CHECK(cudaMemcpyAsync(C->d_src_vel_vec3.get(), h, N * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice, C->so.stream));
		}
	} else if (haveX && haveY && haveZ) {
		C->d_src_vel_x = DeviceMemory<float>(N, C->so.stream);
		C->d_src_vel_y = DeviceMemory<float>(N, C->so.stream);
		C->d_src_vel_z = DeviceMemory<float>(N, C->so.stream);
		if (auto* px = data.pValues<float>("src_vel_x"))
			CUDA_CHECK(cudaMemcpyAsync(C->d_src_vel_x.get(), px, N * sizeof(float), cudaMemcpyHostToDevice, C->so.stream));
		if (auto* py = data.pValues<float>("src_vel_y"))
			CUDA_CHECK(cudaMemcpyAsync(C->d_src_vel_y.get(), py, N * sizeof(float), cudaMemcpyHostToDevice, C->so.stream));
		if (auto* pz = data.pValues<float>("src_vel_z"))
			CUDA_CHECK(cudaMemcpyAsync(C->d_src_vel_z.get(), pz, N * sizeof(float), cudaMemcpyHostToDevice, C->so.stream));
	}

	// Scalar sources: any field starting with "src_"
	for (const auto& name : data.getBlocksOfType<float>()) {
		if (!is_source(name)) continue;
		const float* h = data.pValues<float>(name);
		if (!h) continue;

		DeviceMemory<float> d(N, C->so.stream);
		CUDA_CHECK(cudaMemcpyAsync(d.get(), h, N * sizeof(float), cudaMemcpyHostToDevice, C->so.stream));
		// use insert to avoid copy-assign on mapped_type
		C->d_src.insert(std::make_pair(name, std::move(d)));
	}

	CUDA_CHECK(cudaStreamSynchronize(C->so.stream));
}


extern "C"  void HNS_Advance(void* ctx, int iteration, float dt, const CombustionParams& params) {
	auto* C = reinterpret_cast<SimContext*>(ctx);
	if (!C || C->N == 0) return;

	const size_t N = C->N;
	const float dx = C->voxelSize;
	const float inv_dx = 1.0f / dx;
	const int SUBS = std::max(1, params.substeps);
	const float dt_sub = dt / float(SUBS);
	const int threads = 256;
	const int blocks = int((N + threads - 1) / threads);
	const bool hasCollision = (C->d_collisionSDF.get() != nullptr);

	CUDA_CHECK(cudaMemsetAsync(C->d_advectedVel.get(), 0, C->d_advectedVel.bytes(), C->so.stream));
	CUDA_CHECK(cudaMemsetAsync(C->d_divergence.get(), 0, C->d_divergence.bytes(), C->so.stream));
	CUDA_CHECK(cudaMemsetAsync(C->d_pressure.get(), 0, C->d_pressure.bytes(), C->so.stream));

	if (hasCollision) {
		enforceCollisionBoundaries<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_velocity.get(),
		                                                              C->d_collisionSDF.get(), dx, (int)N);
		CUDA_CHECK(cudaGetLastError());
	}

	// Names of scalars to advect each substep
	std::vector<std::string> advNames;
	advNames.reserve(C->d_state.size());
	for (const auto& kv : C->d_state) advNames.push_back(kv.first);

	for (int s = 0; s < SUBS; ++s) {
		// 1) Advect velocity
		advect_vector<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_velocity.get(), C->d_advectedVel.get(),
		                                                 hasCollision ? C->d_collisionSDF.get() : nullptr, hasCollision, (int)N, dt_sub,
		                                                 inv_dx);
		CUDA_CHECK(cudaGetLastError());

		// Vorticity confinement
		vorticityConfinement<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_advectedVel.get(),
		                                                        C->d_advectedVel.get(), dt_sub, inv_dx, params.vorticityScale,
		                                                        params.factorScale, (int)N);
		CUDA_CHECK(cudaGetLastError());

		// Velocity source
		if (C->d_src_vel_vec3.get()) {
			add_velocity_source<<<blocks, threads, 0, C->so.stream>>>(C->d_advectedVel.get(), C->d_src_vel_vec3.get(), dt_sub, (int)N);
			CUDA_CHECK(cudaGetLastError());
		} else if (C->d_src_vel_x.get() && C->d_src_vel_y.get() && C->d_src_vel_z.get()) {
			add_velocity_source_xyz<<<blocks, threads, 0, C->so.stream>>>(C->d_advectedVel.get(), C->d_src_vel_x.get(), C->d_src_vel_y.get(),
			                                                           C->d_src_vel_z.get(), dt_sub, (int)N);
			CUDA_CHECK(cudaGetLastError());
		}

		// Scalar sources
		struct Pair {
			const char* s;
			const char* d;
		};
		static const Pair pairs[] = {{"src_temperature", "temperature"}, {"src_density", "density"}, {"src_fuel", "fuel"}};
		for (const auto& p : pairs) {
			auto itS = C->d_src.find(p.s);
			auto itD = C->d_state.find(p.d);
			if (itS != C->d_src.end() && itD != C->d_state.end()) {
				add_scalar_source<<<blocks, threads, 0, C->so.stream>>>(itD->second.get(), itS->second.get(), dt_sub, (int)N);
				CUDA_CHECK(cudaGetLastError());
			}
		}

		if (params.gravity != 0.0f) {
			add_gravity<<<blocks, threads, 0, C->so.stream>>>(C->d_advectedVel.get(), params.gravity, dt_sub, (int)N);
			CUDA_CHECK(cudaGetLastError());
		}

		// 2) Divergence
		divergence<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_advectedVel.get(), C->d_divergence.get(), inv_dx,
		                                              (int)N);
		CUDA_CHECK(cudaGetLastError());

		// 3) Combustion + buoyancy
		auto needOrMake = [&](const char* k) -> DeviceMemory<float>& {
			auto it = C->d_state.find(k);
			if (it == C->d_state.end()) {
				auto ins = C->d_state.insert(std::make_pair(std::string(k), DeviceMemory<float>(N, C->so.stream)));
				CUDA_CHECK(cudaMemsetAsync(ins.first->second.get(), 0, N * sizeof(float), C->so.stream));
				return ins.first->second;
			}
			return it->second;
		};
		DeviceMemory<float>& d_fuel = needOrMake("fuel");
		DeviceMemory<float>& d_waste = needOrMake("waste");
		DeviceMemory<float>& d_temp = needOrMake("temperature");
		DeviceMemory<float>& d_flame = needOrMake("flame");

		DeviceMemory<float> o_fuel(N, C->so.stream), o_waste(N, C->so.stream), o_temp(N, C->so.stream), o_flame(N, C->so.stream);
		CUDA_CHECK(cudaMemsetAsync(o_fuel.get(), 0, N * sizeof(float), C->so.stream));
		CUDA_CHECK(cudaMemsetAsync(o_waste.get(), 0, N * sizeof(float), C->so.stream));
		CUDA_CHECK(cudaMemsetAsync(o_temp.get(), 0, N * sizeof(float), C->so.stream));
		CUDA_CHECK(cudaMemsetAsync(o_flame.get(), 0, N * sizeof(float), C->so.stream));

		combustion_oxygen<<<blocks, threads, 0, C->so.stream>>>(d_fuel.get(), d_waste.get(), d_temp.get(), C->d_divergence.get(),
		                                                     d_flame.get(), o_fuel.get(), o_waste.get(), o_temp.get(), o_flame.get(),
		                                                     params.temperatureRelease, params.expansionRate, (int)N);
		CUDA_CHECK(cudaGetLastError());

		temperature_buoyancy<<<blocks, threads, 0, C->so.stream>>>(C->d_advectedVel.get(), o_temp.get(), C->d_advectedVel.get(), dt_sub,
		                                                        params.ambientTemp, params.buoyancyStrength, (int)N);
		CUDA_CHECK(cudaGetLastError());

		// Copy results back into persistent state (no move-assign)
		CUDA_CHECK(cudaMemcpyAsync(d_fuel.get(), o_fuel.get(), N * sizeof(float), cudaMemcpyDeviceToDevice, C->so.stream));
		CUDA_CHECK(cudaMemcpyAsync(d_waste.get(), o_waste.get(), N * sizeof(float), cudaMemcpyDeviceToDevice, C->so.stream));
		CUDA_CHECK(cudaMemcpyAsync(d_temp.get(), o_temp.get(), N * sizeof(float), cudaMemcpyDeviceToDevice, C->so.stream));
		CUDA_CHECK(cudaMemcpyAsync(d_flame.get(), o_flame.get(), N * sizeof(float), cudaMemcpyDeviceToDevice, C->so.stream));

		// 4) Pressure solve
		CUDA_CHECK(cudaMemsetAsync(C->d_pressure.get(), 0, C->d_pressure.bytes(), C->so.stream));
		const float omega = 2.0f / (1.0f + sinf(3.14159f * dx));
		for (int it = 0; it < iteration; ++it) {
			redBlackGaussSeidelUpdate<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_divergence.get(),
			                                                             C->d_pressure.get(), dx, (int)N, 0, omega);
			redBlackGaussSeidelUpdate<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_divergence.get(),
			                                                             C->d_pressure.get(), dx, (int)N, 1, omega);
		}
		CUDA_CHECK(cudaGetLastError());

		// 4.5) Optional disturbance + short re-solve
		if (params.disturbanceEnable > 0.0f) {
			auto itDens = C->d_state.find("density");
			if (itDens != C->d_state.end()) {
				inject_edge_disturbance_vel_from_density<<<blocks, threads, 0, C->so.stream>>>(
				    C->d_grid, C->d_coords.get(), itDens->second.get(), params.maskDensityMin, params.maskDensityMax,
				    hasCollision ? C->d_collisionSDF.get() : nullptr, hasCollision, C->d_advectedVel.get(), (int)N, dt_sub, inv_dx,
				    params.disturbanceThreshold / float(SUBS), params.disturbanceStrength, params.disturbanceSwirl,
				    (int)params.disturbanceFrequency);
				CUDA_CHECK(cudaGetLastError());
			}
			divergence<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_advectedVel.get(), C->d_divergence.get(),
			                                              inv_dx, (int)N);
			CUDA_CHECK(cudaGetLastError());

			CUDA_CHECK(cudaMemsetAsync(C->d_pressure.get(), 0, C->d_pressure.bytes(), C->so.stream));
			const int it2 = std::max(2, iteration / 2);
			for (int it = 0; it < it2; ++it) {
				redBlackGaussSeidelUpdate<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_divergence.get(),
				                                                             C->d_pressure.get(), dx, (int)N, 0, omega);
				redBlackGaussSeidelUpdate<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_divergence.get(),
				                                                             C->d_pressure.get(), dx, (int)N, 1, omega);
			}
			CUDA_CHECK(cudaGetLastError());
		}

		// 5) Projection
		subtractPressureGradient<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), (int)N, C->d_advectedVel.get(),
		                                                            C->d_pressure.get(), C->d_velocity.get(),
		                                                            hasCollision ? C->d_collisionSDF.get() : nullptr, hasCollision, inv_dx);
		CUDA_CHECK(cudaGetLastError());

		if (hasCollision) {
			enforceCollisionBoundaries<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_velocity.get(),
			                                                              C->d_collisionSDF.get(), dx, (int)N);
			CUDA_CHECK(cudaGetLastError());
		}

		// 6) Advect scalars with final velocity
		if (!advNames.empty()) {
			std::vector<float*> inPtrs, outPtrs;
			inPtrs.reserve(advNames.size());
			outPtrs.reserve(advNames.size());

			// scratch outputs
			std::vector<DeviceMemory<float>> outs;
			outs.reserve(advNames.size());

			for (const auto& n : advNames) {
				inPtrs.push_back(C->d_state.at(n).get());
				outs.emplace_back(N, C->so.stream);
				CUDA_CHECK(cudaMemsetAsync(outs.back().get(), 0, N * sizeof(float), C->so.stream));
				outPtrs.push_back(outs.back().get());
			}

			float** d_inArr = nullptr;
			float** d_outArr = nullptr;
			CUDA_CHECK(cudaMalloc(&d_inArr, inPtrs.size() * sizeof(float*)));
			CUDA_CHECK(cudaMalloc(&d_outArr, outPtrs.size() * sizeof(float*)));
			CUDA_CHECK(cudaMemcpyAsync(d_inArr, inPtrs.data(), inPtrs.size() * sizeof(float*), cudaMemcpyHostToDevice, C->so.stream));
			CUDA_CHECK(cudaMemcpyAsync(d_outArr, outPtrs.data(), outPtrs.size() * sizeof(float*), cudaMemcpyHostToDevice, C->so.stream));

			advect_scalars<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_velocity.get(), d_inArr, d_outArr,
			                                                  (int)inPtrs.size(), hasCollision ? C->d_collisionSDF.get() : nullptr,
			                                                  hasCollision, (int)N, dt_sub, inv_dx);
			CUDA_CHECK(cudaGetLastError());

			cudaFree(d_inArr);
			cudaFree(d_outArr);

			// Copy results back to persistent state (no move-assign)
			for (size_t i = 0; i < advNames.size(); ++i) {
				auto& dst = C->d_state.at(advNames[i]);
				CUDA_CHECK(cudaMemcpyAsync(dst.get(), outs[i].get(), N * sizeof(float), cudaMemcpyDeviceToDevice, C->so.stream));
			}
		}
	}

	CUDA_CHECK(cudaStreamSynchronize(C->so.stream));
}

extern "C" void HNS_DownloadForOutput(void* ctx, HNS::GridIndexedData& data) {
	auto* C = reinterpret_cast<SimContext*>(ctx);
	const size_t N = C->N;

	// velocity
	{
		const auto vecNames = data.getBlocksOfType<openvdb::Vec3f>();
		if (vecNames.empty()) return;
		std::string velName = vecNames[0];
		for (auto& n : vecNames)
			if (n == "velocity") {
				velName = n;
				break;
			}
		auto* h_vel = data.pValues<openvdb::Vec3f>(velName);
		CUDA_CHECK(cudaMemcpyAsync(h_vel, C->d_velocity.get(), N * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost, C->so.stream));
	}
	// scalars
	for (auto& kv : C->d_state) {
		auto* h = data.pValues<float>(kv.first);
		if (!h) continue;
		CUDA_CHECK(cudaMemcpyAsync(h, kv.second.get(), N * sizeof(float), cudaMemcpyDeviceToHost, C->so.stream));
	}
	CUDA_CHECK(cudaStreamSynchronize(C->so.stream));
}


void Compute(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const int iteration,
             const float dt, const float voxelSize, const CombustionParams& params, const bool hasCollision, const cudaStream_t& stream) {
	
	if (voxelSize <= 0.0f) throw std::invalid_argument("voxelSize must be positive.");
	if (dt < 0.0f) throw std::invalid_argument("dt (time step) cannot be negative.");
	if (iteration <= 0) throw std::invalid_argument("Number of pressure iterations must be positive.");
	if (handle.isEmpty()) throw std::invalid_argument("Invalid nanovdb::GridHandle provided (null grid).");

	const int SUBS = std::max(1, params.substeps);

	const size_t totalVoxels = data.size();
	if (totalVoxels == 0) return;

	auto gpuGridPtr = handle.deviceGrid<nanovdb::ValueOnIndex>();
	if (!gpuGridPtr) throw std::runtime_error("Failed to get device grid pointer of type ValueOnIndex from handle.");

	const float inv_voxelSize = 1.0f / voxelSize;
	const float dt_sub = dt / float(SUBS);

	// --- Host pointers ---
	const auto vec3fBlocks = data.getBlocksOfType<openvdb::Vec3f>();
	if (vec3fBlocks.empty()) throw std::runtime_error("Expected at least one Vec3f block (velocity).");

	std::string velocityBlockName = vec3fBlocks[0];
	for (const auto& b : vec3fBlocks)
		if (b == "velocity") {
			velocityBlockName = b;
			break;
		}

	nanovdb::Vec3f* h_velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(velocityBlockName));
	if (!h_velocity) throw std::runtime_error("Host velocity data pointer is null for block: " + velocityBlockName);

	nanovdb::Vec3f* h_src_vel_vec3 = nullptr;
	for (const auto& b : vec3fBlocks)
		if (b == "src_vel") {
			h_src_vel_vec3 = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(b));
			break;
		}

	nanovdb::Coord* h_coords = reinterpret_cast<nanovdb::Coord*>(data.pCoords());
	if (!h_coords) throw std::runtime_error("Host coordinate data pointer is null.");

	const auto floatBlockNames = data.getBlocksOfType<float>();
	if (floatBlockNames.empty()) throw std::runtime_error("No float blocks found in input data.");

	const bool have_src_vel_xyz = std::find(floatBlockNames.begin(), floatBlockNames.end(), "src_vel_x") != floatBlockNames.end() &&
	                              std::find(floatBlockNames.begin(), floatBlockNames.end(), "src_vel_y") != floatBlockNames.end() &&
	                              std::find(floatBlockNames.begin(), floatBlockNames.end(), "src_vel_z") != floatBlockNames.end();

	float* h_collisionSDF = nullptr;
	bool hasCollisionData = false;
	if (hasCollision) {
		if (std::find(floatBlockNames.begin(), floatBlockNames.end(), "collision_sdf") != floatBlockNames.end()) {
			h_collisionSDF = data.pValues<float>("collision_sdf");
			hasCollisionData = (h_collisionSDF != nullptr);
		}
	}

	std::vector<float*> h_floatPointers(floatBlockNames.size());
	for (size_t i = 0; i < floatBlockNames.size(); ++i) {
		h_floatPointers[i] = data.pValues<float>(floatBlockNames[i]);
		if (!h_floatPointers[i]) throw std::runtime_error("Host float data pointer is null for block: " + floatBlockNames[i]);
	}

	auto is_source_scalar = [](const std::string& n) -> bool { return n.rfind("src_", 0) == 0; };

	// --- Device buffers ---
	DeviceMemory<nanovdb::Vec3f> d_velocity(totalVoxels, stream);     // state velocity (updated each substep)
	DeviceMemory<nanovdb::Vec3f> d_advectedVel(totalVoxels, stream);  // u* per substep
	DeviceMemory<nanovdb::Coord> d_coords(totalVoxels, stream);
	DeviceMemory<float> d_divergence(totalVoxels, stream);
	DeviceMemory<float> d_pressure(totalVoxels, stream);

	DeviceMemory<float> d_collisionSDF;
	if (hasCollisionData) d_collisionSDF = DeviceMemory<float>(totalVoxels, stream);

	DeviceMemory<nanovdb::Vec3f> d_src_vel_vec3;
	DeviceMemory<float> d_src_vel_x, d_src_vel_y, d_src_vel_z;
	if (h_src_vel_vec3) {
		d_src_vel_vec3 = DeviceMemory<nanovdb::Vec3f>(totalVoxels, stream);
	} else if (have_src_vel_xyz) {
		d_src_vel_x = DeviceMemory<float>(totalVoxels, stream);
		d_src_vel_y = DeviceMemory<float>(totalVoxels, stream);
		d_src_vel_z = DeviceMemory<float>(totalVoxels, stream);
	}

	std::unordered_map<std::string, DeviceMemory<float>> d_inputs;   // live scalar state
	std::unordered_map<std::string, DeviceMemory<float>> d_outputs;  // scratch per-step

	for (const auto& name : floatBlockNames) {
		d_inputs.emplace(std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(totalVoxels, stream));
		d_outputs.emplace(std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(totalVoxels, stream));
	}

	// --- Init device memory ---
	CUDA_CHECK(cudaMemsetAsync(d_advectedVel.get(), 0, d_advectedVel.bytes(), stream));
	CUDA_CHECK(cudaMemsetAsync(d_divergence.get(), 0, d_divergence.bytes(), stream));
	CUDA_CHECK(cudaMemsetAsync(d_pressure.get(), 0, d_pressure.bytes(), stream));
	for (auto& kv : d_outputs) CUDA_CHECK(cudaMemsetAsync(kv.second.get(), 0, kv.second.bytes(), stream));

	CUDA_CHECK(cudaMemcpyAsync(d_velocity.get(), h_velocity, d_velocity.bytes(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_coords.get(), h_coords, d_coords.bytes(), cudaMemcpyHostToDevice, stream));
	if (hasCollisionData) {
		CUDA_CHECK(cudaMemcpyAsync(d_collisionSDF.get(), h_collisionSDF, d_collisionSDF.bytes(), cudaMemcpyHostToDevice, stream));
	}
	for (size_t i = 0; i < floatBlockNames.size(); ++i) {
		const std::string& name = floatBlockNames[i];
		CUDA_CHECK(cudaMemcpyAsync(d_inputs.at(name).get(), h_floatPointers[i], d_inputs.at(name).bytes(), cudaMemcpyHostToDevice, stream));
	}
	if (h_src_vel_vec3) {
		CUDA_CHECK(cudaMemcpyAsync(d_src_vel_vec3.get(), h_src_vel_vec3, d_src_vel_vec3.bytes(), cudaMemcpyHostToDevice, stream));
	} else if (have_src_vel_xyz) {
		CUDA_CHECK(
		    cudaMemcpyAsync(d_src_vel_x.get(), data.pValues<float>("src_vel_x"), d_src_vel_x.bytes(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(
		    cudaMemcpyAsync(d_src_vel_y.get(), data.pValues<float>("src_vel_y"), d_src_vel_y.bytes(), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(
		    cudaMemcpyAsync(d_src_vel_z.get(), data.pValues<float>("src_vel_z"), d_src_vel_z.bytes(), cudaMemcpyHostToDevice, stream));
	}

	// --- launch config ---
	constexpr int PREFERRED_BLOCK_SIZE = 256;
	const int maxThreadsPerBlock = 1024;
	const int blockSize = std::min((int)totalVoxels, std::min(PREFERRED_BLOCK_SIZE, maxThreadsPerBlock));
	const int gridSize = int((totalVoxels + blockSize - 1) / blockSize);

	// Pre-build scalar lists once (advected each substep)
	std::vector<std::string> advScalarNames;
	for (const auto& name : floatBlockNames) {
		if (name == "collision_sdf") continue;
		if (is_source_scalar(name)) continue;
		advScalarNames.push_back(name);
	}

	// Initial collision clamp
	if (hasCollisionData) {
		enforceCollisionBoundaries<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_velocity.get(), d_collisionSDF.get(),
		                                                               voxelSize, totalVoxels);
		CUDA_CHECK(cudaGetLastError());
	}

	// --- Substeps loop ---
	for (int s = 0; s < SUBS; ++s) {
		// 1) Advect velocity: d_velocity -> d_advectedVel
		advect_vector<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_velocity.get(), d_advectedVel.get(),
		                                                  hasCollisionData ? d_collisionSDF.get() : nullptr, hasCollisionData, totalVoxels,
		                                                  dt_sub, inv_voxelSize);
		CUDA_CHECK(cudaGetLastError());

		// Vorticity confinement on u*
		vorticityConfinement<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_advectedVel.get(), d_advectedVel.get(),
		                                                         dt_sub, inv_voxelSize, params.vorticityScale, params.factorScale,
		                                                         totalVoxels);
		CUDA_CHECK(cudaGetLastError());

		// 1.5) Sources on u* and scalars (use dt_sub)
		if (h_src_vel_vec3) {
			add_velocity_source<<<gridSize, blockSize, 0, stream>>>(d_advectedVel.get(), d_src_vel_vec3.get(), dt_sub, (int)totalVoxels);
			CUDA_CHECK(cudaGetLastError());
		} else if (have_src_vel_xyz) {
			add_velocity_source_xyz<<<gridSize, blockSize, 0, stream>>>(d_advectedVel.get(), d_src_vel_x.get(), d_src_vel_y.get(),
			                                                            d_src_vel_z.get(), dt_sub, (int)totalVoxels);
			CUDA_CHECK(cudaGetLastError());
		}
		// scalar sources
		struct Pair {
			const char* src;
			const char* dst;
		};
		const Pair pairs[] = {{"src_temperature", "temperature"}, {"src_density", "density"}, {"src_fuel", "fuel"}};
		for (const auto& p : pairs) {
			auto itS = d_inputs.find(p.src);
			auto itD = d_inputs.find(p.dst);
			if (itS != d_inputs.end() && itD != d_inputs.end()) {
				add_scalar_source<<<gridSize, blockSize, 0, stream>>>(itD->second.get(), itS->second.get(), dt_sub, (int)totalVoxels);
				CUDA_CHECK(cudaGetLastError());
			}
		}
		// gravity
		if (params.gravity > 0.0f) {
			add_gravity<<<gridSize, blockSize, 0, stream>>>(d_advectedVel.get(), params.gravity, dt_sub, (int)totalVoxels);
			CUDA_CHECK(cudaGetLastError());
		}

		// 2) Divergence of u*
		divergence<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_advectedVel.get(), d_divergence.get(), inv_voxelSize,
		                                               totalVoxels);
		CUDA_CHECK(cudaGetLastError());

		// 3) Combustion + buoyancy
		{
			const std::vector<std::string> req = {"fuel", "waste", "temperature", "flame"};
			for (const auto& f : req) {
				if (!d_inputs.count(f)) throw std::runtime_error("Missing input field: " + f);
				if (!d_outputs.count(f)) throw std::runtime_error("Missing output field: " + f);
			}

			combustion_oxygen<<<gridSize, blockSize, 0, stream>>>(
			    d_inputs.at("fuel").get(), d_inputs.at("waste").get(), d_inputs.at("temperature").get(), d_divergence.get(),
			    d_inputs.at("flame").get(), d_outputs.at("fuel").get(), d_outputs.at("waste").get(), d_outputs.at("temperature").get(),
			    d_outputs.at("flame").get(), params.temperatureRelease, params.expansionRate, totalVoxels);
			CUDA_CHECK(cudaGetLastError());

			temperature_buoyancy<<<gridSize, blockSize, 0, stream>>>(d_advectedVel.get(), d_outputs.at("temperature").get(),
			                                                         d_advectedVel.get(), dt_sub, params.ambientTemp,
			                                                         params.buoyancyStrength, totalVoxels);
			CUDA_CHECK(cudaGetLastError());

			// swap combustion-updated fields into inputs; clear new outputs
			for (const auto& f : req) {
				d_inputs.at(f) = std::move(d_outputs.at(f));
				d_outputs.at(f) = DeviceMemory<float>(totalVoxels, stream);
				CUDA_CHECK(cudaMemsetAsync(d_outputs.at(f).get(), 0, d_outputs.at(f).bytes(), stream));
			}
		}

		// 4) Pressure solve
		CUDA_CHECK(cudaMemsetAsync(d_pressure.get(), 0, d_pressure.bytes(), stream));
		{
			const float omega = 2.0f / (1.0f + sinf(3.14159f * voxelSize));
			for (int it = 0; it < iteration; ++it) {
				redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_divergence.get(),
				                                                              d_pressure.get(), voxelSize, totalVoxels, 0, omega);
				redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_divergence.get(),
				                                                              d_pressure.get(), voxelSize, totalVoxels, 1, omega);
			}
			CUDA_CHECK(cudaGetLastError());
		}

		// 4.5) Disturbance + short re-solve (optional)
		if (params.disturbanceEnable > 0.0f) {
			// inject_edge_disturbance_vel<<<gridSize, blockSize, 0, stream>>>(
			//     gpuGridPtr, d_coords.get(), d_pressure.get(), hasCollisionData ? d_collisionSDF.get() : nullptr, hasCollisionData,
			//     d_advectedVel.get(), totalVoxels, dt_sub, inv_voxelSize, params.disturbanceThreshold / SUBS, params.disturbanceStrength,
			//     params.disturbanceSwirl, (int)params.disturbanceFrequency);

			auto densPtr = d_inputs.count("density") ? d_inputs.at("density").get() : nullptr;
			if (densPtr) {
				inject_edge_disturbance_vel_from_density<<<gridSize, blockSize, 0, stream>>>(
				    gpuGridPtr, d_coords.get(), densPtr,
				    params.maskDensityMin,  // e.g. 0.01f
				    params.maskDensityMax,  // e.g. 1.0f
				    hasCollisionData ? d_collisionSDF.get() : nullptr, hasCollisionData, d_advectedVel.get(), totalVoxels, dt_sub,
				    inv_voxelSize,
				    params.disturbanceThreshold / SUBS,  // now a |delta density| threshold
				    params.disturbanceStrength, params.disturbanceSwirl, (int)params.disturbanceFrequency);
				CUDA_CHECK(cudaGetLastError());
			}

			// recompute divergence after velocity change
			divergence<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_advectedVel.get(), d_divergence.get(),
			                                               inv_voxelSize, totalVoxels);
			CUDA_CHECK(cudaGetLastError());

			// reset pressure? not sure if this is needed...
			CUDA_CHECK(cudaMemsetAsync(d_pressure.get(), 0, d_pressure.bytes(), stream));

			const float omega = 2.0f / (1.0f + sinf(3.14159f * voxelSize));
			const int it2 = std::max(2, iteration / 2);
			for (int it = 0; it < it2; ++it) {
				redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_divergence.get(),
				                                                              d_pressure.get(), voxelSize, totalVoxels, 0, omega);
				redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_divergence.get(),
				                                                              d_pressure.get(), voxelSize, totalVoxels, 1, omega);
			}
			CUDA_CHECK(cudaGetLastError());
		}

		// 5) Projection: u = P(u*)
		subtractPressureGradient<<<gridSize, blockSize, 0, stream>>>(
		    gpuGridPtr, d_coords.get(), totalVoxels, d_advectedVel.get(), d_pressure.get(), d_velocity.get(),
		    hasCollisionData ? d_collisionSDF.get() : nullptr, hasCollisionData, inv_voxelSize);
		CUDA_CHECK(cudaGetLastError());

		if (hasCollisionData) {
			enforceCollisionBoundaries<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_velocity.get(),
			                                                               d_collisionSDF.get(), voxelSize, totalVoxels);
			CUDA_CHECK(cudaGetLastError());
		}

		// 6) Advect scalars with final velocity of this substep, swap into inputs
		if (!advScalarNames.empty()) {
			std::vector<float*> inPtrs, outPtrs;
			inPtrs.reserve(advScalarNames.size());
			outPtrs.reserve(advScalarNames.size());
			for (const auto& n : advScalarNames) {
				inPtrs.push_back(d_inputs.at(n).get());
				outPtrs.push_back(d_outputs.at(n).get());
			}

			float** d_inArr = nullptr;
			float** d_outArr = nullptr;
			CUDA_CHECK(cudaMalloc(&d_inArr, inPtrs.size() * sizeof(float*)));
			CUDA_CHECK(cudaMalloc(&d_outArr, outPtrs.size() * sizeof(float*)));
			CUDA_CHECK(cudaMemcpyAsync(d_inArr, inPtrs.data(), inPtrs.size() * sizeof(float*), cudaMemcpyHostToDevice, stream));
			CUDA_CHECK(cudaMemcpyAsync(d_outArr, outPtrs.data(), outPtrs.size() * sizeof(float*), cudaMemcpyHostToDevice, stream));

			advect_scalars<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_velocity.get(), d_inArr, d_outArr,
			                                                   (int)inPtrs.size(), hasCollisionData ? d_collisionSDF.get() : nullptr,
			                                                   hasCollisionData, totalVoxels, dt_sub, inv_voxelSize);
			CUDA_CHECK(cudaGetLastError());

			cudaFree(d_inArr);
			cudaFree(d_outArr);

			for (const auto& n : advScalarNames) {
				d_inputs.at(n) = std::move(d_outputs.at(n));
				d_outputs.at(n) = DeviceMemory<float>(totalVoxels, stream);
				CUDA_CHECK(cudaMemsetAsync(d_outputs.at(n).get(), 0, d_outputs.at(n).bytes(), stream));
			}
		}
	}  // end substeps

	// --- Copy back once ---
	CUDA_CHECK(cudaMemcpyAsync(h_velocity, d_velocity.get(), d_velocity.bytes(), cudaMemcpyDeviceToHost, stream));
	for (size_t i = 0; i < floatBlockNames.size(); ++i) {
		const std::string& name = floatBlockNames[i];
		if (name == "collision_sdf") continue;
		if (is_source_scalar(name)) continue;
		float* h_ptr = h_floatPointers[i];
		CUDA_CHECK(cudaMemcpyAsync(h_ptr, d_inputs.at(name).get(), d_inputs.at(name).bytes(), cudaMemcpyDeviceToHost, stream));
	}

	cudaStreamSynchronize(stream);
}

void create_index_grid(HNS::GridIndexedData& data, nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const float voxelSize) {
	const auto* h_coords = data.pCoords();
	nanovdb::Coord* d_coords = nullptr;
	cudaMalloc(&d_coords, data.size() * sizeof(nanovdb::Coord));
	cudaMemcpy(d_coords, h_coords, data.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice);
	handle = nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, data.size(), voxelSize);
	cudaFree(d_coords);
}

extern "C" void CreateIndexGrid(HNS::GridIndexedData& data, nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle,
                                const float voxelSize) {
	create_index_grid(data, handle, voxelSize);
}

extern "C" void Compute_Sim(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, int iteration,
                            float dt, float voxelSize, const CombustionParams& params, bool hasCollision, const cudaStream_t& stream) {
	Compute(data, handle, iteration, dt, voxelSize, params, hasCollision, stream);
}

