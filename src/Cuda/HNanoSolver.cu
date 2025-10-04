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

	// sparse state
	DeviceMemory<unsigned char> d_active;
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
		
		C->d_active = DeviceMemory<unsigned char>(N, C->so.stream);


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

extern "C" void HNS_Advance(void* ctx, int iteration, float dt, const CombustionParams& params) {
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

	// scratch: advected vel, divergence, pressure
	CUDA_CHECK(cudaMemsetAsync(C->d_advectedVel.get(), 0, C->d_advectedVel.bytes(), C->so.stream));
	CUDA_CHECK(cudaMemsetAsync(C->d_divergence.get(), 0, C->d_divergence.bytes(), C->so.stream));
	CUDA_CHECK(cudaMemsetAsync(C->d_pressure.get(), 0, C->d_pressure.bytes(), C->so.stream));

	if (hasCollision) {
		enforceCollisionBoundaries<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_velocity.get(),
		                                                                 C->d_collisionSDF.get(), dx, (size_t)N);
		CUDA_CHECK(cudaGetLastError());
	}

	// scalar names to advect/persist
	std::vector<std::string> advNames;
	advNames.reserve(C->d_state.size());
	for (const auto& kv : C->d_state) advNames.push_back(kv.first);

	for (int s = 0; s < SUBS; ++s) {
		
		// Activity mask data
		CUDA_CHECK(cudaMemsetAsync(C->d_active.get(), 0, C->d_active.bytes(), C->so.stream));


		const float dens_eps = 1e-6f;
		const float vel_eps = 1e-6f;
		const float* d_density = (C->d_state.count("density") ? C->d_state.at("density").get() : nullptr);

		const nanovdb::Vec3f* src_v3 = C->d_src_vel_vec3.get();
		const float* sx = C->d_src_vel_x.get();
		const float* sy = C->d_src_vel_y.get();
		const float* sz = C->d_src_vel_z.get();
		const float* src_density = (C->d_src.count("src_density") ? C->d_src.at("src_density").get() : nullptr);

		build_activity_mask<<<blocks, threads, 0, C->so.stream>>>(d_density, C->d_velocity.get(), src_density, src_v3, sx, sy, sz,
		                                                          C->d_active.get(), (int)N, dens_eps, vel_eps);
		CUDA_CHECK(cudaGetLastError());

		// 1) Advect velocity (masked)
		advect_vector<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_velocity.get(), C->d_advectedVel.get(),
		                                                    hasCollision ? C->d_collisionSDF.get() : nullptr, hasCollision,
		                                                    /*active*/ C->d_active.get(), (size_t)N, dt_sub, inv_dx);
		CUDA_CHECK(cudaGetLastError());

		// Vorticity confinement (masked)
		vorticityConfinement<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_advectedVel.get(),
		                                                           C->d_advectedVel.get(), dt_sub, inv_dx, params.vorticityScale,
		                                                           params.factorScale,
		                                                           /*active*/ C->d_active.get(), (size_t)N);
		CUDA_CHECK(cudaGetLastError());

		// Velocity sources (masked)
		if (C->d_src_vel_vec3.get()) {
			add_velocity_source<<<blocks, threads, 0, C->so.stream>>>(C->d_advectedVel.get(), C->d_src_vel_vec3.get(), dt_sub,
			                                                          /*active*/ C->d_active.get(), (int)N);
			CUDA_CHECK(cudaGetLastError());
		} else if (C->d_src_vel_x.get() && C->d_src_vel_y.get() && C->d_src_vel_z.get()) {
			add_velocity_source_xyz<<<blocks, threads, 0, C->so.stream>>>(C->d_advectedVel.get(), C->d_src_vel_x.get(),
			                                                              C->d_src_vel_y.get(), C->d_src_vel_z.get(), dt_sub,
			                                                              /*active*/ C->d_active.get(), (int)N);
			CUDA_CHECK(cudaGetLastError());
		}

		// Scalar sources (masked)
		struct Pair {
			const char* s;
			const char* d;
		};
		static const Pair pairs[] = {{"src_temperature", "temperature"}, {"src_density", "density"}, {"src_fuel", "fuel"}};
		for (const auto& p : pairs) {
			auto itS = C->d_src.find(p.s);
			auto itD = C->d_state.find(p.d);
			if (itS != C->d_src.end() && itD != C->d_state.end()) {
				add_scalar_source<<<blocks, threads, 0, C->so.stream>>>(itD->second.get(), itS->second.get(), dt_sub,
				                                                        /*active*/ C->d_active.get(), (int)N);
				CUDA_CHECK(cudaGetLastError());
			}
		}

		// Gravity (masked)
		if (params.gravity != 0.0f) {
			add_gravity<<<blocks, threads, 0, C->so.stream>>>(C->d_advectedVel.get(), params.gravity, dt_sub,
			                                                  /*active*/ C->d_active.get(), (int)N);
			CUDA_CHECK(cudaGetLastError());
		}

		// 2) Divergence (masked)
		divergence<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_advectedVel.get(), C->d_divergence.get(),
		                                                 inv_dx, /*active*/ C->d_active.get(), (size_t)N);
		CUDA_CHECK(cudaGetLastError());

		// 3) Combustion + buoyancy (masked)
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
		                                                        params.temperatureRelease, params.expansionRate,
		                                                        /*active*/ C->d_active.get(), (int)N);
		CUDA_CHECK(cudaGetLastError());

		temperature_buoyancy<<<blocks, threads, 0, C->so.stream>>>(C->d_advectedVel.get(), o_temp.get(), C->d_advectedVel.get(), dt_sub,
		                                                           params.ambientTemp, params.buoyancyStrength,
		                                                           /*active*/ C->d_active.get(), (size_t)N);
		CUDA_CHECK(cudaGetLastError());

		// persist updated scalars
		CUDA_CHECK(cudaMemcpyAsync(d_fuel.get(), o_fuel.get(), N * sizeof(float), cudaMemcpyDeviceToDevice, C->so.stream));
		CUDA_CHECK(cudaMemcpyAsync(d_waste.get(), o_waste.get(), N * sizeof(float), cudaMemcpyDeviceToDevice, C->so.stream));
		CUDA_CHECK(cudaMemcpyAsync(d_temp.get(), o_temp.get(), N * sizeof(float), cudaMemcpyDeviceToDevice, C->so.stream));
		CUDA_CHECK(cudaMemcpyAsync(d_flame.get(), o_flame.get(), N * sizeof(float), cudaMemcpyDeviceToDevice, C->so.stream));

		// 4) Pressure solve (masked updates)
		CUDA_CHECK(cudaMemsetAsync(C->d_pressure.get(), 0, C->d_pressure.bytes(), C->so.stream));
		const float omega = 2.0f / (1.0f + sinf(3.14159f * dx));
		for (int it = 0; it < iteration; ++it) {
			redBlackGaussSeidelUpdate<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_divergence.get(),
			                                                                C->d_pressure.get(), dx, (size_t)N, 0, omega,
			                                                                /*active*/ C->d_active.get());
			redBlackGaussSeidelUpdate<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_divergence.get(),
			                                                                C->d_pressure.get(), dx, (size_t)N, 1, omega,
			                                                                /*active*/ C->d_active.get());
		}
		CUDA_CHECK(cudaGetLastError());

		// 4.5) Optional disturbance (masked) + short re-solve
		if (params.disturbanceEnable > 0.0f) {
			auto itDens = C->d_state.find("density");
			if (itDens != C->d_state.end()) {
				//inject_edge_disturbance_vel_from_density<<<blocks, threads, 0, C->so.stream>>>(
				//    C->d_grid, C->d_coords.get(), itDens->second.get(), params.maskDensityMin, params.maskDensityMax,
				//    hasCollision ? C->d_collisionSDF.get() : nullptr, hasCollision, C->d_advectedVel.get(), (int)N, dt_sub, inv_dx,
				//    params.disturbanceThreshold / float(SUBS), params.disturbanceStrength, params.disturbanceSwirl,
				//    (int)params.disturbanceFrequency, /*active*/ C->d_active.get());
				//CUDA_CHECK(cudaGetLastError());
			}

			divergence<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_advectedVel.get(), C->d_divergence.get(),
			                                                 inv_dx, /*active*/ C->d_active.get(), (size_t)N);
			CUDA_CHECK(cudaGetLastError());

			CUDA_CHECK(cudaMemsetAsync(C->d_pressure.get(), 0, C->d_pressure.bytes(), C->so.stream));
			const int it2 = max(2, iteration / 2);
			for (int it = 0; it < it2; ++it) {
				redBlackGaussSeidelUpdate<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_divergence.get(),
				                                                                C->d_pressure.get(), dx, (size_t)N, 0, omega,
				                                                                /*active*/ C->d_active.get());
				redBlackGaussSeidelUpdate<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_divergence.get(),
				                                                                C->d_pressure.get(), dx, (size_t)N, 1, omega,
				                                                                /*active*/ C->d_active.get());
			}
			CUDA_CHECK(cudaGetLastError());
		}

		// 5) Projection (masked)
		subtractPressureGradient<<<blocks, threads, 0, C->so.stream>>>(
		    C->d_grid, C->d_coords.get(), (size_t)N, C->d_advectedVel.get(), C->d_pressure.get(), C->d_velocity.get(),
		    hasCollision ? C->d_collisionSDF.get() : nullptr, hasCollision, inv_dx, /*active*/ C->d_active.get());
		CUDA_CHECK(cudaGetLastError());

		if (hasCollision) {
			enforceCollisionBoundaries<<<blocks, threads, 0, C->so.stream>>>(C->d_grid, C->d_coords.get(), C->d_velocity.get(),
			                                                                 C->d_collisionSDF.get(), dx, (size_t)N);
			CUDA_CHECK(cudaGetLastError());
		}

		// 6) Advect scalars (masked)
		if (!advNames.empty()) {
			std::vector<float*> inPtrs, outPtrs;
			inPtrs.reserve(advNames.size());
			outPtrs.reserve(advNames.size());
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
			                                                     hasCollision,
			                                                     /*active*/ C->d_active.get(), (int)N, dt_sub, inv_dx);
			CUDA_CHECK(cudaGetLastError());

			cudaFree(d_inArr);
			cudaFree(d_outArr);

			for (size_t i = 0; i < advNames.size(); ++i) {
				auto& dst = C->d_state.at(advNames[i]);
				CUDA_CHECK(cudaMemcpyAsync(dst.get(), outs[i].get(), N * sizeof(float), cudaMemcpyDeviceToDevice, C->so.stream));
			}
		}
	}

	CUDA_CHECK(cudaStreamSynchronize(C->so.stream));
}


extern "C" void HNS_GetActiveMask(void* ctx, std::vector<unsigned char>& hostMask) {
	auto* C = reinterpret_cast<SimContext*>(ctx);
	if (!C || !C->N) {
		hostMask.clear();
		return;
	}
	hostMask.resize(C->d_active.bytes());
	CUDA_CHECK(cudaMemcpy(hostMask.data(), C->d_active.get(), C->d_active.bytes(), cudaMemcpyDeviceToHost));
	
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
	//Compute(data, handle, iteration, dt, voxelSize, params, hasCollision, stream);
}

