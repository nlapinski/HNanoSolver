#include <openvdb/Types.h>

#include "../Utils/GridData.hpp"
#include "../Utils/Stencils.hpp"
#include "Kernels.cuh"
#include "nanovdb/NanoVDB.h"
#include "nanovdb/tools/cuda/PointsToGrid.cuh"

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


			// inject_edge_disturbance_vel_masked<<<gridSize, blockSize, 0, stream>>>(
			//     gpuGridPtr, d_coords.get(), d_pressure.get(), densPtr,
			//     /*densMin=*/params.maskDensityMin,  // e.g. 0.01f
			//     /*densMax=*/params.maskDensityMax,  // e.g. 1.0f
			//     hasCollisionData ? d_collisionSDF.get() : nullptr, hasCollisionData, d_advectedVel.get(), totalVoxels, dt_sub,
			//     inv_voxelSize, params.disturbanceThreshold / SUBS, params.disturbanceStrength, params.disturbanceSwirl,
			//     (int)params.disturbanceFrequency);
			//

			// inject_edge_disturbance_temp<<<gridSize, blockSize, 0, stream>>>(
			//     gpuGridPtr, d_coords.get(), d_pressure.get(), hasCollisionData ? d_collisionSDF.get() : nullptr, hasCollisionData,
			//     d_inputs.at("temperature").get(), totalVoxels, dt_sub, inv_voxelSize, params.disturbanceThreshold / SUBS,
			//     params.disturbanceGain / SUBS,
			//     (int)params.disturbanceFrequency);
			CUDA_CHECK(cudaGetLastError());

			// recompute divergence after velocity change, then short solve
			divergence<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_advectedVel.get(), d_divergence.get(),
			                                               inv_voxelSize, totalVoxels);
			CUDA_CHECK(cudaGetLastError());

			CUDA_CHECK(cudaMemsetAsync(d_pressure.get(), 0, d_pressure.bytes(), stream));
			const float omega = 2.0f / (1.0f + sinf(3.14159f * voxelSize));
			const int it2 = max(2, iteration / 2);
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

