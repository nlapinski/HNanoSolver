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

void old_Compute(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, const int iteration,
             const float dt, const float voxelSize, const CombustionParams& params, const bool hasCollision, const cudaStream_t& stream) {
	// --- Input Validation ---
	if (voxelSize <= 0.0f) {
		throw std::invalid_argument("voxelSize must be positive.");
	}
	if (dt < 0.0f) {  // Allow dt == 0? Might mean no update step.
		throw std::invalid_argument("dt (time step) cannot be negative.");
	}
	if (iteration <= 0) {
		throw std::invalid_argument("Number of pressure iterations must be positive.");
	}
	if (handle.isEmpty()) {
		throw std::invalid_argument("Invalid nanovdb::GridHandle provided (null grid).");
	}

	const size_t totalVoxels = data.size();
	if (totalVoxels == 0) {
		return;  // Nothing to compute
	}

	// Get NanoVDB grid pointer (ensure it's the expected type)
	// Kernels expect nanovdb::NanoGrid<nanovdb::ValueOnIndex>*
	auto gpuGridPtr = handle.deviceGrid<nanovdb::ValueOnIndex>();
	if (!gpuGridPtr) {
		throw std::runtime_error("Failed to get device grid pointer of type ValueOnIndex from handle.");
	}

	const float inv_voxelSize = 1.0f / voxelSize;

	// --- 1. Get Host Data Pointers and Validate ---

	// Velocity
	const auto vec3fBlocks = data.getBlocksOfType<openvdb::Vec3f>();
	if (vec3fBlocks.size() != 1) {
		throw std::runtime_error("Expected exactly one Vec3f block (velocity), found " + std::to_string(vec3fBlocks.size()));
	}
	const std::string& velocityBlockName = vec3fBlocks[0];

	nanovdb::Vec3f* h_velocity = reinterpret_cast<nanovdb::Vec3f*>(data.pValues<openvdb::Vec3f>(velocityBlockName));
	if (!h_velocity) {
		throw std::runtime_error("Host velocity data pointer is null for block: " + velocityBlockName);
	}

	// Coordinates
	nanovdb::Coord* h_coords = reinterpret_cast<nanovdb::Coord*>(data.pCoords());
	if (!h_coords) {
		throw std::runtime_error("Host coordinate data pointer is null.");
	}

	// Scalar Fields (float)
	const auto floatBlockNames = data.getBlocksOfType<float>();
	if (floatBlockNames.empty()) {
		throw std::runtime_error("No float blocks found in input data.");
	}

	// Get collision SDF if available
	float* h_collisionSDF = nullptr;
	bool hasCollisionData = false;
	if (hasCollision) {
		if (std::find(floatBlockNames.begin(), floatBlockNames.end(), "collision_sdf") != floatBlockNames.end()) {
			h_collisionSDF = data.pValues<float>("collision_sdf");
			if (h_collisionSDF) {
				hasCollisionData = true;
			}
		}
	}

	std::vector<float*> h_floatPointers(floatBlockNames.size());
	for (size_t i = 0; i < floatBlockNames.size(); ++i) {
		h_floatPointers[i] = data.pValues<float>(floatBlockNames[i]);
		if (!h_floatPointers[i]) {
			throw std::runtime_error("Host float data pointer is null for block: " + floatBlockNames[i]);
		}
	}

	// --- 2. Allocate Device Memory using RAII ---

	DeviceMemory<nanovdb::Vec3f> d_velocity(totalVoxels, stream);     // Input velocity / Final output velocity
	DeviceMemory<nanovdb::Vec3f> d_advectedVel(totalVoxels, stream);  // Intermediate velocity after advection/buoyancy
	DeviceMemory<nanovdb::Coord> d_coords(totalVoxels, stream);
	DeviceMemory<float> d_divergence(totalVoxels, stream);
	DeviceMemory<float> d_pressure(totalVoxels, stream);

	// Allocate memory for SDF collision if available
	DeviceMemory<float> d_collisionSDF;
	if (hasCollisionData) {
		d_collisionSDF = DeviceMemory<float>(totalVoxels, stream);
	}

	std::unordered_map<std::string, DeviceMemory<float>> d_inputs;
	std::unordered_map<std::string, DeviceMemory<float>> d_outputs;

	for (const auto& name : floatBlockNames) {
		// Use emplace for direct construction within the map
		d_inputs.emplace(std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(totalVoxels, stream));
		d_outputs.emplace(std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(totalVoxels, stream));
	}

	// --- 3. Initialize Device Memory (Memset & Memcpy H->D) ---

	// Memset intermediate/output buffers
	CUDA_CHECK(cudaMemsetAsync(d_advectedVel.get(), 0, d_advectedVel.bytes(), stream));
	CUDA_CHECK(cudaMemsetAsync(d_divergence.get(), 0, d_divergence.bytes(), stream));
	CUDA_CHECK(cudaMemsetAsync(d_pressure.get(), 0, d_pressure.bytes(), stream));
	// Output scalars don't strictly need zeroing if fully overwritten, but doesn't hurt.
	for (auto& [fst, snd] : d_outputs) {
		CUDA_CHECK(cudaMemsetAsync(snd.get(), 0, snd.bytes(), stream));
	}

	// Copy initial data from Host to Device (Asynchronously)
	CUDA_CHECK(cudaMemcpyAsync(d_velocity.get(), h_velocity, d_velocity.bytes(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_coords.get(), h_coords, d_coords.bytes(), cudaMemcpyHostToDevice, stream));

	// Copy collision SDF if available
	if (hasCollisionData) {
		CUDA_CHECK(cudaMemcpyAsync(d_collisionSDF.get(), h_collisionSDF, d_collisionSDF.bytes(), cudaMemcpyHostToDevice, stream));
	}

	for (size_t i = 0; i < floatBlockNames.size(); ++i) {
		const std::string& name = floatBlockNames[i];
		float* h_ptr = h_floatPointers[i];
		auto& d_input_mem = d_inputs.at(name);  // Use .at() for checked access
		CUDA_CHECK(cudaMemcpyAsync(d_input_mem.get(), h_ptr, d_input_mem.bytes(), cudaMemcpyHostToDevice, stream));
	}


	// --- 4. Kernel Launch Configuration ---
	constexpr int PREFERRED_BLOCK_SIZE = 256;
	// Ensure block size is not larger than the maximum allowed or the number of elements
	// Note: Kernels might have internal assumptions about block size (e.g., shared memory).
	// If using _opt kernels, block size must match their requirements (e.g., 8x8x8=512).
	// For the current non-opt kernels, 256 is usually safe.
	int maxThreadsPerBlock = 0;
	maxThreadsPerBlock = 1024;  // Assume a reasonable default if not querying
	dim3 leafDim(8, 8, 8);
	uint32_t leafNum = totalVoxels / 512;

	const int blockSize = std::min((int)totalVoxels, std::min(PREFERRED_BLOCK_SIZE, maxThreadsPerBlock));
	const int gridSize = (totalVoxels + blockSize - 1) / blockSize;

	// --- 5. Simulation Pipeline Kernels ---

	// If collision is enabled, enforce initial collision boundaries
	if (hasCollisionData) {
		enforceCollisionBoundaries<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_velocity.get(), d_collisionSDF.get(),
		                                                               voxelSize, totalVoxels);
		CUDA_CHECK(cudaGetLastError());
	}

	// Step 1: Advect velocity field
	// Input: d_velocity (current state), d_coords
	// Output: d_advectedVel (intermediate buffer)
	{
		ScopedTimerGPU timer("HNanoSolver::Advect::Velocity", 12 * 10 + 12, totalVoxels);
		advect_vector<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(),
		                                                  d_velocity.get(),     // Velocity field to advect (and sample from)
		                                                  d_advectedVel.get(),  // Output buffer for advected result
		                                                  hasCollisionData ? d_collisionSDF.get() : nullptr, hasCollisionData, totalVoxels,
		                                                  dt, inv_voxelSize);
		CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors
	}

	{
		ScopedTimerGPU timer("HNanoSolver::VorticityConfinement", 12 * 43, totalVoxels);
		vorticityConfinement<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_advectedVel.get(), d_advectedVel.get(), dt,
		                                                         inv_voxelSize, params.vorticityScale, params.factorScale, totalVoxels);
	}

	// Step 2: Calculate velocity field divergence
	// Input: d_velocity (using original velocity for divergence, common practice), d_coords
	// Output: d_divergence
	{
		ScopedTimerGPU timer("HNanoSolver::Divergence", 12 + 12 * 6 + 4, totalVoxels);
		// Using the non-optimized divergence kernel as in the original code
		divergence<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(),
		                                               d_advectedVel.get(),
		                                               d_divergence.get(), inv_voxelSize, totalVoxels);
		CUDA_CHECK(cudaGetLastError());
	}

	// Step 3: Combustion and Buoyancy
	{
		// Validate required fields exist before launching kernels
		const std::vector<std::string> required_comb_fields = {"fuel", "waste", "temperature", "flame"};
		for (const auto& field : required_comb_fields) {
			if (d_inputs.find(field) == d_inputs.end()) {
				throw std::runtime_error("Missing required input field for combustion: " + field);
			}
			if (d_outputs.find(field) == d_outputs.end()) {
				throw std::runtime_error("Missing required output field for combustion: " + field);
			}
		}
		// Check temperature exists for buoyancy
		if (d_outputs.find("temperature") == d_outputs.end()) {  // Buoyancy uses output T
			throw std::runtime_error("Missing output buffer for temperature (needed for buoyancy)");
		}


		// Launch Combustion Kernel
		// Input: d_inputs[...], d_divergence (read/write)
		// Output: d_outputs[...]
		{
			ScopedTimerGPU t("HNanoSolver::Combustion::Combust", 4 * 9, totalVoxels);
			combustion_oxygen<<<gridSize, blockSize, 0, stream>>>(
			    d_inputs.at("fuel").get(), d_inputs.at("waste").get(), d_inputs.at("temperature").get(),
			    d_divergence.get(),  // Input/Output - combustion adds expansion
			    d_inputs.at("flame").get(),
			    d_outputs.at("fuel").get(),  // Write results to output buffers
			    d_outputs.at("waste").get(), d_outputs.at("temperature").get(), d_outputs.at("flame").get(), params.temperatureRelease,
			    params.expansionRate, totalVoxels);
			CUDA_CHECK(cudaGetLastError());
		}

		// Launch Buoyancy Kernel
		// Input: d_velocity (original velocity to base force on), d_outputs["temperature"] (result from combustion)
		// Output: d_advectedVel (add buoyancy force to the already advected velocity)
		{
			ScopedTimerGPU t("HNanoSolver::Buoyancy", 12 * 2 + 4, totalVoxels);  // Estimate bytes
			temperature_buoyancy<<<gridSize, blockSize, 0, stream>>>(
			    d_advectedVel.get(),                // Read the intermediate (advected) velocity
			    d_outputs.at("temperature").get(),  // Read temperature *after* combustion
			    d_advectedVel.get(),                // Write buoyancy force additively back to the intermediate velocity
			    dt, params.ambientTemp, params.buoyancyStrength, totalVoxels);
			CUDA_CHECK(cudaGetLastError());
		}

		// SWAP BUFFERS for fields modified by combustion
		// The results are in d_outputs. They need to be the input for the next step (scalar advection).
		// Move the DeviceMemory objects.
		for (const auto& field : required_comb_fields) {
			// Move the result from output map to input map
			d_inputs.at(field) = std::move(d_outputs.at(field));
			// Create a new (empty) output buffer in the output map for the next step
			d_outputs.at(field) = DeviceMemory<float>(totalVoxels, stream);
			// Need to memset the new output buffer if kernels don't guarantee writing all voxels
			CUDA_CHECK(cudaMemsetAsync(d_outputs.at(field).get(), 0, d_outputs.at(field).bytes(), stream));
		}
		// Fields not in required_comb_fields still have original data in d_inputs
		// and (potentially zeroed) buffers in d_outputs.

	}  // End Combustion/Buoyancy block


	// Step 4: Pressure solver (Red-black Gauss-Seidel iterations)
	// Input: d_divergence (potentially modified by combustion), d_coords
	// Output: d_pressure (updated iteratively)
	{
		const float omega = 2.0f / (1.0f + sinf(static_cast<float>(3.14159) * voxelSize));  // Using cmath
		ScopedTimerGPU timer("HNanoSolver::Pressure", 12 + 4 * 9, totalVoxels * iteration);

		for (int iter = 0; iter < iteration; ++iter) {
			// Red phase
			// Using the non-optimized redBlackGaussSeidelUpdate kernel
			redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_divergence.get(), d_pressure.get(),
			                                                              voxelSize, totalVoxels, 0, omega);  // color = 0 for red

			// Black phase
			redBlackGaussSeidelUpdate<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_divergence.get(), d_pressure.get(),
			                                                              voxelSize, totalVoxels, 1, omega);  // color = 1 for black
		}

		CUDA_CHECK(cudaGetLastError());
	}


	// Step 5: Apply pressure gradient (Projection)
	// Input: d_advectedVel (intermediate velocity), d_pressure, d_coords
	// Output: d_velocity (final divergence-free velocity for this timestep)
	{
		constexpr int bytes_per_voxel = sizeof(nanovdb::Vec3f) * 2 + sizeof(float) * 6;  // 48
		ScopedTimerGPU timer("HNanoSolver::Projection", bytes_per_voxel, totalVoxels);
		// Using the non-optimized subtractPressureGradient kernel
		subtractPressureGradient<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), totalVoxels,
		                                                             d_advectedVel.get(),  // Input velocity field to correct (u*)
		                                                             d_pressure.get(),     // Pressure field
		                                                             d_velocity.get(),     // Output projected velocity (u_n+1)
		                                                             hasCollisionData ? d_collisionSDF.get() : nullptr, hasCollisionData,
		                                                             inv_voxelSize);
		CUDA_CHECK(cudaGetLastError());
	}

	// Apply collision boundaries again after projection
	if (hasCollisionData) {
		enforceCollisionBoundaries<<<gridSize, blockSize, 0, stream>>>(gpuGridPtr, d_coords.get(), d_velocity.get(), d_collisionSDF.get(),
		                                                               voxelSize, totalVoxels);
		CUDA_CHECK(cudaGetLastError());
	}

	// Step 6: Advect all scalar fields
	// Input: d_velocity (final projected velocity), d_inputs (state including post-combustion), d_coords
	// Output: d_outputs
	{
		std::vector<float*> d_inDataArrays;
		std::vector<float*> d_outDataArrays;

		// Populate arrays from your maps
		for (const auto& name : floatBlockNames) {
			if (name == "collision_sdf") continue;
			d_inDataArrays.push_back(d_inputs.at(name).get());
			d_outDataArrays.push_back(d_outputs.at(name).get());
		}

		// Allocate device-side array pointers
		float** d_inDataArraysDev;
		float** d_outDataArraysDev;

		cudaMalloc(&d_inDataArraysDev, d_inDataArrays.size() * sizeof(float*));
		cudaMalloc(&d_outDataArraysDev, d_outDataArrays.size() * sizeof(float*));

		cudaMemcpyAsync(d_inDataArraysDev, d_inDataArrays.data(), d_inDataArrays.size() * sizeof(float*), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(d_outDataArraysDev, d_outDataArrays.data(), d_outDataArrays.size() * sizeof(float*), cudaMemcpyHostToDevice,
		                stream);

		ScopedTimerGPU timer("HNanoSolver::Advect::Scalar", 12 * 2 + 12 + 4 * 10, totalVoxels);

		// Launch the single kernel pass
		advect_scalars<<<gridSize, blockSize, 0, stream>>>(
		    gpuGridPtr, d_coords.get(), d_velocity.get(), d_inDataArraysDev, d_outDataArraysDev, static_cast<int>(d_inDataArrays.size()),
		    hasCollisionData ? d_collisionSDF.get() : nullptr, hasCollisionData, totalVoxels, dt, inv_voxelSize);

		// Free temporary arrays
		cudaFree(d_inDataArraysDev);
		cudaFree(d_outDataArraysDev);


		CUDA_CHECK(cudaGetLastError());  // Check each kernel launch
	}

	// --- 6. Copy Results back from Device to Host ---

	// Copy final projected velocity
	CUDA_CHECK(cudaMemcpyAsync(h_velocity, d_velocity.get(), d_velocity.bytes(), cudaMemcpyDeviceToHost, stream));

	// Copy final advected scalar fields (results are in d_outputs after last step)
	for (size_t i = 0; i < floatBlockNames.size(); ++i) {
		const std::string& name = floatBlockNames[i];
		float* h_ptr = h_floatPointers[i];        // Get corresponding host pointer
		auto& d_output_mem = d_outputs.at(name);  // Result is in output buffer from scalar advection
		CUDA_CHECK(cudaMemcpyAsync(h_ptr, d_output_mem.get(), d_output_mem.bytes(), cudaMemcpyDeviceToHost, stream));
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

