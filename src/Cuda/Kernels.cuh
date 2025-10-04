#pragma once
#include "Utils.cuh"
#include "nanovdb/NanoVDB.h"

struct CombustionParams {
	float expansionRate;
	float temperatureRelease;
	float buoyancyStrength;
	float ambientTemp;
	float vorticityScale;
	float factorScale;
	float disturbanceEnable;
	float disturbanceStrength;
	float disturbanceSwirl;
	float disturbanceThreshold;
	float disturbanceGain;
	float disturbanceFrequency;
	int substeps;
	float gravity;
	float maskDensityMin;
	float maskDensityMax;
	int downloadVel;
	int simReset;
};

__global__ void build_activity_mask(const float* density, const nanovdb::Vec3f* vel, const float* src_density,
                                    const nanovdb::Vec3f* src_vel_v3, const float* src_vx, const float* src_vy, const float* src_vz,
                                    unsigned char* active, int N, float dens_eps, float vel_eps);

__global__ void advect_scalar(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* coords,
                              const nanovdb::Vec3f* velocityData, const float* inData, float* outData, const float* collisionSDF,
                              const bool hasCollision, size_t totalVoxels, float dt, float inv_voxelSize);


__global__ void advect_scalars(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* coords,
                               const nanovdb::Vec3f* velocityData, float** inDataArrays, float** outDataArrays, int numScalars,
                               const float* collisionSDF, const bool hasCollision, const unsigned char* active, size_t totalVoxels,
                               float dt, float inv_voxelSize);

__global__ void advect_vector(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* coords,
                              const nanovdb::Vec3f* velocityData, nanovdb::Vec3f* outVelocity, const float* collisionSDF,
                              const bool hasCollision, const unsigned char* active, size_t totalVoxels, float dt, float inv_voxelSize);

__global__ void divergence_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Vec3f* velocityData,
                               float* outDivergence, float inv_dx, int numLeaves);

__global__ void divergence(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coord,
                           const nanovdb::Vec3f* velocityData, float* outDivergence, float inv_dx, const unsigned char* active,
                           size_t totalVoxels);


__global__ void prolongate(const float* coarse, float* fine, nanovdb::Coord coarse_dims, nanovdb::Coord fine_dims);

__global__ void update_pressure(size_t totalVoxels, const float* pressure, const float* correction);

__global__ void restrict_to_2x2x2(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coords,
                                  const float* inData, float* outData, size_t totalVoxels);

__global__ void compute_residual(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coords,
                                 const float* pressure, const float* divergence, float* residual, float dx, size_t totalVoxels);

__global__ void redBlackGaussSeidelUpdate_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const float* divergence,
                                              float* pressure, float dx, size_t totalVoxels, int color, float omega);

__global__ void redBlackGaussSeidelUpdate(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coord,
                                          const float* divergence, float* pressure, float dx, size_t totalVoxels, int color, float omega,
                                          const unsigned char* active);

__global__ void subtractPressureGradient_opt(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Vec3f* velocity,
                                             const float* pressure, nanovdb::Vec3f* out, float inv_voxelSize, size_t numLeaves);

__global__ void subtractPressureGradient(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coords,
                                         size_t totalVoxels, const nanovdb::Vec3f* velocity, const float* pressure, nanovdb::Vec3f* out,
                                         const float* collisionSDF, const bool hasCollision, float inv_voxelSize,
                                         const unsigned char* active);

__global__ void temperature_buoyancy(const nanovdb::Vec3f* velocityData, const float* tempData, nanovdb::Vec3f* outVel, float dt,
                                     float ambient_temp, float buoyancy_strength, const unsigned char* active, size_t totalVoxels);

__global__ void combustion(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coords,
                           const float* fuelData, const float* tempData, float* outFuel, float* outTemp, float dt, float ignition_temp,
                           float combustion_rate, float heat_release, size_t totalVoxels);

__global__ void diffusion(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coords, const float* tempData,
                          const float* fuelData, float* outTemp, float* outFuel, float dt, float temp_diff, float fuel_diff,
                          float ambient_temp, size_t totalVoxels);

__global__ void combustion_oxygen(const float* fuelData, const float* wasteData, const float* temperatureData, float* divergenceData,
                                  const float* flameData, float* outFuel, float* outWaste, float* outTemperature, float* outFlame,
                                  float temp_gain, float expansion, const unsigned char* active, int N);

__global__ void vorticityConfinement(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* d_coord,
                                     const nanovdb::Vec3f* velocityData, nanovdb::Vec3f* outForce, float dt, float inv_dx,
                                     float confinementScale, float factorScale, const unsigned char* active, size_t totalVoxels);

__global__ void enforceCollisionBoundaries(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* domainGrid, const nanovdb::Coord* coords,
                                           nanovdb::Vec3f* velocityData, const float* collisionSDF, const float voxelSize,
                                           size_t totalVoxels);

__global__ void add_scalar_source(float* field, const float* src, float dt, const unsigned char* active, int N);

__global__ void add_velocity_source(nanovdb::Vec3f* vel, const nanovdb::Vec3f* src, float dt, const unsigned char* active, int N);

__global__ void add_velocity_source_xyz(nanovdb::Vec3f* vel, const float* sx, const float* sy, const float* sz, float dt,
                                        const unsigned char* active, int N);

__global__ void add_gravity(nanovdb::Vec3f* vel, float g, float dt, const unsigned char* active, int N);

//___global__ void inject_edge_disturbance_vel_from_density(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* __restrict__ domainGrid,
//                                                          const nanovdb::Coord* __restrict__ d_coords, const float* __restrict__ density,
//                                                          const float densMin, const float densMax, const float* __restrict__ collisionSDF,
//                                                          const bool hasCollision, nanovdb::Vec3f* __restrict__ inoutVel,
//                                                          const size_t totalVoxels, const float dt, const float inv_dx,
//                                                          const float gradThresh, const float strength, const float swirl,
//                                                          const int patternBlock, const unsigned char* __restrict__ active)


