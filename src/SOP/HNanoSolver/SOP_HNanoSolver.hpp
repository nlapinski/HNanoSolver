/*
 * Created by zphrfx on 25/11/2024.
 *
 * Implement the whole Volumetric solver logic in one node to avoid the overhead of the transfer of the data between nodes.
 * This will allow us to use the GPU to its full potential.
 * The SOP_HNanoSolver will be a single node that will take the source VDBs and output the final advected VDBs.
 */

#pragma once

#include <PRM/PRM_TemplateBuilder.h>
#include <SOP/SOP_Node.h>
#include <SOP/SOP_NodeVerb.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/VolumeAdvect.h>

#include "../Utils/GridData.hpp"
#include "SOP_HNanoSolver.proto.h"
#include "nanovdb/GridHandle.h"
#include "nanovdb/cuda/DeviceBuffer.h"

struct CombustionParams {
	float expansionRate;
	float temperatureRelease;
	float buoyancyStrength;
	float ambientTemp;
	float vorticityScale;
	float factorScale;
	// disturb parms
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


class SOP_HNanoSolver final : public SOP_Node {
   public:
	SOP_HNanoSolver(OP_Network* net, const char* name, OP_Operator* op) : SOP_Node(net, name, op) {
		mySopFlags.setManagesDataIDs(true);

		// It means we cook at every frame change.
		OP_Node::flags().setTimeDep(true);
	}
	
	

	~SOP_HNanoSolver() override = default;

	static PRM_Template* buildTemplates();

	static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op) { return new SOP_HNanoSolver(net, name, op); }

	OP_ERROR cookMySop(OP_Context& context) override { return cookMyselfAsVerb(context); }

	const SOP_NodeVerb* cookVerb() const override;

	const char* inputLabel(OP_InputIdx idx) const override {
		switch (idx) {
			case 0:
				return "Input Grids";
			default:
				return "Sourcing Grids";
		}
	}
};

extern "C" void HNS_DestroyContext(void** ctx);

class SOP_HNanoSolverCache final : public SOP_NodeCache {
   public:
	SOP_HNanoSolverCache() : SOP_NodeCache(), user(nullptr) {}
	virtual ~SOP_HNanoSolverCache(){};
	void* user = nullptr;
	nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> nvdb;
	openvdb::MaskGrid::Ptr domainMask;
	openvdb::CoordBBox fixedDomainBox;  // persistent fixed box for the whole sim
	size_t lastN = 0;
	float voxelSize = 0.f;

	void clear() {
		if (user) HNS_DestroyContext(&user);
		user = nullptr;
		nvdb = nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>();
		domainMask.reset();
		fixedDomainBox = openvdb::CoordBBox();
		lastN = 0;
		voxelSize = 0.f;
	}
};

class SOP_HNanoSolverVerb final : public SOP_NodeVerb {
   public:
	SOP_HNanoSolverVerb() = default;
	~SOP_HNanoSolverVerb() override = default;
	virtual SOP_NodeParms* allocParms() const { return new SOP_HNanoSolverParms; }
	virtual SOP_NodeCache* allocCache() const { return new SOP_HNanoSolverCache(); }
	virtual UT_StringHolder name() const { return "HNanoSolver"; }
	SOP_NodeVerb::CookMode cookMode(const SOP_NodeParms* parms) const override { return SOP_NodeVerb::COOK_GENERATOR; }
	void cook(const SOP_NodeVerb::CookParms& cookparms) const override;
	static const SOP_NodeVerb::Register<SOP_HNanoSolverVerb> theVerb;
	static const char* const theDsFile;
};

extern "C" void CreateIndexGrid(HNS::GridIndexedData& data, nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, float voxelSize);
extern "C" void Compute_Sim(HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle, int iteration,
                            float dt, float voxelSize, const CombustionParams& params, bool hasCollision, const cudaStream_t& stream);