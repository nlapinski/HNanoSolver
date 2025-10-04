#include "SOP_HNanoSolver.hpp"

#include <GU/GU_Detail.h>
#include <GU/GU_PrimVDB.h>
#include <UT/UT_DSOVersion.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/VolumeAdvect.h>

#include "../Utils/GridBuilder.hpp"
#include "../Utils/ScopedTimer.hpp"
#include "../Utils/Utils.hpp"
#include "nanovdb/tools/CreateNanoGrid.h"
#include "nanovdb/NanoVDB.h"



void newSopOperator(OP_OperatorTable* table) {
	table->addOperator(new OP_Operator("hnanosolver", "HNanoSolver", SOP_HNanoSolver::myConstructor, SOP_HNanoSolver::buildTemplates(), 2,
	                                   3, nullptr, OP_FLAG_GENERATOR));
}

const char* const SOP_HNanoSolverVerb::theDsFile = R"THEDSFILE(
{
    name        parameters
	parm {
		name "debug_output"
		label "Enable debug printout"
        type    integer
        size    1
        range   { 0 1 }
		default { "0" }
	}
	parm {
		name "download_vel"
		label "Download vel"
        type    integer
        size    1
        range   { 0 1 }
		default { "0" }
	}
	parm {
		name "sim_reset"
		label "Reset Sim"
        type    integer
        size    1
        range   { 0 1 }
		default { "$F==$RFSTART" }
	}
    parm {
        name    "timestep"
        label   "Time Step"
        type    float
        size    1
        default { "1/$FPS" }
    }
	parm {
		name "padding"
		label "Voxel Padding"
        type    integer
        size    1
        range   { 1! 100 }
	}
	parm {
		name "iterations"
		label "Pressure Projection"
        type    integer
        size    1
        range   { 1! 100 }
	}
	parm {
		name "expansion_rate"
		label "Expansion Rate"
		type float
		size 1
		default { "0.1" }
	}
	parm {
		name "temperature_gain"
		label "Temperature Gain"
		type float
		size 1
		default { "0.5" }
	}
	parm {
		name "buoyancy_strength"
		label "Buoyancy Strength"
		type float
		size 1
		default { "1.0" }
	}
	parm {
		name "ambient_temp"
		label "Ambient Temperature"
		type float
		size 1
		default { "23.0" }
	}
	parm {
		name "vorticity"
		label "Vorticity Scale"
		type float
		size 1
		default { "1" }
	}
	parm {
		name "factor_scale"
		label "Vorticity Factor Scale"
		type float
		size 1
		default { "0.5" }
	}
	parm {
		name "disturbance_strength"
		label "Disturbance Strength"
		type float
		size 1
		default { "0.5" }
	}
	parm {
		name "disturbance_swirl"
		label "Disturbance Swirl Size"
		type float
		size 1
		default { "0.05" }
	}
	parm {
		name "disturbance_threshold"
		label "Disturbance Threshold"
		type float
		size 1
		default { "5.00" }
	}
	parm {
		name "disturbance_gain"
		label "Disturbance Gain"
		type float
		size 1
		default { "0.2" }
	}
	parm {
		name "disturbance_enable"
		label "Disturbance Enable"
		type integer
		size 1
		default { "1" }
	}
	parm {
		name "disturbance_frequency"
		label "Disturbance Frequency"
		type integer
		size 1
		default { "1" }
	}
	parm {
		name "substeps"
		label "Substeps"
		type integer
		size 1
		default { "1" }
	}
	parm {
		name "gravity"
		label "Graviy in Y"
		type float
		size 1
		default { "1.00" }
	}
	parm {
		name "maskDensityMin"
		label "Mask Density min"
		type float
		size 1
		default { "0.05" }
	}
	parm {
		name "maskDensityMax"
		label "Mask Density max"
		type float
		size 1
		default { "0.50" }
	}
}
)THEDSFILE";


PRM_Template* SOP_HNanoSolver::buildTemplates() {
	static PRM_TemplateBuilder templ("SOP_HNanoSolver.cpp", SOP_HNanoSolverVerb::theDsFile);
	return templ.templates();
}
const SOP_NodeVerb::Register<SOP_HNanoSolverVerb> SOP_HNanoSolverVerb::theVerb;

const SOP_NodeVerb* SOP_HNanoSolver::cookVerb() const { return SOP_HNanoSolverVerb::theVerb.get(); }


// Opaque GPU context. Implemented in HNanoSolver.cu.
extern "C" void HNS_DownloadForOutput(void* ctx, HNS::GridIndexedData& data);
extern "C" void HNS_Advance(void* ctx, int iteration, float dt, const CombustionParams& params);
extern "C" void HNS_UploadSources(void* ctx, HNS::GridIndexedData& data);
extern "C" void HNS_EnsureContext(void** ctx, HNS::GridIndexedData& data, const nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>& handle,
                                  float voxelSize, bool hasCollision);
extern "C" void HNS_DestroyContext(void** ctx);

void SOP_HNanoSolverVerb::cook(const CookParms& cookparms) const {
	const auto& P = cookparms.parms<SOP_HNanoSolverParms>();
	auto* cache = dynamic_cast<SOP_HNanoSolverCache*>(cookparms.cache());
	if (!cache) {
		std::printf("No sop cache!?\n");
	}

	GU_Detail* gdp = cookparms.gdh().gdpNC();

	CombustionParams params{};
	params.expansionRate = P.getExpansion_rate();
	params.temperatureRelease = P.getTemperature_gain();
	params.buoyancyStrength = P.getBuoyancy_strength();
	params.ambientTemp = P.getAmbient_temp();
	params.vorticityScale = P.getVorticity();
	params.factorScale = P.getFactor_scale();
	params.disturbanceEnable = P.getDisturbance_enable();
	params.disturbanceSwirl = P.getDisturbance_swirl();
	params.disturbanceStrength = P.getDisturbance_strength();
	params.disturbanceThreshold = P.getDisturbance_threshold();
	params.disturbanceGain = P.getDisturbance_gain();
	params.disturbanceFrequency = float(P.getDisturbance_frequency());
	params.substeps = P.getSubsteps();
	params.gravity = P.getGravity();
	params.maskDensityMin = P.getMaskdensitymin();
	params.maskDensityMax = P.getMaskdensitymax();
	params.downloadVel = P.getDownload_vel();
	params.simReset = P.getSim_reset();

	if (params.simReset) {
		cache->clear();
	}

	std::vector<openvdb::GridBase::Ptr> fb, src, col;
	if (auto err = loadGrid(cookparms.inputGeo(0), fb); err != UT_ERROR_NONE) {
		cookparms.sopAddError(SOP_MESSAGE, "Failed to load feedback grid");
		return;
	}
	loadGrid(cookparms.inputGeo(1), src);
	loadGrid(cookparms.inputGeo(2), col);

	std::vector<openvdb::FloatGrid::Ptr> fbF, srcF;
	std::vector<openvdb::VectorGrid::Ptr> fbV, srcV;
	openvdb::FloatGrid::Ptr sdf;
	bool hasCollision = false;

	for (auto& g : fb) {
		if (auto v = openvdb::gridPtrCast<openvdb::VectorGrid>(g))
			fbV.push_back(v);
		else if (auto f = openvdb::gridPtrCast<openvdb::FloatGrid>(g))
			fbF.push_back(f);
	}
	for (auto& g : src) {
		if (auto v = openvdb::gridPtrCast<openvdb::VectorGrid>(g))
			srcV.push_back(v);
		else if (auto f = openvdb::gridPtrCast<openvdb::FloatGrid>(g))
			srcF.push_back(f);
	}
	for (auto& g : col) {
		if (auto f = openvdb::gridPtrCast<openvdb::FloatGrid>(g)) {
			sdf = f;
			hasCollision = true;
			break;
		}
	}

	if (fbV.empty()) {
		cookparms.sopAddError(SOP_MESSAGE, "Velocity VectorGrid required on input 0");
		return;
	}

	const auto primaryV = fbV[0];
	const float voxelSize = float(primaryV->voxelSize()[0]);

	// Domain mask with padding
	openvdb::MaskGrid::Ptr domain = openvdb::MaskGrid::create();
	domain->setTransform(primaryV->transform().copy());
	domain->tree().topologyUnion(primaryV->tree());
	openvdb::tools::morphology::Morphology<openvdb::MaskTree>(domain->tree())
	    .dilateVoxels(P.getPadding(), openvdb::tools::NN_FACE_EDGE_VERTEX, openvdb::tools::IGNORE_TILES);
	if (hasCollision && sdf) {
		domain->tree().topologyUnion(sdf->tree());
	}

	// Tile activity mask from domain leaves
	openvdb::MaskGrid::Ptr tileMask = openvdb::MaskGrid::create(false);
	tileMask->setTransform(domain->transform().copy());
	{
		auto& ttree = tileMask->tree();
		for (auto it = domain->tree().cbeginLeaf(); it; ++it) {
			const openvdb::Coord org = it->origin();
			const openvdb::CoordBBox bbox(org, org + openvdb::Coord(7, 7, 7));
			ttree.fill(bbox, true, true);
		}
		if (hasCollision && sdf) {
			ttree.topologyUnion(sdf->tree());
		}
		openvdb::tools::morphology::Morphology<openvdb::MaskTree>(ttree).dilateVoxels(P.getPadding(), openvdb::tools::NN_FACE_EDGE_VERTEX,
		                                                                              openvdb::tools::IGNORE_TILES);
	}

	// Convert tile mask to float activity grid
	openvdb::FloatGrid::Ptr tileActivity = openvdb::FloatGrid::create(0.0f);
	tileActivity->setName("__tile_activity");
	tileActivity->setTransform(domain->transform().copy());
	tileActivity->tree().topologyUnion(tileMask->tree());
	for (auto it = tileActivity->beginValueOn(); it; ++it) {
		it.setValue(1.0f);
	}

	// Build indexed payload
	HNS::GridIndexedData data;
	HNS::IndexGridBuilder<openvdb::MaskGrid> builder(domain, &data);
	builder.setAllocType(AllocationType::Standard);

	for (auto& f : fbF) builder.addGrid(f, f->getName());
	for (auto& v : fbV) builder.addGrid(v, v->getName());
	if (hasCollision && sdf) builder.addGridSDF(sdf, "collision_sdf");
	for (auto& f : srcF) builder.addGrid(f, f->getName());
	for (auto& v : srcV) builder.addGrid(v, v->getName());
	builder.addGrid(tileActivity, tileActivity->getName());

	builder.build();

	if (data.size() == 0) {
		gdp->clearAndDestroy();
		cookparms.sopAddWarning(SOP_MESSAGE, "No active voxels");
		return;
	}

	try {
		nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle;
		CreateIndexGrid(data, handle, voxelSize);
		cache->nvdb = std::move(handle);
	} catch (const std::exception& e) {
		cookparms.sopAddError(SOP_MESSAGE, e.what());
		return;
	}

	try {
		HNS_EnsureContext(reinterpret_cast<void**>(&cache->user), data, cache->nvdb, voxelSize, hasCollision);
	} catch (const std::exception& e) {
		cookparms.sopAddError(SOP_MESSAGE, e.what());
		return;
	}

	try {
		HNS_UploadSources(cache->user, data);
		HNS_Advance(cache->user, P.getIterations(), float(P.getTimestep()), params);
		gdp->clearAndDestroy();
		HNS_DownloadForOutput(cache->user, data);
	} catch (const std::exception& e) {
		cookparms.sopAddError(SOP_MESSAGE, e.what());
		return;
	}

	for (auto& f : fbF) {
		auto out = builder.writeIndexGrid<openvdb::FloatGrid>(f->getName(), f->voxelSize()[0]);
		GU_PrimVDB::buildFromGrid(*gdp, out, nullptr, out->getName().c_str());
	}
	if (params.downloadVel) {
		for (auto& v : fbV) {
			auto out = builder.writeIndexGrid<openvdb::VectorGrid>(v->getName(), v->voxelSize()[0]);
			GU_PrimVDB::buildFromGrid(*gdp, out, nullptr, out->getName().c_str());
		}
	}
}
