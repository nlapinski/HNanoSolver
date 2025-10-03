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


void SOP_HNanoSolverVerb::cook(const CookParms& cookparms) const {
	const auto& sopparms = cookparms.parms<SOP_HNanoSolverParms>();
	const auto sopcache = dynamic_cast<SOP_HNanoSolverCache*>(cookparms.cache());

	openvdb_houdini::HoudiniInterrupter boss("Computing VDB grids");

	GU_Detail* detail = cookparms.gdh().gdpNC();
	const GU_Detail* feedback_input = cookparms.inputGeo(0);
	const GU_Detail* source_input = cookparms.inputGeo(1);
	const GU_Detail* collision_input = cookparms.inputGeo(2);

	std::vector<openvdb::GridBase::Ptr> feedback_grids;
	std::vector<openvdb::GridBase::Ptr> source_grids;     // Velocity grid ( len = 1 )
	std::vector<openvdb::GridBase::Ptr> collision_grids;  // SDF grid for collisions
	
	int debugOutput = sopparms.getDebug_output();

	if (auto err = loadGrid(feedback_input, feedback_grids); err != UT_ERROR_NONE) {
		err = cookparms.sopAddError(SOP_MESSAGE, "Failed to load feedback grid");
		return;
	}

	if (auto err = loadGrid(source_input, source_grids); err != UT_ERROR_NONE && err != UT_ERROR_ABORT) {
		err = cookparms.sopAddError(SOP_MESSAGE, "Failed to load source grid");
		return;
	}

	// Load collision SDF grid
	openvdb::FloatGrid::Ptr sdf_grid;
	bool has_collision = false;
	if (collision_input) {
		if (auto err = loadGrid(collision_input, collision_grids); err != UT_ERROR_NONE && err != UT_ERROR_ABORT) {
			cookparms.sopAddWarning(SOP_MESSAGE, "Failed to load collision grid, continuing without collision");
		} else if (!collision_grids.empty()) {
			// Find the first float grid to use as SDF
			for (const auto& grid : collision_grids) {
				if (auto float_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid)) {
					sdf_grid = float_grid;
					has_collision = true;
					break;
				}
			}

			if (!has_collision) {
				cookparms.sopAddWarning(SOP_MESSAGE, "No valid SDF grid found in collision input, continuing without collision");
			}
		}
	}

	std::vector<openvdb::FloatGrid::Ptr> feedback_float_grids;
	std::vector<openvdb::FloatGrid::Ptr> source_float_grids;
	std::vector<openvdb::VectorGrid::Ptr> feedback_vector_grids;
	std::vector<openvdb::VectorGrid::Ptr> source_vector_grids;

	for (const auto& grid : feedback_grids) {
		if (auto float_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid)) {
			feedback_float_grids.push_back(float_grid);
		} else if (auto vector_grid = openvdb::gridPtrCast<openvdb::VectorGrid>(grid)) {
			feedback_vector_grids.push_back(vector_grid);
		}
	}

	bool isSourced = !source_grids.empty();

	if (isSourced) {
		if (debugOutput) {
			ScopedTimer timer("HNanoSolver::Sourcing");
		}
		for (const auto& grid : source_grids) {
			if (auto float_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid)) {
				source_float_grids.push_back(float_grid);
			} else if (auto vector_grid = openvdb::gridPtrCast<openvdb::VectorGrid>(grid)) {
				source_vector_grids.push_back(vector_grid);
			}
		}
	}

	// Assume the first vector grid is the primary one for topology/voxel size
	const openvdb::VectorGrid::Ptr primaryVelocityGrid = feedback_vector_grids[0];
	const float voxelSize = static_cast<float>(primaryVelocityGrid->voxelSize()[0]);  // Assuming uniform voxels
	const float deltaTime = static_cast<float>(sopparms.getTimestep());

	openvdb::MaskGrid::Ptr domainGrid = openvdb::MaskGrid::create();
	{
		if (debugOutput) {
			ScopedTimer timer("HNanoSolver::DefineTopology");
		}
		domainGrid->tree().topologyUnion(primaryVelocityGrid->tree());

		openvdb::tools::morphology::Morphology<openvdb::MaskTree> morph(domainGrid->tree());
		morph.dilateVoxels(sopparms.getPadding(), openvdb::tools::NN_FACE_EDGE_VERTEX, openvdb::tools::IGNORE_TILES);

		if (has_collision && sdf_grid) {
			domainGrid->tree().topologyUnion(sdf_grid->tree());
		}
	}

	HNS::GridIndexedData data;
	HNS::IndexGridBuilder<openvdb::MaskGrid> builder(domainGrid, &data);
	builder.debugOutput = debugOutput;
	builder.setAllocType(AllocationType::Standard);
	{
		for (const auto& grid : feedback_float_grids) {
			builder.addGrid(grid, grid->getName());
		}
		for (const auto& grid : feedback_vector_grids) {
			builder.addGrid(grid, grid->getName());
		}

		// Add SDF grid if available
		if (has_collision && sdf_grid) {
			builder.addGridSDF(sdf_grid, "collision_sdf");
		}

		// Add source grids as inputs for CUDA sourcing
		for (const auto& grid : source_float_grids) {
			// Ensure names: src_temperature, src_density, src_fuel
			builder.addGrid(grid, grid->getName());
		}
		for (const auto& grid : source_vector_grids) {
			// Ensure name: src_vel
			builder.addGrid(grid, grid->getName());
		}

		builder.build();
	}


	if (data.size() == 0) {
		cookparms.sopAddWarning(SOP_MESSAGE, "No active voxels found. Simulation skipped.");
		return;  // Nothing to simulate
	}

	// --- 4. Create NanoVDB Acceleration Structure ---
	nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle;
	{
		if (debugOutput) {
			ScopedTimer timer("Building NanoVDB Index Grid");
		}
		try {
			CreateIndexGrid(data, handle, voxelSize);
		} catch (const std::exception& e) {
			cookparms.sopAddError(SOP_MESSAGE, (std::string("Failed to create NanoVDB grid: ") + e.what()).c_str());
			return;
		}
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	{
		const int iterations = sopparms.getIterations();

		CombustionParams params{};
		params.expansionRate = sopparms.getExpansion_rate();
		params.temperatureRelease = sopparms.getTemperature_gain();
		params.buoyancyStrength = sopparms.getBuoyancy_strength();
		params.ambientTemp = sopparms.getAmbient_temp();
		params.vorticityScale = sopparms.getVorticity();
		params.factorScale = sopparms.getFactor_scale();

		params.disturbanceEnable = sopparms.getDisturbance_enable();
		params.disturbanceSwirl = sopparms.getDisturbance_swirl();
		params.disturbanceStrength = sopparms.getDisturbance_strength();
		params.disturbanceThreshold = sopparms.getDisturbance_threshold();
		params.disturbanceGain = sopparms.getDisturbance_gain();
		params.disturbanceFrequency = (float)sopparms.getDisturbance_frequency();
		params.substeps = sopparms.getSubsteps();
		params.gravity = sopparms.getGravity();

		params.maskDensityMin = sopparms.getMaskdensitymin();
		params.maskDensityMax = sopparms.getMaskdensitymax();

		Compute_Sim(data, handle, iterations, deltaTime, primaryVelocityGrid->voxelSize()[0], params, has_collision, stream);
		
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	{
		for (const auto& grid : feedback_float_grids) {
			auto out = builder.writeIndexGrid<openvdb::FloatGrid>(grid->getName(), grid->voxelSize()[0]);
			if (debugOutput) {
				std::printf("Write back grid-> %s\n", grid->getName().c_str());
			}
			GU_PrimVDB::buildFromGrid(*detail, out, nullptr, out->getName().c_str());
		}

		for (const auto& grid : feedback_vector_grids) {
			auto out = builder.writeIndexGrid<openvdb::VectorGrid>(grid->getName(), grid->voxelSize()[0]);
			if (debugOutput) {
				std::printf("Write back grid-> %s\n", grid->getName().c_str());
			}
			GU_PrimVDB::buildFromGrid(*detail, out, nullptr, out->getName().c_str());
		}
	}
}
