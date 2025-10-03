enable_language(CUDA)

set(SRC_FILES 
        Utils.cuh
        Kernels.cuh
        Kernel.cu
        Advection.cu
        PressureProjection.cu
        Combustion.cu
        HNanoSolver.cu)

set(EXTERNAL_INCLUDE
    "${CMAKE_CURRENT_LIST_DIR}/../../externals/nanovdb/tools/cuda/PointsToGrid.cuh"
    "${CMAKE_CURRENT_LIST_DIR}/../../externals/nanovdb/NanoVDB.h"
)

set(HDK_INCLUDE
    "$ENV{HFS}/toolkit/include/openvdb/Types.h"
    "$ENV{HFS}/toolkit/include/openvdb/openvdb.h"
)

set(_EXTRA_INC
    "${CMAKE_CURRENT_LIST_DIR}/../../externals"
    "$ENV{HFS}/toolkit/include"
)
list(JOIN _EXTRA_INC ";" _EXTRA_INC_JOINED)
set(_VS_INCLUDE_PATH "${_EXTRA_INC_JOINED};$(VC_IncludePath);$(WindowsSDK_IncludePath)")


add_library(Kernels STATIC ${SRC_FILES})
set_property(TARGET Kernels PROPERTY VS_GLOBAL_IncludePath "${_VS_INCLUDE_PATH}")

# Make VS include these headers in the IDE
target_sources(Kernels PRIVATE ${EXTERNAL_INCLUDE} ${HDK_INCLUDE})
set_source_files_properties(${EXTERNAL_INCLUDE} ${HDK_INCLUDE} PROPERTIES HEADER_FILE_ONLY TRUE)

target_compile_options(Kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

target_include_directories(Kernels
        PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        "$ENV{HFS}/toolkit/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../externals"
        .  # for your .cuh headers
)

target_link_libraries(Kernels PRIVATE CUDA::cudart)
target_compile_features(Kernels PUBLIC cxx_std_17)

# Make visual studio pull these into the IDE
source_group(TREE "${CMAKE_CURRENT_LIST_DIR}"                 PREFIX "Kernels"   FILES ${SRC_FILES})
source_group(TREE "${CMAKE_CURRENT_LIST_DIR}/../../externals" PREFIX "nanovdb"   FILES ${EXTERNAL_INCLUDE})
source_group(TREE "$ENV{HFS}/toolkit/include"                 PREFIX "HDK"       FILES ${HDK_INCLUDE})