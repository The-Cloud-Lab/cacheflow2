cmake_minimum_required(VERSION 3.18)
project(doca_kv_offload VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_HOST "Build host provider component" ON)
option(BUILD_DPU "Build DPU offloader component" ON)
option(BUILD_PYTHON "Build Python wrapper" OFF)
option(BUILD_EXAMPLES "Build example programs" ON)

# DOCA paths
set(DOCA_INSTALL_PATH "/opt/mellanox/doca" CACHE PATH "DOCA installation path")
set(DOCA_INCLUDE_DIR "${DOCA_INSTALL_PATH}/include")

# Detect architecture for library path
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(DOCA_LIB_DIR "${DOCA_INSTALL_PATH}/lib/aarch64-linux-gnu")
else()
    set(DOCA_LIB_DIR "${DOCA_INSTALL_PATH}/lib/x86_64-linux-gnu")
endif()

# Find DOCA libraries
find_library(DOCA_COMMON_LIB doca_common PATHS ${DOCA_LIB_DIR} REQUIRED)
find_library(DOCA_DMA_LIB doca_dma PATHS ${DOCA_LIB_DIR} REQUIRED)
find_library(DOCA_COMCH_LIB doca_comch PATHS ${DOCA_LIB_DIR} REQUIRED)
find_library(DOCA_ARGP_LIB doca_argp PATHS ${DOCA_LIB_DIR} REQUIRED)

# Common includes
include_directories(
    ${CMAKE_SOURCE_DIR}/common
    ${DOCA_INCLUDE_DIR}
)

# Compiler flags
add_compile_options(
    -Wall
    -Wextra
    -Wno-error=deprecated-declarations
    -O3
)

# Host provider library
if(BUILD_HOST)
    add_library(host_provider STATIC
        host/host_provider.c
        common/comch_utils.c
        common/common.c
        common/message_queue.c
    )
    
    target_include_directories(host_provider PUBLIC
        ${CMAKE_SOURCE_DIR}/host
        ${CMAKE_SOURCE_DIR}/common
        ${DOCA_INCLUDE_DIR}
    )
    
    target_link_libraries(host_provider
        ${DOCA_COMMON_LIB}
        ${DOCA_DMA_LIB}
        ${DOCA_COMCH_LIB}
        ${DOCA_ARGP_LIB}
        pthread
    )
    
    # Host provider shared library (for Python wrapper)
    add_library(host_provider_shared SHARED
        host/host_provider.c
        common/comch_utils.c
        common/common.c
        common/message_queue.c
    )

    target_include_directories(host_provider_shared PUBLIC
        ${CMAKE_SOURCE_DIR}/host
        ${CMAKE_SOURCE_DIR}/common
        ${DOCA_INCLUDE_DIR}
    )

    target_link_libraries(host_provider_shared
        ${DOCA_COMMON_LIB}
        ${DOCA_DMA_LIB}
        ${DOCA_COMCH_LIB}
        ${DOCA_ARGP_LIB}
        pthread
    )

    set_target_properties(host_provider_shared PROPERTIES
        OUTPUT_NAME "host_provider"
        VERSION 1.0.0
        SOVERSION 1
    )

    # Host provider executable (for testing)
    if(BUILD_EXAMPLES)
        add_executable(host_provider_test
            host/host_provider_test.c
        )

        target_link_libraries(host_provider_test
            host_provider
        )

        # Host provider performance benchmark test
        add_executable(host_provider_perf_test
            host/host_provider_perf_test.c
        )

        target_link_libraries(host_provider_perf_test
            host_provider
        )
    endif()
endif()

# DPU offloader executable
if(BUILD_DPU)
    add_executable(dpu_offloader
        dpu/dpu_offloader.c
        dpu/dpu_main.c
        common/comch_utils.c
        common/common.c
    )
    
    target_include_directories(dpu_offloader PRIVATE
        ${CMAKE_SOURCE_DIR}/dpu
        ${CMAKE_SOURCE_DIR}/common
        ${DOCA_INCLUDE_DIR}
    )
    
    target_link_libraries(dpu_offloader
        ${DOCA_COMMON_LIB}
        ${DOCA_DMA_LIB}
        ${DOCA_COMCH_LIB}
        ${DOCA_ARGP_LIB}
        pthread
    )
    
    # DPU standalone test (no ComCh required)
    add_executable(dpu_standalone_test
        dpu/dpu_offloader.c
        dpu/dpu_standalone_test.c
        common/comch_utils.c
        common/common.c
    )
    
    target_include_directories(dpu_standalone_test PRIVATE
        ${CMAKE_SOURCE_DIR}/dpu
        ${CMAKE_SOURCE_DIR}/common
        ${DOCA_INCLUDE_DIR}
    )
    
    target_link_libraries(dpu_standalone_test
        ${DOCA_COMMON_LIB}
        ${DOCA_DMA_LIB}
        ${DOCA_COMCH_LIB}
        ${DOCA_ARGP_LIB}
        pthread
    )
endif()

# Installation
if(BUILD_HOST)
    install(TARGETS host_provider host_provider_shared
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )

    install(FILES
        common/protocol.h
        common/doca_common.h
        host/host_provider.h
        DESTINATION include/doca_kv_offload
    )
endif()

if(BUILD_DPU)
    install(TARGETS dpu_offloader
        RUNTIME DESTINATION bin
    )
endif()

# Print configuration
message(STATUS "=== DOCA KV Cache Offload Configuration ===")
message(STATUS "Build host provider: ${BUILD_HOST}")
message(STATUS "Build DPU offloader: ${BUILD_DPU}")
message(STATUS "Build Python wrapper: ${BUILD_PYTHON}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "DOCA path: ${DOCA_INSTALL_PATH}")
message(STATUS "=========================================")
