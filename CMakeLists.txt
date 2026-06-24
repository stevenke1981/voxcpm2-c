# ══════════════════════════════════════════════════════════
# VoxCPM2-C — Build System
# ══════════════════════════════════════════════════════════
cmake_minimum_required(VERSION 3.20)
project(voxcpm2-c
    VERSION 0.1.0
    DESCRIPTION "Pure C inference engine for VoxCPM2 TTS model"
    LANGUAGES C
)

# ─── C Standard ───
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# ─── Options ───
option(VOXCPM_BUILD_CLI     "Build CLI tool"                     ON)
option(VOXCPM_BUILD_TESTS   "Build tests"                        ON)
option(VOXCPM_BUILD_EXAMPLES "Build examples"                    ON)
option(VOXCPM_BUILD_SHARED  "Build shared library"               OFF)
option(VOXCPM_CUDA          "Enable CUDA backend"                OFF)
option(VOXCPM_OPENMP        "Enable OpenMP parallelization"      ON)
option(VOXCPM_ASAN          "Enable AddressSanitizer"            OFF)
option(VOXCPM_NATIVE        "Optimize for native CPU"            ON)
option(VOXCPM_STATIC_DEPS   "Statically link dependencies"       OFF)

# ─── Compile Warnings ───
if(MSVC)
    add_compile_options(/W4 /utf-8)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wshadow -Wconversion -Wsign-conversion
        -Wdouble-promotion -Wformat=2
        -Wnull-dereference -Wstrict-overflow=5
        -Wno-unused-parameter
    )
endif()

# ─── Debug/Release Flags ───
if(CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
    add_compile_options(-g -O0)
    add_definitions(-DVOXCPM_DEBUG)
else()
    add_compile_options(-O3 -DNDEBUG)
    if(VOXCPM_NATIVE AND NOT MSVC)
        add_compile_options(-march=native -mtune=native)
    endif()
endif()

# ─── Sanitizer ───
if(VOXCPM_ASAN)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# ─── OpenMP ───
if(VOXCPM_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        add_definitions(-DVOXCPM_OPENMP)
        list(APPEND VOXCPM_LIBS OpenMP::OpenMP_C)
    endif()
endif()

# ─── CUDA ───
if(VOXCPM_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_definitions(-DVOXCPM_CUDA)
    
    # Find CUDA toolkit
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND VOXCPM_LIBS ${CUDA_LIBRARIES})
endif()

# ─── Sources ───
set(VOXCPM_CORE_SOURCES
    src/tensor.c
    src/nn.c
    src/model.c
    src/loc_enc.c
    src/tslm.c
    src/ralm.c
    src/loc_dit.c
    src/audio_vae.c
    src/audio.c
    src/tokenizer.c
    src/sampler.c
    src/utils.c
    src/platform.c
)

set(VOXCPM_CUDA_SOURCES
    src/tensor_cuda.cu
)

set(VOXCPM_HEADERS
    include/voxcpm.h
    include/tensor.h
    include/nn.h
    include/model.h
    include/tokenizer.h
    include/audio.h
    include/sampler.h
    include/platform.h
)

# ─── Library ───
add_library(voxcpm2 STATIC ${VOXCPM_CORE_SOURCES} ${VOXCPM_HEADERS})
target_include_directories(voxcpm2 PUBLIC include)
if(NOT MSVC)
    list(APPEND VOXCPM_LIBS m)
endif()
target_link_libraries(voxcpm2 PRIVATE ${VOXCPM_LIBS})

if(VOXCPM_CUDA)
    target_sources(voxcpm2 PRIVATE ${VOXCPM_CUDA_SOURCES})
    set_target_properties(voxcpm2 PROPERTIES CUDA_ARCHITECTURES "80;86")
endif()

if(VOXCPM_BUILD_SHARED)
    add_library(voxcpm2_shared SHARED ${VOXCPM_CORE_SOURCES} ${VOXCPM_HEADERS})
    target_include_directories(voxcpm2_shared PUBLIC include)
    target_link_libraries(voxcpm2_shared PRIVATE ${VOXCPM_LIBS})
    set_target_properties(voxcpm2_shared PROPERTIES
        OUTPUT_NAME voxcpm2
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

# ─── CLI ───
if(VOXCPM_BUILD_CLI)
    add_executable(voxcpm2-c src/main.c)
    target_link_libraries(voxcpm2-c PRIVATE voxcpm2)
    target_include_directories(voxcpm2-c PRIVATE include)
endif()

# ─── Tests ───
if(VOXCPM_BUILD_TESTS AND NOT MSVC)
    enable_testing()
    
    function(add_voxcpm_test name source)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE voxcpm2 ${VOXCPM_LIBS})
        target_include_directories(${name} PRIVATE include tests)
        add_test(NAME ${name} COMMAND ${name})
    endfunction()
    
    add_voxcpm_test(test_tensor        tests/test_tensor.c)
    add_voxcpm_test(test_nn            tests/test_nn.c)
    add_voxcpm_test(test_model_loader  tests/test_model_loader.c)
    add_voxcpm_test(test_tokenizer     tests/test_tokenizer.c)
    add_voxcpm_test(test_audio_vae     tests/test_audio_vae.c)
    add_voxcpm_test(test_audio         tests/test_audio.c)
endif()

# ─── Examples ───
if(VOXCPM_BUILD_EXAMPLES)
    add_executable(tts_minimal examples/tts_minimal.c)
    target_link_libraries(tts_minimal PRIVATE voxcpm2)
    target_include_directories(tts_minimal PRIVATE include)
endif()

# ─── Benchmarks ───
if(VOXCPM_BUILD_TESTS)
    add_executable(bench_rtf benchmarks/bench_rtf.c)
    target_link_libraries(bench_rtf PRIVATE voxcpm2 ${VOXCPM_LIBS})
    target_include_directories(bench_rtf PRIVATE include)
endif()

# ─── Install ───
include(GNUInstallDirs)
install(TARGETS voxcpm2 voxcpm2-c
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES ${VOXCPM_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/voxcpm2)

# ─── Custom Targets ───
add_custom_target(format
    COMMAND find src/ include/ tests/ examples/ -name '*.c' -o -name '*.h'
        | xargs clang-format -i -style=file
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Formatting source files with clang-format"
)

add_custom_target(lint
    COMMAND cppcheck --enable=all --std=c11
        --suppress=missingIncludeSystem
        -I ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running cppcheck static analysis"
)

# ─── Print Summary ───
message(STATUS "══════════════════════════════════════")
message(STATUS "VoxCPM2-C Build Configuration")
message(STATUS "══════════════════════════════════════")
message(STATUS "Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA:           ${VOXCPM_CUDA}")
message(STATUS "OpenMP:         ${VOXCPM_OPENMP}")
message(STATUS "Tests:          ${VOXCPM_BUILD_TESTS}")
message(STATUS "Examples:       ${VOXCPM_BUILD_EXAMPLES}")
message(STATUS "ASAN:           ${VOXCPM_ASAN}")
message(STATUS "C Compiler:     ${CMAKE_C_COMPILER}")
if(VOXCPM_CUDA)
    message(STATUS "CUDA Compiler:  ${CMAKE_CUDA_COMPILER}")
endif()
message(STATUS "══════════════════════════════════════")
