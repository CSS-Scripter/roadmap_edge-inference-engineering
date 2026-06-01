cmake_minimum_required(VERSION 3.18)
project(yolov8n_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ══════════════════════════════════════════════════════════════════════════════
# Options
# ══════════════════════════════════════════════════════════════════════════════

option(ORT_USE_CUDA      "Enable ONNX Runtime CUDA Execution Provider"    ON)
option(ORT_USE_TENSORRT  "Enable ONNX Runtime TensorRT Execution Provider" ON)
option(ORT_USE_PREBUILT  "Download a pre-built ORT release (OFF = build from source)" ON)

set(ORT_VERSION "1.21.0" CACHE STRING "ONNX Runtime version to fetch")

# Common TensorRT install roots to probe (extend as needed)
set(ORT_TENSORRT_ROOT "" CACHE PATH
    "TensorRT install root (e.g. /usr/local/TensorRT). Leave empty to auto-detect.")

# ══════════════════════════════════════════════════════════════════════════════
# CUDA detection
# ══════════════════════════════════════════════════════════════════════════════

set(CUDA_FOUND_AND_WANTED FALSE)
if(ORT_USE_CUDA OR ORT_USE_TENSORRT)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(CUDA_FOUND_AND_WANTED TRUE)
        message(STATUS "CUDA toolkit ${CUDAToolkit_VERSION} found")
    else()
        if(ORT_USE_TENSORRT)
            message(WARNING
                "ORT_USE_TENSORRT=ON but no CUDA toolkit found — "
                "disabling TensorRT EP")
            set(ORT_USE_TENSORRT OFF)
        endif()
        if(ORT_USE_CUDA)
            message(WARNING
                "ORT_USE_CUDA=ON but no CUDA toolkit found — falling back to CPU-only")
        endif()
    endif()
endif()

# ══════════════════════════════════════════════════════════════════════════════
# TensorRT detection
#
# TensorRT ships its own libraries (libnvinfer, libnvinfer_plugin, …) that
# are NOT bundled inside the ORT prebuilt tarball.  They must already be
# installed on the system (from the TensorRT .deb/.tar.gz or via the
# nvcr.io/nvidia/tensorrt Docker image).
#
# We locate them here so we can:
#   1. Fail fast at configure time with a clear message if they are missing.
#   2. Add the TensorRT lib dir to RPATH so the binary finds them at runtime
#      without LD_LIBRARY_PATH.
#
# Typical install paths:
#   /usr/lib/x86_64-linux-gnu/         (TRT installed via .deb)
#   /usr/local/TensorRT/lib/           (TRT installed from tar.gz)
#   /usr/local/cuda/lib64/             (some Jetson / CUDA-bundled builds)
#   /usr/lib/aarch64-linux-gnu/        (Jetson / aarch64 .deb)
# ══════════════════════════════════════════════════════════════════════════════

set(TRT_FOUND FALSE)
set(_trt_lib_dir "")

if(ORT_USE_TENSORRT AND CUDA_FOUND_AND_WANTED)

    # Build candidate search paths
    set(_trt_search_roots
        "${ORT_TENSORRT_ROOT}"            # user override (may be empty)
        "/usr/local/TensorRT"
        "/usr/local/tensorrt"
        "/opt/TensorRT"
        "${CUDAToolkit_ROOT}"
    )

    # Also probe standard system library dirs for .deb-installed TRT
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        list(APPEND _trt_search_roots "/usr")
        set(_trt_lib_suffix "lib/x86_64-linux-gnu")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
        list(APPEND _trt_search_roots "/usr")
        set(_trt_lib_suffix "lib/aarch64-linux-gnu")
    else()
        set(_trt_lib_suffix "lib")
    endif()

    # libnvinfer is the sentinel for the whole TRT installation
    find_library(TRT_NVINFER_LIB
        NAMES nvinfer
        HINTS
            ${_trt_search_roots}
        PATH_SUFFIXES
            lib lib64 "${_trt_lib_suffix}"
    )

    find_library(TRT_NVINFER_PLUGIN_LIB
        NAMES nvinfer_plugin
        HINTS
            ${_trt_search_roots}
        PATH_SUFFIXES
            lib lib64 "${_trt_lib_suffix}"
    )

    find_library(TRT_NVONNXPARSER_LIB
        NAMES nvonnxparser
        HINTS
            ${_trt_search_roots}
        PATH_SUFFIXES
            lib lib64 "${_trt_lib_suffix}"
    )

    find_path(TRT_INCLUDE_DIR
        NAMES NvInfer.h
        HINTS
            ${_trt_search_roots}
        PATH_SUFFIXES
            include
    )

    if(TRT_NVINFER_LIB AND TRT_NVINFER_PLUGIN_LIB AND TRT_INCLUDE_DIR)
        set(TRT_FOUND TRUE)
        cmake_path(GET TRT_NVINFER_LIB PARENT_PATH _trt_lib_dir)
        message(STATUS "TensorRT found")
        message(STATUS "  nvinfer         : ${TRT_NVINFER_LIB}")
        message(STATUS "  nvinfer_plugin  : ${TRT_NVINFER_PLUGIN_LIB}")
        message(STATUS "  nvonnxparser    : ${TRT_NVONNXPARSER_LIB}")
        message(STATUS "  include         : ${TRT_INCLUDE_DIR}")
        message(STATUS "  lib dir for RPATH: ${_trt_lib_dir}")
    else()
        message(WARNING
            "ORT_USE_TENSORRT=ON but TensorRT libraries not found.\n"
            "Install TensorRT (https://developer.nvidia.com/tensorrt) and either:\n"
            "  a) Let the .deb installer put libs in /usr/lib/<arch>-linux-gnu/\n"
            "  b) Pass -DORT_TENSORRT_ROOT=/path/to/TensorRT to cmake\n"
            "Disabling TensorRT EP for this build.")
        set(ORT_USE_TENSORRT OFF)
    endif()
endif()

# ══════════════════════════════════════════════════════════════════════════════
# Fetch / build ONNX Runtime
# ══════════════════════════════════════════════════════════════════════════════

if(ORT_USE_PREBUILT)

    # ── Determine GitHub release asset name ──────────────────────────────────
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(_os "linux")
        set(_ext "tgz")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(_os "win")
        set(_ext "zip")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(_os "osx")
        set(_ext "tgz")
    else()
        message(FATAL_ERROR "ORT_USE_PREBUILT: unsupported OS '${CMAKE_SYSTEM_NAME}'")
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(_arch "x64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
        set(_arch "aarch64")
    else()
        message(FATAL_ERROR "ORT_USE_PREBUILT: unsupported arch '${CMAKE_SYSTEM_PROCESSOR}'")
    endif()

    # The -gpu prebuilt includes both the CUDA EP and the TensorRT EP .so files.
    # The CPU-only tarball has neither.
    set(_variant "")
    if(CUDA_FOUND_AND_WANTED AND _os STREQUAL "linux" AND _arch STREQUAL "x64")
        set(_variant "-gpu")
    endif()

    set(_asset_name "onnxruntime-${_os}-${_arch}${_variant}-${ORT_VERSION}")
    set(_url
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${_asset_name}.${_ext}")

    message(STATUS "Fetching pre-built ORT: ${_url}")

    FetchContent_Declare(
        onnxruntime_prebuilt
        URL                        ${_url}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime_prebuilt)

    set(_ort_root "${onnxruntime_prebuilt_SOURCE_DIR}")

    find_path(ORT_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS "${_ort_root}/include"
        NO_DEFAULT_PATH REQUIRED)

    find_library(ORT_LIBRARY
        NAMES onnxruntime
        PATHS "${_ort_root}/lib"
        NO_DEFAULT_PATH REQUIRED)

    message(STATUS "ORT include : ${ORT_INCLUDE_DIR}")
    message(STATUS "ORT library : ${ORT_LIBRARY}")

    add_library(onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION             "${ORT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
    )

    set(_ort_lib_dir "${_ort_root}/lib")

else()   # ── Build from source ─────────────────────────────────────────────

    message(STATUS "Building ONNX Runtime ${ORT_VERSION} from source (30-60 min)…")

    set(onnxruntime_BUILD_SHARED_LIB  ON  CACHE BOOL "" FORCE)
    set(onnxruntime_BUILD_UNIT_TESTS  OFF CACHE BOOL "" FORCE)
    set(onnxruntime_ENABLE_PYTHON     OFF CACHE BOOL "" FORCE)
    set(onnxruntime_BUILD_CSHARP      OFF CACHE BOOL "" FORCE)
    set(onnxruntime_BUILD_JAVA        OFF CACHE BOOL "" FORCE)
    set(onnxruntime_BUILD_OBJC        OFF CACHE BOOL "" FORCE)
    set(FETCHCONTENT_QUIET            OFF CACHE BOOL "" FORCE)

    if(CUDA_FOUND_AND_WANTED)
        set(onnxruntime_USE_CUDA ON CACHE BOOL "" FORCE)
        set(onnxruntime_CUDA_HOME "${CUDAToolkit_ROOT}" CACHE PATH "" FORCE)
    endif()

    if(TRT_FOUND)
        set(onnxruntime_USE_TENSORRT     ON               CACHE BOOL "" FORCE)
        set(onnxruntime_TENSORRT_HOME    "${_trt_lib_dir}/.." CACHE PATH "" FORCE)
    endif()

    FetchContent_Declare(
        onnxruntime_src
        GIT_REPOSITORY         https://github.com/microsoft/onnxruntime.git
        GIT_TAG                v${ORT_VERSION}
        GIT_SHALLOW            TRUE
        GIT_SUBMODULES_RECURSE TRUE
    )
    FetchContent_GetProperties(onnxruntime_src)
    if(NOT onnxruntime_src_POPULATED)
        FetchContent_Populate(onnxruntime_src)
    endif()

    add_subdirectory(
        "${onnxruntime_src_SOURCE_DIR}/cmake"
        "${onnxruntime_src_BINARY_DIR}"
        EXCLUDE_FROM_ALL
    )

    set(_ort_lib_dir "")

endif()

# ══════════════════════════════════════════════════════════════════════════════
# Our executable
# ══════════════════════════════════════════════════════════════════════════════

add_executable(yolov8n_infer yolov8n_infer.cpp)

if(ORT_USE_PREBUILT)
    # Flat prebuilt layout: also expose the parent so nested #includes resolve
    cmake_path(GET ORT_INCLUDE_DIR PARENT_PATH _ort_include_parent)
    target_include_directories(yolov8n_infer PRIVATE "${_ort_include_parent}")
endif()

target_link_libraries(yolov8n_infer PRIVATE onnxruntime)

# ── RPATH: collect every lib dir the binary needs at runtime ─────────────────
#
# The binary dlopen()s EP .so files at runtime (libonnxruntime_providers_tensorrt.so,
# libonnxruntime_providers_cuda.so, etc.).  Those in turn need libnvinfer*.so.
# Embedding all relevant lib dirs in RPATH means neither LD_LIBRARY_PATH nor
# ldconfig entries are required.
#
set(_rpath_dirs "")
if(_ort_lib_dir)
    list(APPEND _rpath_dirs "${_ort_lib_dir}")
endif()
if(TRT_FOUND AND _trt_lib_dir)
    list(APPEND _rpath_dirs "${_trt_lib_dir}")
endif()
if(CUDA_FOUND_AND_WANTED)
    # cudart / cublas live here
    list(APPEND _rpath_dirs "${CUDAToolkit_LIBRARY_DIR}")
endif()
list(REMOVE_DUPLICATES _rpath_dirs)

if(_rpath_dirs)
    # CMake wants a semicolon-separated list for [BUILD_]RPATH
    string(JOIN ";" _rpath_str ${_rpath_dirs})
    set_target_properties(yolov8n_infer PROPERTIES
        BUILD_RPATH   "${_rpath_str}"
        INSTALL_RPATH "${_rpath_str}"
    )
    message(STATUS "Runtime RPATH: ${_rpath_str}")
endif()

# ── Compile definitions so C++ code knows what EPs are available ─────────────
if(CUDA_FOUND_AND_WANTED)
    target_link_libraries(yolov8n_infer PRIVATE CUDA::cudart)
    target_compile_definitions(yolov8n_infer PRIVATE ORT_USE_CUDA=1)
endif()

if(TRT_FOUND)
    target_compile_definitions(yolov8n_infer PRIVATE ORT_USE_TENSORRT=1)
    # Link directly so the linker (and RPATH logic above) knows about these
    target_link_libraries(yolov8n_infer PRIVATE
        "${TRT_NVINFER_LIB}"
        "${TRT_NVINFER_PLUGIN_LIB}"
    )
    if(TRT_NVONNXPARSER_LIB)
        target_link_libraries(yolov8n_infer PRIVATE "${TRT_NVONNXPARSER_LIB}")
    endif()
    target_include_directories(yolov8n_infer PRIVATE "${TRT_INCLUDE_DIR}")
    message(STATUS "TensorRT EP enabled")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(yolov8n_infer PRIVATE OpenMP::OpenMP_CXX)
endif()

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "┌─ Build summary ───────────────────────────────┐")
message(STATUS "│  ORT version   : ${ORT_VERSION}")
message(STATUS "│  CUDA EP       : ${CUDA_FOUND_AND_WANTED}")
message(STATUS "│  TensorRT EP   : ${TRT_FOUND}")
message(STATUS "│  Prebuilt ORT  : ${ORT_USE_PREBUILT}")
message(STATUS "└───────────────────────────────────────────────┘")
message(STATUS "")

install(TARGETS yolov8n_infer DESTINATION bin)
