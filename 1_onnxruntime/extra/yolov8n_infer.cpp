#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <chrono>

// ONNX Runtime headers
#include <onnxruntime_cxx_api.h>

// ─────────────────────────────────────────────
// YOLOv8n model constants
// ─────────────────────────────────────────────
static constexpr int INPUT_WIDTH        = 640;
static constexpr int INPUT_HEIGHT       = 640;
static constexpr int NUM_CHANNELS       = 3;
static constexpr int TOTAL_INPUT_SIZE   = NUM_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT;

std::string MODEL = "../models/yolov8n.onnx";

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main(int argc, char** argv) {
    // ── ONNX Runtime setup ──────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov8n");

    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(4);
    session_opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_opts.AddConfigEntry("session.intra_op.allow_spinning", "1");
    session_opts.EnableProfiling("1");

    // OrtCUDAProviderOptions cuda_opts{};
    // cuda_opts.device_id = 0;
    // session_opts.AppendExecutionProvider_CUDA(cuda_opts);

    OrtTensorRTProviderOptionsV2* trt_opts = nullptr;
    Ort::GetApi().CreateTensorRTProviderOptions(&trt_opts);

    const char* keys[] = {
        "device_id",
        "trt_max_workspace_size",
        "trt_fp16_enable",
        "trt_engine_cache_enable",
        "trt_engine_cache_path",
        "trt_timing_cache_enable",  // V2-only: reuse TRT build timing across sessions
        "trt_cuda_graph_enable",    // V2-only: CUDA graph for lower CPU overhead
    };
    const char* values[] = {
        "0",
        "1073741824",               // 1 GB
        "1",
        "1",
        "./trt_cache",
        "1",
        "0",                        // keep off unless input shapes are truly static
    };
    Ort::GetApi().UpdateTensorRTProviderOptions(
        trt_opts, keys, values, sizeof(keys) / sizeof(keys[0]));

    session_opts.AppendExecutionProvider_TensorRT_V2(*trt_opts);

    Ort::Session session(env, MODEL.c_str(), session_opts);
    Ort::GetApi().ReleaseTensorRTProviderOptions(trt_opts);
    Ort::AllocatorWithDefaultOptions allocator;

    // ── Inspect model I/O ───────────────────────
    size_t num_inputs  = session.GetInputCount();
    size_t num_outputs = session.GetOutputCount();

    // ── Build dummy input tensor ─────────────────
    std::vector<float> input_data(TOTAL_INPUT_SIZE, 0.f);
    std::array<int64_t, 4> input_dims = {1, NUM_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        input_data.data(), TOTAL_INPUT_SIZE,
        input_dims.data(), input_dims.size());

    // ── Run inference ────────────────────────────
    const char* input_names[]  = { "images" };
    const char* output_names[] = { "output0" };

    std::cout << "\nWarming up...\n";
    session.Run(
        Ort::RunOptions{nullptr},
        input_names,  &input_tensor, 1,
        output_names, 1);

    std::cout << "\nRunning inference...\n";
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        session.Run(
            Ort::RunOptions{nullptr},
            input_names,  &input_tensor, 1,
            output_names, 1);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Inference time: " << ms << " ms\n";
    }

    return 0;
}
