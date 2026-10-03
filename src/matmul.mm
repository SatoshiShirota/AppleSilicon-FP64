#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "apple_fp64/matmul.hpp"
#include "arithmetic.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace apple_fp64 {
namespace {

/** @brief 経過時間の測定に使用する単調な時計。 */
using clock_t = std::chrono::steady_clock;

/**
 * @brief 指定した時点からの経過秒数を返す。
 * @param[in] start 測定開始時点。
 * @return 経過した秒数。
 */
double elapsed(clock_t::time_point start) {
    return std::chrono::duration<double>(clock_t::now() - start).count();
}

/**
 * @brief 要素数とサイズの積を、オーバーフローせずに求める。
 * @param[in] factors 乗算する非負の値。
 * @return 積。
 * @exception std::invalid_argument 積がsize_tの範囲を超える。
 */
std::size_t checked_size(std::initializer_list<std::size_t> factors) {
    std::size_t size = 1;
    for (std::size_t factor : factors) {
        if (factor != 0 && size > std::numeric_limits<std::size_t>::max() / factor)
            throw std::invalid_argument("行列のサイズがsize_tの範囲を超えます。");
        size *= factor;
    }
    return size;
}

/**
 * @brief 使用する法と復元係数を、必要な数値範囲から決める。
 * @param[in] k 内積の項数。正の値。
 * @param[in] options 整数幅。
 * @return CRTの係数。
 * @exception std::invalid_argument 利用可能な法の積では整数幅を支えられない。
 */
crt_plan_s make_plan(std::uint32_t k, options_s options) {
    std::uint64_t precision = std::uint64_t(options.precision_a) + options.precision_b;
    if (precision + word_bit_length(k) + 1 > MAX_LIMBS * DIGIT_BITS)
        throw std::invalid_argument("整数幅と内積の長さに必要な範囲が、利用可能な法の積を超えます。");
    big_uint_s bound = {};
    for (word_t bit = 0; bit < 32; ++bit) {
        if ((k >> bit) & 1u) {
            word_t position = word_t(precision) + 1 + bit;
            bound.digits[position / DIGIT_BITS] |= 1u << (position % DIGIT_BITS);
        }
    }
    crt_plan_s plan = {};
    plan.product.digits[0] = 1;
    for (word_t value : CRT_MODULI) {
        modulus_s modulus = {value, word_t((1ull << 32) / value), word_t((1ull << 32) % value)};
        word_t t = plan.count++;
        plan.moduli[t] = modulus;
        plan.prefixes[t] = plan.product;
        word_t remainder = big_mod(plan.product, modulus, MAX_LIMBS);
        for (word_t inverse = 1; inverse < value; ++inverse) {
            if (word_mod(remainder * inverse, modulus) == 1) {
                plan.inverses[t] = inverse;
                break;
            }
        }
        assert(plan.inverses[t] != 0);
        big_add_scaled(plan.product, plan.product, value - 1, MAX_LIMBS);
        plan.stage_limbs[t] = (big_bit_length(plan.product, MAX_LIMBS) + DIGIT_BITS - 1) / DIGIT_BITS;
        if (big_compare(plan.product, bound, MAX_LIMBS) > 0) {
            plan.limbs = plan.stage_limbs[t];
            word_t carry = 0;
            for (int i = int(plan.limbs) - 1; i >= 0; --i) {
                plan.half_product.digits[i] = (plan.product.digits[i] >> 1) | (carry << (DIGIT_BITS - 1));
                carry = plan.product.digits[i] & 1u;
            }
            word_t precision_limit = std::max(options.precision_a, options.precision_b);
            for (word_t index = 0; index < plan.count; ++index) {
                word_t remainder = 1;
                for (word_t exponent = 0; exponent < precision_limit; ++exponent) {
                    plan.powers[index][exponent] = static_cast<unsigned char>(remainder);
                    remainder *= 2;
                    if (remainder >= plan.moduli[index].value) remainder -= plan.moduli[index].value;
                }
            }
            return plan;
        }
    }
    throw std::invalid_argument("整数幅と内積の長さに必要な範囲が、利用可能な法の積を超えます。");
}

/**
 * @brief Metalの実行を待ち、成功した場合だけGPUの実行時間を返す。
 * @param[in] command 投入済みの実行指示。
 * @param[in,out] measurement 待ち時間の加算先。
 * @return GPUの実行秒数。
 * @exception std::runtime_error 実行が正常完了しない。
 */
double wait_command(id<MTLCommandBuffer> command, measurement_s& measurement) {
    auto start = clock_t::now();
    [command waitUntilCompleted];
    measurement.wait_seconds += elapsed(start);
    if (command.status != MTLCommandBufferStatusCompleted) {
        std::string reason = command.error ? command.error.localizedDescription.UTF8String : "実行結果が正常完了ではありません。";
        throw std::runtime_error("Metalの実行に失敗しました: " + reason);
    }
    return command.GPUEndTime - command.GPUStartTime;
}

} // namespace

/** @brief CPUから所有するMetalの計算資源。 */
struct implementation_s {
    id<MTLDevice> device; /**< 使用するデバイス。 */
    id<MTLCommandQueue> queue; /**< 順序を保持する実行待ち行列。 */
    id<MTLComputePipelineState> scales; /**< 指数を求めるパイプライン。 */
    id<MTLComputePipelineState> residues; /**< 入力の余りを生成するパイプライン。 */
    id<MTLComputePipelineState> product; /**< 余りの行列積のパイプライン。 */
    id<MTLComputePipelineState> reconstruct; /**< CRTと丸めのパイプライン。 */
    id<MTLBuffer> plan_buffer = nil; /**< CPUから渡す復元係数。 */
    id<MTLBuffer> input_a = nil; /**< CPUから渡すAのビット列。 */
    id<MTLBuffer> input_b = nil; /**< CPUから渡すBのビット列。 */
    id<MTLBuffer> output = nil; /**< CPUへ返す結果のビット列。 */
    id<MTLBuffer> scales_a = nil; /**< Aの行の指数を保持する作業領域。 */
    id<MTLBuffer> scales_b = nil; /**< Bの列の指数を保持する作業領域。 */
    id<MTLBuffer> a_residues = nil; /**< Aの行のまとまりの余りを保持する作業領域。 */
    id<MTLBuffer> b_residues = nil; /**< Bの余りを保持する作業領域。 */
    id<MTLBuffer> c_residues = nil; /**< 出力の余りを保持する作業領域。 */

    /**
     * @brief デバイスと全パイプラインを作成する。
     * @param[in] path コンパイル済みのMetalライブラリー。
     * @exception std::runtime_error 資源を作成できない。
     */
    explicit implementation_s(const std::filesystem::path& path) {
        device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("Metalのデバイスを取得できません。");
        queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("Metalの実行待ち行列を作成できません。");
        NSError* error = nil;
        NSString* name = [NSString stringWithUTF8String:path.c_str()];
        id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:name] error:&error];
        if (!library) throw std::runtime_error("Metalライブラリーを読み込めません: " + std::string(error.localizedDescription.UTF8String));
        scales = pipeline(library, @"find_scales");
        residues = pipeline(library, @"make_residues");
        product = pipeline(library, @"residue_matmul");
        reconstruct = pipeline(library, @"reconstruct");
    }

    /**
     * @brief 一つのカーネルから計算パイプラインを作成する。
     * @param[in] library コンパイル済みライブラリー。
     * @param[in] name カーネル名。
     * @return 計算パイプライン。
     * @exception std::runtime_error カーネルの取得または作成に失敗する。
     */
    id<MTLComputePipelineState> pipeline(id<MTLLibrary> library, NSString* name) {
        id<MTLFunction> function = [library newFunctionWithName:name];
        if (!function) throw std::runtime_error("Metalカーネルを取得できません: " + std::string(name.UTF8String));
        NSError* error = nil;
        auto state = [device newComputePipelineStateWithFunction:function error:&error];
        if (!state) throw std::runtime_error("Metalパイプラインを作成できません: " + std::string(error.localizedDescription.UTF8String));
        return state;
    }

    /**
     * @brief 必要な容量のバッファーを再利用または確保し、使用量を加算する。
     * @param[in,out] workspace 同じ用途の呼び出しで保持するバッファー。
     * @param[in] bytes 確保するバイト数。
     * @param[in] storage CPUと共有するか、GPUだけで使うかの指定。
     * @param[in,out] measurement 使用量の加算先。
     * @return workspaceが所有するバッファー。
     * @pre 同じworkspaceには同じstorageを指定すること。
     * @exception std::runtime_error デバイスの上限を超えるか、確保に失敗する。
     */
    id<MTLBuffer> buffer(id<MTLBuffer> __strong& workspace, std::size_t bytes, MTLResourceOptions storage, measurement_s& measurement) {
        if (bytes > device.maxBufferLength) throw std::runtime_error("作業領域がMetalデバイスのバッファー上限を超えます。");
        if (!workspace || workspace.length < bytes) {
            auto value = [device newBufferWithLength:bytes options:storage];
            if (!value) throw std::runtime_error("Metalの作業領域を確保できません。");
            workspace = value;
        }
        measurement.workspace_bytes += workspace.length;
        return workspace;
    }

    /**
     * @brief 未投入の実行指示を作成する。
     * @return 実行指示。
     * @exception std::runtime_error 作成に失敗する。
     */
    id<MTLCommandBuffer> command() {
        auto value = [queue commandBuffer];
        if (!value) throw std::runtime_error("Metalの実行指示を作成できません。");
        return value;
    }

    /**
     * @brief 計算エンコーダーを作成する。
     * @param[in] command 実行指示。
     * @return エンコーダー。
     * @exception std::runtime_error 作成に失敗する。
     */
    id<MTLComputeCommandEncoder> encoder(id<MTLCommandBuffer> command) {
        auto value = [command computeCommandEncoder];
        if (!value) throw std::runtime_error("Metalの計算エンコーダーを作成できません。");
        return value;
    }

    /**
     * @brief 線形の要素列を、二次元のGPU実行範囲へ割り当てる。
     * @param[in] encoder 設定済みのエンコーダー。
     * @param[in] pipeline 実行するパイプライン。
     * @param[in] count 要素数。
     */
    void dispatch_elements(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, std::size_t count) {
        std::size_t width = std::min<std::size_t>(count, 65536);
        NSUInteger threads = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(width, (count + width - 1) / width, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }

    /**
     * @brief GPUの指数取得と余りの生成を記録する。
     * @param[in] command 実行指示。
     * @param[in] input FP64のビット列。
     * @param[in] scale_buffer 指数の書き込み先。
     * @param[in] output 余りの書き込み先。
     * @param[in] parameters 行列の寸法と整数幅。
     * @param[in] plan_buffer CRTの係数。
     * @param[in] columns Bを処理する場合は1、Aの場合は0。
     */
    void encode_prepare(id<MTLCommandBuffer> command, id<MTLBuffer> input, id<MTLBuffer> scale_buffer,
                        id<MTLBuffer> output, batch_parameters_s parameters, id<MTLBuffer> plan_buffer, word_t columns) {
        auto scale_encoder = encoder(command);
        [scale_encoder setComputePipelineState:scales];
        [scale_encoder setBuffer:input offset:0 atIndex:0];
        [scale_encoder setBuffer:scale_buffer offset:0 atIndex:1];
        [scale_encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
        [scale_encoder setBytes:&columns length:sizeof(columns) atIndex:3];
        [scale_encoder dispatchThreadgroups:MTLSizeMake(columns ? parameters.columns : parameters.rows, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [scale_encoder endEncoding];
        auto residue_encoder = encoder(command);
        [residue_encoder setComputePipelineState:residues];
        [residue_encoder setBuffer:input offset:0 atIndex:0];
        [residue_encoder setBuffer:scale_buffer offset:0 atIndex:1];
        [residue_encoder setBuffer:output offset:0 atIndex:2];
        [residue_encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
        [residue_encoder setBuffer:plan_buffer offset:0 atIndex:4];
        [residue_encoder setBytes:&columns length:sizeof(columns) atIndex:5];
        dispatch_elements(residue_encoder, residues, columns ? std::size_t(parameters.inner) * parameters.columns
                                                             : std::size_t(parameters.rows) * parameters.inner);
        [residue_encoder endEncoding];
    }

    /**
     * @brief すべての法の行列積を一括して記録する。
     * @param[in] command 実行指示。
     * @param[in] a Aの余り。
     * @param[in] b Bの余り。
     * @param[in] c 出力の余りの保存先。
     * @param[in] parameters 行列の寸法。
     * @param[in] plan_buffer CRTの係数。
     * @param[in] count 法の個数。
     */
    void encode_product(id<MTLCommandBuffer> command, id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c,
                        batch_parameters_s parameters, id<MTLBuffer> plan_buffer, word_t count) {
        auto product_encoder = encoder(command);
        [product_encoder setComputePipelineState:product];
        [product_encoder setBuffer:a offset:0 atIndex:0];
        [product_encoder setBuffer:b offset:0 atIndex:1];
        [product_encoder setBuffer:c offset:0 atIndex:2];
        [product_encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
        [product_encoder setBuffer:plan_buffer offset:0 atIndex:4];
        [product_encoder dispatchThreadgroups:MTLSizeMake((parameters.columns + TILE_COLUMNS - 1) / TILE_COLUMNS,
                                                         (parameters.rows + TILE_ROWS - 1) / TILE_ROWS, count)
                         threadsPerThreadgroup:MTLSizeMake(product.threadExecutionWidth * SIMD_GROUPS, 1, 1)];
        [product_encoder endEncoding];
    }

    /**
     * @brief GPUのCRTと丸めを記録する。
     * @param[in] command 実行指示。
     * @param[in] c 出力の余り。
     * @param[in] scales_a Aの行の指数。
     * @param[in] scales_b Bの列の指数。
     * @param[in] output FP64のビット列の保存先。
     * @param[in] parameters 行列の寸法と整数幅。
     * @param[in] plan_buffer CRTの係数。
     */
    void encode_reconstruct(id<MTLCommandBuffer> command, id<MTLBuffer> c, id<MTLBuffer> scales_a,
                            id<MTLBuffer> scales_b, id<MTLBuffer> output, batch_parameters_s parameters, id<MTLBuffer> plan_buffer) {
        auto reconstruct_encoder = encoder(command);
        [reconstruct_encoder setComputePipelineState:reconstruct];
        [reconstruct_encoder setBuffer:c offset:0 atIndex:0];
        [reconstruct_encoder setBuffer:scales_a offset:0 atIndex:1];
        [reconstruct_encoder setBuffer:scales_b offset:0 atIndex:2];
        [reconstruct_encoder setBuffer:output offset:0 atIndex:3];
        [reconstruct_encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
        [reconstruct_encoder setBuffer:plan_buffer offset:0 atIndex:5];
        dispatch_elements(reconstruct_encoder, reconstruct, std::size_t(parameters.rows) * parameters.columns);
        [reconstruct_encoder endEncoding];
    }
};

multiplier_c::multiplier_c(const std::filesystem::path& path) {
    @autoreleasepool {
        implementation = std::make_unique<implementation_s>(path);
    }
}

multiplier_c::~multiplier_c() = default;

std::string multiplier_c::device_name() const {
    return implementation->device.name.UTF8String;
}

result_s multiplier_c::multiply(std::span<const double> a, std::span<const double> b,
                                std::uint32_t m, std::uint32_t n, std::uint32_t k,
                                options_s options) {
    @autoreleasepool {
        auto start = clock_t::now();
        if (m > INT_MAX || n > INT_MAX || k > INT_MAX)
            throw std::invalid_argument("行列の次元はMetalの符号付き32ビット整数の範囲で指定してください。");
        if (a.size() != checked_size({m, k}) || b.size() != checked_size({k, n}))
            throw std::invalid_argument("入力配列の長さが行列の寸法と一致しません。");
        if (options.precision_a == 0 || options.precision_b == 0 || options.batch_rows == 0)
            throw std::invalid_argument("整数幅と行のまとまりの大きさは正の値で指定してください。");
        std::size_t output_bytes = checked_size({m, n, sizeof(double)});
        result_s result;
        result.values.resize(output_bytes / sizeof(double), 0);
        if (m == 0 || n == 0 || k == 0) {
            result.measurement.total_seconds = elapsed(start);
            return result;
        }
        auto plan = make_plan(k, options);
        auto& measurement = result.measurement;
        measurement.modulus_count = plan.count;
        auto& backend = *implementation;
        auto plan_buffer = backend.buffer(backend.plan_buffer, sizeof(plan), MTLResourceStorageModeShared, measurement);
        std::memcpy(plan_buffer.contents, &plan, sizeof(plan));
        word_t batch_rows = std::min(m, options.batch_rows);
        batch_parameters_s parameters = {batch_rows, n, k, 0, options.precision_a, options.precision_b};
        auto b_residues = backend.buffer(backend.b_residues, checked_size({plan.count, k, n}), MTLResourceStorageModePrivate, measurement);
        auto input_a = backend.buffer(backend.input_a, checked_size({a.size(), sizeof(double)}), MTLResourceStorageModeShared, measurement);
        auto input_b = backend.buffer(backend.input_b, checked_size({b.size(), sizeof(double)}), MTLResourceStorageModeShared, measurement);
        auto output = backend.buffer(backend.output, output_bytes, MTLResourceStorageModeShared, measurement);
        std::memcpy(input_a.contents, a.data(), a.size_bytes());
        std::memcpy(input_b.contents, b.data(), b.size_bytes());
        auto scales_a = backend.buffer(backend.scales_a, checked_size({batch_rows, sizeof(int)}), MTLResourceStorageModePrivate, measurement);
        auto scales_b = backend.buffer(backend.scales_b, checked_size({n, sizeof(int)}), MTLResourceStorageModePrivate, measurement);
        auto a_residues = backend.buffer(backend.a_residues, checked_size({plan.count, batch_rows, k}), MTLResourceStorageModePrivate, measurement);
        auto c_residues = backend.buffer(backend.c_residues, checked_size({plan.count, batch_rows, n}), MTLResourceStorageModePrivate, measurement);
        std::vector<id<MTLCommandBuffer>> prepare_commands, product_commands, reconstruct_commands;
        auto prepare_b = backend.command();
        backend.encode_prepare(prepare_b, input_b, scales_b, b_residues, parameters, plan_buffer, 1);
        [prepare_b commit];
        prepare_commands.push_back(prepare_b);
        for (word_t row = 0; row < m;) {
            parameters.row_begin = row;
            parameters.rows = std::min(batch_rows, m - row);
            auto prepare_a = backend.command();
            backend.encode_prepare(prepare_a, input_a, scales_a, a_residues, parameters, plan_buffer, 0);
            [prepare_a commit];
            prepare_commands.push_back(prepare_a);
            auto product = backend.command();
            backend.encode_product(product, a_residues, b_residues, c_residues, parameters, plan_buffer, plan.count);
            [product commit];
            product_commands.push_back(product);
            auto reconstruct = backend.command();
            backend.encode_reconstruct(reconstruct, c_residues, scales_a, scales_b, output, parameters, plan_buffer);
            [reconstruct commit];
            reconstruct_commands.push_back(reconstruct);
            row += parameters.rows;
        }
        // 中間結果をCPUへ取り出さず、最後の指示の完了後に各指示の成否と時間を取得する。
        wait_command(reconstruct_commands.back(), measurement);
        for (auto command : prepare_commands) measurement.prepare_seconds += wait_command(command, measurement);
        for (auto command : product_commands) measurement.product_seconds += wait_command(command, measurement);
        for (auto command : reconstruct_commands) measurement.reconstruct_seconds += wait_command(command, measurement);
        std::memcpy(result.values.data(), output.contents, output_bytes);
        measurement.total_seconds = elapsed(start);
        return result;
    }
}

} // namespace apple_fp64
