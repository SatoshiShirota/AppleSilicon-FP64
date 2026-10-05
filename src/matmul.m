#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "apple_fp64/matmul.h"
#include "arithmetic.h"
#include "timing.h"

#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/** @brief 計算器が保持するMetalバッファーの用途。 */
typedef enum fp64_workspace_e {
    FP64_PLAN, /**< 入力の値に依存しない復元係数。 */
    FP64_ANALYSIS, /**< GPUが求めた整数幅と非有限値の有無。 */
    FP64_INPUT_A, /**< AのFP64のビット列。 */
    FP64_INPUT_B, /**< BのFP64のビット列。 */
    FP64_SCALES_A, /**< Aの解析結果。調整量を求めるときは列、それ以外は行を扱う。 */
    FP64_SCALES_B, /**< Bの解析結果。調整量を求めるときは行、それ以外は列を扱う。 */
    FP64_INNER_SHIFTS, /**< 内積方向の指数の調整量。 */
    FP64_RESIDUES_A, /**< Aの行のまとまりの余り。 */
    FP64_RESIDUES_B, /**< Bの余り。 */
    FP64_RESIDUES_C, /**< 出力の余り。 */
    FP64_OPERANDS_A, /**< Strassen法で使うAのブロックの和と差。 */
    FP64_OPERANDS_B, /**< Strassen法で使うBのブロックの和と差。 */
    FP64_COMBINED_C, /**< Strassen法の七つの積から組み立てた出力の余り。 */
    FP64_WORKSPACE_COUNT /**< 保持するバッファーの個数。 */
} fp64_workspace_t;

/** @brief 計算器が所有するMetalの資源。 */
@interface AppleFP64Multiplier : NSObject {
@public
    id<MTLDevice> device; /**< 使用するデバイス。 */
    id<MTLCommandQueue> queue; /**< 順序を保持する実行待ち行列。 */
    id<MTLComputePipelineState> row_scales[2]; /**< 行の指数を求めるパイプライン。添字1は指数調整を使う。 */
    id<MTLComputePipelineState> column_scales[2]; /**< 列の指数を求めるパイプライン。添字1は指数調整を使う。 */
    id<MTLComputePipelineState> inner_shifts; /**< 内積方向の指数の調整量を求めるパイプライン。 */
    id<MTLComputePipelineState> analysis; /**< 入力全体に必要な整数幅を求めるパイプライン。 */
    id<MTLComputePipelineState> floating[2]; /**< FP64の積和演算。添字1は有限の入力だけを扱う。 */
    id<MTLComputePipelineState> residues[2]; /**< 入力の余りを生成するパイプライン。添字1は指数調整を使う。 */
    id<MTLComputePipelineState> operands[2][2]; /**< 指数調整の有無とAまたはBに対応する、Strassen法の入力の生成。 */
    id<MTLComputePipelineState> product[2]; /**< 一回の内積と、部分内積を蓄積する行列積のパイプライン。 */
    id<MTLComputePipelineState> combine; /**< 七つの積から出力を組み立てるパイプライン。 */
    id<MTLComputePipelineState> reconstruct[FP64_RECONSTRUCTION_COUNT]; /**< 整数配列の容量ごとのCRTと丸めのパイプライン。 */
    uint32_t plan_inner; /**< 復元係数を作成した内積の項数。係数の作成前は0。 */
    uint32_t plan_precision_a; /**< 復元係数を作成したAの整数幅。 */
    uint32_t plan_precision_b; /**< 復元係数を作成したBの整数幅。 */
    char *device_name; /**< 計算器が所有するUTF-8のデバイス名。 */
    id<MTLBuffer> workspace[FP64_WORKSPACE_COUNT]; /**< 用途ごとに再利用する作業領域。 */
}
@end

@implementation AppleFP64Multiplier

/** @brief ARCで管理されないデバイス名を解放する。 */
- (void)dealloc
{
    free(device_name);
}

@end

/**
 * @brief 診断メッセージを複写して、操作の失敗を返す。
 * @param[out] error 診断の格納先。NULLを許容する。
 * @param[in] status 失敗の分類。
 * @param[in] message UTF-8の失敗理由。
 * @return status。メッセージの確保に失敗した場合はAPPLE_FP64_OUT_OF_MEMORY。
 */
static apple_fp64_status_t fp64_fail(apple_fp64_error_t *error,
                                    apple_fp64_status_t status, const char *message)
{
    if (error != NULL) {
        error->message = strdup(message);
        if (error->message == NULL) return APPLE_FP64_OUT_OF_MEMORY;
    }
    return status;
}

/**
 * @brief 行列の容量を、size_tの範囲内で求める。
 * @param[in] rows 行数。
 * @param[in] columns 列数。
 * @param[in] element_bytes 一要素に必要なバイト数。
 * @param[out] bytes 容量。
 * @return 積を表現できる場合はtrue。
 */
static bool fp64_checked_size(size_t rows, size_t columns, size_t element_bytes, size_t *bytes)
{
    if (columns != 0 && rows > SIZE_MAX / columns) return false;
    size_t count = rows * columns;
    if (element_bytes != 0 && count > SIZE_MAX / element_bytes) return false;
    *bytes = count * element_bytes;
    return true;
}

/**
 * @brief 使用する法と復元係数を、必要な数値範囲から決める。
 * @param[in] k 正の内積の項数。
 * @param[in] precision_a Aの整数幅。
 * @param[in] precision_b Bの整数幅。
 * @param[out] plan 入力の値に依存しない係数。
 * @return 必要な数値範囲を法の積で表せる場合はtrue。
 */
static bool fp64_make_plan(uint32_t k, uint32_t precision_a, uint32_t precision_b,
                           fp64_crt_plan_t *plan)
{
    uint64_t precision = (uint64_t)precision_a + precision_b;
    if (precision + fp64_word_bit_length(k) + 1 > FP64_MAX_LIMBS * FP64_DIGIT_BITS)
        return false;
    fp64_big_uint_t bound = {0};
    for (fp64_word_t bit = 0; bit < 32; ++bit) {
        if ((k >> bit) & 1u) {
            fp64_word_t position = (fp64_word_t)precision + 1 + bit;
            bound.digits[position / FP64_DIGIT_BITS] |= 1u << (position % FP64_DIGIT_BITS);
        }
    }
    memset(plan, 0, sizeof(*plan));
    plan->product.digits[0] = 1;
    for (fp64_word_t index = 0; index < FP64_MAX_MODULI; ++index) {
        fp64_word_t value = FP64_CRT_MODULI[index];
        fp64_modulus_t modulus = {value, (fp64_word_t)((1ull << 32) / value),
                                 (fp64_word_t)((1ull << 32) % value)};
        fp64_word_t t = plan->count++;
        plan->moduli[t] = modulus;
        fp64_big_add_scaled(&plan->product, plan->product, value - 1, FP64_MAX_LIMBS);
        if (fp64_big_compare(plan->product, bound, FP64_MAX_LIMBS) > 0) {
            plan->limbs = (fp64_big_bit_length(plan->product, FP64_MAX_LIMBS)
                           + fp64_word_bit_length(plan->count * 255u) + FP64_DIGIT_BITS - 1) / FP64_DIGIT_BITS;
            fp64_word_t carry = 0;
            for (int i = (int)plan->limbs - 1; i >= 0; --i) {
                plan->half_product.digits[i] = (plan->product.digits[i] >> 1)
                                               | (carry << (FP64_DIGIT_BITS - 1));
                carry = plan->product.digits[i] & 1u;
            }
            fp64_word_t precision_limit = precision_a > precision_b ? precision_a : precision_b;
            for (fp64_word_t index = 0; index < plan->count; ++index) {
                fp64_modulus_t modulus = plan->moduli[index];
                fp64_big_uint_t basis = {0};
                fp64_word_t remainder = 0;
                for (int digit = (int)plan->limbs - 1; digit >= 0; --digit) {
                    fp64_word_t dividend = (remainder << FP64_DIGIT_BITS) | plan->product.digits[digit];
                    basis.digits[digit] = dividend / modulus.value;
                    remainder = dividend % modulus.value;
                }
                assert(remainder == 0);
                remainder = fp64_big_mod(basis, modulus, plan->limbs);
                fp64_word_t inverse = 1;
                while (fp64_word_mod(remainder * inverse, modulus) != 1) ++inverse;
                fp64_big_add_scaled(&plan->coefficients[index], basis, inverse, plan->limbs);
                plan->ratios[index] = (float)((double)inverse / modulus.value);
                fp64_word_t low = 1, high = modulus.word_weight;
                for (fp64_word_t exponent = 0; exponent < precision_limit; ++exponent) {
                    plan->powers[index][exponent] = (unsigned short)(low | (high << 8));
                    low *= 2;
                    if (low >= modulus.value) low -= modulus.value;
                    high *= 2;
                    if (high >= modulus.value) high -= modulus.value;
                }
            }
            return true;
        }
    }
    return false;
}

/**
 * @brief 必要な容量のMetalバッファーを確保または再利用する。
 * @param[in,out] backend 作業領域を所有する計算器。
 * @param[in] kind バッファーの用途。
 * @param[in] bytes 必要なバイト数。
 * @param[in,out] measurement 使用量の加算先。
 * @param[out] error 診断の格納先。NULLを許容する。
 * @return 操作の成否。
 */
static apple_fp64_status_t fp64_reserve_buffer(AppleFP64Multiplier *backend, fp64_workspace_t kind,
                                              size_t bytes, apple_fp64_measurement_t *measurement,
                                              apple_fp64_error_t *error)
{
    if (bytes > backend->device.maxBufferLength)
        return fp64_fail(error, APPLE_FP64_METAL_ERROR, "作業領域がMetalデバイスのバッファー上限を超えます。");
    id<MTLBuffer> buffer = backend->workspace[kind];
    if (buffer == nil || buffer.length < bytes) {
        MTLResourceOptions storage = kind <= FP64_INPUT_B ? MTLResourceStorageModeShared
                                                         : MTLResourceStorageModePrivate;
        buffer = [backend->device newBufferWithLength:bytes options:storage];
        if (buffer == nil)
            return fp64_fail(error, APPLE_FP64_METAL_ERROR, "Metalの作業領域を確保できません。");
        backend->workspace[kind] = buffer;
    }
    measurement->workspace_bytes += buffer.length;
    return APPLE_FP64_SUCCESS;
}

/**
 * @brief 入力解析の結果に対応するCRT係数を再利用または作成する。
 * @param[in,out] backend 復元係数を所有する計算器。
 * @param[in] inner 内積の項数。
 * @param[in] analysis 入力の整数幅と非有限値の有無。
 * @param[out] modular CRTで必要な整数幅を保持できる場合はtrue。
 * @param[out] error 診断の格納先。NULLを許容する。
 * @return 係数の準備の成否。CRTの範囲を超える場合も正常終了する。
 */
static apple_fp64_status_t fp64_prepare_plan(AppleFP64Multiplier *backend, uint32_t inner,
                                            const fp64_input_analysis_t *analysis, bool *modular,
                                            apple_fp64_error_t *error)
{
    *modular = analysis->nonfinite == 0;
    if (!*modular || (backend->plan_inner == inner && backend->plan_precision_a == analysis->precision_a
                                                  && backend->plan_precision_b == analysis->precision_b))
        return APPLE_FP64_SUCCESS;
    fp64_crt_plan_t coefficients;
    *modular = fp64_make_plan(inner, analysis->precision_a, analysis->precision_b, &coefficients);
    if (!*modular) return APPLE_FP64_SUCCESS;
    apple_fp64_measurement_t allocation_measurement = {0};
    apple_fp64_status_t status = fp64_reserve_buffer(backend, FP64_PLAN, sizeof(coefficients), &allocation_measurement, error);
    if (status != APPLE_FP64_SUCCESS) return status;
    memcpy(backend->workspace[FP64_PLAN].contents, &coefficients, sizeof(coefficients));
    backend->plan_inner = inner;
    backend->plan_precision_a = analysis->precision_a;
    backend->plan_precision_b = analysis->precision_b;
    return APPLE_FP64_SUCCESS;
}

/**
 * @brief 一つのMetalカーネルから計算パイプラインを作成する。
 * @param[in] backend 使用するデバイスを所有する計算器。
 * @param[in] library コンパイル済みのMetalライブラリー。
 * @param[in] name カーネル名。
 * @param[out] status 失敗時の分類。
 * @param[out] error 診断の格納先。NULLを許容する。
 * @return 作成したパイプライン。失敗時はnil。
 */
static id<MTLComputePipelineState> fp64_pipeline(AppleFP64Multiplier *backend, id<MTLLibrary> library,
                                               NSString *name, apple_fp64_status_t *status,
                                               apple_fp64_error_t *error)
{
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        *status = fp64_fail(error, APPLE_FP64_METAL_ERROR,
                           [NSString stringWithFormat:@"Metalカーネルを取得できません: %@", name].UTF8String);
        return nil;
    }
    NSError *detail = nil;
    id<MTLComputePipelineState> pipeline = [backend->device newComputePipelineStateWithFunction:function error:&detail];
    if (pipeline == nil)
        *status = fp64_fail(error, APPLE_FP64_METAL_ERROR,
                           [NSString stringWithFormat:@"Metalパイプラインを作成できません: %@",
                                                      detail.localizedDescription].UTF8String);
    return pipeline;
}

/**
 * @brief 未投入のMetalの実行指示を作成する。
 * @param[in] backend 実行待ち行列を所有する計算器。
 * @param[out] status 失敗時の分類。
 * @param[out] error 診断の格納先。NULLを許容する。
 * @return 作成した指示。失敗時はnil。
 */
static id<MTLCommandBuffer> fp64_command(AppleFP64Multiplier *backend, apple_fp64_status_t *status,
                                        apple_fp64_error_t *error)
{
    id<MTLCommandBuffer> command = [backend->queue commandBuffer];
    if (command == nil) *status = fp64_fail(error, APPLE_FP64_METAL_ERROR, "Metalの実行指示を作成できません。");
    return command;
}

/**
 * @brief Metalの計算エンコーダーを作成する。
 * @param[in] command 未投入の指示。
 * @param[in] dispatch_type 同じエンコーダーの処理を実行する順序。
 * @param[out] status 失敗時の分類。
 * @param[out] error 診断の格納先。NULLを許容する。
 * @return 作成したエンコーダー。失敗時はnil。
 */
static id<MTLComputeCommandEncoder> fp64_encoder(id<MTLCommandBuffer> command, MTLDispatchType dispatch_type,
                                                apple_fp64_status_t *status, apple_fp64_error_t *error)
{
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoderWithDispatchType:dispatch_type];
    if (encoder == nil) *status = fp64_fail(error, APPLE_FP64_METAL_ERROR, "Metalの計算エンコーダーを作成できません。");
    return encoder;
}

/**
 * @brief 線形の要素列を、二次元のGPU実行範囲へ割り当てる。
 * @param[in] encoder 設定済みのエンコーダー。
 * @param[in] pipeline 実行するパイプライン。
 * @param[in] count 正の要素数。
 * @param[in] copies 同じ形で実行する範囲の個数。
 */
static void fp64_dispatch_elements(id<MTLComputeCommandEncoder> encoder,
                                   id<MTLComputePipelineState> pipeline, size_t count, size_t copies)
{
    size_t width = count < 65536 ? count : 65536;
    NSUInteger threads = pipeline.maxTotalThreadsPerThreadgroup < 256
                       ? pipeline.maxTotalThreadsPerThreadgroup : 256;
    [encoder dispatchThreads:MTLSizeMake(width, (count + width - 1) / width, copies)
         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

/**
 * @brief GPUの指数取得を記録する。
 * @param[in] backend パイプラインと作業領域を所有する計算器。
 * @param[in] encoder 入力解析をまとめるエンコーダー。
 * @param[in] parameters 入力全体の寸法。
 * @param[in] input_b Bを処理する場合はtrue、Aの場合はfalse。
 * @param[in] inner_axis Aの列またはBの行を解析し、指数調整の準備を行う場合はtrue。
 */
static void fp64_encode_scales(AppleFP64Multiplier *backend, id<MTLComputeCommandEncoder> encoder,
                               fp64_batch_parameters_t parameters, bool input_b, bool inner_axis)
{
    bool columns = input_b != inner_axis;
    bool shifted = parameters.shifted && !inner_axis;
    if (inner_axis) {
        if (input_b) {
            parameters.rows = parameters.inner;
            parameters.inner = parameters.columns;
        } else {
            parameters.columns = parameters.inner;
            parameters.inner = parameters.rows;
        }
    }
    [encoder setComputePipelineState:columns ? backend->column_scales[shifted] : backend->row_scales[shifted]];
    [encoder setBuffer:backend->workspace[input_b ? FP64_INPUT_B : FP64_INPUT_A] offset:0 atIndex:0];
    [encoder setBuffer:backend->workspace[input_b ? FP64_SCALES_B : FP64_SCALES_A] offset:0 atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    if (shifted) [encoder setBuffer:backend->workspace[FP64_INNER_SHIFTS] offset:0 atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(columns ? (parameters.columns + 31) / 32 : parameters.rows, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(columns ? 32 * FP64_COLUMN_SIMD_GROUPS : 32, 1, 1)];
}

/**
 * @brief GPUで入力を解析し、計算方式を選ぶための情報を受け取る。
 * @param[in] backend パイプラインと作業領域を所有する計算器。
 * @param[in] parameters 入力全体の寸法。
 * @param[in,out] measurement GPUの解析時間とCPUの待ち時間の加算先。
 * @param[out] error 診断の格納先。NULLを許容する。
 * @return 操作の成否。
 */
static apple_fp64_status_t fp64_analyse_inputs(AppleFP64Multiplier *backend,
                                              fp64_batch_parameters_t parameters,
                                              apple_fp64_measurement_t *measurement, apple_fp64_error_t *error)
{
    apple_fp64_status_t status = APPLE_FP64_SUCCESS;
    id<MTLCommandBuffer> command = fp64_command(backend, &status, error);
    if (command == nil) return status;
    id<MTLComputeCommandEncoder> encoder = fp64_encoder(command, MTLDispatchTypeConcurrent, &status, error);
    if (encoder == nil) return status;
    if (parameters.shifted) {
        for (fp64_word_t input_b = 0; input_b < 2; ++input_b)
            fp64_encode_scales(backend, encoder, parameters, input_b, true);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:backend->inner_shifts];
        [encoder setBuffer:backend->workspace[FP64_SCALES_A] offset:0 atIndex:0];
        [encoder setBuffer:backend->workspace[FP64_SCALES_B] offset:0 atIndex:1];
        [encoder setBuffer:backend->workspace[FP64_INNER_SHIFTS] offset:0 atIndex:2];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
        fp64_dispatch_elements(encoder, backend->inner_shifts, parameters.inner, 1);
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    for (fp64_word_t columns = 0; columns < 2; ++columns)
        fp64_encode_scales(backend, encoder, parameters, columns, false);
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:backend->analysis];
    [encoder setBuffer:backend->workspace[FP64_SCALES_A] offset:0 atIndex:0];
    [encoder setBuffer:backend->workspace[FP64_SCALES_B] offset:0 atIndex:1];
    [encoder setBuffer:backend->workspace[FP64_ANALYSIS] offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    [command commit];
    double wait_start = fp64_monotonic_seconds();
    [command waitUntilCompleted];
    measurement->wait_seconds += fp64_monotonic_seconds() - wait_start;
    if (command.status != MTLCommandBufferStatusCompleted)
        return fp64_fail(error, APPLE_FP64_METAL_ERROR,
                         [NSString stringWithFormat:@"Metalの入力解析に失敗しました: %@",
                                                    command.error.localizedDescription].UTF8String);
    measurement->prepare_seconds += command.GPUEndTime - command.GPUStartTime;
    return APPLE_FP64_SUCCESS;
}

/**
 * @brief GPUの余りの生成を記録する。
 * @param[in] backend パイプラインと作業領域を所有する計算器。
 * @param[in] command 未投入の指示。
 * @param[in] parameters 行列の寸法と整数幅。
 * @param[in] columns Bを処理する場合は1、Aの場合は0。
 * @param[out] error 診断の格納先。NULLを許容する。
 * @return 操作の成否。
 */
static apple_fp64_status_t fp64_encode_prepare(AppleFP64Multiplier *backend, id<MTLCommandBuffer> command,
                                              fp64_batch_parameters_t parameters, fp64_word_t columns,
                                              apple_fp64_error_t *error)
{
    apple_fp64_status_t status = APPLE_FP64_SUCCESS;
    id<MTLBuffer> input = backend->workspace[columns ? FP64_INPUT_B : FP64_INPUT_A];
    id<MTLBuffer> scales = backend->workspace[columns ? FP64_SCALES_B : FP64_SCALES_A];
    id<MTLBuffer> output = backend->workspace[parameters.strassen ? (columns ? FP64_OPERANDS_B : FP64_OPERANDS_A)
                                                                 : (columns ? FP64_RESIDUES_B : FP64_RESIDUES_A)];
    id<MTLComputeCommandEncoder> encoder = fp64_encoder(command, MTLDispatchTypeSerial, &status, error);
    if (encoder == nil) return status;
    id<MTLComputePipelineState> pipeline = parameters.strassen ? backend->operands[parameters.shifted][columns]
                                                              : backend->residues[parameters.shifted];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:scales offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder setBuffer:backend->workspace[FP64_PLAN] offset:0 atIndex:4];
    if (!parameters.strassen) [encoder setBytes:&columns length:sizeof(columns) atIndex:5];
    if (parameters.shifted)
        [encoder setBuffer:backend->workspace[FP64_INNER_SHIFTS] offset:0 atIndex:parameters.strassen ? 5 : 6];
    size_t count = columns ? (size_t)parameters.inner * parameters.columns : (size_t)parameters.rows * parameters.inner;
    fp64_dispatch_elements(encoder, pipeline, parameters.strassen ? count / 4 : count, 1);
    [encoder endEncoding];
    return APPLE_FP64_SUCCESS;
}

/**
 * @brief すべての法の行列積を一括して記録する。
 * @param[in] backend パイプラインと作業領域を所有する計算器。
 * @param[in] command 未投入の指示。
 * @param[in] parameters 行列の寸法。
 * @param[in] count 法の個数。
 * @param[out] error 診断の格納先。NULLを許容する。
 * @return 操作の成否。
 */
static apple_fp64_status_t fp64_encode_product(AppleFP64Multiplier *backend, id<MTLCommandBuffer> command,
                                              fp64_batch_parameters_t parameters, fp64_word_t count,
                                              apple_fp64_error_t *error)
{
    apple_fp64_status_t status = APPLE_FP64_SUCCESS;
    id<MTLComputeCommandEncoder> encoder = fp64_encoder(command, MTLDispatchTypeSerial, &status, error);
    if (encoder == nil) return status;
    bool strassen = parameters.strassen != 0;
    if (strassen) {
        parameters.rows /= 2;
        parameters.columns /= 2;
        parameters.inner /= 2;
        count *= 7;
    }
    id<MTLComputePipelineState> pipeline = backend->product[parameters.inner <= FP64_K_CHUNK ? 0 : 1];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:backend->workspace[strassen ? FP64_OPERANDS_A : FP64_RESIDUES_A] offset:0 atIndex:0];
    [encoder setBuffer:backend->workspace[strassen ? FP64_OPERANDS_B : FP64_RESIDUES_B] offset:0 atIndex:1];
    [encoder setBuffer:backend->workspace[FP64_RESIDUES_C] offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder setBuffer:backend->workspace[FP64_PLAN] offset:0 atIndex:4];
    [encoder setBuffer:backend->workspace[FP64_SCALES_A] offset:0 atIndex:5];
    [encoder setBuffer:backend->workspace[FP64_SCALES_B] offset:0 atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake((parameters.columns + FP64_TILE_COLUMNS - 1) / FP64_TILE_COLUMNS,
                                             (parameters.rows + FP64_TILE_ROWS - 1) / FP64_TILE_ROWS, count)
           threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth * FP64_SIMD_GROUPS, 1, 1)];
    [encoder endEncoding];
    return APPLE_FP64_SUCCESS;
}

/**
 * @brief GPUのCRTと丸めを記録する。
 * @param[in] backend パイプラインと作業領域を所有する計算器。
 * @param[in] command 未投入の指示。
 * @param[in] parameters 行列の寸法と整数幅。
 * @param[in] limbs 必要な整数の桁数。
 * @param[out] output 呼び出し側へ返す出力と共有するMetalバッファー。
 * @param[out] error 診断の格納先。NULLを許容する。
 * @return 操作の成否。
 */
static apple_fp64_status_t fp64_encode_reconstruct(AppleFP64Multiplier *backend, id<MTLCommandBuffer> command,
                                                  fp64_batch_parameters_t parameters, fp64_word_t limbs,
                                                  id<MTLBuffer> output,
                                                  apple_fp64_error_t *error)
{
    apple_fp64_status_t status = APPLE_FP64_SUCCESS;
    if (parameters.strassen) {
        id<MTLComputeCommandEncoder> encoder = fp64_encoder(command, MTLDispatchTypeSerial, &status, error);
        if (encoder == nil) return status;
        [encoder setComputePipelineState:backend->combine];
        [encoder setBuffer:backend->workspace[FP64_RESIDUES_C] offset:0 atIndex:0];
        [encoder setBuffer:backend->workspace[FP64_COMBINED_C] offset:0 atIndex:1];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
        [encoder setBuffer:backend->workspace[FP64_PLAN] offset:0 atIndex:3];
        fp64_dispatch_elements(encoder, backend->combine,
                               (size_t)(parameters.rows / 2) * (parameters.columns / 2), 4);
        [encoder endEncoding];
    }
    id<MTLComputeCommandEncoder> encoder = fp64_encoder(command, MTLDispatchTypeSerial, &status, error);
    if (encoder == nil) return status;
    size_t index = 0;
    while (limbs > FP64_RECONSTRUCTION_CAPACITIES[index]) ++index;
    id<MTLComputePipelineState> pipeline = backend->reconstruct[index];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:backend->workspace[parameters.strassen ? FP64_COMBINED_C : FP64_RESIDUES_C] offset:0 atIndex:0];
    [encoder setBuffer:backend->workspace[FP64_SCALES_A] offset:0 atIndex:1];
    [encoder setBuffer:backend->workspace[FP64_SCALES_B] offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
    [encoder setBuffer:backend->workspace[FP64_PLAN] offset:0 atIndex:5];
    fp64_dispatch_elements(encoder, pipeline, (size_t)parameters.rows * parameters.columns, 1);
    [encoder endEncoding];
    return APPLE_FP64_SUCCESS;
}

apple_fp64_options_t apple_fp64_default_options(void)
{
    return (apple_fp64_options_t){256};
}

apple_fp64_status_t apple_fp64_multiplier_create(const char *library_path,
                                                apple_fp64_multiplier_t **multiplier, apple_fp64_error_t *error)
{
    assert(library_path != NULL && multiplier != NULL);
    *multiplier = NULL;
    if (error != NULL) error->message = NULL;
    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:library_path];
        if (path == nil) return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, "MetalライブラリーのパスはUTF-8で指定してください。");
        AppleFP64Multiplier *backend = [AppleFP64Multiplier new];
        if (backend == nil) return APPLE_FP64_OUT_OF_MEMORY;
        backend->device = MTLCreateSystemDefaultDevice();
        if (backend->device == nil) return fp64_fail(error, APPLE_FP64_METAL_ERROR, "Metalのデバイスを取得できません。");
        backend->queue = [backend->device newCommandQueue];
        if (backend->queue == nil) return fp64_fail(error, APPLE_FP64_METAL_ERROR, "Metalの実行待ち行列を作成できません。");
        backend->device_name = strdup(backend->device.name.UTF8String);
        if (backend->device_name == NULL) return APPLE_FP64_OUT_OF_MEMORY;
        NSError *detail = nil;
        id<MTLLibrary> library = [backend->device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&detail];
        if (library == nil)
            return fp64_fail(error, APPLE_FP64_METAL_ERROR,
                             [NSString stringWithFormat:@"Metalライブラリーを読み込めません: %@",
                                                        detail.localizedDescription].UTF8String);
        apple_fp64_status_t status = APPLE_FP64_SUCCESS;
        for (unsigned shifted = 0; shifted < 2; ++shifted) {
            backend->row_scales[shifted] = fp64_pipeline(backend, library,
                shifted ? @"find_shifted_row_scales" : @"find_row_scales", &status, error);
            if (backend->row_scales[shifted] == nil) return status;
            backend->column_scales[shifted] = fp64_pipeline(backend, library,
                shifted ? @"find_shifted_column_scales" : @"find_column_scales", &status, error);
            if (backend->column_scales[shifted] == nil) return status;
            backend->residues[shifted] = fp64_pipeline(backend, library,
                shifted ? @"make_shifted_residues" : @"make_residues", &status, error);
            if (backend->residues[shifted] == nil) return status;
            backend->operands[shifted][0] = fp64_pipeline(backend, library,
                shifted ? @"strassen_shifted_operands_a" : @"strassen_operands_a", &status, error);
            if (backend->operands[shifted][0] == nil) return status;
            backend->operands[shifted][1] = fp64_pipeline(backend, library,
                shifted ? @"strassen_shifted_operands_b" : @"strassen_operands_b", &status, error);
            if (backend->operands[shifted][1] == nil) return status;
        }
        backend->inner_shifts = fp64_pipeline(backend, library, @"find_inner_shifts", &status, error);
        if (backend->inner_shifts == nil) return status;
        backend->analysis = fp64_pipeline(backend, library, @"analyse_inputs", &status, error);
        if (backend->analysis == nil) return status;
        backend->floating[0] = fp64_pipeline(backend, library, @"floating_matmul_general", &status, error);
        if (backend->floating[0] == nil) return status;
        backend->floating[1] = fp64_pipeline(backend, library, @"floating_matmul_finite", &status, error);
        if (backend->floating[1] == nil) return status;
        backend->product[0] = fp64_pipeline(backend, library, @"residue_matmul_chunk", &status, error);
        if (backend->product[0] == nil) return status;
        backend->product[1] = fp64_pipeline(backend, library, @"residue_matmul_accumulate", &status, error);
        if (backend->product[1] == nil) return status;
        backend->combine = fp64_pipeline(backend, library, @"strassen_combine", &status, error);
        if (backend->combine == nil) return status;
        for (size_t index = 0; index < FP64_RECONSTRUCTION_COUNT; ++index) {
            NSString *name = [NSString stringWithUTF8String:FP64_RECONSTRUCTION_NAMES[index]];
            backend->reconstruct[index] = fp64_pipeline(backend, library, name, &status, error);
            if (backend->reconstruct[index] == nil) return status;
        }
        *multiplier = (__bridge_retained apple_fp64_multiplier_t *)backend;
        return APPLE_FP64_SUCCESS;
    }
}

void apple_fp64_multiplier_destroy(apple_fp64_multiplier_t *multiplier)
{
    @autoreleasepool {
        id owner = (__bridge_transfer id)multiplier;
        (void)owner;
    }
}

const char *apple_fp64_device_name(const apple_fp64_multiplier_t *multiplier)
{
    assert(multiplier != NULL);
    AppleFP64Multiplier *backend = (__bridge AppleFP64Multiplier *)multiplier;
    return backend->device_name;
}

apple_fp64_status_t apple_fp64_multiply(apple_fp64_multiplier_t *multiplier,
                                      const double *a, size_t a_count, const double *b, size_t b_count,
                                      uint32_t m, uint32_t n, uint32_t k, apple_fp64_options_t options,
                                      apple_fp64_result_t *result, apple_fp64_error_t *error)
{
    assert(multiplier != NULL && result != NULL);
    assert((a != NULL || a_count == 0) && (b != NULL || b_count == 0));
    *result = (apple_fp64_result_t){0};
    if (error != NULL) error->message = NULL;
    @autoreleasepool {
        double start = fp64_monotonic_seconds();
        if (m > INT_MAX || n > INT_MAX || k > INT_MAX)
            return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, "行列の次元はMetalの符号付き32ビット整数の範囲で指定してください。");
        if (a_count != (size_t)m * k || b_count != (size_t)k * n)
            return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, "入力配列の長さが行列の寸法と一致しません。");
        if (options.batch_rows == 0)
            return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, "行のまとまりの大きさは正の値で指定してください。");
        size_t output_bytes;
        if (!fp64_checked_size(m, n, sizeof(double), &output_bytes))
            return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, "行列のサイズがsize_tの範囲を超えます。");
        if (m == 0 || n == 0 || k == 0) {
            if (output_bytes != 0) {
                result->values = calloc(1, output_bytes);
                if (result->values == NULL) return APPLE_FP64_OUT_OF_MEMORY;
            }
            result->count = output_bytes / sizeof(double);
            result->measurement.total_seconds = fp64_monotonic_seconds() - start;
            return APPLE_FP64_SUCCESS;
        }
        AppleFP64Multiplier *backend = (__bridge AppleFP64Multiplier *)multiplier;
        apple_fp64_status_t status = APPLE_FP64_SUCCESS;
        apple_fp64_measurement_t measurement = {0};
        size_t bytes[FP64_WORKSPACE_COUNT] = {0};
        bytes[FP64_ANALYSIS] = sizeof(fp64_input_analysis_t);
        if (!fp64_checked_size(m, k, sizeof(double), &bytes[FP64_INPUT_A])
            || !fp64_checked_size(k, n, sizeof(double), &bytes[FP64_INPUT_B])
            || !fp64_checked_size(m, 1, sizeof(fp64_scale_t), &bytes[FP64_SCALES_A])
            || !fp64_checked_size(n, 1, sizeof(fp64_scale_t), &bytes[FP64_SCALES_B]))
            return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, "行列のサイズがsize_tの範囲を超えます。");
        for (fp64_workspace_t kind = FP64_ANALYSIS; kind <= FP64_SCALES_B; ++kind) {
            status = fp64_reserve_buffer(backend, kind, bytes[kind], &measurement, error);
            if (status != APPLE_FP64_SUCCESS) return status;
        }
        memcpy(backend->workspace[FP64_INPUT_A].contents, a, bytes[FP64_INPUT_A]);
        memcpy(backend->workspace[FP64_INPUT_B].contents, b, bytes[FP64_INPUT_B]);
        fp64_batch_parameters_t parameters = {m, n, k, 0, 1, 1, m, 0, 0};
        status = fp64_analyse_inputs(backend, parameters, &measurement, error);
        if (status != APPLE_FP64_SUCCESS) return status;
        const fp64_input_analysis_t *analysis = backend->workspace[FP64_ANALYSIS].contents;
        bool modular;
        status = fp64_prepare_plan(backend, k, analysis, &modular, error);
        if (status != APPLE_FP64_SUCCESS) return status;
        if (!modular && analysis->nonfinite == 0) {
            if (!fp64_checked_size(m > k ? m : k, 1, sizeof(fp64_scale_t), &bytes[FP64_SCALES_A])
                || !fp64_checked_size(n > k ? n : k, 1, sizeof(fp64_scale_t), &bytes[FP64_SCALES_B])
                || !fp64_checked_size(k, 1, sizeof(int), &bytes[FP64_INNER_SHIFTS]))
                return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, "行列のサイズがsize_tの範囲を超えます。");
            for (fp64_workspace_t kind = FP64_SCALES_A; kind <= FP64_INNER_SHIFTS; ++kind) {
                if (kind != FP64_INNER_SHIFTS) measurement.workspace_bytes -= backend->workspace[kind].length;
                status = fp64_reserve_buffer(backend, kind, bytes[kind], &measurement, error);
                if (status != APPLE_FP64_SUCCESS) return status;
            }
            parameters.shifted = 1;
            status = fp64_analyse_inputs(backend, parameters, &measurement, error);
            if (status != APPLE_FP64_SUCCESS) return status;
            status = fp64_prepare_plan(backend, k, analysis, &modular, error);
            if (status != APPLE_FP64_SUCCESS) return status;
        }
        parameters.precision_a = analysis->precision_a;
        parameters.precision_b = analysis->precision_b;
        const fp64_crt_plan_t *plan = modular ? backend->workspace[FP64_PLAN].contents : NULL;
        measurement.modulus_count = modular ? plan->count : 0;
        uint32_t batch_rows = m < options.batch_rows ? m : options.batch_rows;
        bool strassen = modular && m >= 1024 && n >= 1024 && m % 2 == 0 && n % 128 == 0
                     && k % 2048 == 0 && batch_rows >= 128;
        if (strassen) batch_rows = (batch_rows / 2) * 2;
        uint32_t row_limit = strassen ? m / 2 : m;
        uint32_t row_step = strassen ? batch_rows / 2 : batch_rows;
        if (modular) {
            bytes[FP64_PLAN] = sizeof(*plan);
            if (!fp64_checked_size(batch_rows, n, plan->count, &bytes[FP64_RESIDUES_C]))
                return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, "行列のサイズがsize_tの範囲を超えます。");
            if (strassen) {
                bytes[FP64_COMBINED_C] = bytes[FP64_RESIDUES_C];
                if (!fp64_checked_size(batch_rows / 2, k / 2, 7 * plan->count, &bytes[FP64_OPERANDS_A])
                    || !fp64_checked_size(k / 2, n / 2, 7 * plan->count, &bytes[FP64_OPERANDS_B])
                    || !fp64_checked_size(batch_rows / 2, n / 2, 7 * plan->count, &bytes[FP64_RESIDUES_C]))
                    return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, "行列のサイズがsize_tの範囲を超えます。");
            } else if (!fp64_checked_size(batch_rows, k, plan->count, &bytes[FP64_RESIDUES_A])
                       || !fp64_checked_size(k, n, plan->count, &bytes[FP64_RESIDUES_B])) {
                return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, "行列のサイズがsize_tの範囲を超えます。");
            }
        }
        size_t alignment = (size_t)getpagesize();
        if (output_bytes > (backend->device.maxBufferLength & ~(alignment - 1)))
            return fp64_fail(error, APPLE_FP64_METAL_ERROR, "出力がMetalデバイスのバッファー上限を超えます。");
        size_t buffer_bytes = (output_bytes + alignment - 1) & ~(alignment - 1);
        void *allocation = NULL;
        if (posix_memalign(&allocation, alignment, buffer_bytes) != 0) return APPLE_FP64_OUT_OF_MEMORY;
        double *values = allocation;
        // 出力の記憶領域を解放する前に、その領域を参照するMetalの資源を破棄する。
        @autoreleasepool {
            id<MTLBuffer> output = [backend->device newBufferWithBytesNoCopy:values length:buffer_bytes
                                                                  options:MTLResourceStorageModeShared deallocator:nil];
            if (output == nil) {
                free(values);
                return fp64_fail(error, APPLE_FP64_METAL_ERROR, "Metalの出力バッファーを作成できません。");
            }
            size_t command_capacity = modular ? 1 + 3 * (((size_t)row_limit + row_step - 1) / row_step) : 1;
            // ARCが参照を保持する要素をnilで初期化する。freeの前に各要素をnilへ戻して参照を解放する。
            id<MTLCommandBuffer> __strong *commands = (id<MTLCommandBuffer> __strong *)calloc(command_capacity, sizeof(*commands));
            if (commands == NULL) {
                output = nil;
                free(values);
                return APPLE_FP64_OUT_OF_MEMORY;
            }
            size_t submitted = 0;
            id<MTLCommandBuffer> current = nil;
            measurement.workspace_bytes += buffer_bytes;
            for (fp64_workspace_t kind = FP64_PLAN; kind < FP64_WORKSPACE_COUNT; ++kind) {
                if (bytes[kind] == 0 || (kind >= FP64_ANALYSIS && kind <= FP64_INNER_SHIFTS)) continue;
                status = fp64_reserve_buffer(backend, kind, bytes[kind], &measurement, error);
                if (status != APPLE_FP64_SUCCESS) goto finish;
            }
            if (!modular) {
                current = fp64_command(backend, &status, error);
                if (current == nil) goto finish;
                id<MTLComputeCommandEncoder> encoder = fp64_encoder(current, MTLDispatchTypeSerial, &status, error);
                if (encoder == nil) goto finish;
                [encoder setComputePipelineState:backend->floating[analysis->nonfinite == 0]];
                [encoder setBuffer:backend->workspace[FP64_INPUT_A] offset:0 atIndex:0];
                [encoder setBuffer:backend->workspace[FP64_INPUT_B] offset:0 atIndex:1];
                [encoder setBuffer:output offset:0 atIndex:2];
                [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake((n + 15) / 16, (m + 15) / 16, 1)
                       threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
                [encoder endEncoding];
                commands[submitted++] = current;
                [current commit];
                goto finish;
            }
            parameters.rows = batch_rows;
            parameters.strassen = strassen;
            current = fp64_command(backend, &status, error);
            if (current == nil) goto finish;
            status = fp64_encode_prepare(backend, current, parameters, 1, error);
            if (status != APPLE_FP64_SUCCESS) goto finish;
            commands[submitted++] = current;
            [current commit];
            for (uint32_t row = 0; row < row_limit;) {
                parameters.row_begin = row;
                uint32_t rows = row_step < row_limit - row ? row_step : row_limit - row;
                parameters.rows = strassen ? 2 * rows : rows;
                current = fp64_command(backend, &status, error);
                if (current == nil) goto finish;
                status = fp64_encode_prepare(backend, current, parameters, 0, error);
                if (status != APPLE_FP64_SUCCESS) goto finish;
                commands[submitted++] = current;
                [current commit];
                current = fp64_command(backend, &status, error);
                if (current == nil) goto finish;
                status = fp64_encode_product(backend, current, parameters, plan->count, error);
                if (status != APPLE_FP64_SUCCESS) goto finish;
                commands[submitted++] = current;
                [current commit];
                current = fp64_command(backend, &status, error);
                if (current == nil) goto finish;
                status = fp64_encode_reconstruct(backend, current, parameters, plan->limbs, output, error);
                if (status != APPLE_FP64_SUCCESS) goto finish;
                commands[submitted++] = current;
                [current commit];
                row += rows;
            }
        finish:
            if (submitted != 0) {
                double wait_start = fp64_monotonic_seconds();
                for (size_t index = submitted; index != 0; --index)
                    [commands[index - 1] waitUntilCompleted];
                measurement.wait_seconds += fp64_monotonic_seconds() - wait_start;
            }
            if (status == APPLE_FP64_SUCCESS) {
                for (size_t index = 0; index < submitted; ++index) {
                    id<MTLCommandBuffer> command = commands[index];
                    if (command.status != MTLCommandBufferStatusCompleted) {
                        status = fp64_fail(error, APPLE_FP64_METAL_ERROR,
                                           [NSString stringWithFormat:@"Metalの実行に失敗しました: %@",
                                                                      command.error.localizedDescription].UTF8String);
                        break;
                    }
                    double seconds = command.GPUEndTime - command.GPUStartTime;
                    if (!modular) measurement.product_seconds += seconds;
                    else if (index == 0 || (index - 1) % 3 == 0) measurement.prepare_seconds += seconds;
                    else if ((index - 1) % 3 == 1) measurement.product_seconds += seconds;
                    else measurement.reconstruct_seconds += seconds;
                }
            }
            if (status == APPLE_FP64_SUCCESS) {
                result->values = values;
                result->count = output_bytes / sizeof(double);
                measurement.total_seconds = fp64_monotonic_seconds() - start;
                result->measurement = measurement;
            }
            for (size_t index = 0; index < submitted; ++index) commands[index] = nil;
            free(commands);
        }
        if (status != APPLE_FP64_SUCCESS) free(values);
        return status;
    }
}

void apple_fp64_result_destroy(apple_fp64_result_t *result)
{
    assert(result != NULL);
    free(result->values);
    *result = (apple_fp64_result_t){0};
}

void apple_fp64_error_destroy(apple_fp64_error_t *error)
{
    assert(error != NULL);
    free(error->message);
    error->message = NULL;
}
