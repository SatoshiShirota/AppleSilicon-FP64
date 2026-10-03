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

/** @brief 計算器が保持するMetalバッファーの用途。 */
typedef enum fp64_workspace_e {
    FP64_PLAN, /**< 入力の値に依存しない復元係数。 */
    FP64_INPUT_A, /**< AのFP64のビット列。 */
    FP64_INPUT_B, /**< BのFP64のビット列。 */
    FP64_OUTPUT, /**< 出力のFP64のビット列。 */
    FP64_SCALES_A, /**< Aの行の指数。 */
    FP64_SCALES_B, /**< Bの列の指数。 */
    FP64_RESIDUES_A, /**< Aの行のまとまりの余り。 */
    FP64_RESIDUES_B, /**< Bの余り。 */
    FP64_RESIDUES_C, /**< 出力の余り。 */
    FP64_WORKSPACE_COUNT /**< 保持するバッファーの個数。 */
} fp64_workspace_t;

/** @brief 計算器が所有するMetalの資源。 */
@interface AppleFP64Multiplier : NSObject {
@public
    id<MTLDevice> device; /**< 使用するデバイス。 */
    id<MTLCommandQueue> queue; /**< 順序を保持する実行待ち行列。 */
    id<MTLComputePipelineState> scales; /**< 指数を求めるパイプライン。 */
    id<MTLComputePipelineState> residues; /**< 入力の余りを生成するパイプライン。 */
    id<MTLComputePipelineState> product; /**< 余りの行列積のパイプライン。 */
    id<MTLComputePipelineState> reconstruct; /**< CRTと丸めのパイプライン。 */
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
 * @param[in] options 計算の整数幅。
 * @param[out] plan 入力の値に依存しない係数。
 * @param[out] error 診断の格納先。NULLを許容する。
 * @return 操作の成否。
 */
static apple_fp64_status_t fp64_make_plan(uint32_t k, apple_fp64_options_t options,
                                        fp64_crt_plan_t *plan, apple_fp64_error_t *error)
{
    const char *range_error = "整数幅と内積の長さに必要な範囲が、利用可能な法の積を超えます。";
    uint64_t precision = (uint64_t)options.precision_a + options.precision_b;
    if (precision + fp64_word_bit_length(k) + 1 > FP64_MAX_LIMBS * FP64_DIGIT_BITS)
        return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, range_error);
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
        plan->prefixes[t] = plan->product;
        fp64_word_t remainder = fp64_big_mod(plan->product, modulus, FP64_MAX_LIMBS);
        for (fp64_word_t inverse = 1; inverse < value; ++inverse) {
            if (fp64_word_mod(remainder * inverse, modulus) == 1) {
                plan->inverses[t] = inverse;
                break;
            }
        }
        assert(plan->inverses[t] != 0);
        fp64_big_add_scaled(&plan->product, plan->product, value - 1, FP64_MAX_LIMBS);
        plan->stage_limbs[t] = (fp64_big_bit_length(plan->product, FP64_MAX_LIMBS)
                               + FP64_DIGIT_BITS - 1) / FP64_DIGIT_BITS;
        if (fp64_big_compare(plan->product, bound, FP64_MAX_LIMBS) > 0) {
            plan->limbs = plan->stage_limbs[t];
            fp64_word_t carry = 0;
            for (int i = (int)plan->limbs - 1; i >= 0; --i) {
                plan->half_product.digits[i] = (plan->product.digits[i] >> 1)
                                               | (carry << (FP64_DIGIT_BITS - 1));
                carry = plan->product.digits[i] & 1u;
            }
            fp64_word_t precision_limit = options.precision_a > options.precision_b
                                        ? options.precision_a : options.precision_b;
            for (fp64_word_t index = 0; index < plan->count; ++index) {
                fp64_word_t remainder = 1;
                for (fp64_word_t exponent = 0; exponent < precision_limit; ++exponent) {
                    plan->powers[index][exponent] = (unsigned char)remainder;
                    remainder *= 2;
                    if (remainder >= plan->moduli[index].value) remainder -= plan->moduli[index].value;
                }
            }
            return APPLE_FP64_SUCCESS;
        }
    }
    return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, range_error);
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
        MTLResourceOptions storage = kind <= FP64_OUTPUT ? MTLResourceStorageModeShared
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
 * @param[out] status 失敗時の分類。
 * @param[out] error 診断の格納先。NULLを許容する。
 * @return 作成したエンコーダー。失敗時はnil。
 */
static id<MTLComputeCommandEncoder> fp64_encoder(id<MTLCommandBuffer> command,
                                                apple_fp64_status_t *status, apple_fp64_error_t *error)
{
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) *status = fp64_fail(error, APPLE_FP64_METAL_ERROR, "Metalの計算エンコーダーを作成できません。");
    return encoder;
}

/**
 * @brief 線形の要素列を、二次元のGPU実行範囲へ割り当てる。
 * @param[in] encoder 設定済みのエンコーダー。
 * @param[in] pipeline 実行するパイプライン。
 * @param[in] count 正の要素数。
 */
static void fp64_dispatch_elements(id<MTLComputeCommandEncoder> encoder,
                                   id<MTLComputePipelineState> pipeline, size_t count)
{
    size_t width = count < 65536 ? count : 65536;
    NSUInteger threads = pipeline.maxTotalThreadsPerThreadgroup < 256
                       ? pipeline.maxTotalThreadsPerThreadgroup : 256;
    [encoder dispatchThreads:MTLSizeMake(width, (count + width - 1) / width, 1)
         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

/**
 * @brief GPUの指数取得と余りの生成を記録する。
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
    id<MTLBuffer> output = backend->workspace[columns ? FP64_RESIDUES_B : FP64_RESIDUES_A];
    id<MTLComputeCommandEncoder> encoder = fp64_encoder(command, &status, error);
    if (encoder == nil) return status;
    [encoder setComputePipelineState:backend->scales];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:scales offset:0 atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [encoder setBytes:&columns length:sizeof(columns) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(columns ? parameters.columns : parameters.rows, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    encoder = fp64_encoder(command, &status, error);
    if (encoder == nil) return status;
    [encoder setComputePipelineState:backend->residues];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:scales offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder setBuffer:backend->workspace[FP64_PLAN] offset:0 atIndex:4];
    [encoder setBytes:&columns length:sizeof(columns) atIndex:5];
    fp64_dispatch_elements(encoder, backend->residues,
                           columns ? (size_t)parameters.inner * parameters.columns
                                   : (size_t)parameters.rows * parameters.inner);
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
    id<MTLComputeCommandEncoder> encoder = fp64_encoder(command, &status, error);
    if (encoder == nil) return status;
    [encoder setComputePipelineState:backend->product];
    [encoder setBuffer:backend->workspace[FP64_RESIDUES_A] offset:0 atIndex:0];
    [encoder setBuffer:backend->workspace[FP64_RESIDUES_B] offset:0 atIndex:1];
    [encoder setBuffer:backend->workspace[FP64_RESIDUES_C] offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder setBuffer:backend->workspace[FP64_PLAN] offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((parameters.columns + FP64_TILE_COLUMNS - 1) / FP64_TILE_COLUMNS,
                                             (parameters.rows + FP64_TILE_ROWS - 1) / FP64_TILE_ROWS, count)
           threadsPerThreadgroup:MTLSizeMake(backend->product.threadExecutionWidth * FP64_SIMD_GROUPS, 1, 1)];
    [encoder endEncoding];
    return APPLE_FP64_SUCCESS;
}

/**
 * @brief GPUのCRTと丸めを記録する。
 * @param[in] backend パイプラインと作業領域を所有する計算器。
 * @param[in] command 未投入の指示。
 * @param[in] parameters 行列の寸法と整数幅。
 * @param[out] error 診断の格納先。NULLを許容する。
 * @return 操作の成否。
 */
static apple_fp64_status_t fp64_encode_reconstruct(AppleFP64Multiplier *backend, id<MTLCommandBuffer> command,
                                                  fp64_batch_parameters_t parameters, apple_fp64_error_t *error)
{
    apple_fp64_status_t status = APPLE_FP64_SUCCESS;
    id<MTLComputeCommandEncoder> encoder = fp64_encoder(command, &status, error);
    if (encoder == nil) return status;
    [encoder setComputePipelineState:backend->reconstruct];
    [encoder setBuffer:backend->workspace[FP64_RESIDUES_C] offset:0 atIndex:0];
    [encoder setBuffer:backend->workspace[FP64_SCALES_A] offset:0 atIndex:1];
    [encoder setBuffer:backend->workspace[FP64_SCALES_B] offset:0 atIndex:2];
    [encoder setBuffer:backend->workspace[FP64_OUTPUT] offset:0 atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
    [encoder setBuffer:backend->workspace[FP64_PLAN] offset:0 atIndex:5];
    fp64_dispatch_elements(encoder, backend->reconstruct, (size_t)parameters.rows * parameters.columns);
    [encoder endEncoding];
    return APPLE_FP64_SUCCESS;
}

apple_fp64_options_t apple_fp64_default_options(void)
{
    return (apple_fp64_options_t){60, 60, 256};
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
        backend->scales = fp64_pipeline(backend, library, @"find_scales", &status, error);
        if (backend->scales == nil) return status;
        backend->residues = fp64_pipeline(backend, library, @"make_residues", &status, error);
        if (backend->residues == nil) return status;
        backend->product = fp64_pipeline(backend, library, @"residue_matmul", &status, error);
        if (backend->product == nil) return status;
        backend->reconstruct = fp64_pipeline(backend, library, @"reconstruct", &status, error);
        if (backend->reconstruct == nil) return status;
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
        if (options.precision_a == 0 || options.precision_b == 0 || options.batch_rows == 0)
            return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, "整数幅と行のまとまりの大きさは正の値で指定してください。");
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
        fp64_crt_plan_t plan;
        apple_fp64_status_t status = fp64_make_plan(k, options, &plan, error);
        if (status != APPLE_FP64_SUCCESS) return status;
        uint32_t batch_rows = m < options.batch_rows ? m : options.batch_rows;
        size_t bytes[FP64_WORKSPACE_COUNT];
        bytes[FP64_PLAN] = sizeof(plan);
        bytes[FP64_OUTPUT] = output_bytes;
        if (!fp64_checked_size(m, k, sizeof(double), &bytes[FP64_INPUT_A])
            || !fp64_checked_size(k, n, sizeof(double), &bytes[FP64_INPUT_B])
            || !fp64_checked_size(batch_rows, 1, sizeof(int), &bytes[FP64_SCALES_A])
            || !fp64_checked_size(n, 1, sizeof(int), &bytes[FP64_SCALES_B])
            || !fp64_checked_size(batch_rows, k, plan.count, &bytes[FP64_RESIDUES_A])
            || !fp64_checked_size(k, n, plan.count, &bytes[FP64_RESIDUES_B])
            || !fp64_checked_size(batch_rows, n, plan.count, &bytes[FP64_RESIDUES_C]))
            return fp64_fail(error, APPLE_FP64_INVALID_ARGUMENT, "行列のサイズがsize_tの範囲を超えます。");
        double *values = malloc(output_bytes);
        if (values == NULL) return APPLE_FP64_OUT_OF_MEMORY;
        size_t command_capacity = 1 + 3 * (((size_t)m + batch_rows - 1) / batch_rows);
        // ARCが参照を保持する要素をnilで初期化する。freeの前に各要素をnilへ戻して参照を解放する。
        id<MTLCommandBuffer> __strong *commands = (id<MTLCommandBuffer> __strong *)calloc(command_capacity, sizeof(*commands));
        if (commands == NULL) {
            free(values);
            return APPLE_FP64_OUT_OF_MEMORY;
        }
        size_t submitted = 0;
        AppleFP64Multiplier *backend = (__bridge AppleFP64Multiplier *)multiplier;
        id<MTLCommandBuffer> current = nil;
        apple_fp64_measurement_t measurement = {0};
        measurement.modulus_count = plan.count;
        for (fp64_workspace_t kind = FP64_PLAN; kind < FP64_WORKSPACE_COUNT; ++kind) {
            status = fp64_reserve_buffer(backend, kind, bytes[kind], &measurement, error);
            if (status != APPLE_FP64_SUCCESS) goto finish;
        }
        memcpy(backend->workspace[FP64_PLAN].contents, &plan, sizeof(plan));
        memcpy(backend->workspace[FP64_INPUT_A].contents, a, bytes[FP64_INPUT_A]);
        memcpy(backend->workspace[FP64_INPUT_B].contents, b, bytes[FP64_INPUT_B]);
        fp64_batch_parameters_t parameters = {batch_rows, n, k, 0, options.precision_a, options.precision_b};
        current = fp64_command(backend, &status, error);
        if (current == nil) goto finish;
        status = fp64_encode_prepare(backend, current, parameters, 1, error);
        if (status != APPLE_FP64_SUCCESS) goto finish;
        commands[submitted++] = current;
        [current commit];
        for (uint32_t row = 0; row < m;) {
            parameters.row_begin = row;
            parameters.rows = batch_rows < m - row ? batch_rows : m - row;
            current = fp64_command(backend, &status, error);
            if (current == nil) goto finish;
            status = fp64_encode_prepare(backend, current, parameters, 0, error);
            if (status != APPLE_FP64_SUCCESS) goto finish;
            commands[submitted++] = current;
            [current commit];
            current = fp64_command(backend, &status, error);
            if (current == nil) goto finish;
            status = fp64_encode_product(backend, current, parameters, plan.count, error);
            if (status != APPLE_FP64_SUCCESS) goto finish;
            commands[submitted++] = current;
            [current commit];
            current = fp64_command(backend, &status, error);
            if (current == nil) goto finish;
            status = fp64_encode_reconstruct(backend, current, parameters, error);
            if (status != APPLE_FP64_SUCCESS) goto finish;
            commands[submitted++] = current;
            [current commit];
            row += parameters.rows;
        }
    finish:
        if (submitted != 0) {
            double wait_start = fp64_monotonic_seconds();
            for (size_t index = submitted; index != 0; --index)
                [commands[index - 1] waitUntilCompleted];
            measurement.wait_seconds = fp64_monotonic_seconds() - wait_start;
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
                if (index == 0 || (index - 1) % 3 == 0) measurement.prepare_seconds += seconds;
                else if ((index - 1) % 3 == 1) measurement.product_seconds += seconds;
                else measurement.reconstruct_seconds += seconds;
            }
        }
        if (status == APPLE_FP64_SUCCESS) {
            memcpy(values, backend->workspace[FP64_OUTPUT].contents, output_bytes);
            result->values = values;
            result->count = output_bytes / sizeof(double);
            measurement.total_seconds = fp64_monotonic_seconds() - start;
            result->measurement = measurement;
        } else {
            free(values);
        }
        for (size_t index = 0; index < submitted; ++index) commands[index] = nil;
        free(commands);
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
