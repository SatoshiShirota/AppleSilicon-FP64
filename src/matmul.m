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

/**
 * @~japanese
 * @brief 計算器が保持して再利用するMetalバッファーの用途。
 * @~english
 * @brief Roles of the Metal workspace buffers retained and reused by the multiplier.
 * @~
 */
typedef enum fp64_workspace_e {
    FP64_PLAN, /**< @~japanese 入力の具体的な要素値に依存しないCRTの定数表。
                * @~english CRT constant tables independent of specific input element values.
                * @~
                */
    FP64_ANALYSIS, /**< @~japanese GPUでの入力解析で求めた整数幅と非有限値の有無。
                    * @~english Integer widths and nonfinite flag determined by GPU input analysis.
                    * @~
                    */
    FP64_INPUT_A, /**< @~japanese 入力行列AのFP64ビット列。
                   * @~english FP64 bit patterns of input matrix A.
                   * @~
                   */
    FP64_INPUT_B, /**< @~japanese 入力行列BのFP64ビット列。
                   * @~english FP64 bit patterns of input matrix B.
                   * @~
                   */
    FP64_SCALES_A, /**< @~japanese 行列Aの解析結果。内積方向の調整量を求めるときは各列の解析結果を、それ以外のときは各行の解析結果を格納する。
                    * @~english Analysis of matrix A: stores per-column results when computing inner-dimension
                    * adjustments, and per-row results otherwise.
                    * @~
                    */
    FP64_SCALES_B, /**< @~japanese 行列Bの解析結果。内積方向の調整量を求めるときは各行の解析結果を、それ以外のときは各列の解析結果を格納する。
                    * @~english Analysis of matrix B: stores per-row results when computing inner-dimension
                    * adjustments, and per-column results otherwise.
                    * @~
                    */
    FP64_INNER_SHIFTS, /**< @~japanese 内積方向の指数の調整量。
                        * @~english Exponent adjustments along the inner dimension.
                        * @~
                        */
    FP64_RESIDUES_A, /**< @~japanese 行列Aの1つの行のまとまりに対するINT8の余り。
                      * @~english INT8 residues for one row batch of matrix A.
                      * @~
                      */
    FP64_RESIDUES_B, /**< @~japanese 行列B全体に対するINT8の余り。
                      * @~english INT8 residues for the entire matrix B.
                      * @~
                      */
    FP64_RESIDUES_C, /**< @~japanese INT8の行列積が出力した法ごとの余り。
                      * @~english Per-modulus output residues produced by INT8 matrix multiplication.
                      * @~
                      */
    FP64_OPERANDS_A, /**< @~japanese Strassen法で使う行列Aのブロックの和と差の余り。
                      * @~english Residue sums and differences of matrix A blocks for Strassen multiplication.
                      * @~
                      */
    FP64_OPERANDS_B, /**< @~japanese Strassen法で使う行列Bのブロックの和と差の余り。
                      * @~english Residue sums and differences of matrix B blocks for Strassen multiplication.
                      * @~
                      */
    FP64_COMBINED_C, /**< @~japanese Strassen法の7つのブロック積から組み立てた出力ブロックの余り。
                      * @~english Output block residues assembled from the seven Strassen block products.
                      * @~
                      */
    FP64_WORKSPACE_COUNT /**< @~japanese 計算器が管理する作業領域バッファーの種類数。
                          * @~english Number of workspace buffer roles managed by the multiplier.
                          * @~
                          */
} fp64_workspace_t;

/**
 * @~japanese
 * @brief 計算器が所有するMetalのパイプラインと作業領域バッファー。
 * @~english
 * @brief Metal pipelines and workspace buffers owned by the multiplier.
 * @~
 */
@interface AppleFP64Multiplier : NSObject {
@public
    id<MTLDevice> device; /**< @~japanese 計算を実行するMetalデバイス。
                           * @~english Metal device executing the computations.
                           * @~
                           */
    id<MTLCommandQueue> queue; /**< @~japanese コマンドバッファーを投入順に実行するコマンドキュー。
                                * @~english Command queue executing command buffers in submission order.
                                * @~
                                */
    id<MTLComputePipelineState> row_scales[2]; /**< @~japanese 行ごとの指数を解析するパイプライン。添字0は調整なし、添字1は内積方向の指数調整あり。
                                                * @~english Pipelines for per-row exponent analysis: index 0 is
                                                * unadjusted, index 1 applies inner-dimension adjustments.
                                                * @~
                                                */
    id<MTLComputePipelineState> column_scales[2]; /**< @~japanese 列ごとの指数を解析するパイプライン。添字0は調整なし、添字1は内積方向の指数調整あり。
                                                   * @~english Pipelines for per-column exponent analysis: index
                                                   * 0 is unadjusted, index 1 applies inner-dimension
                                                   * adjustments.
                                                   * @~
                                                   */
    id<MTLComputePipelineState> inner_shifts; /**< @~japanese 内積方向の指数の調整量を求めるパイプライン。
                                               * @~english Pipeline computing exponent adjustments along the
                                               * inner dimension.
                                               * @~
                                               */
    id<MTLComputePipelineState> analysis; /**< @~japanese 行と列の解析結果から全体の整数幅を集計するパイプライン。
                                           * @~english Pipeline aggregating per-row and per-column analyses into
                                           * full-matrix integer widths.
                                           * @~
                                           */
    id<MTLComputePipelineState> floating[2]; /**< @~japanese GPUの整数演算によるFP64の積和演算パイプライン。添字0はNaNや無限大を含む一般入力用、添字1は有限値専用。
                                              * @~english Integer-based FP64 fused multiply-add pipelines: index
                                              * 0 supports general inputs including NaN/Inf, index 1 is
                                              * specialized for finite inputs.
                                              * @~
                                              */
    id<MTLComputePipelineState> residues[2]; /**< @~japanese 入力要素からINT8の余りを生成するパイプライン。添字0は調整なし、添字1は内積方向の指数調整あり。
                                              * @~english Pipelines generating INT8 residues from inputs: index
                                              * 0 is unadjusted, index 1 applies inner-dimension adjustments.
                                              * @~
                                              */
    id<MTLComputePipelineState> operands[2][2]; /**< @~japanese Strassen法の7つのブロック演算の入力を生成するパイプライン。第1添字は指数調整の有無、第2添字はA（0）またはB（1）。
                                                 * @~english Pipelines generating inputs for the seven Strassen
                                                 * block products, indexed by [shifted][is_matrix_b].
                                                 * @~
                                                 */
    id<MTLComputePipelineState> product[2]; /**< @~japanese INT8の余りの行列積パイプライン。添字0は単一区間（1024項以下）用、添字1は複数区間の加算用。
                                             * @~english INT8 modular matrix multiplication pipelines: index 0
                                             * for a single chunk (at most 1024 terms), index 1 for accumulating
                                             * multiple chunks.
                                             * @~
                                             */
    id<MTLComputePipelineState> combine; /**< @~japanese Strassen法の7つのブロック積から4つの出力ブロックの余りを組み立てるパイプライン。
                                          * @~english Pipeline assembling four output blocks from the seven
                                          * Strassen block products.
                                          * @~
                                          */
    id<MTLComputePipelineState> reconstruct[FP64_RECONSTRUCTION_COUNT]; /**< @~japanese 多倍長整数の桁数ごとに用意したCRT復元とFP64丸めのパイプライン。
                                                                         * @~english CRT reconstruction and FP64
                                                                         * rounding pipelines for each
                                                                         * multi-precision digit capacity.
                                                                         * @~
                                                                         */
    uint32_t plan_inner; /**< @~japanese 現在のCRT定数表を作成したときの内積の項数。未作成の場合は0。
                          * @~english Dot-product length for which the current CRT constant table was built, or
                          * 0 if none has been built yet.
                          * @~
                          */
    uint32_t plan_precision_a; /**< @~japanese 現在のCRT定数表を作成したときの行列Aの整数幅。
                                * @~english Integer width of matrix A for which the current CRT constant table
                                * was built.
                                * @~
                                */
    uint32_t plan_precision_b; /**< @~japanese 現在のCRT定数表を作成したときの行列Bの整数幅。
                                * @~english Integer width of matrix B for which the current CRT constant table
                                * was built.
                                * @~
                                */
    char *device_name; /**< @~japanese 計算器が所有するUTF-8のデバイス名文字列。
                        * @~english UTF-8 device name string owned by the multiplier.
                        * @~
                        */
    id<MTLBuffer> workspace[FP64_WORKSPACE_COUNT]; /**< @~japanese 用途ごとに確保して再利用するMetalバッファー。
                                                    * @~english Metal buffers allocated and reused for each
                                                    * workspace role.
                                                    * @~
                                                    */
}
@end

@implementation AppleFP64Multiplier

/**
 * @~japanese
 * @brief Objective-CのARC管理対象外であるC文字列のデバイス名を解放する。
 * @~english
 * @brief Free the C-string device name not managed by Objective-C ARC.
 * @~
 */
- (void)dealloc
{
    free(device_name);
}

@end

/**
 * @~japanese
 * @brief 失敗時の診断メッセージを複製してerrorへ格納し、状態コードを返す。
 * @param[out] error 診断メッセージの格納先。NULLを指定した場合はメッセージを格納しない。
 * @param[in] status 返す失敗の状態コード。
 * @param[in] message UTF-8の診断メッセージ文字列。
 * @return 引数のstatus。ただし診断メッセージのメモリー確保に失敗した場合はAPPLE_FP64_OUT_OF_MEMORY。
 * @~english
 * @brief Duplicate a diagnostic message into error and return the failure status code.
 * @param[out] error Destination for the diagnostic message, or NULL if no message is requested.
 * @param[in] status Failure status code to return.
 * @param[in] message UTF-8 diagnostic message string.
 * @return status, or APPLE_FP64_OUT_OF_MEMORY if duplicating the message fails.
 * @~
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
 * @~japanese
 * @brief 行数、列数、および1要素あたりのバイト数の積を、size_tのオーバーフローを検査しながら求める。
 * @param[in] rows 行数。
 * @param[in] columns 列数。
 * @param[in] element_bytes 1要素あたりのバイト数。
 * @param[out] bytes 算出した合計バイト数の格納先。
 * @return 積がsize_tの範囲内に収まる場合はtrue、オーバーフローする場合はfalse。
 * @~english
 * @brief Compute rows * columns * element_bytes while checking for size_t overflow.
 * @param[in] rows Number of rows.
 * @param[in] columns Number of columns.
 * @param[in] element_bytes Number of bytes per element.
 * @param[out] bytes Destination for the total byte count.
 * @return true if the product fits in size_t, false on overflow.
 * @~
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
 * @~japanese
 * @brief 内積の項数と整数幅から、必要な法の組み合わせとCRT復元用の定数表を作成する。
 * @param[in] k 内積の項数（1以上）。
 * @param[in] precision_a 行列Aの整数幅。
 * @param[in] precision_b 行列Bの整数幅。
 * @param[out] plan 作成したCRT定数表の格納先。
 * @return 利用可能な法の積で内積の数値範囲を表現できる場合はtrue、表現範囲を超える場合はfalse。
 * @~english
 * @brief Build the active modulus set and CRT constant tables for the given dot-product length and integer
 * widths.
 * @param[in] k Positive number of terms in each dot product.
 * @param[in] precision_a Integer width of matrix A.
 * @param[in] precision_b Integer width of matrix B.
 * @param[out] plan Destination for the constructed CRT constant tables.
 * @return true if the product of available moduli covers the required dot-product range, otherwise false.
 * @~
 */
static bool fp64_make_plan(uint32_t k, uint32_t precision_a, uint32_t precision_b,
                           fp64_crt_plan_t *plan)
{
    uint64_t precision = (uint64_t)precision_a + precision_b;
    if (precision + fp64_word_bit_length(k) + 1 > FP64_MAX_LIMBS * FP64_DIGIT_BITS)
        return false;
    // 符号付きの厳密な内積は [-K * 2^(p_A + p_B), K * 2^(p_A + p_B)] に収まるため、
    // 法の積 P が bound = 2 * K * 2^(p_A + p_B) を超えるまで大きい順から法を追加する。
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
 * @~japanese
 * @brief 指定された容量以上のMetalバッファーを計算器の作業領域から再利用するか、新しく確保する。
 * @param[in,out] backend 作業領域を所有する計算器。
 * @param[in] kind 確保または再利用するバッファーの用途。
 * @param[in] bytes 必要な最小バイト数。
 * @param[in,out] measurement 確保したバッファー容量を加算する測定値。
 * @param[out] error 失敗時の診断メッセージの格納先。NULLを指定できる。
 * @return 成功した場合はAPPLE_FP64_SUCCESS、失敗した場合はAPPLE_FP64_METAL_ERROR。
 * @~english
 * @brief Reuse an existing Metal workspace buffer of sufficient capacity or allocate a new one.
 * @param[in,out] backend Multiplier owning the workspace buffers.
 * @param[in] kind Workspace buffer role.
 * @param[in] bytes Minimum required capacity in bytes.
 * @param[in,out] measurement Measurement structure to which the buffer's capacity is added.
 * @param[out] error Destination for the failure diagnostic message. May be NULL.
 * @return APPLE_FP64_SUCCESS on success, or APPLE_FP64_METAL_ERROR on failure.
 * @~
 */
static apple_fp64_status_t fp64_reserve_buffer(AppleFP64Multiplier *backend, fp64_workspace_t kind,
                                              size_t bytes, apple_fp64_measurement_t *measurement,
                                              apple_fp64_error_t *error)
{
    if (bytes > backend->device.maxBufferLength)
        return fp64_fail(error, APPLE_FP64_METAL_ERROR, "作業領域がMetalデバイスのバッファー上限を超えます。");
    id<MTLBuffer> buffer = backend->workspace[kind];
    if (buffer == nil || buffer.length < bytes) {
        // CPUとやり取りする定数表・解析結果・入力行列はSharedモード、GPU内部だけで使う中間領域はPrivateモードで確保する。
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
 * @~japanese
 * @brief 入力解析で得られた整数幅に対応するCRT定数表を再利用するか、新しく作成してMetalバッファーへ書き込む。
 * @param[in,out] backend CRT定数表を保持する計算器。
 * @param[in] inner 内積の項数。
 * @param[in] analysis GPUが求めた整数幅と非有限値の有無。
 * @param[out] modular CRTによる行列積を実行できる場合はtrue、範囲外または非有限値を含む場合はfalse。
 * @param[out] error 失敗時の診断メッセージの格納先。NULLを指定できる。
 * @return バッファー確保の成否。整数幅がCRTの範囲を超える場合も関数自体はAPPLE_FP64_SUCCESSを返し、*modularをfalseにする。
 * @~english
 * @brief Reuse or build the CRT constant table matching the analyzed integer widths and copy it into the Metal
 * buffer.
 * @param[in,out] backend Multiplier retaining the CRT constant table.
 * @param[in] inner Number of terms in each dot product.
 * @param[in] analysis GPU analysis containing the required integer widths and nonfinite flag.
 * @param[out] modular Set to true if modular CRT multiplication is usable, or false if out of range or
 * nonfinite.
 * @param[out] error Destination for the failure diagnostic message. May be NULL.
 * @return Buffer allocation status. Exceeding the CRT range is not an error and returns APPLE_FP64_SUCCESS with
 * *modular set to false.
 * @~
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
 * @~japanese
 * @brief コンパイル済みのMetalライブラリーから指定した名前のカーネルを取得し、計算パイプラインを作成する。
 * @param[in] backend デバイスを所有する計算器。
 * @param[in] library 読み込み済みのMetalライブラリー。
 * @param[in] name 取得するカーネルの関数名。
 * @param[out] status 失敗したときの状態コードの格納先。
 * @param[out] error 失敗時の診断メッセージの格納先。NULLを指定できる。
 * @return 作成した計算パイプライン。失敗した場合はnil。
 * @~english
 * @brief Look up a kernel function by name in the compiled Metal library and create a compute pipeline state.
 * @param[in] backend Multiplier owning the Metal device.
 * @param[in] library Loaded Metal library.
 * @param[in] name Kernel function name.
 * @param[out] status Destination for the failure status code.
 * @param[out] error Destination for the failure diagnostic message. May be NULL.
 * @return Created compute pipeline state, or nil on failure.
 * @~
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
 * @~japanese
 * @brief コマンドキューから新しい未投入のMetalコマンドバッファーを作成する。
 * @param[in] backend コマンドキューを所有する計算器。
 * @param[out] status 失敗したときの状態コードの格納先。
 * @param[out] error 失敗時の診断メッセージの格納先。NULLを指定できる。
 * @return 作成したコマンドバッファー。失敗した場合はnil。
 * @~english
 * @brief Create a new uncommitted Metal command buffer from the command queue.
 * @param[in] backend Multiplier owning the command queue.
 * @param[out] status Destination for the failure status code.
 * @param[out] error Destination for the failure diagnostic message. May be NULL.
 * @return Created command buffer, or nil on failure.
 * @~
 */
static id<MTLCommandBuffer> fp64_command(AppleFP64Multiplier *backend, apple_fp64_status_t *status,
                                        apple_fp64_error_t *error)
{
    id<MTLCommandBuffer> command = [backend->queue commandBuffer];
    if (command == nil) *status = fp64_fail(error, APPLE_FP64_METAL_ERROR, "Metalの実行指示を作成できません。");
    return command;
}

/**
 * @~japanese
 * @brief コマンドバッファーから指定したディスパッチ方式のMetal計算エンコーダーを作成する。
 * @param[in] command エンコーダーを作成するコマンドバッファー。
 * @param[in] dispatch_type エンコーダー内のカーネルを直列（Serial）または並行（Concurrent）のどちらで実行するかの指定。
 * @param[out] status 失敗したときの状態コードの格納先。
 * @param[out] error 失敗時の診断メッセージの格納先。NULLを指定できる。
 * @return 作成した計算エンコーダー。失敗した場合はnil。
 * @~english
 * @brief Create a Metal compute command encoder with the specified dispatch type from a command buffer.
 * @param[in] command Command buffer from which to create the encoder.
 * @param[in] dispatch_type Serial or concurrent dispatch execution mode within the encoder.
 * @param[out] status Destination for the failure status code.
 * @param[out] error Destination for the failure diagnostic message. May be NULL.
 * @return Created compute command encoder, or nil on failure.
 * @~
 */
static id<MTLComputeCommandEncoder> fp64_encoder(id<MTLCommandBuffer> command, MTLDispatchType dispatch_type,
                                                apple_fp64_status_t *status, apple_fp64_error_t *error)
{
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoderWithDispatchType:dispatch_type];
    if (encoder == nil) *status = fp64_fail(error, APPLE_FP64_METAL_ERROR, "Metalの計算エンコーダーを作成できません。");
    return encoder;
}

/**
 * @~japanese
 * @brief 1次元の要素列を最大幅65536の2次元グリッドへ折り返してGPUスレッドをディスパッチする。
 * @param[in] encoder ディスパッチを記録する計算エンコーダー。
 * @param[in] pipeline 実行する計算パイプライン。
 * @param[in] count 1面あたりの要素数（1以上）。
 * @param[in] copies Z方向に並べて同時にディスパッチする面の個数。
 * @~english
 * @brief Dispatch a linear sequence of elements across a 2D GPU grid of width at most 65536.
 * @param[in] encoder Compute command encoder recording the dispatch.
 * @param[in] pipeline Compute pipeline state to execute.
 * @param[in] count Positive number of elements per plane.
 * @param[in] copies Number of planes dispatched along the Z dimension.
 * @~
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
 * @~japanese
 * @brief 行ごとまたは列ごとの指数の範囲を調べるカーネルを計算エンコーダーへ記録する。
 * @param[in] backend パイプラインと作業領域を所有する計算器。
 * @param[in] encoder コマンドを記録する計算エンコーダー。
 * @param[in] parameters 行列全体の寸法。
 * @param[in] input_b 行列Bを解析する場合はtrue、行列Aを解析する場合はfalse。
 * @param[in] inner_axis 内積方向の調整量を求めるためにAの列またはBの行を解析する場合はtrue、通常のAの行またはBの列を解析する場合はfalse。
 * @~english
 * @brief Encode a row-scale or column-scale exponent analysis kernel into the compute encoder.
 * @param[in] backend Multiplier owning the pipelines and workspace buffers.
 * @param[in] encoder Compute command encoder recording the dispatch.
 * @param[in] parameters Dimensions of the full matrices.
 * @param[in] input_b true when analyzing matrix B, false when analyzing matrix A.
 * @param[in] inner_axis true when analyzing columns of A or rows of B to compute inner-dimension adjustments,
 * false when analyzing rows of A or columns of B.
 * @~
 */
static void fp64_encode_scales(AppleFP64Multiplier *backend, id<MTLComputeCommandEncoder> encoder,
                               fp64_batch_parameters_t parameters, bool input_b, bool inner_axis)
{
    bool columns = input_b != inner_axis;
    bool shifted = parameters.shifted && !inner_axis;
    // 内積方向の調整量を求める準備（inner_axis == true）ではAの列とBの行を走査するため、寸法のフィールドをカーネルの走査方向へ入れ替える。
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
 * @~japanese
 * @brief GPUで入力の指数範囲を解析し、全体の整数幅p_A, p_Bと非有限値の有無をFP64_ANALYSISバッファーへ取得する。
 * @param[in] backend パイプラインと作業領域を所有する計算器。
 * @param[in] parameters 行列全体の寸法と指数調整の有無。
 * @param[in,out] measurement GPUの解析時間とCPUの待機時間を加算する測定値。
 * @param[out] error 失敗時の診断メッセージの格納先。NULLを指定できる。
 * @return 解析の成否。
 * @~english
 * @brief Run input exponent analysis on the GPU and store the required integer widths p_A, p_B and nonfinite
 * flag into the FP64_ANALYSIS buffer.
 * @param[in] backend Multiplier owning the pipelines and workspace buffers.
 * @param[in] parameters Full-matrix dimensions and whether exponent adjustment is enabled.
 * @param[in,out] measurement Measurement structure accumulating GPU analysis time and CPU wait time.
 * @param[out] error Destination for the failure diagnostic message. May be NULL.
 * @return Status of the analysis execution.
 * @~
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
 * @~japanese
 * @brief 入力行列からINT8の余り（またはStrassen法のブロックの和と差の余り）を生成するカーネルをコマンドバッファーへ記録する。
 * @param[in] backend パイプラインと作業領域を所有する計算器。
 * @param[in] command 記録先の未投入コマンドバッファー。
 * @param[in] parameters 行列の寸法、整数幅、およびStrassen法・指数調整の有無。
 * @param[in] columns 行列Bを変換する場合は1、行列Aの行のまとまりを変換する場合は0。
 * @param[out] error 失敗時の診断メッセージの格納先。NULLを指定できる。
 * @return 記録の成否。
 * @~english
 * @brief Encode the kernel that converts an input matrix into INT8 residues (or Strassen block residue sums and
 * differences).
 * @param[in] backend Multiplier owning the pipelines and workspace buffers.
 * @param[in] command Uncommitted command buffer to record into.
 * @param[in] parameters Matrix dimensions, integer widths, and Strassen/exponent-adjustment flags.
 * @param[in] columns One when converting matrix B, zero when converting a row batch of matrix A.
 * @param[out] error Destination for the failure diagnostic message. May be NULL.
 * @return Status of encoding the command.
 * @~
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
 * @~japanese
 * @brief 使用するすべての法（Strassen法ではすべての法と7つのブロック積）に対するINT8の行列積を一括で記録する。
 * @param[in] backend パイプラインと作業領域を所有する計算器。
 * @param[in] command 記録先の未投入コマンドバッファー。
 * @param[in] parameters 行のまとまりの寸法と整数幅。
 * @param[in] count 使用する法の個数。
 * @param[out] error 失敗時の診断メッセージの格納先。NULLを指定できる。
 * @return 記録の成否。
 * @~english
 * @brief Encode INT8 modular matrix multiplication across all active moduli (and all seven Strassen products
 * when active) in a single dispatch.
 * @param[in] backend Multiplier owning the pipelines and workspace buffers.
 * @param[in] command Uncommitted command buffer to record into.
 * @param[in] parameters Row-batch dimensions and integer widths.
 * @param[in] count Number of active moduli.
 * @param[out] error Destination for the failure diagnostic message. May be NULL.
 * @return Status of encoding the command.
 * @~
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
 * @~japanese
 * @brief CRTによる多倍長整数の復元とFP64への丸め（Strassen法では出力ブロックの組み立てを含む）をコマンドバッファーへ記録する。
 * @param[in] backend パイプラインと作業領域を所有する計算器。
 * @param[in] command 記録先の未投入コマンドバッファー。
 * @param[in] parameters 行のまとまりの寸法と整数幅。
 * @param[in] limbs CRTの復元に要する多倍長整数の桁数。
 * @param[out] output 呼び出し側へ返す出力配列とメモリーを共有するMetalバッファー。
 * @param[out] error 失敗時の診断メッセージの格納先。NULLを指定できる。
 * @return 記録の成否。
 * @~english
 * @brief Encode CRT reconstruction and FP64 rounding (preceded by Strassen output block assembly when active).
 * @param[in] backend Multiplier owning the pipelines and workspace buffers.
 * @param[in] command Uncommitted command buffer to record into.
 * @param[in] parameters Row-batch dimensions and integer widths.
 * @param[in] limbs Number of multi-precision integer digits required for CRT reconstruction.
 * @param[out] output Metal buffer sharing memory with the output array returned to the caller.
 * @param[out] error Destination for the failure diagnostic message. May be NULL.
 * @return Status of encoding the command.
 * @~
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
                // SCALES_A と SCALES_B は行と列の両方を収められる容量へ再確保するため、直前に加算した容量を一度差し引いて二重計上を防ぐ。
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
        // 失敗時に values を解放する前に、そのメモリー領域を直接参照する Metal バッファー（output）を内側の autoreleasepool で確実に破棄する。
        @autoreleasepool {
            id<MTLBuffer> output = [backend->device newBufferWithBytesNoCopy:values length:buffer_bytes
                                                                  options:MTLResourceStorageModeShared deallocator:nil];
            if (output == nil) {
                free(values);
                return fp64_fail(error, APPLE_FP64_METAL_ERROR, "Metalの出力バッファーを作成できません。");
            }
            size_t command_capacity = modular ? 1 + 3 * (((size_t)row_limit + row_step - 1) / row_step) : 1;
            // ARC環境でObjective-Cオブジェクトのポインター配列を動的に確保するため、callocでnilに初期化し、freeの前に各要素へnilを代入して参照を解放する。
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
