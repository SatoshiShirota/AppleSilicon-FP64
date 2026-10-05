#ifndef APPLE_FP64_MATMUL_H
#define APPLE_FP64_MATMUL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @~japanese
 * @brief 計算に必要な資源の作成と行列積の成否。
 * @~english
 * @brief Status of resource creation and matrix multiplication.
 * @~
 */
typedef enum apple_fp64_status_e {
    APPLE_FP64_SUCCESS, /**< @~japanese 操作が正常に完了した。
                         * @~english The operation completed successfully.
                         * @~
                         */
    APPLE_FP64_INVALID_ARGUMENT, /**< @~japanese 寸法、入力の長さ、設定またはパスの表現が条件を満たさない。
                                  * @~english The dimensions, input lengths, options, or path representation
                                  * do not satisfy the requirements.
                                  * @~
                                  */
    APPLE_FP64_OUT_OF_MEMORY, /**< @~japanese CPUのメモリーを確保できない。
                               * @~english CPU memory could not be allocated.
                               * @~
                               */
    APPLE_FP64_METAL_ERROR /**< @~japanese Metalの資源の作成または実行に失敗した。
                            * @~english Metal resource creation or execution failed.
                            * @~
                            */
} apple_fp64_status_t;

/**
 * @~japanese
 * @brief 作業領域に保持する行数。
 * @~english
 * @brief Number of rows held in the workspace.
 * @~
 */
typedef struct apple_fp64_options_s {
    uint32_t batch_rows; /**< @~japanese CRTの中間配列で一度に保持する行数。正の値で指定する。
                          * @~english Number of rows held at once in the intermediate CRT arrays. Must be
                          * positive.
                          * @~
                          */
} apple_fp64_options_t;

/**
 * @~japanese
 * @brief 一回の積について取得した処理時間と資源量。
 * @~english
 * @brief Execution times and resource usage measured for one matrix product.
 * @~
 */
typedef struct apple_fp64_measurement_s {
    double prepare_seconds; /**< @~japanese 入力の解析と変換に費やしたGPUの実行時間。
                             * @~english GPU execution time spent analyzing and converting the inputs.
                             * @~
                             */
    double product_seconds; /**< @~japanese 行列積に費やしたGPUの実行時間。
                             * @~english GPU execution time spent computing the matrix product.
                             * @~
                             */
    double reconstruct_seconds; /**< @~japanese 復元に費やしたGPUの実行時間。
                                 * @~english GPU execution time spent reconstructing the output.
                                 * @~
                                 */
    double wait_seconds; /**< @~japanese CPUがGPUの完了を待った実時間。
                          * @~english Wall-clock time the CPU spent waiting for GPU completion.
                          * @~
                          */
    double total_seconds; /**< @~japanese 出力と作業領域の確保、転送、同期を含む実時間。
                           * @~english Wall-clock time including output and workspace allocation, transfers,
                           * and synchronization.
                           * @~
                           */
    size_t workspace_bytes; /**< @~japanese 計算に使用したMetalバッファーの容量の合計。
                             * @~english Sum of the capacities of the Metal buffers used for the computation.
                             * @~
                             */
    uint32_t modulus_count; /**< @~japanese 使用した法の個数。FP64の積和演算と空の積では0。
                             * @~english Number of moduli used. Zero for FP64 fused multiply-add and empty
                             * products.
                             * @~
                             */
} apple_fp64_measurement_t;

/**
 * @~japanese
 * @brief 呼び出し側が所有する出力と、その計算についての測定値。
 * @~english
 * @brief Caller-owned output and measurements for its computation.
 * @~
 */
typedef struct apple_fp64_result_s {
    double *values; /**< @~japanese 行優先の連続した出力。apple_fp64_result_destroyで解放する。
                     * @~english Contiguous row-major output. Release with apple_fp64_result_destroy.
                     * @~
                     */
    size_t count; /**< @~japanese 出力の要素数。空の場合はvaluesがNULLとなる。
                   * @~english Number of output elements. values is NULL when the output is empty.
                   * @~
                   */
    apple_fp64_measurement_t measurement; /**< @~japanese 計算に費やした時間と資源量。
                                           * @~english Time and resource usage for the computation.
                                           * @~
                                           */
} apple_fp64_result_t;

/**
 * @~japanese
 * @brief 呼び出し側が所有する失敗理由。
 * @~english
 * @brief Caller-owned failure diagnostic.
 * @~
 */
typedef struct apple_fp64_error_s {
    char *message; /**< @~japanese UTF-8の診断メッセージ。成功時とCPUの確保失敗時はNULLとなる。
                    * @~english UTF-8 diagnostic message. NULL on success and on CPU allocation failure.
                    * @~
                    */
} apple_fp64_error_t;

/**
 * @~japanese
 * @brief Metalのパイプラインと作業領域を再利用する計算器。
 * @note 同じ計算器への呼び出しを並行して行ってはならない。
 * @~english
 * @brief Multiplier that reuses Metal pipelines and workspace.
 * @note Calls using the same multiplier must not run concurrently.
 * @~
 */
typedef struct apple_fp64_multiplier_s apple_fp64_multiplier_t;

/**
 * @~japanese
 * @brief 既定の行のまとまりを返す。
 * @return 計算の設定。
 * @~english
 * @brief Return the default row batch settings.
 * @return Computation options.
 * @~
 */
apple_fp64_options_t apple_fp64_default_options(void);

/**
 * @~japanese
 * @brief コンパイル済みのMetalライブラリーから計算器を生成する。
 * @param[in] library_path CMakeで生成したfp64.metallibのUTF-8のパス。
 * @param[out] multiplier 成功時の計算器。失敗時はNULLとなる。
 * @param[out] error 失敗理由の格納先。診断が不要な場合はNULLを指定できる。
 * @return 操作の成否。
 * @pre library_pathとmultiplierはNULLではないこと。
 * @pre errorを指定する場合、解放されていないメッセージを所有していないこと。
 * @note 診断メッセージを確保できない場合はAPPLE_FP64_OUT_OF_MEMORYを返す。
 * @~english
 * @brief Create a multiplier from a compiled Metal library.
 * @param[in] library_path UTF-8 path to fp64.metallib generated by CMake.
 * @param[out] multiplier Multiplier on success. NULL on failure.
 * @param[out] error Destination for the failure diagnostic. May be NULL if no diagnostic is needed.
 * @return Status of the operation.
 * @pre library_path and multiplier must not be NULL.
 * @pre If error is supplied, it must not own an unreleased message.
 * @note Return APPLE_FP64_OUT_OF_MEMORY if the diagnostic message cannot be allocated.
 * @~
 */
apple_fp64_status_t apple_fp64_multiplier_create(const char *library_path,
                                                apple_fp64_multiplier_t **multiplier,
                                                apple_fp64_error_t *error);

/**
 * @~japanese
 * @brief 計算器と、計算器が所有するMetalの資源を解放する。
 * @param[in] multiplier 生成済みの計算器。NULLの場合は何もしない。
 * @pre NULL以外の場合、まだ破棄されていない計算器であること。
 * @~english
 * @brief Release the multiplier and its Metal resources.
 * @param[in] multiplier Created multiplier. NULL causes no action.
 * @pre A non-NULL multiplier must not have been destroyed.
 * @~
 */
void apple_fp64_multiplier_destroy(apple_fp64_multiplier_t *multiplier);

/**
 * @~japanese
 * @brief 使用しているMetalデバイスの名前を返す。
 * @param[in] multiplier 生成済みの計算器。
 * @return 計算器が所有するUTF-8のデバイス名。計算器の破棄まで有効。
 * @pre multiplierはNULLではなく、まだ破棄されていないこと。
 * @~english
 * @brief Return the name of the Metal device in use.
 * @param[in] multiplier Created multiplier.
 * @return UTF-8 device name owned by the multiplier. Valid until the multiplier is destroyed.
 * @pre multiplier must not be NULL and must not have been destroyed.
 * @~
 */
const char *apple_fp64_device_name(const apple_fp64_multiplier_t *multiplier);

/**
 * @~japanese
 * @brief 二つのFP64行列の積を求める。
 * @param[in,out] multiplier 生成済みの計算器。
 * @param[in] a M行K列の連続した入力。呼び出しが完了するまで変更してはならない。
 * @param[in] a_count Aの要素数。
 * @param[in] b K行N列の連続した入力。呼び出しが完了するまで変更してはならない。
 * @param[in] b_count Bの要素数。
 * @param[in] m Aと出力の行数。
 * @param[in] n Bと出力の列数。
 * @param[in] k 内積の項数。
 * @param[in] options 行のまとまりの大きさ。
 * @param[out] result 出力と測定値。失敗時は全フィールドがゼロとなる。
 * @param[out] error 失敗理由の格納先。診断が不要な場合はNULLを指定できる。
 * @return 操作の成否。
 * @pre multiplierとresultはNULLではないこと。multiplierはまだ破棄されていないこと。
 * @pre aとbはFP64のビット列を保持すること。要素数がゼロの場合だけNULLを指定できる。
 * @pre resultとerrorは解放されていない出力やメッセージを所有していないこと。
 * @post 関数が戻る時点で、投入したGPUの処理は完了している。
 * @note 計算の数値的な意味はREADME.md「計算の定義」で定める。
 * @~english
 * @brief Compute the product of two FP64 matrices.
 * @param[in,out] multiplier Created multiplier.
 * @param[in] a Contiguous M-by-K input. Must not be modified until the call completes.
 * @param[in] a_count Number of elements in A.
 * @param[in] b Contiguous K-by-N input. Must not be modified until the call completes.
 * @param[in] b_count Number of elements in B.
 * @param[in] m Number of rows in A and the output.
 * @param[in] n Number of columns in B and the output.
 * @param[in] k Number of terms in each dot product.
 * @param[in] options Row batch size.
 * @param[out] result Output and measurements. All fields are zero on failure.
 * @param[out] error Destination for the failure diagnostic. May be NULL if no diagnostic is needed.
 * @return Status of the operation.
 * @pre multiplier and result must not be NULL. multiplier must not have been destroyed.
 * @pre a and b must hold FP64 bit patterns. NULL is allowed only when the corresponding element count is
 * zero.
 * @pre result and error must not own any unreleased output or message.
 * @post All submitted GPU work has completed when the function returns.
 * @note Numerical semantics are defined in README.en.md, section "Numerical semantics".
 * @~
 */
apple_fp64_status_t apple_fp64_multiply(apple_fp64_multiplier_t *multiplier,
                                      const double *a, size_t a_count,
                                      const double *b, size_t b_count,
                                      uint32_t m, uint32_t n, uint32_t k,
                                      apple_fp64_options_t options,
                                      apple_fp64_result_t *result,
                                      apple_fp64_error_t *error);

/**
 * @~japanese
 * @brief 出力の配列を解放し、結果の全フィールドをゼロにする。
 * @param[in,out] result 行列積の結果、または全フィールドがゼロの結果。
 * @pre resultはNULLではないこと。
 * @~english
 * @brief Release the output array and zero all result fields.
 * @param[in,out] result Matrix multiplication result, or a result whose fields are all zero.
 * @pre result must not be NULL.
 * @~
 */
void apple_fp64_result_destroy(apple_fp64_result_t *result);

/**
 * @~japanese
 * @brief 診断メッセージを解放し、messageをNULLにする。
 * @param[in,out] error 操作の診断、またはmessageがNULLの診断。
 * @pre errorはNULLではないこと。
 * @~english
 * @brief Release the diagnostic message and set message to NULL.
 * @param[in,out] error Operation diagnostic, or a diagnostic whose message is NULL.
 * @pre error must not be NULL.
 * @~
 */
void apple_fp64_error_destroy(apple_fp64_error_t *error);

#ifdef __cplusplus
}
#endif

#endif
