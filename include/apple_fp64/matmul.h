#ifndef APPLE_FP64_MATMUL_H
#define APPLE_FP64_MATMUL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @~japanese
 * @brief 計算器の作成または行列積の計算における成否を表す状態コード。
 * @~english
 * @brief Status code indicating whether multiplier creation or matrix multiplication succeeded.
 * @~
 */
typedef enum apple_fp64_status_e {
    APPLE_FP64_SUCCESS, /**< @~japanese 操作が正常に完了した。
                         * @~english The operation completed successfully.
                         * @~
                         */
    APPLE_FP64_INVALID_ARGUMENT, /**< @~japanese 行列の寸法、入力配列の長さ、計算の設定、またはファイルパスの文字列が前提条件を満たさない。
                                  * @~english The matrix dimensions, input array lengths, computation options,
                                  * or file path string do not satisfy the preconditions.
                                  * @~
                                  */
    APPLE_FP64_OUT_OF_MEMORY, /**< @~japanese CPU側のメモリーを確保できなかった。
                               * @~english CPU memory could not be allocated.
                               * @~
                               */
    APPLE_FP64_METAL_ERROR /**< @~japanese Metalの資源の作成、またはGPUでの計算の実行に失敗した。
                            * @~english Failed to create Metal resources or to execute the computation on the GPU.
                            * @~
                            */
} apple_fp64_status_t;

/**
 * @~japanese
 * @brief 行列積の計算における作業領域の設定。
 * @~english
 * @brief Workspace configuration for matrix multiplication.
 * @~
 */
typedef struct apple_fp64_options_s {
    uint32_t batch_rows; /**< @~japanese 整数剰余による行列積の中間バッファーで一度に処理する行数。1以上の値を指定する。
                          * @~english Number of rows processed at once in the intermediate buffers for modular
                          * matrix multiplication. Must be at least 1.
                          * @~
                          */
} apple_fp64_options_t;

/**
 * @~japanese
 * @brief 1回の行列積の計算で測定した処理時間とメモリー使用量。
 * @~english
 * @brief Execution times and memory usage measured for a single matrix multiplication.
 * @~
 */
typedef struct apple_fp64_measurement_s {
    double prepare_seconds; /**< @~japanese 入力の解析と余りへの変換に要したGPUの実行時間（秒）。
                             * @~english GPU execution time (in seconds) spent analyzing the inputs and
                             * converting them to residues.
                             * @~
                             */
    double product_seconds; /**< @~japanese 行列積のカーネルに要したGPUの実行時間（秒）。
                             * @~english GPU execution time (in seconds) spent in the matrix multiplication
                             * kernels.
                             * @~
                             */
    double reconstruct_seconds; /**< @~japanese CRTによる復元とFP64への丸めに要したGPUの実行時間（秒）。
                                 * @~english GPU execution time (in seconds) spent reconstructing values with
                                 * CRT and rounding them to FP64.
                                 * @~
                                 */
    double wait_seconds; /**< @~japanese CPUがGPUの完了を待機した実時間（秒）。
                          * @~english Wall-clock time (in seconds) that the CPU spent waiting for GPU
                          * completion.
                          * @~
                          */
    double total_seconds; /**< @~japanese メモリーの確保、データの転送、GPUの待機を含む呼び出し全体の実時間（秒）。
                           * @~english Total wall-clock time (in seconds) for the call, including memory
                           * allocation, data transfer, and waiting for the GPU.
                           * @~
                           */
    size_t workspace_bytes; /**< @~japanese 計算に使用したすべてのMetalバッファーの容量の合計（バイト）。
                             * @~english Total capacity (in bytes) of all Metal buffers used for the
                             * computation.
                             * @~
                             */
    uint32_t modulus_count; /**< @~japanese 整数剰余による行列積で使用した法の個数。FP64の積和演算を使った場合と空の行列積では0となる。
                             * @~english Number of moduli used in modular matrix multiplication. Zero when FP64
                             * fused multiply-add is used or when the matrix product is empty.
                             * @~
                             */
} apple_fp64_measurement_t;

/**
 * @~japanese
 * @brief 行列積の出力配列と、その計算で記録された測定値。
 * @~english
 * @brief Output array of a matrix multiplication and the measurements recorded during its computation.
 * @~
 */
typedef struct apple_fp64_result_s {
    double *values; /**< @~japanese 行優先で連続して並ぶ出力の配列（M * N要素）。使用後はapple_fp64_result_destroyで解放する。
                     * @~english Contiguous row-major output array of M * N elements. Release with
                     * apple_fp64_result_destroy after use.
                     * @~
                     */
    size_t count; /**< @~japanese 出力配列の要素数（M * N）。要素数が0の場合はvaluesがNULLとなる。
                   * @~english Number of elements in the output array (M * N). When count is 0, values is NULL.
                   * @~
                   */
    apple_fp64_measurement_t measurement; /**< @~japanese この行列積の計算で記録された処理時間とメモリー使用量。
                                           * @~english Execution times and memory usage recorded for this matrix
                                           * multiplication.
                                           * @~
                                           */
} apple_fp64_result_t;

/**
 * @~japanese
 * @brief 失敗した操作の診断メッセージを保持する構造体。
 * @~english
 * @brief Structure holding the diagnostic message for a failed operation.
 * @~
 */
typedef struct apple_fp64_error_s {
    char *message; /**< @~japanese UTF-8の診断メッセージ。使用後はapple_fp64_error_destroyで解放する。成功時およびCPUのメモリー確保に失敗したときはNULLとなる。
                    * @~english UTF-8 diagnostic message. Release with apple_fp64_error_destroy after use. NULL
                    * on success and when CPU memory allocation fails.
                    * @~
                    */
} apple_fp64_error_t;

/**
 * @~japanese
 * @brief コンパイル済みのMetalパイプラインと作業領域のバッファーを保持し、複数の行列積で再利用する計算器。
 * @note 同じ計算器に対して複数のスレッドから同時に呼び出してはならない。
 * @~english
 * @brief Multiplier that retains compiled Metal pipelines and workspace buffers for reuse across multiple matrix multiplications.
 * @note Do not invoke functions on the same multiplier concurrently from multiple threads.
 * @~
 */
typedef struct apple_fp64_multiplier_s apple_fp64_multiplier_t;

/**
 * @~japanese
 * @brief 既定の作業領域の設定（一度に処理する行数256）を返す。
 * @return 既定値を設定したapple_fp64_options_t。
 * @~english
 * @brief Return the default workspace options (256 rows per batch).
 * @return apple_fp64_options_t initialized with default values.
 * @~
 */
apple_fp64_options_t apple_fp64_default_options(void);

/**
 * @~japanese
 * @brief コンパイル済みのMetalライブラリー（fp64.metallib）を読み込み、計算器を作成する。
 * @param[in] library_path CMakeでビルドまたはインストールしたfp64.metallibへのUTF-8のファイルパス。
 * @param[out] multiplier 作成した計算器の格納先。失敗したときはNULLが書き込まれる。
 * @param[out] error 失敗時の診断メッセージの格納先。診断メッセージが不要な場合はNULLを指定できる。
 * @return 作成に成功した場合はAPPLE_FP64_SUCCESS、失敗した場合は失敗の種類を表す状態コード。
 * @pre library_pathとmultiplierはNULLではないこと。
 * @pre errorにNULL以外を渡す場合、error->messageは未解放のメモリーを指していないこと。
 * @note 失敗時の診断メッセージ用のメモリーを確保できない場合は、error->messageをNULLにしたままAPPLE_FP64_OUT_OF_MEMORYを返す。
 * @~english
 * @brief Load the compiled Metal library (fp64.metallib) and create a multiplier.
 * @param[in] library_path UTF-8 file path to fp64.metallib built or installed by CMake.
 * @param[out] multiplier Destination for the created multiplier. Set to NULL on failure.
 * @param[out] error Destination for the failure diagnostic message. May be NULL if no diagnostic is needed.
 * @return APPLE_FP64_SUCCESS on success, or a status code indicating the failure kind.
 * @pre library_path and multiplier must not be NULL.
 * @pre If error is non-NULL, error->message must not point to unreleased memory.
 * @note If memory for the diagnostic message cannot be allocated on failure, error->message remains NULL and APPLE_FP64_OUT_OF_MEMORY is returned.
 * @~
 */
apple_fp64_status_t apple_fp64_multiplier_create(const char *library_path,
                                                apple_fp64_multiplier_t **multiplier,
                                                apple_fp64_error_t *error);

/**
 * @~japanese
 * @brief 計算器を破棄し、保持しているMetalパイプラインと作業領域のバッファーをすべて解放する。
 * @param[in] multiplier 破棄する計算器。NULLを渡した場合は何もしない。
 * @pre multiplierがNULLでない場合、まだ破棄されていない計算器であること。
 * @~english
 * @brief Destroy the multiplier and release all retained Metal pipelines and workspace buffers.
 * @param[in] multiplier Multiplier to destroy. Passing NULL causes no action.
 * @pre If multiplier is non-NULL, it must not have been destroyed yet.
 * @~
 */
void apple_fp64_multiplier_destroy(apple_fp64_multiplier_t *multiplier);

/**
 * @~japanese
 * @brief 計算器が使用しているMetalデバイス（GPU）の名前を返す。
 * @param[in] multiplier 作成済みの計算器。
 * @return 計算器が所有するUTF-8のデバイス名文字列。返されたポインターは計算器を破棄するまで有効である。
 * @pre multiplierはNULLではなく、まだ破棄されていないこと。
 * @~english
 * @brief Return the name of the Metal device (GPU) used by the multiplier.
 * @param[in] multiplier Created multiplier.
 * @return UTF-8 device name string owned by the multiplier. The returned pointer remains valid until the multiplier is destroyed.
 * @pre multiplier must not be NULL and must not have been destroyed.
 * @~
 */
const char *apple_fp64_device_name(const apple_fp64_multiplier_t *multiplier);

/**
 * @~japanese
 * @brief 2つのFP64行列A（M行K列）とB（K行N列）の積C = ABをGPUで計算する。
 * @param[in,out] multiplier 作成済みの計算器。
 * @param[in] a 行優先で連続して並ぶM行K列の入力配列。関数の呼び出しが完了するまで書き換えてはならない。
 * @param[in] a_count 配列aの要素数（M * Kと一致する必要がある）。
 * @param[in] b 行優先で連続して並ぶK行N列の入力配列。関数の呼び出しが完了するまで書き換えてはならない。
 * @param[in] b_count 配列bの要素数（K * Nと一致する必要がある）。
 * @param[in] m 行列Aおよび出力行列Cの行数。
 * @param[in] n 行列Bおよび出力行列Cの列数。
 * @param[in] k 行列Aの列数および行列Bの行数（内積の項数）。
 * @param[in] options 作業領域で一度に処理する行数の設定。
 * @param[out] result 出力配列と測定値の格納先。失敗したときはすべてのフィールドがゼロになる。
 * @param[out] error 失敗時の診断メッセージの格納先。診断メッセージが不要な場合はNULLを指定できる。
 * @return 計算に成功した場合はAPPLE_FP64_SUCCESS、失敗した場合は失敗の種類を表す状態コード。
 * @pre multiplierとresultはNULLではないこと。また、multiplierはまだ破棄されていないこと。
 * @pre aとbはそれぞれa_count個、b_count個のFP64値を保持すること。要素数が0の場合に限りNULLを渡せる。
 * @pre result->valuesおよびerror->message（errorがNULLでない場合）は、未解放のメモリーを指していないこと。
 * @post 関数が戻る時点で、投入したすべてのGPUの処理が完了している。
 * @note 計算の数値的な意味はREADME.md「計算の定義」で定める。
 * @~english
 * @brief Compute the product C = AB of two FP64 matrices A (M by K) and B (K by N) on the GPU.
 * @param[in,out] multiplier Created multiplier.
 * @param[in] a Contiguous row-major M-by-K input array. Must not be modified until the call returns.
 * @param[in] a_count Number of elements in array a (must equal M * K).
 * @param[in] b Contiguous row-major K-by-N input array. Must not be modified until the call returns.
 * @param[in] b_count Number of elements in array b (must equal K * N).
 * @param[in] m Number of rows in matrix A and output matrix C.
 * @param[in] n Number of columns in matrix B and output matrix C.
 * @param[in] k Number of columns in matrix A and rows in matrix B (number of terms in each dot product).
 * @param[in] options Configuration specifying the number of rows processed per batch in the workspace.
 * @param[out] result Destination for the output array and measurements. All fields are zeroed on failure.
 * @param[out] error Destination for the failure diagnostic message. May be NULL if no diagnostic is needed.
 * @return APPLE_FP64_SUCCESS on success, or a status code indicating the failure kind.
 * @pre multiplier and result must not be NULL, and multiplier must not have been destroyed.
 * @pre a and b must hold a_count and b_count FP64 values respectively. NULL may be passed only when the corresponding element count is 0.
 * @pre result->values and error->message (when error is non-NULL) must not point to unreleased memory.
 * @post All submitted GPU work has completed by the time the function returns.
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
 * @brief 行列積の出力配列を解放し、resultのすべてのフィールドをゼロで初期化する。
 * @param[in,out] result 解放する結果の構造体。すでにゼロで初期化されている構造体を渡してもよい。
 * @pre resultはNULLではないこと。
 * @~english
 * @brief Release the matrix multiplication output array and zero all fields of result.
 * @param[in,out] result Result structure to release. May also be a structure whose fields are already zero.
 * @pre result must not be NULL.
 * @~
 */
void apple_fp64_result_destroy(apple_fp64_result_t *result);

/**
 * @~japanese
 * @brief 診断メッセージのメモリーを解放し、error->messageをNULLにする。
 * @param[in,out] error 解放する診断の構造体。messageがすでにNULLである構造体を渡してもよい。
 * @pre errorはNULLではないこと。
 * @~english
 * @brief Release the diagnostic message memory and set error->message to NULL.
 * @param[in,out] error Diagnostic structure to release. May also be a structure whose message is already NULL.
 * @pre error must not be NULL.
 * @~
 */
void apple_fp64_error_destroy(apple_fp64_error_t *error);

#ifdef __cplusplus
}
#endif

#endif
