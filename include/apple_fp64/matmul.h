#ifndef APPLE_FP64_MATMUL_H
#define APPLE_FP64_MATMUL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 計算に必要な資源の作成と行列積の成否。 */
typedef enum apple_fp64_status_e {
    APPLE_FP64_SUCCESS, /**< 操作が正常に完了した。 */
    APPLE_FP64_INVALID_ARGUMENT, /**< 寸法、入力の長さ、設定またはパスの表現が条件を満たさない。 */
    APPLE_FP64_OUT_OF_MEMORY, /**< CPUのメモリーを確保できない。 */
    APPLE_FP64_METAL_ERROR /**< Metalの資源の作成または実行に失敗した。 */
} apple_fp64_status_t;

/** @brief 計算の整数幅と、作業領域に保持する行数。 */
typedef struct apple_fp64_options_s {
    uint32_t precision_a; /**< Aの整数幅。正の値で指定する。 */
    uint32_t precision_b; /**< Bの整数幅。正の値で指定する。 */
    uint32_t batch_rows; /**< 一度に処理する行数。正の値で指定する。 */
} apple_fp64_options_t;

/** @brief 一回の積について取得した処理時間と資源量。 */
typedef struct apple_fp64_measurement_s {
    double prepare_seconds; /**< 入力変換に費やしたGPUの実行時間。 */
    double product_seconds; /**< 余りの行列積に費やしたGPUの実行時間。 */
    double reconstruct_seconds; /**< 復元に費やしたGPUの実行時間。 */
    double wait_seconds; /**< CPUがGPUの完了を待った実時間。 */
    double total_seconds; /**< 出力と作業領域の確保、転送、同期を含む実時間。 */
    size_t workspace_bytes; /**< 計算に使用したMetalバッファーの容量の合計。 */
    uint32_t modulus_count; /**< 使用した法の個数。 */
} apple_fp64_measurement_t;

/** @brief 呼び出し側が所有する出力と、その計算についての測定値。 */
typedef struct apple_fp64_result_s {
    double *values; /**< 行優先の連続した出力。apple_fp64_result_destroyで解放する。 */
    size_t count; /**< 出力の要素数。空の場合はvaluesがNULLとなる。 */
    apple_fp64_measurement_t measurement; /**< 計算に費やした時間と資源量。 */
} apple_fp64_result_t;

/** @brief 呼び出し側が所有する失敗理由。 */
typedef struct apple_fp64_error_s {
    char *message; /**< UTF-8の診断メッセージ。成功時とCPUの確保失敗時はNULLとなる。 */
} apple_fp64_error_t;

/**
 * @brief Metalのパイプラインと作業領域を再利用する計算器。
 * @note 同じ計算器への呼び出しを並行して行ってはならない。
 */
typedef struct apple_fp64_multiplier_s apple_fp64_multiplier_t;

/** @brief 既定の整数幅と行のまとまりを返す。 @return 計算の設定。 */
apple_fp64_options_t apple_fp64_default_options(void);

/**
 * @brief コンパイル済みのMetalライブラリーから計算器を生成する。
 * @param[in] library_path CMakeで生成したfp64.metallibのUTF-8のパス。
 * @param[out] multiplier 成功時の計算器。失敗時はNULLとなる。
 * @param[out] error 失敗理由の格納先。診断が不要な場合はNULLを指定できる。
 * @return 操作の成否。
 * @pre library_pathとmultiplierはNULLではないこと。
 * @pre errorを指定する場合、解放されていないメッセージを所有していないこと。
 * @note 診断メッセージを確保できない場合はAPPLE_FP64_OUT_OF_MEMORYを返す。
 */
apple_fp64_status_t apple_fp64_multiplier_create(const char *library_path,
                                                apple_fp64_multiplier_t **multiplier,
                                                apple_fp64_error_t *error);

/**
 * @brief 計算器と、計算器が所有するMetalの資源を解放する。
 * @param[in] multiplier 生成済みの計算器。NULLの場合は何もしない。
 * @pre NULL以外の場合、まだ破棄されていない計算器であること。
 */
void apple_fp64_multiplier_destroy(apple_fp64_multiplier_t *multiplier);

/**
 * @brief 使用しているMetalデバイスの名前を返す。
 * @param[in] multiplier 生成済みの計算器。
 * @return 計算器が所有するUTF-8のデバイス名。計算器の破棄まで有効。
 * @pre multiplierはNULLではなく、まだ破棄されていないこと。
 */
const char *apple_fp64_device_name(const apple_fp64_multiplier_t *multiplier);

/**
 * @brief 二つのFP64行列の積を求める。
 * @param[in,out] multiplier 生成済みの計算器。
 * @param[in] a M行K列の連続した入力。呼び出しが完了するまで変更してはならない。
 * @param[in] a_count Aの要素数。
 * @param[in] b K行N列の連続した入力。呼び出しが完了するまで変更してはならない。
 * @param[in] b_count Bの要素数。
 * @param[in] m Aと出力の行数。
 * @param[in] n Bと出力の列数。
 * @param[in] k 内積の項数。
 * @param[in] options 整数幅と行のまとまりの大きさ。
 * @param[out] result 出力と測定値。失敗時は全フィールドがゼロとなる。
 * @param[out] error 失敗理由の格納先。診断が不要な場合はNULLを指定できる。
 * @return 操作の成否。
 * @pre multiplierとresultはNULLではないこと。multiplierはまだ破棄されていないこと。
 * @pre aとbは有限のFP64値だけを含むこと。要素数がゼロの場合だけNULLを指定できる。
 * @pre resultとerrorは解放されていない出力やメッセージを所有していないこと。
 * @post 関数が戻る時点で、投入したGPUの処理は完了している。
 * @note 計算の数値的な意味はREADME.md「計算の定義」で定める。
 */
apple_fp64_status_t apple_fp64_multiply(apple_fp64_multiplier_t *multiplier,
                                      const double *a, size_t a_count,
                                      const double *b, size_t b_count,
                                      uint32_t m, uint32_t n, uint32_t k,
                                      apple_fp64_options_t options,
                                      apple_fp64_result_t *result,
                                      apple_fp64_error_t *error);

/**
 * @brief 出力の配列を解放し、結果の全フィールドをゼロにする。
 * @param[in,out] result 行列積の結果、または全フィールドがゼロの結果。
 * @pre resultはNULLではないこと。
 */
void apple_fp64_result_destroy(apple_fp64_result_t *result);

/**
 * @brief 診断メッセージを解放し、messageをNULLにする。
 * @param[in,out] error 操作の診断、またはmessageがNULLの診断。
 * @pre errorはNULLではないこと。
 */
void apple_fp64_error_destroy(apple_fp64_error_t *error);

#ifdef __cplusplus
}
#endif

#endif
