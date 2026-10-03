#include "apple_fp64/matmul.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief 一定値の行列について、同じ計算器による積の全ビットを検証する。
 * @param[in,out] multiplier 積を実行する計算器。
 * @param[in] m 出力の行数。
 * @param[in] n 出力の列数。
 * @param[in] k 内積の項数。
 * @param[in] a_value Aの全要素の値。
 * @param[in] b_value Bの全要素の値。
 * @param[in] options 整数幅と行のまとまりの大きさ。
 * @return 出力の長さと値が一致した場合はtrue。
 * @pre 入力の整数化と期待値の算出に丸めが発生しない値を指定すること。
 */
static bool fp64_check_product(apple_fp64_multiplier_t *multiplier, uint32_t m, uint32_t n,
                               uint32_t k, double a_value, double b_value, apple_fp64_options_t options)
{
    size_t a_count = (size_t)m * k, b_count = (size_t)k * n;
    double *a = a_count != 0 ? malloc(a_count * sizeof(double)) : NULL;
    double *b = b_count != 0 ? malloc(b_count * sizeof(double)) : NULL;
    if ((a_count != 0 && a == NULL) || (b_count != 0 && b == NULL)) {
        free(a);
        free(b);
        return false;
    }
    for (size_t index = 0; index < a_count; ++index) a[index] = a_value;
    for (size_t index = 0; index < b_count; ++index) b[index] = b_value;
    apple_fp64_result_t result = {0};
    apple_fp64_error_t error = {0};
    apple_fp64_status_t status = apple_fp64_multiply(multiplier, a, a_count, b, b_count,
                                                    m, n, k, options, &result, &error);
    bool matches = status == APPLE_FP64_SUCCESS && result.count == (size_t)m * n;
    double expected = k == 0 ? 0.0 : (double)k * a_value * b_value;
    for (size_t index = 0; matches && index < result.count; ++index)
        matches = memcmp(&result.values[index], &expected, sizeof(double)) == 0;
    if (!matches) fprintf(stderr, "寸法と入力を変えた積が一致しません: %s\n",
                          error.message != NULL ? error.message : "出力の長さまたは値が異なります。");
    apple_fp64_result_destroy(&result);
    apple_fp64_error_destroy(&error);
    free(a);
    free(b);
    return matches;
}

/**
 * @brief 入力の長さが不正な場合、所有する出力を返さずに診断を返すことを検証する。
 * @param[in,out] multiplier 積を実行する計算器。
 * @return 定義された失敗を返した場合はtrue。
 */
static bool fp64_check_rejected_length(apple_fp64_multiplier_t *multiplier)
{
    double b = 1;
    apple_fp64_result_t result = {0};
    apple_fp64_error_t error = {0};
    apple_fp64_status_t status = apple_fp64_multiply(multiplier, NULL, 0, &b, 1,
                                                    1, 1, 1, apple_fp64_default_options(), &result, &error);
    apple_fp64_measurement_t measurement = result.measurement;
    bool rejected = status == APPLE_FP64_INVALID_ARGUMENT && result.values == NULL && result.count == 0
                 && measurement.prepare_seconds == 0 && measurement.product_seconds == 0
                 && measurement.reconstruct_seconds == 0 && measurement.wait_seconds == 0
                 && measurement.total_seconds == 0 && measurement.workspace_bytes == 0
                 && measurement.modulus_count == 0 && error.message != NULL && error.message[0] != '\0';
    apple_fp64_result_destroy(&result);
    apple_fp64_error_destroy(&error);
    return rejected && error.message == NULL;
}

/**
 * @brief 読めないMetalライブラリーについて、計算器を返さずに診断を返すことを検証する。
 * @param[in] library_path コンパイル済みライブラリーのパス。
 * @return 定義された失敗を返した場合はtrue。
 */
static bool fp64_check_unreadable_library(const char *library_path)
{
    size_t length = strlen(library_path);
    char *missing = malloc(length + sizeof("/fp64.metallib"));
    if (missing == NULL) return false;
    memcpy(missing, library_path, length);
    memcpy(missing + length, "/fp64.metallib", sizeof("/fp64.metallib"));
    apple_fp64_multiplier_t *multiplier = NULL;
    apple_fp64_error_t error = {0};
    apple_fp64_status_t status = apple_fp64_multiplier_create(missing, &multiplier, &error);
    bool rejected = status == APPLE_FP64_METAL_ERROR && multiplier == NULL
                 && error.message != NULL && error.message[0] != '\0';
    apple_fp64_multiplier_destroy(multiplier);
    apple_fp64_error_destroy(&error);
    free(missing);
    return rejected;
}

/**
 * @brief 寸法、入力と整数幅を変え、同じ計算器を繰り返し使う。
 * @param[in] argc 引数の個数。
 * @param[in] argv Metalライブラリーのパスを含む引数。
 * @return すべての積と失敗動作が一致した場合は0。
 * @pre argcは2であること。
 */
int main(int argc, char **argv)
{
    assert(argc == 2);
    (void)argc;
    apple_fp64_multiplier_t *multiplier = NULL;
    apple_fp64_error_t error = {0};
    apple_fp64_status_t status = apple_fp64_multiplier_create(argv[1], &multiplier, &error);
    if (status != APPLE_FP64_SUCCESS) {
        fprintf(stderr, "%s\n", error.message != NULL ? error.message : "CPUのメモリーを確保できません。");
        apple_fp64_error_destroy(&error);
        return 1;
    }
    const char *device_name = apple_fp64_device_name(multiplier);
    bool matches = device_name[0] != '\0'
                && fp64_check_product(multiplier, 2, 3, 2, 1.5, -2, apple_fp64_default_options())
                && fp64_check_product(multiplier, 65, 37, 33, 0.5, 4, (apple_fp64_options_t){60, 60, 17})
                && fp64_check_product(multiplier, 1, 2, 1, -3, 0.25, (apple_fp64_options_t){3, 4, 1})
                && fp64_check_product(multiplier, 4, 3, 0, 0, 0, apple_fp64_default_options())
                && fp64_check_rejected_length(multiplier)
                && fp64_check_product(multiplier, 33, 65, 7, 2, -0.5, (apple_fp64_options_t){80, 54, 11})
                && fp64_check_product(multiplier, 33, 65, 7, 0, 4, (apple_fp64_options_t){80, 54, 11})
                && fp64_check_unreadable_library(argv[1])
                && strcmp(device_name, apple_fp64_device_name(multiplier)) == 0;
    apple_fp64_multiplier_destroy(multiplier);
    if (!matches) fprintf(stderr, "公開APIの結果が一致しません。\n");
    return matches ? 0 : 1;
}
