#include "apple_fp64/matmul.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @~japanese
 * @brief 一定値の行列について、同じ計算器による積の全ビットを検証する。
 * @param[in,out] multiplier 積を実行する計算器。
 * @param[in] m 出力の行数。
 * @param[in] n 出力の列数。
 * @param[in] k 内積の項数。
 * @param[in] a_value Aの全要素の値。
 * @param[in] b_value Bの全要素の値。
 * @param[in] options 行のまとまりの大きさ。
 * @return 出力の長さと値が一致した場合はtrue。
 * @pre 入力の整数化と期待値の算出に丸めが発生しない値を指定すること。
 * @~english
 * @brief Verify every output bit of constant-valued matrix products using the same multiplier.
 * @param[in,out] multiplier Multiplier computing the product.
 * @param[in] m Number of output rows.
 * @param[in] n Number of output columns.
 * @param[in] k Number of terms in each dot product.
 * @param[in] a_value Value of every element in A.
 * @param[in] b_value Value of every element in B.
 * @param[in] options Row batch size.
 * @return true if the output length and values match.
 * @pre Input integer conversion and expected-value calculation must not require rounding.
 * @~
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
 * @~japanese
 * @brief 行と列の係数を持つ大きな長方形の積を、厳密な整数の期待値と比較する。
 * @param[in,out] multiplier 積を実行する計算器。
 * @param[in] wide_exponents 内積方向に逆向きの指数を掛け、入力の範囲を広げる場合はtrue。
 * @return 全出力のビット列が一致した場合はtrue。
 * @note 行のまとまりに奇数を指定し、符号と指数が異なる行と列を含める。
 * @note 上下と左右のブロックで、整数化した値の下位ゼロビット数を変える。
 * @~english
 * @brief Compare a large rectangular product with row and column coefficients against exact integer
 * expectations.
 * @param[in,out] multiplier Multiplier computing the product.
 * @param[in] wide_exponents true when applying opposite exponents along the inner dimension to widen the
 * input range.
 * @return true if every output bit pattern matches.
 * @note Use an odd row batch size and include rows and columns with different signs and exponents.
 * @note Vary the number of trailing zero bits in integer-converted values between the top, bottom, left, and
 * right blocks.
 * @~
 */
static bool fp64_check_factored_product(apple_fp64_multiplier_t *multiplier, bool wide_exponents)
{
    const uint32_t m = 1026, n = 1152, k = 2048;
    size_t a_count = (size_t)m * k, b_count = (size_t)k * n;
    double *a = malloc(a_count * sizeof(double)), *b = malloc(b_count * sizeof(double));
    if (a == NULL || b == NULL) {
        free(a);
        free(b);
        return false;
    }
    int64_t inner_sum = 0;
    for (uint32_t inner = 0; inner < k; ++inner) {
        int first = (int)(inner % 11) - 5, second = (int)(inner % 13) - 6;
        int exponent = wide_exponents ? ((int)(inner % 5) - 2) * 400 : 0;
        inner_sum += first * second;
        for (uint32_t row = 0; row < m; ++row) {
            int factor = (int)(row % 7) - 3;
            if (row >= m / 2 && row < m / 2 + 128) factor += 8;
            a[(size_t)row * k + inner] = ldexp(factor * first, exponent);
        }
        for (uint32_t column = 0; column < n; ++column) {
            int factor = (int)(column % 17) - 8;
            if (column >= n / 2 && column < n / 2 + 64) factor += 16;
            b[(size_t)inner * n + column] = ldexp(second * factor, -exponent);
        }
    }
    apple_fp64_result_t result = {0};
    apple_fp64_error_t error = {0};
    apple_fp64_options_t options = {255};
    apple_fp64_status_t status = apple_fp64_multiply(multiplier, a, a_count, b, b_count,
                                                    m, n, k, options, &result, &error);
    bool matches = status == APPLE_FP64_SUCCESS && result.count == (size_t)m * n;
    for (uint32_t row = 0; matches && row < m; ++row) {
        int row_factor = (int)(row % 7) - 3;
        if (row >= m / 2 && row < m / 2 + 128) row_factor += 8;
        for (uint32_t column = 0; matches && column < n; ++column) {
            int column_factor = (int)(column % 17) - 8;
            if (column >= n / 2 && column < n / 2 + 64) column_factor += 16;
            double expected = (double)(row_factor * column_factor * inner_sum);
            matches = memcmp(&result.values[(size_t)row * n + column], &expected, sizeof(expected)) == 0;
        }
    }
    if (!matches) fprintf(stderr, "大きな長方形の積が整数の期待値と一致しません: %s\n",
                          error.message != NULL ? error.message : "出力の長さまたは値が異なります。");
    apple_fp64_result_destroy(&result);
    apple_fp64_error_destroy(&error);
    free(a);
    free(b);
    return matches;
}

/**
 * @~japanese
 * @brief 同じ寸法で入力の指数範囲と特殊値を変え、計算器を再利用する。
 * @param[in,out] multiplier 積を実行する計算器。
 * @return 各入力の数値結果が一致した場合はtrue。
 * @~english
 * @brief Reuse the multiplier with the same dimensions while varying input exponent ranges and special
 * values.
 * @param[in,out] multiplier Multiplier computing the product.
 * @return true if the numerical results match for each input.
 * @~
 */
static bool fp64_check_input_ranges(apple_fp64_multiplier_t *multiplier)
{
    const double a[][2] = {{1, 0}, {0x1p500, 0x1p-500}, {1 + 0x1p-27, 0x1p-200},
                           {DBL_MAX, 0x1p-1074}, {INFINITY, 1}, {1, -1}, {1, 0x1p-80}};
    const double b[][2] = {{1, 1}, {0x1p-500, 0x1p500}, {1 - 0x1p-27, -0x1p-200},
                           {0, 1}, {0, 1}, {1, 1}, {0, 0x1p80}};
    const uint64_t expected[] = {UINT64_C(0x3ff0000000000000), UINT64_C(0x4000000000000000),
                                 UINT64_C(0x3ff0000000000000), UINT64_C(1),
                                 UINT64_C(0x7ff8000000000000), UINT64_C(0), UINT64_C(0x3ff0000000000000)};
    for (size_t index = 0; index < sizeof(expected) / sizeof(expected[0]); ++index) {
        apple_fp64_result_t result = {0};
        apple_fp64_error_t error = {0};
        apple_fp64_status_t status = apple_fp64_multiply(multiplier, a[index], 2, b[index], 2,
                                                        1, 1, 2, apple_fp64_default_options(), &result, &error);
        bool matches = status == APPLE_FP64_SUCCESS && result.count == 1
                    && memcmp(result.values, &expected[index], sizeof(double)) == 0;
        apple_fp64_result_destroy(&result);
        apple_fp64_error_destroy(&error);
        if (!matches) return false;
    }
    return true;
}

/**
 * @~japanese
 * @brief 入力の長さが不正な場合、所有する出力を返さずに診断を返すことを検証する。
 * @param[in,out] multiplier 積を実行する計算器。
 * @return 定義された失敗を返した場合はtrue。
 * @~english
 * @brief Verify that invalid input lengths produce a diagnostic without returning owned output.
 * @param[in,out] multiplier Multiplier computing the product.
 * @return true if the defined failure is returned.
 * @~
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
 * @~japanese
 * @brief 読めないMetalライブラリーについて、計算器を返さずに診断を返すことを検証する。
 * @param[in] library_path コンパイル済みライブラリーのパス。
 * @return 定義された失敗を返した場合はtrue。
 * @~english
 * @brief Verify that an unreadable Metal library produces a diagnostic without returning a multiplier.
 * @param[in] library_path Path to the compiled library.
 * @return true if the defined failure is returned.
 * @~
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
 * @~japanese
 * @brief 寸法と入力を変え、同じ計算器を繰り返し使う。
 * @param[in] argc 引数の個数。
 * @param[in] argv Metalライブラリーのパスを含む引数。
 * @return すべての積と失敗動作が一致した場合は0。
 * @pre argcは2であること。
 * @~english
 * @brief Use the same multiplier repeatedly with different dimensions and inputs.
 * @param[in] argc Number of arguments.
 * @param[in] argv Arguments including the Metal library path.
 * @return Zero if every product and failure behavior matches.
 * @pre argc must be 2.
 * @~
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
                && fp64_check_product(multiplier, 65, 37, 33, 0.5, 4, (apple_fp64_options_t){17})
                && fp64_check_product(multiplier, 1, 2, 1, -3, 0.25, (apple_fp64_options_t){1})
                && fp64_check_product(multiplier, 4, 3, 0, 0, 0, apple_fp64_default_options())
                && fp64_check_rejected_length(multiplier)
                && fp64_check_product(multiplier, 33, 65, 7, 2, -0.5, (apple_fp64_options_t){11})
                && fp64_check_product(multiplier, 33, 65, 7, 0, 4, (apple_fp64_options_t){11})
                && fp64_check_unreadable_library(argv[1])
                && fp64_check_input_ranges(multiplier)
                && fp64_check_factored_product(multiplier, false)
                && fp64_check_factored_product(multiplier, true)
                && fp64_check_product(multiplier, 9, 3, 11, 2, -0.5, (apple_fp64_options_t){2})
                && strcmp(device_name, apple_fp64_device_name(multiplier)) == 0;
    apple_fp64_multiplier_destroy(multiplier);
    if (!matches) fprintf(stderr, "公開APIの結果が一致しません。\n");
    return matches ? 0 : 1;
}
