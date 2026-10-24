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
 * @brief すべての要素が同じ値である行列AとBについて、計算器による積の全ビットが期待値と一致するか検証する。
 * @param[in,out] multiplier 行列積を実行する計算器。
 * @param[in] m 出力行列の行数。
 * @param[in] n 出力行列の列数。
 * @param[in] k 内積の項数。
 * @param[in] a_value 行列Aの全要素に入れる値。
 * @param[in] b_value 行列Bの全要素に入れる値。
 * @param[in] options 行のバッチサイズなどの計算設定。
 * @return 出力の要素数とすべてのビット列が期待値と一致した場合はtrue。
 * @pre 入力の整数化と期待値の計算で丸め誤差が生じない値を指定すること。
 * @~english
 * @brief Verify that every output bit of the product of constant-valued matrices A and B matches the expected value.
 * @param[in,out] multiplier Multiplier computing the matrix product.
 * @param[in] m Number of output rows.
 * @param[in] n Number of output columns.
 * @param[in] k Number of terms in each dot product.
 * @param[in] a_value Value assigned to every element of matrix A.
 * @param[in] b_value Value assigned to every element of matrix B.
 * @param[in] options Computation options including the row batch size.
 * @return true if the output element count and all bit patterns match the expected values.
 * @pre Values must be chosen so that integer conversion and expected-value calculation incur no rounding error.
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
 * @brief 行ごとおよび列ごとの係数を持つ大きな長方形の行列積を計算し、厳密な整数の期待値と一致するか検証する。
 * @param[in,out] multiplier 行列積を実行する計算器。
 * @param[in] wide_exponents 内積の方向に逆向きの指数スケーリングを与えて、入力の指数範囲を広げる場合はtrue。
 * @return すべての出力要素のビット列が期待値と一致した場合はtrue。
 * @note 奇数のバッチサイズ（255）を指定し、符号や指数が異なる行と列を混在させる。
 * @note 行と列の中央付近のブロックで係数を変え、整数化した値の下位ゼロビット数がブロックごとに異なるようにする。
 * @~english
 * @brief Compute a large rectangular matrix product with row-wise and column-wise factors and verify it against exact integer expectations.
 * @param[in,out] multiplier Multiplier computing the matrix product.
 * @param[in] wide_exponents true to apply opposite exponent scaling along the inner dimension and widen the input exponent range.
 * @return true if the bit pattern of every output element matches the expected value.
 * @note Uses an odd row batch size (255) and mixes rows and columns with varying signs and exponents.
 * @note Adjusts factors in central row and column blocks so that the trailing-zero bit counts of integer-scaled values vary across blocks.
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
 * @brief 同じ行列寸法のまま入力の指数範囲や特殊値（無限大など）を切り替え、同じ計算器を繰り返し使用できるか検証する。
 * @param[in,out] multiplier 行列積を実行する計算器。
 * @return すべての入力パターンで計算結果が期待値と一致した場合はtrue。
 * @~english
 * @brief Verify that the same multiplier can be reused across inputs with fixed dimensions but varying exponent ranges and special values.
 * @param[in,out] multiplier Multiplier computing the matrix product.
 * @return true if the computed result matches the expected value for every input pattern.
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
 * @brief 入力配列の要素数が不正なとき、出力のメモリーを確保せずにエラー情報を返すことを検証する。
 * @param[in,out] multiplier 行列積を実行する計算器。
 * @return 仕様どおりのエラーと空の出力が返された場合はtrue。
 * @~english
 * @brief Verify that an invalid input element count returns an error diagnostic without allocating output memory.
 * @param[in,out] multiplier Multiplier computing the matrix product.
 * @return true if the specified error status and empty output are returned.
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
 * @brief 読み込めないMetalライブラリーのパスを渡したとき、計算器を作成せずにエラー情報を返すことを検証する。
 * @param[in] library_path コンパイル済みMetalライブラリーのパス。
 * @return 仕様どおりのエラーとNULLの計算器が返された場合はtrue。
 * @~english
 * @brief Verify that passing an unreadable Metal library path returns an error diagnostic without creating a multiplier.
 * @param[in] library_path Path to the compiled Metal library.
 * @return true if the specified error status and a NULL multiplier are returned.
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
 * @brief 行列の寸法や入力の性質を変えながら同じ計算器を繰り返し呼び出し、計算結果とエラー時の動作を検証する。
 * @param[in] argc 引数の個数。
 * @param[in] argv Metalライブラリーのパスを含むコマンドライン引数。
 * @return すべての検証に成功した場合は0、それ以外は1。
 * @pre argcは2であること。
 * @~english
 * @brief Repeatedly invoke the same multiplier across varying matrix dimensions and input characteristics to verify results and error behavior.
 * @param[in] argc Number of arguments.
 * @param[in] argv Command-line arguments including the Metal library path.
 * @return Zero if all verifications succeed, otherwise one.
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
