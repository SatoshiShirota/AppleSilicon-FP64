#include <apple_fp64/matmul.h>
#include <stdio.h>

/**
 * @~japanese
 * @brief 外部のCMakeプロジェクトから公開ヘッダーとライブラリーをリンクし、行列積が正しく計算できるか検証する。
 * @param[in] argc 引数の個数。
 * @param[in] argv 配布されたMetalライブラリーのパスを含むコマンドライン引数。
 * @return 計算結果が期待値と一致した場合は0、それ以外は1。
 * @~english
 * @brief Link the public header and library from an external CMake project and verify matrix multiplication.
 * @param[in] argc Number of arguments.
 * @param[in] argv Command-line arguments including the distributed Metal library path.
 * @return Zero if the computed product matches the expected values, otherwise one.
 * @~
 */
int main(int argc, char **argv)
{
    if (argc != 2) return 1;
    const double a[] = {1, 2, 3, 4};
    const double b[] = {5, 6, 7, 8};
    const double expected[] = {19, 22, 43, 50};
    apple_fp64_multiplier_t *multiplier = NULL;
    apple_fp64_result_t result = {0};
    apple_fp64_error_t error = {0};
    apple_fp64_status_t status = apple_fp64_multiplier_create(argv[1], &multiplier, &error);
    if (status == APPLE_FP64_SUCCESS) {
        status = apple_fp64_multiply(multiplier, a, 4, b, 4, 2, 2, 2,
                                     apple_fp64_default_options(), &result, &error);
    }
    int matches = status == APPLE_FP64_SUCCESS && result.count == 4;
    if (matches) {
        for (size_t index = 0; index < result.count; ++index) {
            if (result.values[index] != expected[index]) matches = 0;
        }
    }
    if (!matches) {
        fprintf(stderr, "%s\n", error.message != NULL ? error.message : "外部から呼び出した行列積が一致しません。");
    }
    apple_fp64_result_destroy(&result);
    apple_fp64_error_destroy(&error);
    apple_fp64_multiplier_destroy(multiplier);
    return matches ? 0 : 1;
}
