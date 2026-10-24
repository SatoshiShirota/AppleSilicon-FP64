#include "apple_fp64/matmul.h"

#include <cassert>
#include <cstdio>
#include <memory>

/**
 * @~japanese
 * @brief C++から公開Cヘッダーをインクルードし、計算器の作成と行列積の実行ができるか検証する。
 * @param[in] argc 引数の個数。
 * @param[in] argv Metalライブラリーのパスを含むコマンドライン引数。
 * @return C++から行列積を正しく実行できた場合は0、それ以外は1。
 * @pre argcは2であること。
 * @~english
 * @brief Include the public C header from C++ and verify multiplier creation and matrix multiplication.
 * @param[in] argc Number of arguments.
 * @param[in] argv Command-line arguments including the Metal library path.
 * @return Zero if matrix multiplication succeeds from C++, otherwise one.
 * @pre argc must be 2.
 * @~
 */
int main(int argc, char **argv)
{
    assert(argc == 2);
    (void)argc;
    apple_fp64_multiplier_t *raw = nullptr;
    if (apple_fp64_multiplier_create(argv[1], &raw, nullptr) != APPLE_FP64_SUCCESS) return 1;
    std::unique_ptr<apple_fp64_multiplier_t, decltype(&apple_fp64_multiplier_destroy)>
        multiplier(raw, apple_fp64_multiplier_destroy);
    const double a[] = {1, 2};
    const double b[] = {3, 4};
    apple_fp64_result_t result = {};
    apple_fp64_status_t status = apple_fp64_multiply(multiplier.get(), a, 2, b, 2, 1, 1, 2,
                                                    apple_fp64_default_options(), &result, nullptr);
    bool matches = status == APPLE_FP64_SUCCESS && result.count == 1 && result.values[0] == 11
                && result.measurement.modulus_count != 0 && apple_fp64_device_name(multiplier.get())[0] != '\0';
    apple_fp64_result_destroy(&result);
    if (!matches) std::fprintf(stderr, "C++から呼び出した行列積が一致しません。\n");
    return matches ? 0 : 1;
}
