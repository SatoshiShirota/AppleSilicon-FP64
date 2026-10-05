#include "apple_fp64/matmul.h"

#include <cassert>
#include <cstdio>
#include <memory>

/**
 * @~japanese
 * @brief C++からCのヘッダーを使い、計算器と結果を操作する。
 * @param[in] argc 引数の個数。
 * @param[in] argv Metalライブラリーのパスを含む引数。
 * @return C++から行列積を利用できた場合は0。
 * @pre argcは2であること。
 * @~english
 * @brief Use the C header from C++ to manage the multiplier and result.
 * @param[in] argc Number of arguments.
 * @param[in] argv Arguments including the Metal library path.
 * @return Zero if matrix multiplication was usable from C++.
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
