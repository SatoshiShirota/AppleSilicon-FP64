#include "apple_fp64/matmul.hpp"

#include <bit>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

/**
 * @brief 一定値の行列について、同じインスタンスによる積の全ビットを検証する。
 * @param[in,out] multiplier 積を実行するインスタンス。
 * @param[in] m 出力の行数。
 * @param[in] n 出力の列数。
 * @param[in] k 内積の項数。
 * @param[in] a_value Aの全要素の値。
 * @param[in] b_value Bの全要素の値。
 * @param[in] options 整数幅と行のまとまりの大きさ。
 * @pre 入力の整数化と期待値の算出に丸めが発生しない値を指定すること。
 * @exception std::runtime_error 出力の長さまたは値が一致しない。
 */
void check_product(apple_fp64::multiplier_c& multiplier, std::uint32_t m, std::uint32_t n,
                   std::uint32_t k, double a_value, double b_value, apple_fp64::options_s options = {}) {
    std::vector<double> a(std::size_t(m) * k, a_value), b(std::size_t(k) * n, b_value);
    auto result = multiplier.multiply(a, b, m, n, k, options);
    double expected = k == 0 ? 0.0 : double(k) * a_value * b_value;
    if (result.values.size() != std::size_t(m) * n)
        throw std::runtime_error("繰り返し呼び出した積の出力の長さが一致しません。");
    for (double value : result.values) {
        if (std::bit_cast<std::uint64_t>(value) != std::bit_cast<std::uint64_t>(expected))
            throw std::runtime_error("寸法と入力を変えた積の値が一致しません。");
    }
}

} // namespace

/**
 * @brief 寸法、入力と整数幅を変え、同じインスタンスを繰り返し使う。
 * @param[in] argc 引数の個数。
 * @param[in] argv Metalライブラリーのパスを含む引数。
 * @return すべての積が一致した場合は0、実行または比較が失敗した場合は1。
 * @pre argcは2であること。
 */
int main(int argc, char** argv) {
    assert(argc == 2);
    (void)argc;
    try {
        apple_fp64::multiplier_c multiplier(argv[1]);
        check_product(multiplier, 2, 3, 2, 1.5, -2);
        check_product(multiplier, 65, 37, 33, 0.5, 4, {60, 60, 17});
        check_product(multiplier, 1, 2, 1, -3, 0.25, {3, 4, 1});
        check_product(multiplier, 4, 3, 0, 0, 0);
        check_product(multiplier, 33, 65, 7, 2, -0.5, {80, 54, 11});
        check_product(multiplier, 33, 65, 7, 0, 4, {80, 54, 11});
        return 0;
    } catch (const std::runtime_error& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
