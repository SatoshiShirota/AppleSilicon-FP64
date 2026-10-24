#include "fp64_fma.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/**
 * @~japanese
 * @brief 指数部と仮数部のあらゆるビットパターンを生成するため、乱数生成器の状態を進める。
 * @param[in,out] state 非ゼロの乱数状態。
 * @return 次の64ビット乱数値。
 * @~english
 * @brief Advance the random number generator state to produce bit patterns across the full exponent and significand range.
 * @param[in,out] state Non-zero generator state.
 * @return Next 64-bit random value.
 * @~
 */
static uint64_t fp64_next_bits(uint64_t *state)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return *state * UINT64_C(2685821657736338717);
}

/**
 * @~japanese
 * @brief CPUとGPUで共有するソフトウェアFMA実装の出力を、標準Cライブラリーのfma関数の結果とビット単位で比較する。
 * @return すべてのテストケースでビット列が一致した場合は0、不一致があった場合は1。
 * @~english
 * @brief Compare the output of the software FMA implementation shared between CPU and GPU bit-for-bit against standard C fma.
 * @return Zero if all test cases match bit-for-bit, otherwise one.
 * @~
 */
int main(void)
{
    uint64_t state = UINT64_C(0x7adcc327e89b2041);
    for (unsigned index = 0; index < 20000; ++index) {
        uint64_t inputs[3] = {fp64_next_bits(&state), fp64_next_bits(&state), fp64_next_bits(&state)};
        double values[3];
        memcpy(values, inputs, sizeof(values));
        if (index % 3 != 0) {
            double product = values[0] * values[1];
            values[2] = index % 3 == 1 ? -product : -nextafter(product, INFINITY);
            memcpy(&inputs[2], &values[2], sizeof(double));
        }
        fp64_bits_t operands[3];
        for (unsigned i = 0; i < 3; ++i)
            operands[i] = (fp64_bits_t){(uint32_t)inputs[i], (uint32_t)(inputs[i] >> 32)};
        fp64_bits_t result = fp64_fused_multiply_add(fp64_unpack_operand(operands[0]), fp64_unpack_operand(operands[1]), operands[2],
                                                   isfinite(values[0]) && isfinite(values[1]));
        uint64_t actual = ((uint64_t)result.high << 32) | result.low;
        double reference = fma(values[0], values[1], values[2]);
        uint64_t expected;
        memcpy(&expected, &reference, sizeof(expected));
        if (isnan(reference)) expected = UINT64_C(0x7ff8000000000000);
        if (actual != expected) {
            fprintf(stderr, "FP64の積和が一致しません: %016" PRIx64 " * %016" PRIx64
                    " + %016" PRIx64 ": %016" PRIx64 " != %016" PRIx64 "\n",
                    inputs[0], inputs[1], inputs[2], actual, expected);
            return 1;
        }
    }
    puts("全指数範囲と桁の打ち消しでFP64の積和のビット列が一致しました。");
    return 0;
}
