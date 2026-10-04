#include "fp64_fma.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief 全指数範囲のビット列を作るための乱数の状態を進める。
 * @param[in,out] state 非ゼロの乱数の状態。
 * @return 次の64ビットの値。
 */
static uint64_t fp64_next_bits(uint64_t *state)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return *state * UINT64_C(2685821657736338717);
}

/**
 * @brief CPUとGPUで共有する積和演算を、CPUのFP64の積和演算と比較する。
 * @return ビット列が一致した場合は0。
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
        fp64_bits_t result = fp64_fused_multiply_add(operands[0], operands[1], operands[2],
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
