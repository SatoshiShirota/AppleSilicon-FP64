#ifndef FP64_FMA_H
#define FP64_FMA_H

#include "arithmetic.h"

#define FP64_NONFINITE_SCALE (972) /**< @~japanese 無限大とNaNの乗数を分解したときの指数。
                                    * @~english Exponent used when unpacking infinite and NaN multiplicands.
                                    * @~
                                    */

/**
 * @~japanese
 * @brief 複数の積和演算で再利用する、分解済みのFP64の乗数。
 * @~english
 * @brief Unpacked FP64 multiplicand reused across fused multiply-add operations.
 * @~
 */
typedef struct fp64_fma_operand_s {
    fp64_bits_t significand; /**< @~japanese 非負の仮数と、上位語の最上位ビットに置く符号。
                              * @~english Nonnegative significand with the sign stored in the most significant
                              * bit of the high word.
                              * @~
                              */
    int scale; /**< @~japanese 仮数に掛ける2のべき乗の指数。非有限値ではFP64_NONFINITE_SCALE。
                * @~english Exponent of the power of two multiplying the significand. FP64_NONFINITE_SCALE for
                * nonfinite values.
                * @~
                */
} fp64_fma_operand_t;

/**
 * @~japanese
 * @brief FP64の乗数を、符号付きの仮数と指数へ分解する。
 * @param[in] bits FP64のビット列。
 * @return 積和演算で再利用できる乗数。非有限値では仮数部のビット列を保持する。
 * @~english
 * @brief Unpack an FP64 multiplicand into a signed significand and exponent.
 * @param[in] bits FP64 bit pattern.
 * @return Multiplicand reusable in fused multiply-add operations. Retains the fraction bit pattern for
 * nonfinite values.
 * @~
 */
static inline fp64_fma_operand_t fp64_unpack_operand(fp64_bits_t bits) {
    fp64_word_t exponent = (bits.high >> 20) & 2047u;
    fp64_word_t high = bits.high & 0x800fffffu;
    if (exponent != 0 && exponent != 2047) high |= 0x100000u;
    return (fp64_fma_operand_t){{bits.low, high}, exponent != 0 ? (int)exponent - 1075 : -1074};
}

/**
 * @~japanese
 * @brief FP64の積と加数を揃える128ビットの符号なし整数。
 * @~english
 * @brief Unsigned 128-bit integer used to align an FP64 product and addend.
 * @~
 */
typedef struct fp64_uint128_s {
    fp64_word_t words[4]; /**< @~japanese 下位から並ぶ32ビットの語。
                           * @~english 32-bit words ordered from least to most significant.
                           * @~
                           */
} fp64_uint128_t;

/**
 * @~japanese
 * @brief 32ビットの積の上位を求める。
 * @param[in] a 第一の整数。
 * @param[in] b 第二の整数。
 * @return 積の上位32ビット。
 * @~english
 * @brief Compute the high word of a 32-bit multiplication.
 * @param[in] a First integer.
 * @param[in] b Second integer.
 * @return High 32 bits of the product.
 * @~
 */
static inline fp64_word_t fp64_multiply_high(fp64_word_t a, fp64_word_t b) {
#ifdef __METAL_VERSION__
    return metal::mulhi(a, b);
#else
    return (fp64_word_t)(((uint64_t)a * b) >> 32);
#endif
}

/**
 * @~japanese
 * @brief 128ビット整数の有効な桁数を求める。
 * @param[in] value 非負整数。
 * @return 最上位ビットの位置に1を加えた値。ゼロでは0。
 * @~english
 * @brief Compute the bit length of a 128-bit integer.
 * @param[in] value Nonnegative integer.
 * @return One plus the position of the most significant bit. Zero for a zero value.
 * @~
 */
static inline fp64_word_t fp64_uint128_length(fp64_uint128_t value) {
    for (int i = 3; i >= 0; --i)
        if (value.words[i] != 0) return (fp64_word_t)i * 32 + fp64_word_bit_length(value.words[i]);
    return 0;
}

/**
 * @~japanese
 * @brief 下位の非ゼロの情報を最下位ビットへ集めながら指数を揃える。
 * @param[in] value 非負整数。
 * @param[in] shift 正なら左、負なら右へずらすビット数。
 * @return 指数を揃えた整数。
 * @pre 左シフトした値が128ビットに収まること。
 * @~english
 * @brief Align exponents while collecting information about discarded nonzero bits in the least significant
 * bit.
 * @param[in] value Nonnegative integer.
 * @param[in] shift Bit shift count. Positive shifts left; negative shifts right.
 * @return Integer with its exponent aligned.
 * @pre The left-shifted value must fit in 128 bits.
 * @~
 */
static inline fp64_uint128_t fp64_uint128_align(fp64_uint128_t value, int shift) {
    if (shift >= 0) {
        if (shift & 64) value = (fp64_uint128_t){{0, 0, value.words[0], value.words[1]}};
        if (shift & 32) value = (fp64_uint128_t){{0, value.words[0], value.words[1], value.words[2]}};
        fp64_word_t tail = (fp64_word_t)shift & 31u;
        if (tail == 0) return value;
        return (fp64_uint128_t){{value.words[0] << tail,
                                 (value.words[1] << tail) | (value.words[0] >> (32 - tail)),
                                 (value.words[2] << tail) | (value.words[1] >> (32 - tail)),
                                 (value.words[3] << tail) | (value.words[2] >> (32 - tail))}};
    }
    fp64_word_t amount = (fp64_word_t)-shift;
    if (amount >= 128)
        return (fp64_uint128_t){{(value.words[0] | value.words[1] | value.words[2] | value.words[3]) != 0, 0, 0, 0}};
    fp64_word_t lost = 0;
    if (amount & 64) {
        lost = value.words[0] | value.words[1];
        value = (fp64_uint128_t){{value.words[2], value.words[3], 0, 0}};
    }
    if (amount & 32) {
        lost |= value.words[0];
        value = (fp64_uint128_t){{value.words[1], value.words[2], value.words[3], 0}};
    }
    fp64_word_t tail = amount & 31u;
    if (tail != 0) {
        lost |= value.words[0] & ((1u << tail) - 1);
        value = (fp64_uint128_t){{(value.words[0] >> tail) | (value.words[1] << (32 - tail)),
                                 (value.words[1] >> tail) | (value.words[2] << (32 - tail)),
                                 (value.words[2] >> tail) | (value.words[3] << (32 - tail)),
                                 value.words[3] >> tail}};
    }
    value.words[0] |= lost != 0;
    return value;
}

/**
 * @~japanese
 * @brief 128ビット整数と指数から最近接偶数丸めしたFP64を求める。
 * @param[in] value 絶対値の整数。
 * @param[in] negative 符号。
 * @param[in] scale 2のべき乗の指数。
 * @return FP64のビット列。整数がゼロの場合は正のゼロ。
 * @~english
 * @brief Convert a 128-bit integer and exponent to FP64 using round-to-nearest, ties-to-even.
 * @param[in] value Integer magnitude.
 * @param[in] negative Sign.
 * @param[in] scale Exponent of the power of two.
 * @return FP64 bit pattern. Positive zero when the integer is zero.
 * @~
 */
static inline fp64_bits_t fp64_uint128_pack(fp64_uint128_t value, bool negative, int scale) {
    fp64_word_t length = fp64_uint128_length(value);
    fp64_word_t sign = negative ? 0x80000000u : 0;
    if (length == 0) return (fp64_bits_t){0, 0};
    int exponent = (int)length - 1 + scale;
    if (exponent > 1023) return (fp64_bits_t){0, sign | 0x7ff00000u};
    bool subnormal = exponent < -1022;
    int shift = subnormal ? -1074 - scale : (int)length - 53;
    fp64_bits_t mantissa;
    if (shift <= 0) {
        mantissa = fp64_shift_left((fp64_bits_t){value.words[0], value.words[1]}, (fp64_word_t)-shift);
    } else {
        // 丸め位置の直下に2ビットを残す。上側が中間値のビットで、下側には失われた非ゼロの情報を含める。
        fp64_uint128_t rounded = fp64_uint128_align(value, 2 - shift);
        mantissa = (fp64_bits_t){(rounded.words[0] >> 2) | (rounded.words[1] << 30), rounded.words[1] >> 2};
        fp64_word_t increment = ((rounded.words[0] >> 1) & 1u) & ((rounded.words[0] | mantissa.low) & 1u);
        fp64_word_t low = mantissa.low + increment;
        mantissa.high += low < mantissa.low;
        mantissa.low = low;
    }
    if (subnormal) return (fp64_bits_t){mantissa.low, sign | mantissa.high};
    if (mantissa.high & 0x200000u) {
        mantissa = fp64_shift_right(mantissa, 1);
        ++exponent;
    }
    if (exponent > 1023) return (fp64_bits_t){0, sign | 0x7ff00000u};
    return (fp64_bits_t){mantissa.low, sign | ((fp64_word_t)(exponent + 1023) << 20) | (mantissa.high & 0xfffffu)};
}

/**
 * @~japanese
 * @brief 53ビットの仮数同士を誤差なく掛ける。
 * @param[in] a 第一の仮数。
 * @param[in] b 第二の仮数。
 * @return 最大106ビットの積。
 * @~english
 * @brief Multiply two 53-bit significands exactly.
 * @param[in] a First significand.
 * @param[in] b Second significand.
 * @return Product of at most 106 bits.
 * @~
 */
static inline fp64_uint128_t fp64_multiply_significands(fp64_bits_t a, fp64_bits_t b) {
    fp64_word_t first = a.low * b.high, second = a.high * b.low;
    fp64_word_t middle = fp64_multiply_high(a.low, b.low);
    fp64_word_t sum = middle + first;
    fp64_word_t carry = sum < middle;
    middle = sum + second;
    carry += middle < sum;
    fp64_word_t upper = fp64_multiply_high(a.low, b.high) + fp64_multiply_high(a.high, b.low) + carry;
    fp64_word_t high = a.high * b.high;
    sum = high + upper;
    return (fp64_uint128_t){{a.low * b.low, middle, sum,
                             fp64_multiply_high(a.high, b.high) + (sum < high)}};
}

/**
 * @~japanese
 * @brief FP64の積と加算を合わせて最近接偶数丸めする。
 * @param[in] a 分解済みの第一の乗数。
 * @param[in] b 分解済みの第二の乗数。
 * @param[in] c 加数のビット列。
 * @param[in] finite_inputs 両乗数が有限であることが確定している場合はtrue。
 * @return README.md「計算の定義」に従うFP64のビット列。
 * @pre aとbはfp64_unpack_operandで分解した値であること。
 * @pre finite_inputsがtrueの場合、aとbは有限であること。
 * @note CPUの浮動小数点演算と例外フラグを使用しない。
 * @~english
 * @brief Round the combined FP64 product and addition using round-to-nearest, ties-to-even.
 * @param[in] a First unpacked multiplicand.
 * @param[in] b Second unpacked multiplicand.
 * @param[in] c Bit pattern of the addend.
 * @param[in] finite_inputs true when both multiplicands are known to be finite.
 * @return FP64 bit pattern following README.en.md, section "Numerical semantics".
 * @pre a and b must have been unpacked by fp64_unpack_operand.
 * @pre a and b must be finite when finite_inputs is true.
 * @note Does not use CPU floating-point arithmetic or exception flags.
 * @~
 */
static inline fp64_bits_t fp64_fused_multiply_add(fp64_fma_operand_t a, fp64_fma_operand_t b, fp64_bits_t c, bool finite_inputs) {
    fp64_word_t ec = (c.high >> 20) & 2047u;
    fp64_bits_t ma = {a.significand.low, a.significand.high & 0x1fffffu};
    fp64_bits_t mb = {b.significand.low, b.significand.high & 0x1fffffu}, mc = {c.low, c.high & 0xfffffu};
    bool product_negative = ((a.significand.high ^ b.significand.high) >> 31) != 0, c_negative = (c.high >> 31) != 0;
    bool za = (ma.low | ma.high) == 0, zb = (mb.low | mb.high) == 0;
    fp64_bits_t nan = {0, 0x7ff80000u};
    if (ec == 2047 && (mc.low | mc.high) != 0) return nan;
    if (!finite_inputs) {
        if ((a.scale == FP64_NONFINITE_SCALE && (ma.low | ma.high) != 0)
            || (b.scale == FP64_NONFINITE_SCALE && (mb.low | mb.high) != 0)) return nan;
        if (a.scale == FP64_NONFINITE_SCALE || b.scale == FP64_NONFINITE_SCALE) {
            if ((za && a.scale != FP64_NONFINITE_SCALE) || (zb && b.scale != FP64_NONFINITE_SCALE)
                || (ec == 2047 && product_negative != c_negative)) return nan;
            return (fp64_bits_t){0, (product_negative ? 0x80000000u : 0) | 0x7ff00000u};
        }
    }
    if (ec == 2047) return c;
    if (za || zb) {
        if ((c.high & 0x7fffffffu) == 0 && c.low == 0)
            return (fp64_bits_t){0, product_negative && c_negative ? 0x80000000u : 0};
        return c;
    }
    if (ec != 0) mc.high |= 0x100000u;
    int product_scale = a.scale + b.scale;
    int c_scale = ec != 0 ? (int)ec - 1075 : -1074;
    fp64_uint128_t product = fp64_multiply_significands(ma, mb), addend = {{mc.low, mc.high, 0, 0}};
    fp64_word_t product_length = fp64_uint128_length(product), c_length = fp64_uint128_length(addend);
    if (c_length == 0) return fp64_uint128_pack(product, product_negative, product_scale);
    int product_top = product_scale + (int)product_length - 1, c_top = c_scale + (int)c_length - 1;
    int scale = (product_top > c_top ? product_top : c_top) - 126;
    // 積は106ビット、加数は53ビットである。桁の打ち消しが起こる範囲は128ビットに収まり、
    // 範囲外へずらす場合は、最下位ビットに集めた情報だけで最終丸めの方向を決められる。
    product = fp64_uint128_align(product, product_scale - scale);
    addend = fp64_uint128_align(addend, c_scale - scale);
    fp64_uint128_t magnitude;
    bool negative = product_negative;
    if (product_negative == c_negative) {
        fp64_word_t carry = 0;
        for (int i = 0; i < 4; ++i) {
            fp64_word_t sum = product.words[i] + addend.words[i];
            magnitude.words[i] = sum + carry;
            carry = (sum < product.words[i]) | (magnitude.words[i] < sum);
        }
    } else {
        fp64_word_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            fp64_word_t difference = product.words[i] - addend.words[i];
            magnitude.words[i] = difference - borrow;
            borrow = (product.words[i] < addend.words[i]) | (difference < borrow);
        }
        // 最上位からの借りを、符号と二の補数からの絶対値化に使う。
        // マスクがゼロの場合も同じ命令列を使い、SIMD内で符号が異なる場合の分岐を避ける。
        negative = negative != (borrow != 0);
        fp64_word_t mask = 0u - borrow, carry = borrow;
        for (int i = 0; i < 4; ++i) {
            fp64_word_t inverted = magnitude.words[i] ^ mask;
            magnitude.words[i] = inverted + carry;
            carry = magnitude.words[i] < inverted;
        }
    }
    return fp64_uint128_pack(magnitude, negative, scale);
}

#endif
