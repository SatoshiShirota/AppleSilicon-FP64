#ifndef FP64_FMA_H
#define FP64_FMA_H

#include "arithmetic.h"

#define FP64_NONFINITE_SCALE (972) /**< @~japanese 無限大またはNaNの乗数を分解したときにscaleへ設定する指数値（2047 - 1075）。
                                    * @~english Sentinel exponent value (2047 - 1075) stored in scale when
                                    * unpacking an infinite or NaN multiplicand.
                                    * @~
                                    */

/**
 * @~japanese
 * @brief 複数のFP64積和演算で再利用できるよう、あらかじめ仮数と指数へ分解した乗数。
 * @~english
 * @brief Multiplicand pre-unpacked into significand and exponent for reuse across multiple FP64 fused
 * multiply-add operations.
 * @~
 */
typedef struct fp64_fma_operand_s {
    fp64_bits_t significand; /**< @~japanese 非負の仮数（下位53ビット）と、上位語の最上位ビット（ビット31）に保持した符号ビット。
                              * @~english Nonnegative significand (low 53 bits) with the sign bit retained in
                              * bit 31 of the high word.
                              * @~
                              */
    int scale; /**< @~japanese 整数として見た仮数に掛ける2のべき乗の指数。NaNまたは無限大ではFP64_NONFINITE_SCALE。
                * @~english Power-of-two exponent scaling the integer significand, or FP64_NONFINITE_SCALE for
                * NaN and infinity.
                * @~
                */
} fp64_fma_operand_t;

/**
 * @~japanese
 * @brief FP64のビット列を、積和演算で再利用できる符号付きの仮数と2のべき乗の指数へ分解する。
 * @param[in] bits 分解するFP64値のビット列。
 * @return 仮数と指数へ分解した乗数。NaNや無限大の場合は仮数部のビット列をそのまま保持する。
 * @~english
 * @brief Unpack an FP64 bit pattern into a signed significand and power-of-two exponent for fused
 * multiply-add.
 * @param[in] bits FP64 bit pattern to unpack.
 * @return Unpacked multiplicand. Preserves the fraction bits when the input is NaN or infinity.
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
 * @brief 仮数どうしの積（最大106ビット）と加数（最大53ビット）の指数を揃えて加減算するための128ビット符号なし整数。
 * @~english
 * @brief Unsigned 128-bit integer used to align and add or subtract a 106-bit significand product and a 53-bit
 * addend.
 * @~
 */
typedef struct fp64_uint128_s {
    fp64_word_t words[4]; /**< @~japanese 最下位語（words[0]）から最上位語（words[3]）の順に並ぶ32ビット符号なし整数の配列。
                           * @~english Array of four 32-bit unsigned words ordered from least significant
                           * (words[0]) to most significant (words[3]).
                           * @~
                           */
} fp64_uint128_t;

/**
 * @~japanese
 * @brief 2つの32ビット符号なし整数の積（64ビット）の上位32ビットを求める。
 * @param[in] a 被乗数。
 * @param[in] b 乗数。
 * @return 64ビット積の上位32ビット。
 * @~english
 * @brief Compute the upper 32 bits of the 64-bit product of two unsigned 32-bit integers.
 * @param[in] a Multiplicand.
 * @param[in] b Multiplier.
 * @return High 32 bits of the 64-bit product.
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
 * @brief 128ビット符号なし整数を表現するために必要な二進数の桁数を求める。
 * @param[in] value 調べる128ビット符号なし整数。
 * @return 値が0の場合は0、それ以外の場合は最上位の1のビット位置（0始まり）に1を加えた値。
 * @~english
 * @brief Compute the number of binary digits needed to represent an unsigned 128-bit integer.
 * @param[in] value Unsigned 128-bit integer to examine.
 * @return Zero when value is 0; otherwise 1 plus the zero-based position of the most significant set bit.
 * @~
 */
static inline fp64_word_t fp64_uint128_length(fp64_uint128_t value) {
    for (int i = 3; i >= 0; --i)
        if (value.words[i] != 0) return (fp64_word_t)i * 32 + fp64_word_bit_length(value.words[i]);
    return 0;
}

/**
 * @~japanese
 * @brief 128ビット符号なし整数をシフトして指数を揃え、右シフトではみ出した非ゼロビットの情報を最下位ビットへ集約する。
 * @param[in] value シフトする128ビット符号なし整数。
 * @param[in] shift シフトするビット数。正の場合は左シフト、負の場合は右シフト。
 * @return 指数を揃えた128ビット符号なし整数。
 * @pre shiftが正の場合、左シフト後の値が128ビットに収まること。
 * @~english
 * @brief Shift an unsigned 128-bit integer to align exponents, folding any nonzero bits shifted out on the
 * right into the least significant bit.
 * @param[in] value Unsigned 128-bit integer to shift.
 * @param[in] shift Signed shift count: positive shifts left, negative shifts right.
 * @return Exponent-aligned 128-bit unsigned integer.
 * @pre When shift is positive, the left-shifted value must fit in 128 bits.
 * @~
 */
static inline fp64_uint128_t fp64_uint128_align(fp64_uint128_t value, int shift) {
    // 配列を可変の添字で参照する処理を避けるため、64ビット、32ビット、32ビット未満の段階に分けてシフトする。
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
 * @brief 128ビット符号なし整数の絶対値と2のべき乗の指数から、最近接偶数丸めしたFP64のビット列を組み立てる。
 * @param[in] value 絶対値を表す128ビット符号なし整数。
 * @param[in] negative 値が負である場合はtrue。
 * @param[in] scale 掛ける2のべき乗の指数。
 * @return 最近接偶数丸めしたFP64のビット列。valueが0の場合は正のゼロ（+0.0）。
 * @~english
 * @brief Pack an unsigned 128-bit magnitude and base-2 exponent into an FP64 bit pattern using
 * round-to-nearest, ties-to-even.
 * @param[in] value Unsigned 128-bit integer magnitude.
 * @param[in] negative true if the value is negative.
 * @param[in] scale Base-2 exponent scaling the magnitude.
 * @return Rounded FP64 bit pattern, or positive zero (+0.0) when value is 0.
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
        // 保持する53ビットの仮数の直下に2ビットを残して右シフトする。
        // ビット1が丸めの境界（中間値）を表すビットとなり、ビット0がそれより下位の非ゼロ情報をまとめたビットとなる。
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
 * @brief 2つの53ビット仮数の厳密な積（最大106ビット）を、32ビットの部分積と繰り上がりから求める。
 * @param[in] a 第1の仮数（下位53ビットが有効）。
 * @param[in] b 第2の仮数（下位53ビットが有効）。
 * @return 誤差のない最大106ビットの積。
 * @~english
 * @brief Multiply two 53-bit significands exactly using 32-bit partial products and carries.
 * @param[in] a First 53-bit significand.
 * @param[in] b Second 53-bit significand.
 * @return Exact product of at most 106 bits.
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
 * @brief 分解済みの2つの乗数a, bと加数cについて、厳密な積和 a * b + c を一度だけ最近接偶数丸めしたFP64のビット列を求める。
 * @param[in] a fp64_unpack_operandで分解した第1の乗数。
 * @param[in] b fp64_unpack_operandで分解した第2の乗数。
 * @param[in] c 加数のFP64ビット列。
 * @param[in] finite_inputs 乗数aとbの両方が有限値であると分かっている場合はtrue。
 * @return README.md「計算の定義」に従うFP64のビット列。
 * @pre aとbはfp64_unpack_operandで分解した値であること。
 * @pre finite_inputsがtrueの場合、aとbはNaNや無限大ではないこと。
 * @note すべての計算を整数のビット演算で行うため、CPUの浮動小数点例外フラグは変化しない。
 * @~english
 * @brief Compute the fused multiply-add a * b + c of two unpacked multiplicands and an addend, rounding once
 * to FP64 using round-to-nearest, ties-to-even.
 * @param[in] a First multiplicand unpacked by fp64_unpack_operand.
 * @param[in] b Second multiplicand unpacked by fp64_unpack_operand.
 * @param[in] c FP64 bit pattern of the addend.
 * @param[in] finite_inputs true when both multiplicands a and b are known to be finite.
 * @return FP64 bit pattern following README.en.md, section "Numerical semantics".
 * @pre a and b must have been unpacked by fp64_unpack_operand.
 * @pre When finite_inputs is true, a and b must not be NaN or infinity.
 * @note Uses only integer and bitwise operations and does not touch CPU floating-point exception flags.
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
    // 仮数の積は最大106ビット、加数は最大53ビットである。
    // 大きい方の最上位ビットを128ビット整数のビット126へ合わせると、桁の打ち消しが起きる範囲はすべて128ビット内に収まり、
    // 下端からはみ出すビットは最下位ビットへ集約した非ゼロの有無だけで最終丸めの向きが決まる。
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
        // 最上位語からの借り（borrow == 1）が出た場合は積より加数の絶対値の方が大きいため、結果の符号を反転し、
        // 2の補数を取って絶対値へ戻す。borrow == 0 の場合もマスク演算で同じ命令列を通し、SIMDレーン間の分岐を避ける。
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
