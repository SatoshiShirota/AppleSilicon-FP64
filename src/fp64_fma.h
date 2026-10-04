#ifndef FP64_FMA_H
#define FP64_FMA_H

#include "arithmetic.h"

/**
 * @brief FP64の積と加数を揃える128ビットの符号なし整数。
 */
typedef struct fp64_uint128_s {
    fp64_word_t words[4]; /**< 下位から並ぶ32ビットの語。 */
} fp64_uint128_t;

/**
 * @brief 32ビットの積の上位を求める。
 * @param[in] a 第一の整数。
 * @param[in] b 第二の整数。
 * @return 積の上位32ビット。
 */
static inline fp64_word_t fp64_multiply_high(fp64_word_t a, fp64_word_t b) {
#ifdef __METAL_VERSION__
    return metal::mulhi(a, b);
#else
    return (fp64_word_t)(((uint64_t)a * b) >> 32);
#endif
}

/**
 * @brief 128ビットの和を求める。
 * @param[in] a 第一の整数。
 * @param[in] b 第二の整数。
 * @return 和。
 * @pre 和が128ビットに収まること。
 */
static inline fp64_uint128_t fp64_uint128_add(fp64_uint128_t a, fp64_uint128_t b) {
    fp64_uint128_t result = {0};
    fp64_word_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        fp64_word_t sum = a.words[i] + b.words[i];
        fp64_word_t value = sum + carry;
        carry = (sum < a.words[i]) | (value < sum);
        result.words[i] = value;
    }
    return result;
}

/**
 * @brief 128ビットの大小を比較する。
 * @param[in] a 左辺。
 * @param[in] b 右辺。
 * @return 小さい場合は-1、等しい場合は0、大きい場合は1。
 */
static inline int fp64_uint128_compare(fp64_uint128_t a, fp64_uint128_t b) {
    for (int i = 3; i >= 0; --i)
        if (a.words[i] != b.words[i]) return a.words[i] < b.words[i] ? -1 : 1;
    return 0;
}

/**
 * @brief 128ビットの非負の差を求める。
 * @param[in] a 被減数。
 * @param[in] b 減数。
 * @return 差。
 * @pre aがb以上であること。
 */
static inline fp64_uint128_t fp64_uint128_subtract(fp64_uint128_t a, fp64_uint128_t b) {
    fp64_uint128_t result = {0};
    fp64_word_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        fp64_word_t difference = a.words[i] - b.words[i];
        result.words[i] = difference - borrow;
        borrow = (a.words[i] < b.words[i]) | (difference < borrow);
    }
    return result;
}

/**
 * @brief 128ビット整数の有効な桁数を求める。
 * @param[in] value 非負整数。
 * @return 最上位ビットの位置に1を加えた値。ゼロでは0。
 */
static inline fp64_word_t fp64_uint128_length(fp64_uint128_t value) {
    for (int i = 3; i >= 0; --i)
        if (value.words[i] != 0) return (fp64_word_t)i * 32 + fp64_word_bit_length(value.words[i]);
    return 0;
}

/**
 * @brief 下位の非ゼロの情報を最下位ビットへ集めながら指数を揃える。
 * @param[in] value 非負整数。
 * @param[in] shift 正なら左、負なら右へずらすビット数。
 * @return 指数を揃えた整数。
 * @pre 左シフトした値が128ビットに収まること。
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
 * @brief 128ビット整数と指数から最近接偶数丸めしたFP64を求める。
 * @param[in] value 絶対値の整数。
 * @param[in] negative 符号。
 * @param[in] scale 2のべき乗の指数。
 * @return FP64のビット列。
 */
static inline fp64_bits_t fp64_uint128_pack(fp64_uint128_t value, bool negative, int scale) {
    fp64_word_t length = fp64_uint128_length(value);
    fp64_word_t sign = negative ? 0x80000000u : 0;
    if (length == 0) return (fp64_bits_t){0, sign};
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
        if ((rounded.words[0] & 2u) && ((rounded.words[0] & 1u) || (mantissa.low & 1u))) {
            ++mantissa.low;
            if (mantissa.low == 0) ++mantissa.high;
        }
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
 * @brief 53ビットの仮数同士を誤差なく掛ける。
 * @param[in] a 第一の仮数。
 * @param[in] b 第二の仮数。
 * @return 最大106ビットの積。
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
 * @brief FP64の積と加算を合わせて最近接偶数丸めする。
 * @param[in] a 第一の乗数のビット列。
 * @param[in] b 第二の乗数のビット列。
 * @param[in] c 加数のビット列。
 * @param[in] finite_inputs 両乗数が有限であることが確定している場合はtrue。
 * @return README.md「計算の定義」に従うFP64のビット列。
 * @pre finite_inputsがtrueの場合、aとbは有限であること。
 * @note CPUの浮動小数点演算と例外フラグを使用しない。
 */
static inline fp64_bits_t fp64_fused_multiply_add(fp64_bits_t a, fp64_bits_t b, fp64_bits_t c, bool finite_inputs) {
    fp64_word_t ea = (a.high >> 20) & 2047u, eb = (b.high >> 20) & 2047u, ec = (c.high >> 20) & 2047u;
    fp64_bits_t ma = {a.low, a.high & 0xfffffu}, mb = {b.low, b.high & 0xfffffu}, mc = {c.low, c.high & 0xfffffu};
    bool product_negative = ((a.high ^ b.high) >> 31) != 0, c_negative = (c.high >> 31) != 0;
    bool za = ea == 0 && (ma.low | ma.high) == 0, zb = eb == 0 && (mb.low | mb.high) == 0;
    fp64_bits_t nan = {0, 0x7ff80000u};
    if (ec == 2047 && (mc.low | mc.high) != 0) return nan;
    if (!finite_inputs) {
        if ((ea == 2047 && (ma.low | ma.high) != 0) || (eb == 2047 && (mb.low | mb.high) != 0)) return nan;
        if (ea == 2047 || eb == 2047) {
            if (za || zb || (ec == 2047 && product_negative != c_negative)) return nan;
            return (fp64_bits_t){0, (product_negative ? 0x80000000u : 0) | 0x7ff00000u};
        }
    }
    if (ec == 2047) return c;
    if (za || zb) {
        if ((c.high & 0x7fffffffu) == 0 && c.low == 0)
            return (fp64_bits_t){0, product_negative && c_negative ? 0x80000000u : 0};
        return c;
    }
    if (ea != 0) ma.high |= 0x100000u;
    if (eb != 0) mb.high |= 0x100000u;
    if (ec != 0) mc.high |= 0x100000u;
    int product_scale = (ea != 0 ? (int)ea - 1075 : -1074) + (eb != 0 ? (int)eb - 1075 : -1074);
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
    if (product_negative == c_negative)
        return fp64_uint128_pack(fp64_uint128_add(product, addend), product_negative, scale);
    int order = fp64_uint128_compare(product, addend);
    if (order == 0) return (fp64_bits_t){0, 0};
    return fp64_uint128_pack(order > 0 ? fp64_uint128_subtract(product, addend) : fp64_uint128_subtract(addend, product),
                             order > 0 ? product_negative : c_negative, scale);
}

#endif
