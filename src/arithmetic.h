#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <bit>
#include <cstdint>
#endif

namespace apple_fp64 {

#ifdef __METAL_VERSION__
using word_t = metal::uint; /**< 算術で使用する32ビット符号なし整数。 */
#define FP64_CONSTANT constant /**< Metalの定数用のアドレス空間。 */
#else
using word_t = std::uint32_t; /**< 算術で使用する32ビット符号なし整数。 */
#define FP64_CONSTANT /**< CPUではアドレス空間の修飾を付けない。 */
#endif

FP64_CONSTANT constexpr word_t DIGIT_BITS = 24; /**< 小整数との積が32ビットに収まる桁幅。 */
FP64_CONSTANT constexpr word_t DIGIT_MASK = (1u << DIGIT_BITS) - 1u; /**< 一桁の値域を取り出すマスク。 */
FP64_CONSTANT constexpr word_t MAX_LIMBS = 15; /**< 全法の積の342ビットを格納する桁数。 */
FP64_CONSTANT constexpr word_t CRT_MODULI[] = { /**< INT8で対称な余りを表せる互いに素な法。 */
    256, 251, 241, 239, 233, 229, 227, 223, 211, 199, 197, 193, 191, 181,
    179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127, 113, 109, 107,
    103, 101, 97, 89, 83, 79, 73, 71, 67, 61, 59, 53, 47, 43, 41, 37,
    31, 29, 23, 19, 17, 13, 11, 7, 5, 3
};
FP64_CONSTANT constexpr word_t MAX_MODULI = sizeof(CRT_MODULI) / sizeof(CRT_MODULI[0]); /**< 法の候補数。 */
FP64_CONSTANT constexpr int ZERO_EXPONENT = -2147483647; /**< 最大値の取得でゼロを除外する値。 */
FP64_CONSTANT constexpr word_t K_CHUNK = 1024; /**< INT8の積の絶対値の総和を2の24乗以下に保つ項数。 */
FP64_CONSTANT constexpr word_t K_ACCUMULATE = 65536; /**< 余りと部分内積の合計が符号付き32ビットに収まる項数。 */
FP64_CONSTANT constexpr word_t TILE_ROWS = 64; /**< 行列積のスレッドグループが担当する行数。 */
FP64_CONSTANT constexpr word_t TILE_COLUMNS = 64; /**< 行列積のスレッドグループが担当する列数。 */
FP64_CONSTANT constexpr word_t SIMD_GROUPS = 4; /**< 行列積を協調実行するSIMDグループ数。 */

#undef FP64_CONSTANT

/** @brief FP64形式のビット列を、下位と上位に分けて保持する。 */
struct fp64_bits_s {
    word_t low; /**< 下位32ビット。 */
    word_t high; /**< 符号と指数を含む上位32ビット。 */
};

/** @brief 基数2の24乗で表す、非負の固定長整数。 */
struct big_uint_s {
    word_t digits[MAX_LIMBS]; /**< 下位桁から並ぶ値。各桁はDIGIT_MASK以下。 */
};

#ifdef __METAL_VERSION__
using big_uint_input_t = big_uint_s; /**< 異なるMetalのアドレス空間から受け取れる整数の入力型。 */
using big_uint_output_t = thread big_uint_s&; /**< GPUのスレッドが所有する整数への参照。 */
#else
using big_uint_input_t = const big_uint_s&; /**< CPUで読み取り専用の整数を複写せずに渡す入力型。 */
using big_uint_output_t = big_uint_s&; /**< CPUの処理が更新する整数への参照。 */
#endif

/** @brief 一つの法と、32ビット整数の余りを求める係数。 */
struct modulus_s {
    word_t value; /**< 256以下の法。 */
    word_t reciprocal; /**< 2の32乗をvalueで割った商。 */
    word_t word_weight; /**< 2の32乗をvalueで割った余り。 */
};

/** @brief 入力の値に依存しないCRTの係数。 */
struct crt_plan_s {
    word_t count; /**< 使用する法の個数。 */
    word_t limbs; /**< 使用する法の積を格納する桁数。 */
    modulus_s moduli[MAX_MODULI]; /**< 使用順に並ぶ法と余りの計算に使う係数。 */
    word_t inverses[MAX_MODULI]; /**< 各法に対する、それ以前の法の積の逆元。 */
    word_t stage_limbs[MAX_MODULI]; /**< 各法を取り込んだ整数を保持するための桁数。 */
    big_uint_s prefixes[MAX_MODULI]; /**< 各法より前に使用した法の積。 */
    big_uint_s product; /**< 使用する法の積。 */
    big_uint_s half_product; /**< 使用する法の積の半分。 */
    unsigned char powers[MAX_MODULI][MAX_LIMBS * DIGIT_BITS]; /**< 各法に対する2のべき乗の余り。指数の昇順で並ぶ。 */
};

/** @brief 一つの行のまとまりをGPUに渡す寸法と整数幅。 */
struct batch_parameters_s {
    word_t rows; /**< 行のまとまりに含まれる行数。 */
    word_t columns; /**< 出力の列数。 */
    word_t inner; /**< 内積の項数。 */
    word_t row_begin; /**< 入力Aにおける先頭行。 */
    word_t precision_a; /**< Aの整数幅。 */
    word_t precision_b; /**< Bの整数幅。 */
};

/**
 * @brief 非負の32ビット整数が必要とするビット数を求める。
 * @param[in] value 対象の整数。
 * @return ゼロの場合は0。それ以外は最上位ビットの位置に1を加えた値。
 */
inline word_t word_bit_length(word_t value) {
#ifdef __METAL_VERSION__
    return 32u - metal::clz(value);
#else
    return 32u - std::countl_zero(value);
#endif
}

/**
 * @brief 有限のFP64値の絶対値を囲む2のべき乗の指数を求める。
 * @param[in] bits 有限のFP64値のビット列。
 * @return 非ゼロ値ではREADME.md「計算の定義」の指数。ゼロではZERO_EXPONENT。
 * @pre NaNと無限大を渡してはならない。
 */
inline int fp64_exponent(fp64_bits_s bits) {
    word_t exponent = (bits.high >> 20) & 2047u;
    if (exponent != 0) return int(exponent) - 1022;
    word_t high = bits.high & 0xfffffu;
    if (high != 0) return int(32 + word_bit_length(high)) - 1074;
    if (bits.low != 0) return int(word_bit_length(bits.low)) - 1074;
    return ZERO_EXPONENT;
}

/**
 * @brief 二語の非負整数を右へずらす。
 * @param[in] value 下位、上位の順で保持する整数。
 * @param[in] shift ずらすビット数。
 * @return 切り捨てた整数。64ビット以上ずらす場合はゼロ。
 */
inline fp64_bits_s shift_right(fp64_bits_s value, word_t shift) {
    if (shift >= 64) return {0, 0};
    if (shift >= 32) return {value.high >> (shift - 32), 0};
    if (shift == 0) return value;
    return {(value.low >> shift) | (value.high << (32 - shift)), value.high >> shift};
}

/**
 * @brief 二語の非負整数を左へずらす。
 * @param[in] value 下位、上位の順で保持する整数。
 * @param[in] shift ずらすビット数。64未満。
 * @return 64ビット以内の結果。
 * @pre 結果が64ビットに収まること。
 */
inline fp64_bits_s shift_left(fp64_bits_s value, word_t shift) {
    if (shift >= 32) return {0, value.low << (shift - 32)};
    if (shift == 0) return value;
    return {value.low << shift, (value.high << shift) | (value.low >> (32 - shift))};
}

/**
 * @brief 32ビット整数の余りを、乗算の上位ビットから求める。
 * @param[in] value 非負の32ビット整数。
 * @param[in] modulus 法と、その法から求めた係数。
 * @return 0以上modulus.value未満の余り。
 */
inline word_t word_mod(word_t value, modulus_s modulus) {
#ifdef __METAL_VERSION__
    word_t quotient = metal::mulhi(value, modulus.reciprocal);
#else
    word_t quotient = word_t((std::uint64_t(value) * modulus.reciprocal) >> 32);
#endif
    // 商は正しい商以下で、差は最大1である。そのため、一回の補正だけで余りが定まる。
    word_t remainder = value - quotient * modulus.value;
    return remainder >= modulus.value ? remainder - modulus.value : remainder;
}

/**
 * @brief 二語の整数と2のべき乗の積を、対称な余りへ変換する。
 * @param[in] mantissa 整数化で残る仮数の絶対値。
 * @param[in] negative 入力の符号。
 * @param[in] power 2のべき乗をmodulusで割った余り。
 * @param[in] modulus 256以下の法と、その法から求めた係数。
 * @return 符号付きINT8に収まる余り。
 */
inline int signed_residue(fp64_bits_s mantissa, bool negative, word_t power, modulus_s modulus) {
    word_t remainder = word_mod(word_mod(mantissa.high, modulus) * modulus.word_weight + word_mod(mantissa.low, modulus), modulus);
    remainder = word_mod(remainder * power, modulus);
    if (negative && remainder != 0) remainder = modulus.value - remainder;
    return remainder >= (modulus.value + 1) / 2 ? int(remainder) - int(modulus.value) : int(remainder);
}

/**
 * @brief 固定長整数を比較する。
 * @param[in] left 左辺。
 * @param[in] right 右辺。
 * @param[in] limbs 有効な桁数。
 * @return 左辺が小さい場合は-1、等しい場合は0、大きい場合は1。
 */
inline int big_compare(big_uint_input_t left, big_uint_input_t right, word_t limbs) {
    for (int i = int(limbs) - 1; i >= 0; --i) {
        if (left.digits[i] != right.digits[i]) return left.digits[i] < right.digits[i] ? -1 : 1;
    }
    return 0;
}

/**
 * @brief 非負の差を求める。
 * @param[in] left 被減数。
 * @param[in] right 減数。
 * @param[in] limbs 有効な桁数。
 * @return leftからrightを引いた値。
 * @pre leftがright以上であること。
 */
inline big_uint_s big_subtract(big_uint_input_t left, big_uint_input_t right, word_t limbs) {
    big_uint_s result = {};
    word_t borrow = 0;
    for (word_t i = 0; i < limbs; ++i) {
        word_t subtrahend = right.digits[i] + borrow;
        result.digits[i] = (left.digits[i] - subtrahend) & DIGIT_MASK;
        borrow = left.digits[i] < subtrahend;
    }
    return result;
}

/**
 * @brief 固定長整数に別の整数と小整数の積を加える。
 * @param[in,out] value 加算先。
 * @param[in] basis 乗算する整数。
 * @param[in] multiplier 255以下の乗数。
 * @param[in] limbs 有効な桁数。
 * @pre 結果がlimbs桁に収まること。
 */
inline void big_add_scaled(big_uint_output_t value, big_uint_input_t basis, word_t multiplier, word_t limbs) {
    word_t carry = 0;
    for (word_t i = 0; i < limbs; ++i) {
        word_t total = value.digits[i] + basis.digits[i] * multiplier + carry;
        value.digits[i] = total & DIGIT_MASK;
        carry = total >> DIGIT_BITS;
    }
}

/**
 * @brief 固定長整数を小整数で割った余りを求める。
 * @param[in] value 非負整数。
 * @param[in] modulus 256以下の法と、その法から求めた係数。
 * @param[in] limbs 有効な桁数。
 * @return 0以上modulus.value未満の余り。
 */
inline word_t big_mod(big_uint_input_t value, modulus_s modulus, word_t limbs) {
    word_t remainder = 0;
    for (int i = int(limbs) - 1; i >= 0; --i) {
        remainder = word_mod((remainder << DIGIT_BITS) | value.digits[i], modulus);
    }
    return remainder;
}

/**
 * @brief 一つの法の余りをCRTへ取り込む。
 * @param[in,out] value それ以前の法について復元済みの整数。取り込んだ法にも一致する整数へ更新する。
 * @param[in] prefix それ以前の法の積。
 * @param[in] modulus 取り込む法。
 * @param[in] inverse prefixの逆元。
 * @param[in] residue 0以上modulus.value未満の余り。
 * @param[in] limbs 有効な桁数。
 */
inline void crt_step(big_uint_output_t value, big_uint_input_t prefix, modulus_s modulus,
                     word_t inverse, word_t residue, word_t limbs) {
    word_t difference = word_mod(residue + modulus.value - big_mod(value, modulus, limbs), modulus);
    big_add_scaled(value, prefix, word_mod(difference * inverse, modulus), limbs);
}

/**
 * @brief 固定長整数が必要とするビット数を求める。
 * @param[in] value 非負整数。
 * @param[in] limbs 有効な桁数。
 * @return ゼロでは0。それ以外では最上位ビットの位置に1を加えた値。
 */
inline word_t big_bit_length(big_uint_input_t value, word_t limbs) {
    for (int i = int(limbs) - 1; i >= 0; --i) {
        if (value.digits[i] != 0) return word_t(i) * DIGIT_BITS + word_bit_length(value.digits[i]);
    }
    return 0;
}

/**
 * @brief 任意のビット位置から32ビットを取り出す。
 * @param[in] value 非負整数。
 * @param[in] start 最下位ビットの位置。
 * @param[in] limbs 有効な桁数。
 * @return 範囲外をゼロとした32ビット。
 */
inline word_t big_window(big_uint_input_t value, word_t start, word_t limbs) {
    word_t digit = start / DIGIT_BITS, offset = start % DIGIT_BITS;
    if (digit >= limbs) return 0;
    word_t result = value.digits[digit] >> offset;
    if (digit + 1 < limbs) result |= value.digits[digit + 1] << (24 - offset);
    if (offset > 16 && digit + 2 < limbs) result |= value.digits[digit + 2] << (48 - offset);
    return result;
}

/**
 * @brief 指定位置より下位に非ゼロのビットがあるかを調べる。
 * @param[in] value 非負整数。
 * @param[in] end 調べる範囲の上端。この位置のビットは含めない。
 * @param[in] limbs 有効な桁数。
 * @return 範囲内に非ゼロのビットがあればtrue。
 */
inline bool big_any_below(big_uint_input_t value, word_t end, word_t limbs) {
    word_t whole = end / DIGIT_BITS, tail = end % DIGIT_BITS;
    for (word_t i = 0; i < limbs && i < whole; ++i) {
        if (value.digits[i] != 0) return true;
    }
    return whole < limbs && (value.digits[whole] & ((1u << tail) - 1u)) != 0;
}

/**
 * @brief 整数と2のべき乗の積をFP64のビット列へ直接丸める。
 * @param[in] magnitude 絶対値を表す整数。
 * @param[in] negative 符号。
 * @param[in] scale 2のべき乗の指数。
 * @param[in] limbs 有効な桁数。
 * @return README.md「計算の定義」に従うFP64のビット列。
 */
inline fp64_bits_s pack_fp64(big_uint_input_t magnitude, bool negative, int scale, word_t limbs) {
    word_t length = big_bit_length(magnitude, limbs);
    if (length == 0) return {0, 0};
    word_t sign = negative ? 0x80000000u : 0;
    int exponent = int(length) - 1 + scale;
    if (exponent > 1023) return {0, sign | 0x7ff00000u};
    bool subnormal = exponent < -1022;
    int shift = subnormal ? -1074 - scale : int(length) - 53;
    fp64_bits_s mantissa;
    if (shift >= 0) {
        mantissa = {big_window(magnitude, word_t(shift), limbs), big_window(magnitude, word_t(shift) + 32, limbs)};
    } else {
        mantissa = shift_left({big_window(magnitude, 0, limbs), big_window(magnitude, 32, limbs)}, word_t(-shift));
    }
    if (shift > 0) {
        bool guard = (big_window(magnitude, word_t(shift - 1), limbs) & 1u) != 0;
        bool sticky = big_any_below(magnitude, word_t(shift - 1), limbs);
        if (guard && (sticky || (mantissa.low & 1u))) {
            ++mantissa.low;
            if (mantissa.low == 0) ++mantissa.high;
        }
    }
    if (subnormal) return {mantissa.low, sign | mantissa.high};
    if (mantissa.high & 0x200000u) {
        mantissa = shift_right(mantissa, 1);
        ++exponent;
    }
    if (exponent > 1023) return {0, sign | 0x7ff00000u};
    return {mantissa.low, sign | (word_t(exponent + 1023) << 20) | (mantissa.high & 0xfffffu)};
}

} // namespace apple_fp64
