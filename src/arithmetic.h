#ifndef FP64_ARITHMETIC_H
#define FP64_ARITHMETIC_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdbool.h>
#include <stdint.h>
#endif

#ifdef __METAL_VERSION__
typedef metal::uint fp64_word_t; /**< @~japanese 算術で使用する32ビット符号なし整数。
                                  * @~english Unsigned 32-bit integer used for arithmetic.
                                  * @~
                                  */
#define FP64_CONSTANT constant constexpr /**< @~japanese Metalの定数用のアドレス空間。
                                          * @~english Metal address space for constants.
                                          * @~
                                          */
#else
typedef uint32_t fp64_word_t; /**< @~japanese 算術で使用する32ビット符号なし整数。
                               * @~english Unsigned 32-bit integer used for arithmetic.
                               * @~
                               */
#define FP64_CONSTANT static const /**< @~japanese CPUでは各翻訳単位に定数を保持する。
                                    * @~english Keep CPU constants in each translation unit.
                                    * @~
                                    */
#endif

#define FP64_DIGIT_BITS (18u) /**< @~japanese CRT係数と余りの積の総和が32ビットに収まる桁幅。
                               * @~english Digit width that keeps the sum of CRT coefficient-residue products
                               * within 32 bits.
                               * @~
                               */
#define FP64_DIGIT_MASK ((1u << FP64_DIGIT_BITS) - 1u) /**< @~japanese 一桁の値域を取り出すマスク。
                                                        * @~english Mask that extracts the value of one digit.
                                                        * @~
                                                        */
#define FP64_MAX_LIMBS (20u) /**< @~japanese CRT係数の総和と法の積の倍数を格納する桁数。
                              * @~english Number of digits storing the CRT coefficient sum and multiples of
                              * the modulus product.
                              * @~
                              */
FP64_CONSTANT fp64_word_t FP64_CRT_MODULI[] = { /**< @~japanese INT8で対称な余りを表せる互いに素な法。
                                                 * @~english Pairwise coprime moduli whose symmetric residues
                                                 * fit in INT8.
                                                 * @~
                                                 */
    256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199,
    197, 193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131,
    127, 113, 109, 107, 103, 101, 97, 89, 83, 79, 73, 71, 67, 61, 59,
    53, 47, 43, 41, 37, 29
};
#define FP64_MAX_MODULI (sizeof(FP64_CRT_MODULI) / sizeof(FP64_CRT_MODULI[0])) /**< @~japanese 法の候補数。
                                                                                * @~english Number of candidate moduli.
                                                                                * @~
                                                                                */
#define FP64_ZERO_EXPONENT (-2147483647) /**< @~japanese 最大値の取得でゼロを除外する値。
                                          * @~english Value that excludes zero from the maximum calculation.
                                          * @~
                                          */
#define FP64_K_CHUNK (1024u) /**< @~japanese INT8の積の絶対値の総和を2の24乗以下に保つ項数。
                              * @~english Number of terms that keeps the sum of absolute INT8 products at most
                              * 2 to the power of 24.
                              * @~
                              */
#define FP64_K_ACCUMULATE (65536u) /**< @~japanese 余りと部分内積の合計が符号付き32ビットに収まる項数。
                                    * @~english Number of terms that keeps the sum of a residue and a partial
                                    * dot product within signed 32 bits.
                                    * @~
                                    */
#define FP64_TILE_ROWS (64u) /**< @~japanese 行列積のスレッドグループが担当する行数。
                              * @~english Number of rows assigned to each matrix multiplication threadgroup.
                              * @~
                              */
#define FP64_TILE_COLUMNS (64u) /**< @~japanese 行列積のスレッドグループが担当する列数。
                                 * @~english Number of columns assigned to each matrix multiplication
                                 * threadgroup.
                                 * @~
                                 */
#define FP64_SIMD_GROUPS (4u) /**< @~japanese 行列積を協調実行するSIMDグループ数。
                               * @~english Number of SIMD groups cooperating on matrix multiplication.
                               * @~
                               */
#define FP64_COLUMN_SIMD_GROUPS (4u) /**< @~japanese 同じ列の指数取得を分担するSIMDグループ数。
                                      * @~english Number of SIMD groups sharing the exponent analysis of each
                                      * column.
                                      * @~
                                      */

/**
 * @~japanese
 * @brief CPUとGPUで共有する、復元カーネルの名前と整数の桁数。
 * @param[in] apply 名前と桁数を受け取る宣言用のマクロ。
 * @~english
 * @brief Reconstruction kernel names and integer digit counts shared by the CPU and GPU.
 * @param[in] apply Declaration macro receiving a name and digit count.
 * @~
 */
#define FP64_RECONSTRUCTION_KERNELS(apply) \
    apply(small, 4u) \
    apply(medium, 8u) \
    apply(large, 12u) \
    apply(full, FP64_MAX_LIMBS)

/**
 * @~japanese
 * @brief 復元カーネルの配置から、整数の桁数を取り出す。
 * @param[in] name カーネルの名前。
 * @param[in] size 整数の桁数。
 * @~english
 * @brief Extract integer digit counts from the reconstruction kernel layout.
 * @param[in] name Kernel name.
 * @param[in] size Number of integer digits.
 * @~
 */
#define FP64_RECONSTRUCTION_SIZE(name, size) size,
FP64_CONSTANT fp64_word_t FP64_RECONSTRUCTION_CAPACITIES[] = { /**< @~japanese 復元カーネルごとの整数の桁数。昇順で並ぶ。
                                                                * @~english Integer digit counts for the
                                                                * reconstruction kernels, in ascending order.
                                                                * @~
                                                                */
    FP64_RECONSTRUCTION_KERNELS(FP64_RECONSTRUCTION_SIZE)
};
#undef FP64_RECONSTRUCTION_SIZE
#define FP64_RECONSTRUCTION_COUNT (sizeof(FP64_RECONSTRUCTION_CAPACITIES) / sizeof(FP64_RECONSTRUCTION_CAPACITIES[0])) /**< @~japanese 復元カーネルの個数。
                                                                                                                        * @~english Number of reconstruction
                                                                                                                        * kernels.
                                                                                                                        * @~
                                                                                                                        */

#ifndef __METAL_VERSION__
/**
 * @~japanese
 * @brief 復元カーネルの配置から、Metalで使用する名前を取り出す。
 * @param[in] name カーネルの名前。
 * @param[in] size 整数の桁数。
 * @~english
 * @brief Extract the names used by Metal from the reconstruction kernel layout.
 * @param[in] name Kernel name.
 * @param[in] size Number of integer digits.
 * @~
 */
#define FP64_RECONSTRUCTION_NAME(name, size) "reconstruct_" #name,
static const char *const FP64_RECONSTRUCTION_NAMES[] = { /**< @~japanese Metalライブラリーから取得する復元カーネルの名前。
                                                          * @~english Reconstruction kernel names to retrieve
                                                          * from the Metal library.
                                                          * @~
                                                          */
    FP64_RECONSTRUCTION_KERNELS(FP64_RECONSTRUCTION_NAME)
};
#undef FP64_RECONSTRUCTION_NAME
#endif

#undef FP64_CONSTANT

/**
 * @~japanese
 * @brief FP64形式のビット列を、下位と上位に分けて保持する。
 * @~english
 * @brief FP64 bit pattern stored as low and high words.
 * @~
 */
typedef struct fp64_bits_s {
    fp64_word_t low; /**< @~japanese 下位32ビット。
                      * @~english Low 32 bits.
                      * @~
                      */
    fp64_word_t high; /**< @~japanese 符号と指数を含む上位32ビット。
                       * @~english High 32 bits, including the sign and exponent.
                       * @~
                       */
} fp64_bits_t;

/**
 * @~japanese
 * @brief 行または列に含まれる有限値の指数と、非有限値の有無。
 * @~english
 * @brief Exponents of finite values in a row or column and the presence of nonfinite values.
 * @~
 */
typedef struct fp64_scale_s {
    int exponent; /**< @~japanese 最大値を囲む2のべき乗の指数。
                   * @~english Exponent of the power of two bounding the maximum value.
                   * @~
                   */
    int lowest_exponent; /**< @~japanese 非ゼロの最下位ビットの最小指数。非ゼロの有限値がない場合はINT_MAX。
                          * @~english Minimum exponent of the least significant nonzero bit. INT_MAX if there
                          * are no nonzero finite values.
                          * @~
                          */
    fp64_word_t nonfinite; /**< @~japanese NaNまたは無限大を含む場合は1。
                            * @~english One if a NaN or infinity is present.
                            * @~
                            */
} fp64_scale_t;

/**
 * @~japanese
 * @brief GPUが求めた、損失のない整数化に必要な幅と非有限値の有無。
 * @~english
 * @brief GPU analysis of the widths needed for lossless integer conversion and the presence of nonfinite
 * values.
 * @~
 */
typedef struct fp64_input_analysis_s {
    fp64_word_t precision_a; /**< @~japanese Aの各行を損失なく整数化できる幅の最大値。
                              * @~english Maximum width needed for lossless integer conversion of each row of
                              * A.
                              * @~
                              */
    fp64_word_t precision_b; /**< @~japanese Bの各列を損失なく整数化できる幅の最大値。
                              * @~english Maximum width needed for lossless integer conversion of each column
                              * of B.
                              * @~
                              */
    fp64_word_t nonfinite; /**< @~japanese いずれかの入力がNaNまたは無限大を含む場合は1。
                            * @~english One if either input contains a NaN or infinity.
                            * @~
                            */
} fp64_input_analysis_t;

/**
 * @~japanese
 * @brief FP64_DIGIT_BITSの桁幅で表す、非負の固定長整数。
 * @~english
 * @brief Nonnegative fixed-length integer with FP64_DIGIT_BITS bits per digit.
 * @~
 */
#ifdef __METAL_VERSION__
template <unsigned limb_count>
struct fp64_big_uint_s {
    fp64_word_t digits[limb_count]; /**< @~japanese 下位桁から並ぶ値。各桁はFP64_DIGIT_MASK以下。
                                     * @~english Digits ordered from least to most significant. Each digit is
                                     * at most FP64_DIGIT_MASK.
                                     * @~
                                     */
};
typedef fp64_big_uint_s<FP64_MAX_LIMBS> fp64_big_uint_t; /**< @~japanese CPUと同じ配置を持つ復元係数の整数。
                                                          * @~english Reconstruction coefficient integer with
                                                          * the same layout as on the CPU.
                                                          * @~
                                                          */
#define FP64_BIG_FUNCTION template <unsigned limb_count> /**< @~japanese GPUでは配列の容量ごとに算術をコンパイルする。
                                                          * @~english Compile GPU arithmetic for each array
                                                          * capacity.
                                                          * @~
                                                          */
#define FP64_BIG_VALUE fp64_big_uint_s<limb_count> /**< @~japanese GPUのスレッドが保持する整数。
                                                    * @~english Integer held by a GPU thread.
                                                    * @~
                                                    */
#else
typedef struct fp64_big_uint_s {
    fp64_word_t digits[FP64_MAX_LIMBS]; /**< @~japanese 下位桁から並ぶ値。各桁はFP64_DIGIT_MASK以下。
                                         * @~english Digits ordered from least to most significant. Each digit
                                         * is at most FP64_DIGIT_MASK.
                                         * @~
                                         */
} fp64_big_uint_t;
#define FP64_BIG_FUNCTION /**< @~japanese CPUでは復元係数の最大容量を使用する。
                           * @~english Use the maximum reconstruction coefficient capacity on the CPU.
                           * @~
                           */
#define FP64_BIG_VALUE fp64_big_uint_t /**< @~japanese CPUが保持する整数。
                                        * @~english Integer held by the CPU.
                                        * @~
                                        */
#endif

#ifdef __METAL_VERSION__
typedef thread fp64_big_uint_t *fp64_big_uint_output_t; /**< @~japanese GPUのスレッドが更新する整数。
                                                         * @~english Integer updated by a GPU thread.
                                                         * @~
                                                         */
#else
typedef fp64_big_uint_t *fp64_big_uint_output_t; /**< @~japanese CPUの処理が更新する整数。
                                                  * @~english Integer updated by CPU code.
                                                  * @~
                                                  */
#endif

/**
 * @~japanese
 * @brief 一つの法と、32ビット整数の余りを求める係数。
 * @~english
 * @brief One modulus and coefficients for reducing 32-bit integers.
 * @~
 */
typedef struct fp64_modulus_s {
    fp64_word_t value; /**< @~japanese 256以下の法。
                        * @~english Modulus at most 256.
                        * @~
                        */
    fp64_word_t reciprocal; /**< @~japanese 2の32乗をvalueで割った商。
                             * @~english Quotient of 2 to the power of 32 divided by value.
                             * @~
                             */
    fp64_word_t word_weight; /**< @~japanese 2の32乗をvalueで割った余り。
                              * @~english Remainder of 2 to the power of 32 divided by value.
                              * @~
                              */
} fp64_modulus_t;

/**
 * @~japanese
 * @brief 入力の値に依存しないCRTの係数。
 * @~english
 * @brief CRT coefficients independent of input values.
 * @~
 */
typedef struct fp64_crt_plan_s {
    fp64_word_t count; /**< @~japanese 使用する法の個数。
                        * @~english Number of moduli used.
                        * @~
                        */
    fp64_word_t limbs; /**< @~japanese CRT係数の総和を格納する桁数。
                        * @~english Number of digits storing the sum of CRT coefficients.
                        * @~
                        */
    fp64_modulus_t moduli[FP64_MAX_MODULI]; /**< @~japanese 使用順に並ぶ法と余りの計算に使う係数。
                                             * @~english Moduli and reduction coefficients in their order of
                                             * use.
                                             * @~
                                             */
    fp64_big_uint_t coefficients[FP64_MAX_MODULI]; /**< @~japanese 各法に対応するCRT係数。
                                                    * @~english CRT coefficient for each modulus.
                                                    * @~
                                                    */
    float ratios[FP64_MAX_MODULI]; /**< @~japanese 各CRT係数を法の積で割った比。候補の選択にだけ使う。
                                    * @~english Ratio of each CRT coefficient to the modulus product. Used
                                    * only to select a candidate.
                                    * @~
                                    */
    fp64_big_uint_t product; /**< @~japanese 使用する法の積。
                              * @~english Product of the moduli used.
                              * @~
                              */
    fp64_big_uint_t half_product; /**< @~japanese 使用する法の積の半分。
                                   * @~english Half the product of the moduli used.
                                   * @~
                                   */
    unsigned short powers[FP64_MAX_MODULI][FP64_MAX_LIMBS * FP64_DIGIT_BITS]; /**< @~japanese 下位8ビットは2のべき乗の余り、上位8ビットは32ビット高いべき乗の余り。
                                                                               * @~english Low 8 bits store a
                                                                               * power-of-two residue; high 8 bits store
                                                                               * the residue for a power 32 bits higher.
                                                                               * @~
                                                                               */
} fp64_crt_plan_t;

/**
 * @~japanese
 * @brief 一つの行のまとまりをGPUに渡す寸法と整数幅。
 * @~english
 * @brief Dimensions and integer widths passed to the GPU for one row batch.
 * @~
 */
typedef struct fp64_batch_parameters_s {
    fp64_word_t rows; /**< @~japanese 行のまとまりに含まれる行数。
                       * @~english Number of rows in the batch.
                       * @~
                       */
    fp64_word_t columns; /**< @~japanese 出力の列数。
                          * @~english Number of output columns.
                          * @~
                          */
    fp64_word_t inner; /**< @~japanese 内積の項数。
                        * @~english Number of terms in each dot product.
                        * @~
                        */
    fp64_word_t row_begin; /**< @~japanese 入力Aにおける先頭行。
                            * @~english First row in input A.
                            * @~
                            */
    fp64_word_t precision_a; /**< @~japanese Aの整数幅。
                              * @~english Integer width of A.
                              * @~
                              */
    fp64_word_t precision_b; /**< @~japanese Bの整数幅。
                              * @~english Integer width of B.
                              * @~
                              */
    fp64_word_t total_rows; /**< @~japanese 入力Aの行数。上下のブロックの対応に使う。
                             * @~english Number of rows in input A. Used to map the top and bottom blocks.
                             * @~
                             */
    fp64_word_t strassen; /**< @~japanese Strassen法でブロックを処理する場合は1、それ以外は0。
                           * @~english One when processing blocks with Strassen multiplication, otherwise
                           * zero.
                           * @~
                           */
    fp64_word_t shifted; /**< @~japanese 内積方向の指数を調整する場合は1、それ以外は0。
                          * @~english One when adjusting exponents along the inner dimension, otherwise zero.
                          * @~
                          */
} fp64_batch_parameters_t;

/**
 * @~japanese
 * @brief 非負の32ビット整数が必要とするビット数を求める。
 * @param[in] value 対象の整数。
 * @return ゼロの場合は0。それ以外は最上位ビットの位置に1を加えた値。
 * @~english
 * @brief Compute the bit length of a nonnegative 32-bit integer.
 * @param[in] value Integer to examine.
 * @return Zero for a zero value. Otherwise, one plus the position of the most significant bit.
 * @~
 */
static inline fp64_word_t fp64_word_bit_length(fp64_word_t value) {
#ifdef __METAL_VERSION__
    return 32u - metal::clz(value);
#else
    return value == 0 ? 0u : 32u - (fp64_word_t)__builtin_clz(value);
#endif
}

/**
 * @~japanese
 * @brief 有限のFP64値の絶対値を囲む2のべき乗の指数を求める。
 * @param[in] bits 有限のFP64値のビット列。
 * @return 非ゼロ値ではREADME.md「計算の定義」の指数。ゼロではFP64_ZERO_EXPONENT。
 * @pre NaNと無限大を渡してはならない。
 * @~english
 * @brief Compute the exponent of the power of two bounding the magnitude of a finite FP64 value.
 * @param[in] bits Bit pattern of a finite FP64 value.
 * @return For a nonzero value, the exponent defined in README.en.md, section "Numerical semantics".
 * FP64_ZERO_EXPONENT for zero.
 * @pre NaNs and infinities must not be supplied.
 * @~
 */
static inline int fp64_exponent(fp64_bits_t bits) {
    fp64_word_t exponent = (bits.high >> 20) & 2047u;
    if (exponent != 0) return (int)(exponent) - 1022;
    fp64_word_t high = bits.high & 0xfffffu;
    if (high != 0) return (int)(32 + fp64_word_bit_length(high)) - 1074;
    if (bits.low != 0) return (int)(fp64_word_bit_length(bits.low)) - 1074;
    return FP64_ZERO_EXPONENT;
}

/**
 * @~japanese
 * @brief 二語の非負整数を右へずらす。
 * @param[in] value 下位、上位の順で保持する整数。
 * @param[in] shift ずらすビット数。
 * @return 切り捨てた整数。64ビット以上ずらす場合はゼロ。
 * @~english
 * @brief Shift a nonnegative two-word integer to the right.
 * @param[in] value Integer stored as the low word followed by the high word.
 * @param[in] shift Number of bits to shift.
 * @return Truncated integer. Zero when shifting by at least 64 bits.
 * @~
 */
static inline fp64_bits_t fp64_shift_right(fp64_bits_t value, fp64_word_t shift) {
    if (shift >= 64) return (fp64_bits_t){0, 0};
    if (shift >= 32) return (fp64_bits_t){value.high >> (shift - 32), 0};
    if (shift == 0) return value;
    return (fp64_bits_t){(value.low >> shift) | (value.high << (32 - shift)), value.high >> shift};
}

/**
 * @~japanese
 * @brief 二語の非負整数を左へずらす。
 * @param[in] value 下位、上位の順で保持する整数。
 * @param[in] shift ずらすビット数。64未満。
 * @return 64ビット以内の結果。
 * @pre 結果が64ビットに収まること。
 * @~english
 * @brief Shift a nonnegative two-word integer to the left.
 * @param[in] value Integer stored as the low word followed by the high word.
 * @param[in] shift Number of bits to shift. Must be less than 64.
 * @return Result within 64 bits.
 * @pre The result must fit in 64 bits.
 * @~
 */
static inline fp64_bits_t fp64_shift_left(fp64_bits_t value, fp64_word_t shift) {
    if (shift >= 32) return (fp64_bits_t){0, value.low << (shift - 32)};
    if (shift == 0) return value;
    return (fp64_bits_t){value.low << shift, (value.high << shift) | (value.low >> (32 - shift))};
}

/**
 * @~japanese
 * @brief 32ビット整数の余りを、乗算の上位ビットから求める。
 * @param[in] value 非負の32ビット整数。
 * @param[in] modulus 法と、その法から求めた係数。
 * @return 0以上modulus.value未満の余り。
 * @~english
 * @brief Reduce a 32-bit integer using the high bits of a multiplication.
 * @param[in] value Nonnegative 32-bit integer.
 * @param[in] modulus Modulus and coefficients derived from it.
 * @return Remainder greater than or equal to zero and less than modulus.value.
 * @~
 */
static inline fp64_word_t fp64_word_mod(fp64_word_t value, fp64_modulus_t modulus) {
#ifdef __METAL_VERSION__
    fp64_word_t quotient = metal::mulhi(value, modulus.reciprocal);
#else
    fp64_word_t quotient = (fp64_word_t)(((uint64_t)(value) * modulus.reciprocal) >> 32);
#endif
    // 商は正しい商以下で、差は最大1である。そのため、一回の補正だけで余りが定まる。
    fp64_word_t remainder = value - quotient * modulus.value;
    return remainder >= modulus.value ? remainder - modulus.value : remainder;
}

/**
 * @~japanese
 * @brief 二語の整数と2のべき乗の積を、対称な余りへ変換する。
 * @param[in] mantissa 整数化で残る仮数の絶対値。
 * @param[in] negative 入力の符号。
 * @param[in] powers 仮数の上位と下位に掛ける係数。fp64_crt_plan_tのpowersと同じ配置。
 * @param[in] modulus 256以下の法と、その法から求めた係数。
 * @return 符号付きINT8に収まる余り。
 * @pre mantissaの上位は21ビット以下であること。
 * @~english
 * @brief Convert the product of a two-word integer and a power of two to a symmetric residue.
 * @param[in] mantissa Magnitude of the significand retained by integer conversion.
 * @param[in] negative Input sign.
 * @param[in] powers Coefficients multiplying the high and low significand words. Same layout as powers in
 * fp64_crt_plan_t.
 * @param[in] modulus Modulus at most 256 and coefficients derived from it.
 * @return Residue that fits in signed INT8.
 * @pre The high word of mantissa must use at most 21 bits.
 * @~
 */
static inline int fp64_signed_residue(fp64_bits_t mantissa, bool negative, fp64_word_t powers, fp64_modulus_t modulus) {
    // 仮数の上位は21ビット以下であるため、係数との積と下位の余りの和は32ビットに収まる。
    fp64_word_t remainder = fp64_word_mod(mantissa.high * (powers >> 8)
                                         + fp64_word_mod(mantissa.low, modulus) * (powers & 255u), modulus);
    if (negative && remainder != 0) remainder = modulus.value - remainder;
    return remainder >= (modulus.value + 1) / 2 ? (int)(remainder) - (int)(modulus.value) : (int)(remainder);
}

/**
 * @~japanese
 * @brief 固定長整数を比較する。
 * @param[in] left 左辺。
 * @param[in] right 右辺。
 * @param[in] limbs 有効な桁数。
 * @return 左辺が小さい場合は-1、等しい場合は0、大きい場合は1。
 * @~english
 * @brief Compare fixed-length integers.
 * @param[in] left Left operand.
 * @param[in] right Right operand.
 * @param[in] limbs Number of active digits.
 * @return Minus one if the left operand is smaller, zero if equal, and one if greater.
 * @~
 */
FP64_BIG_FUNCTION
static inline int fp64_big_compare(FP64_BIG_VALUE left, FP64_BIG_VALUE right, fp64_word_t limbs) {
    for (int i = (int)(limbs) - 1; i >= 0; --i) {
        if (left.digits[i] != right.digits[i]) return left.digits[i] < right.digits[i] ? -1 : 1;
    }
    return 0;
}

/**
 * @~japanese
 * @brief 非負の差を求める。
 * @param[in] left 被減数。
 * @param[in] right 減数。
 * @param[in] limbs 有効な桁数。
 * @return leftからrightを引いた値。
 * @pre leftがright以上であること。
 * @~english
 * @brief Compute a nonnegative difference.
 * @param[in] left Minuend.
 * @param[in] right Subtrahend.
 * @param[in] limbs Number of active digits.
 * @return Value of left minus right.
 * @pre left must be greater than or equal to right.
 * @~
 */
FP64_BIG_FUNCTION
static inline FP64_BIG_VALUE fp64_big_subtract(FP64_BIG_VALUE left, FP64_BIG_VALUE right, fp64_word_t limbs) {
    FP64_BIG_VALUE result = {0};
    fp64_word_t borrow = 0;
    for (fp64_word_t i = 0; i < limbs; ++i) {
        fp64_word_t subtrahend = right.digits[i] + borrow;
        result.digits[i] = (left.digits[i] - subtrahend) & FP64_DIGIT_MASK;
        borrow = left.digits[i] < subtrahend;
    }
    return result;
}

/**
 * @~japanese
 * @brief 固定長整数に別の整数と小整数の積を加える。
 * @param[in,out] value 加算先。
 * @param[in] basis 乗算する整数。
 * @param[in] multiplier 255以下の乗数。
 * @param[in] limbs 有効な桁数。
 * @pre 結果がlimbs桁に収まること。
 * @~english
 * @brief Add the product of another integer and a small integer to a fixed-length integer.
 * @param[in,out] value Destination of the addition.
 * @param[in] basis Integer to multiply.
 * @param[in] multiplier Multiplier at most 255.
 * @param[in] limbs Number of active digits.
 * @pre The result must fit in limbs digits.
 * @~
 */
static inline void fp64_big_add_scaled(fp64_big_uint_output_t value, fp64_big_uint_t basis, fp64_word_t multiplier, fp64_word_t limbs) {
    fp64_word_t carry = 0;
    for (fp64_word_t i = 0; i < limbs; ++i) {
        fp64_word_t total = value->digits[i] + basis.digits[i] * multiplier + carry;
        value->digits[i] = total & FP64_DIGIT_MASK;
        carry = total >> FP64_DIGIT_BITS;
    }
}

/**
 * @~japanese
 * @brief 固定長整数を小整数で割った余りを求める。
 * @param[in] value 非負整数。
 * @param[in] modulus 256以下の法と、その法から求めた係数。
 * @param[in] limbs 有効な桁数。
 * @return 0以上modulus.value未満の余り。
 * @~english
 * @brief Compute the remainder of a fixed-length integer divided by a small integer.
 * @param[in] value Nonnegative integer.
 * @param[in] modulus Modulus at most 256 and coefficients derived from it.
 * @param[in] limbs Number of active digits.
 * @return Remainder greater than or equal to zero and less than modulus.value.
 * @~
 */
static inline fp64_word_t fp64_big_mod(fp64_big_uint_t value, fp64_modulus_t modulus, fp64_word_t limbs) {
    fp64_word_t remainder = 0;
    for (int i = (int)(limbs) - 1; i >= 0; --i) {
        remainder = fp64_word_mod((remainder << FP64_DIGIT_BITS) | value.digits[i], modulus);
    }
    return remainder;
}

/**
 * @~japanese
 * @brief 固定長整数が必要とするビット数を求める。
 * @param[in] value 非負整数。
 * @param[in] limbs 有効な桁数。
 * @return ゼロでは0。それ以外では最上位ビットの位置に1を加えた値。
 * @~english
 * @brief Compute the bit length of a fixed-length integer.
 * @param[in] value Nonnegative integer.
 * @param[in] limbs Number of active digits.
 * @return Zero for a zero value. Otherwise, one plus the position of the most significant bit.
 * @~
 */
FP64_BIG_FUNCTION
static inline fp64_word_t fp64_big_bit_length(FP64_BIG_VALUE value, fp64_word_t limbs) {
    for (int i = (int)(limbs) - 1; i >= 0; --i) {
        if (value.digits[i] != 0) return (fp64_word_t)(i) * FP64_DIGIT_BITS + fp64_word_bit_length(value.digits[i]);
    }
    return 0;
}

/**
 * @~japanese
 * @brief 任意のビット位置から32ビットを取り出す。
 * @param[in] value 非負整数。
 * @param[in] start 最下位ビットの位置。
 * @param[in] limbs 有効な桁数。
 * @return 範囲外をゼロとした32ビット。
 * @~english
 * @brief Extract 32 bits starting at an arbitrary bit position.
 * @param[in] value Nonnegative integer.
 * @param[in] start Position of the least significant bit to extract.
 * @param[in] limbs Number of active digits.
 * @return 32 bits with positions outside the range treated as zero.
 * @~
 */
FP64_BIG_FUNCTION
static inline fp64_word_t fp64_big_window(FP64_BIG_VALUE value, fp64_word_t start, fp64_word_t limbs) {
    fp64_word_t digit = start / FP64_DIGIT_BITS, offset = start % FP64_DIGIT_BITS;
    if (digit >= limbs) return 0;
    fp64_word_t result = value.digits[digit] >> offset;
    if (digit + 1 < limbs) result |= value.digits[digit + 1] << (FP64_DIGIT_BITS - offset);
    if (offset > 2 * FP64_DIGIT_BITS - 32 && digit + 2 < limbs)
        result |= value.digits[digit + 2] << (2 * FP64_DIGIT_BITS - offset);
    return result;
}

/**
 * @~japanese
 * @brief 指定位置より下位に非ゼロのビットがあるかを調べる。
 * @param[in] value 非負整数。
 * @param[in] end 調べる範囲の上端。この位置のビットは含めない。
 * @param[in] limbs 有効な桁数。
 * @return 範囲内に非ゼロのビットがあればtrue。
 * @~english
 * @brief Check for nonzero bits below the specified position.
 * @param[in] value Nonnegative integer.
 * @param[in] end Upper end of the range to examine. The bit at this position is excluded.
 * @param[in] limbs Number of active digits.
 * @return true if a nonzero bit exists within the range.
 * @~
 */
FP64_BIG_FUNCTION
static inline bool fp64_big_any_below(FP64_BIG_VALUE value, fp64_word_t end, fp64_word_t limbs) {
    fp64_word_t whole = end / FP64_DIGIT_BITS, tail = end % FP64_DIGIT_BITS;
    for (fp64_word_t i = 0; i < limbs && i < whole; ++i) {
        if (value.digits[i] != 0) return true;
    }
    return whole < limbs && (value.digits[whole] & ((1u << tail) - 1u)) != 0;
}

/**
 * @~japanese
 * @brief 整数と2のべき乗の積をFP64のビット列へ直接丸める。
 * @param[in] magnitude 絶対値を表す整数。
 * @param[in] negative 符号。
 * @param[in] scale 2のべき乗の指数。
 * @param[in] limbs 有効な桁数。
 * @return README.md「計算の定義」に従うFP64のビット列。
 * @~english
 * @brief Round the product of an integer and a power of two directly to an FP64 bit pattern.
 * @param[in] magnitude Integer representing the magnitude.
 * @param[in] negative Sign.
 * @param[in] scale Exponent of the power of two.
 * @param[in] limbs Number of active digits.
 * @return FP64 bit pattern following README.en.md, section "Numerical semantics".
 * @~
 */
FP64_BIG_FUNCTION
static inline fp64_bits_t fp64_pack_fp64(FP64_BIG_VALUE magnitude, bool negative, int scale, fp64_word_t limbs) {
    fp64_word_t length = fp64_big_bit_length(magnitude, limbs);
    if (length == 0) return (fp64_bits_t){0, 0};
    fp64_word_t sign = negative ? 0x80000000u : 0;
    int exponent = (int)(length) - 1 + scale;
    if (exponent > 1023) return (fp64_bits_t){0, sign | 0x7ff00000u};
    bool subnormal = exponent < -1022;
    int shift = subnormal ? -1074 - scale : (int)(length) - 53;
    fp64_bits_t mantissa;
    if (shift >= 0) {
        mantissa = (fp64_bits_t){fp64_big_window(magnitude, (fp64_word_t)(shift), limbs), fp64_big_window(magnitude, (fp64_word_t)(shift) + 32, limbs)};
    } else {
        mantissa = fp64_shift_left((fp64_bits_t){fp64_big_window(magnitude, 0, limbs), fp64_big_window(magnitude, 32, limbs)}, (fp64_word_t)(-shift));
    }
    if (shift > 0) {
        bool guard = (fp64_big_window(magnitude, (fp64_word_t)(shift - 1), limbs) & 1u) != 0;
        bool sticky = fp64_big_any_below(magnitude, (fp64_word_t)(shift - 1), limbs);
        if (guard && (sticky || (mantissa.low & 1u))) {
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

#undef FP64_BIG_FUNCTION
#undef FP64_BIG_VALUE

#endif
