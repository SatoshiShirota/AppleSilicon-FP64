#ifndef FP64_ARITHMETIC_H
#define FP64_ARITHMETIC_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdbool.h>
#include <stdint.h>
#endif

#ifdef __METAL_VERSION__
typedef metal::uint fp64_word_t; /**< @~japanese 多倍長の整数およびビット列の操作に使う32ビットの符号なし整数。
                                  * @~english Unsigned 32-bit integer used for multi-precision integers and bit
                                  * manipulation.
                                  * @~
                                  */
#define FP64_CONSTANT constant constexpr /**< @~japanese Metalの定数アドレス空間に定数を配置する修飾子。
                                          * @~english Qualifier placing constants in the Metal constant address
                                          * space.
                                          * @~
                                          */
#else
typedef uint32_t fp64_word_t; /**< @~japanese 多倍長の整数およびビット列の操作に使う32ビットの符号なし整数。
                               * @~english Unsigned 32-bit integer used for multi-precision integers and bit
                               * manipulation.
                               * @~
                               */
#define FP64_CONSTANT static const /**< @~japanese CPU側で各翻訳単位に定数を配置する修飾子。
                                    * @~english Qualifier keeping constants in each CPU translation unit.
                                    * @~
                                    */
#endif

#define FP64_DIGIT_BITS (18u) /**< @~japanese 多倍長整数の1桁あたりのビット数。CRTの係数と余りの積和が下位からの繰り上がりを含めて32ビットに収まる幅。
                               * @~english Number of bits per multi-precision integer digit, chosen so that the
                               * sum of CRT coefficient-residue products plus carry fits in 32 bits.
                               * @~
                               */
#define FP64_DIGIT_MASK ((1u << FP64_DIGIT_BITS) - 1u) /**< @~japanese 多倍長整数の下位1桁分（18ビット）を取り出すビットマスク。
                                                        * @~english Bit mask extracting one 18-bit digit of a
                                                        * multi-precision integer.
                                                        * @~
                                                        */
#define FP64_MAX_LIMBS (20u) /**< @~japanese 多倍長整数の最大桁数（20桁、合計360ビット）。CRTの係数の総和と法の積の倍数を格納する。
                              * @~english Maximum number of digits in a multi-precision integer (20 digits, 360
                              * bits total), sufficient to hold the CRT coefficient sum and multiples of the
                              * modulus product.
                              * @~
                              */
FP64_CONSTANT fp64_word_t FP64_CRT_MODULI[] = { /**< @~japanese CRTで使用する互いに素な法の候補。符号付きINT8で対称な余りを表せる256以下の整数を大きい順に並べる。
                                                 * @~english Candidate pairwise coprime moduli for CRT, ordered
                                                 * descending from 256 so that symmetric residues fit in signed
                                                 * INT8.
                                                 * @~
                                                 */
    256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199,
    197, 193, 191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131,
    127, 113, 109, 107, 103, 101, 97, 89, 83, 79, 73, 71, 67, 61, 59,
    53, 47, 43, 41, 37, 29
};
#define FP64_MAX_MODULI (sizeof(FP64_CRT_MODULI) / sizeof(FP64_CRT_MODULI[0])) /**< @~japanese CRTで使用できる法の最大個数。
                                                                                * @~english Maximum number of candidate moduli for CRT.
                                                                                * @~
                                                                                */
#define FP64_ZERO_EXPONENT (-2147483647) /**< @~japanese 最大の指数を求める集計で、値がゼロの要素を候補から除外するための最小値。
                                          * @~english Sentinel minimum value used to exclude zero elements when
                                          * finding the maximum exponent.
                                          * @~
                                          */
#define FP64_K_CHUNK (1024u) /**< @~japanese 1回のINT8行列積で計算する内積の最大項数。各項の絶対値の和が2の24乗以下に収まる項数。
                              * @~english Maximum number of dot-product terms per INT8 matrix multiplication,
                              * keeping the sum of absolute INT8 products at most 2 to the power of 24.
                              * @~
                              */
#define FP64_K_ACCUMULATE (65536u) /**< @~japanese INT32で部分内積を加算する間に、法で割った余りへ戻す周期の項数。符号付き32ビット整数のオーバーフローを防ぐ。
                                    * @~english Number of dot-product terms accumulated before reducing modulo
                                    * the modulus, preventing signed 32-bit integer overflow.
                                    * @~
                                    */
#define FP64_TILE_ROWS (64u) /**< @~japanese INT8の行列積で1つのスレッドグループが担当する出力タイルの行数。
                              * @~english Number of output tile rows assigned to each threadgroup in INT8
                              * matrix multiplication.
                              * @~
                              */
#define FP64_TILE_COLUMNS (64u) /**< @~japanese INT8の行列積で1つのスレッドグループが担当する出力タイルの列数。
                                 * @~english Number of output tile columns assigned to each threadgroup in INT8
                                 * matrix multiplication.
                                 * @~
                                 */
#define FP64_SIMD_GROUPS (4u) /**< @~japanese 1つの出力タイルの行列積を分担して実行するSIMDグループの個数。
                               * @~english Number of SIMD groups cooperating to compute one output tile in
                               * matrix multiplication.
                               * @~
                               */
#define FP64_COLUMN_SIMD_GROUPS (4u) /**< @~japanese 列方向の入力解析で、同じ32列の内積方向を分担して走査するSIMDグループの個数。
                                      * @~english Number of SIMD groups sharing the scan along the inner
                                      * dimension for the same 32 columns during column analysis.
                                      * @~
                                      */

/**
 * @~japanese
 * @brief CRTの復元用カーネルの名称と、各スレッドが保持する多倍長整数の桁数の組を列挙するXマクロ。
 * @param[in] apply カーネルの接尾辞と桁数の組を受け取るマクロ。
 * @~english
 * @brief X-macro listing the CRT reconstruction kernel suffixes paired with the number of multi-precision
 * digits per thread.
 * @param[in] apply Macro receiving a kernel suffix and digit count.
 * @~
 */
#define FP64_RECONSTRUCTION_KERNELS(apply) \
    apply(small, 4u) \
    apply(medium, 8u) \
    apply(large, 12u) \
    apply(full, FP64_MAX_LIMBS)

/**
 * @~japanese
 * @brief 復元用カーネルの定義から、多倍長整数の桁数だけを取り出す補助マクロ。
 * @param[in] name カーネルの接尾辞。
 * @param[in] size 多倍長整数の桁数。
 * @~english
 * @brief Helper macro extracting the multi-precision digit count from a reconstruction kernel entry.
 * @param[in] name Kernel suffix.
 * @param[in] size Number of multi-precision integer digits.
 * @~
 */
#define FP64_RECONSTRUCTION_SIZE(name, size) size,
FP64_CONSTANT fp64_word_t FP64_RECONSTRUCTION_CAPACITIES[] = { /**< @~japanese 各復元用カーネルが保持できる多倍長整数の桁数（昇順）。
                                                                * @~english Multi-precision integer digit
                                                                * capacities of the reconstruction kernels, in
                                                                * ascending order.
                                                                * @~
                                                                */
    FP64_RECONSTRUCTION_KERNELS(FP64_RECONSTRUCTION_SIZE)
};
#undef FP64_RECONSTRUCTION_SIZE
#define FP64_RECONSTRUCTION_COUNT (sizeof(FP64_RECONSTRUCTION_CAPACITIES) / sizeof(FP64_RECONSTRUCTION_CAPACITIES[0])) /**< @~japanese 桁数ごとに用意した復元用カーネルの種類数。
                                                                                                                        * @~english Number of reconstruction kernel variants.
                                                                                                                        * @~
                                                                                                                        */

#ifndef __METAL_VERSION__
/**
 * @~japanese
 * @brief 復元用カーネルの定義から、Metalライブラリー内の関数名文字列を取り出す補助マクロ。
 * @param[in] name カーネルの接尾辞。
 * @param[in] size 多倍長整数の桁数。
 * @~english
 * @brief Helper macro building the Metal library function name string from a reconstruction kernel entry.
 * @param[in] name Kernel suffix.
 * @param[in] size Number of multi-precision integer digits.
 * @~
 */
#define FP64_RECONSTRUCTION_NAME(name, size) "reconstruct_" #name,
static const char *const FP64_RECONSTRUCTION_NAMES[] = { /**< @~japanese Metalライブラリーから検索する復元用カーネルの関数名一覧。
                                                          * @~english Function names of the reconstruction
                                                          * kernels looked up in the Metal library.
                                                          * @~
                                                          */
    FP64_RECONSTRUCTION_KERNELS(FP64_RECONSTRUCTION_NAME)
};
#undef FP64_RECONSTRUCTION_NAME
#endif

#undef FP64_CONSTANT

/**
 * @~japanese
 * @brief 64ビットのFP64値のビット列を、下位32ビットと上位32ビットの2語に分けて保持する構造体。
 * @~english
 * @brief 64-bit FP64 bit pattern stored as a pair of low and high 32-bit words.
 * @~
 */
typedef struct fp64_bits_s {
    fp64_word_t low; /**< @~japanese 仮数の下位32ビット。
                      * @~english Low 32 bits of the significand.
                      * @~
                      */
    fp64_word_t high; /**< @~japanese 符号、指数、および仮数の上位ビットを含む上位32ビット。
                       * @~english High 32 bits containing the sign, exponent, and upper significand bits.
                       * @~
                       */
} fp64_bits_t;

/**
 * @~japanese
 * @brief 1つの行または列に含まれる有限値の指数の範囲と、NaNまたは無限大の有無を保持する構造体。
 * @~english
 * @brief Exponent range of finite values in a single row or column and whether any nonfinite value is present.
 * @~
 */
typedef struct fp64_scale_s {
    int exponent; /**< @~japanese 行または列の絶対値の最大値を囲む2のべき乗の指数。全要素がゼロの場合は0。
                   * @~english Power-of-two exponent bounding the maximum magnitude in the row or column. Zero
                   * if all elements are zero.
                   * @~
                   */
    int lowest_exponent; /**< @~japanese 非ゼロの有限値における最下位の非ゼロビットの最小指数。非ゼロの有限値がない場合はINT_MAX。
                          * @~english Minimum exponent of the least significant nonzero bit among nonzero
                          * finite values. INT_MAX if there are no nonzero finite values.
                          * @~
                          */
    fp64_word_t nonfinite; /**< @~japanese 行または列にNaNまたは無限大が含まれる場合は1、すべて有限値の場合は0。
                            * @~english One if the row or column contains a NaN or infinity, otherwise zero.
                            * @~
                            */
} fp64_scale_t;

/**
 * @~japanese
 * @brief GPUでの入力解析で求めた、行列AとBの整数幅および非有限値の有無を保持する構造体。
 * @~english
 * @brief Integer widths of matrices A and B and the presence of nonfinite values determined by GPU input
 * analysis.
 * @~
 */
typedef struct fp64_input_analysis_s {
    fp64_word_t precision_a; /**< @~japanese 行列Aの各行を切り捨てずに整数化するために必要なビット幅の最大値。
                              * @~english Maximum bit width needed to convert every row of matrix A to integers
                              * without truncation.
                              * @~
                              */
    fp64_word_t precision_b; /**< @~japanese 行列Bの各列を切り捨てずに整数化するために必要なビット幅の最大値。
                              * @~english Maximum bit width needed to convert every column of matrix B to
                              * integers without truncation.
                              * @~
                              */
    fp64_word_t nonfinite; /**< @~japanese AまたはBのどちらかにNaNまたは無限大が含まれる場合は1、すべて有限値の場合は0。
                            * @~english One if either A or B contains a NaN or infinity, otherwise zero.
                            * @~
                            */
} fp64_input_analysis_t;

/**
 * @~japanese
 * @brief 1桁あたりFP64_DIGIT_BITSビット（基数2の18乗）で表す非負の多倍長整数。
 * @~english
 * @brief Nonnegative multi-precision integer represented in base 2^FP64_DIGIT_BITS (18 bits per digit).
 * @~
 */
#ifdef __METAL_VERSION__
template <unsigned limb_count>
struct fp64_big_uint_s {
    fp64_word_t digits[limb_count]; /**< @~japanese 最下位桁から順に並ぶ配列。正規化後の各桁はFP64_DIGIT_MASK以下。
                                     * @~english Digits ordered from least to most significant. Each normalized
                                     * digit is at most FP64_DIGIT_MASK.
                                     * @~
                                     */
};
typedef fp64_big_uint_s<FP64_MAX_LIMBS> fp64_big_uint_t; /**< @~japanese CPU側と同じ最大桁数（FP64_MAX_LIMBS）を持つ、CRT定数表用の多倍長整数型。
                                                          * @~english Multi-precision integer type with the
                                                          * maximum capacity (FP64_MAX_LIMBS), matching the CPU
                                                          * layout for CRT constant tables.
                                                          * @~
                                                          */
#define FP64_BIG_FUNCTION template <unsigned limb_count> /**< @~japanese GPUでは保持する桁数ごとに多倍長整数の関数を特殊化する。
                                                          * @~english Specialize multi-precision integer
                                                          * functions for each digit capacity on the GPU.
                                                          * @~
                                                          */
#define FP64_BIG_VALUE fp64_big_uint_s<limb_count> /**< @~japanese GPUの各スレッドがローカルに保持する多倍長整数の型。
                                                    * @~english Multi-precision integer type held locally by
                                                    * each GPU thread.
                                                    * @~
                                                    */
#else
typedef struct fp64_big_uint_s {
    fp64_word_t digits[FP64_MAX_LIMBS]; /**< @~japanese 最下位桁から順に並ぶ配列。正規化後の各桁はFP64_DIGIT_MASK以下。
                                         * @~english Digits ordered from least to most significant. Each
                                         * normalized digit is at most FP64_DIGIT_MASK.
                                         * @~
                                         */
} fp64_big_uint_t;
#define FP64_BIG_FUNCTION /**< @~japanese CPUでは最大桁数（FP64_MAX_LIMBS）の関数として定義する。
                           * @~english Define multi-precision functions for the maximum digit capacity on the
                           * CPU.
                           * @~
                           */
#define FP64_BIG_VALUE fp64_big_uint_t /**< @~japanese CPU側で使用する多倍長整数の型。
                                        * @~english Multi-precision integer type used on the CPU.
                                        * @~
                                        */
#endif

#ifdef __METAL_VERSION__
typedef thread fp64_big_uint_t *fp64_big_uint_output_t; /**< @~japanese GPUのスレッド内で更新する多倍長整数へのポインター。
                                                         * @~english Pointer to a multi-precision integer
                                                         * updated within a GPU thread.
                                                         * @~
                                                         */
#else
typedef fp64_big_uint_t *fp64_big_uint_output_t; /**< @~japanese CPU側で更新する多倍長整数へのポインター。
                                                  * @~english Pointer to a multi-precision integer updated on
                                                  * the CPU.
                                                  * @~
                                                  */
#endif

/**
 * @~japanese
 * @brief 1つの法と、除算命令を使わずに32ビット整数の余りを求めるための定数。
 * @~english
 * @brief Single modulus paired with constants for reducing 32-bit integers without division instructions.
 * @~
 */
typedef struct fp64_modulus_s {
    fp64_word_t value; /**< @~japanese 256以下の法。
                        * @~english Modulus value (at most 256).
                        * @~
                        */
    fp64_word_t reciprocal; /**< @~japanese 2の32乗をvalueで割った商の切り捨て。上位32ビット乗算（mulhi）で近似商を求めるために使う。
                             * @~english Floor of 2^32 divided by value, used with high-half 32-bit
                             * multiplication (mulhi) to approximate the quotient.
                             * @~
                             */
    fp64_word_t word_weight; /**< @~japanese 2の32乗をvalueで割った余り。仮数の上位語に対する2のべき乗の余りを求めるために使う。
                              * @~english Remainder of 2^32 divided by value, used to compute power-of-two
                              * residues for the upper word of the significand.
                              * @~
                              */
} fp64_modulus_t;

/**
 * @~japanese
 * @brief 内積の項数と整数幅から決まる、CRTの変換と復元のための定数表。入力の具体的な要素値には依存しない。
 * @~english
 * @brief Constant tables for CRT conversion and reconstruction determined solely by the dot-product length and
 * integer widths, independent of the specific element values.
 * @~
 */
typedef struct fp64_crt_plan_s {
    fp64_word_t count; /**< @~japanese 使用する法の個数。
                        * @~english Number of moduli in use.
                        * @~
                        */
    fp64_word_t limbs; /**< @~japanese CRTの係数の総和を収めるために必要な多倍長整数の桁数。
                        * @~english Number of multi-precision digits needed to hold the sum of CRT
                        * coefficient-residue products.
                        * @~
                        */
    fp64_modulus_t moduli[FP64_MAX_MODULI]; /**< @~japanese 使用する順に並んだ法と剰余計算用の定数。
                                             * @~english Active moduli and their reduction constants in order
                                             * of use.
                                             * @~
                                             */
    fp64_big_uint_t coefficients[FP64_MAX_MODULI]; /**< @~japanese 各法m_tに対するCRTの重み係数 (P / m_t) * ((P / m_t)^{-1} mod m_t)。
                                                    * @~english CRT weight coefficient (P / m_t) *
                                                    * ((P / m_t)^{-1} mod m_t) for each modulus m_t.
                                                    * @~
                                                    */
    float ratios[FP64_MAX_MODULI]; /**< @~japanese 各CRT係数を法の積Pで割った比のFP32近似値。Pの何倍を引くかの候補を選ぶためにだけ使う。
                                    * @~english FP32 approximation of each CRT coefficient divided by the
                                    * modulus product P, used only to estimate the multiple of P to subtract.
                                    * @~
                                    */
    fp64_big_uint_t product; /**< @~japanese 使用するすべての法の積P。
                              * @~english Product P of all active moduli.
                              * @~
                              */
    fp64_big_uint_t half_product; /**< @~japanese 法の積Pの半分（floor(P / 2)）。対称な範囲への符号判定に使う。
                                   * @~english Half of the modulus product (floor(P / 2)), used to determine
                                   * the sign of the symmetric residue.
                                   * @~
                                   */
    unsigned short powers[FP64_MAX_MODULI][FP64_MAX_LIMBS * FP64_DIGIT_BITS]; /**< @~japanese 指数ごとの2のべき乗の余り。下位8ビットに2^s mod m、上位8ビットに2^(s+32) mod mを格納する。
                                                                               * @~english Power-of-two residues
                                                                               * per exponent: the low 8 bits
                                                                               * hold 2^s mod m and the high 8
                                                                               * bits hold 2^(s+32) mod m.
                                                                               * @~
                                                                               */
} fp64_crt_plan_t;

/**
 * @~japanese
 * @brief 行のまとまり（または行列全体）の処理でGPUカーネルへ渡す寸法、整数幅、および実行モードの引数。
 * @~english
 * @brief Dimensions, integer widths, and execution mode parameters passed to GPU kernels for a row batch (or
 * the full matrix).
 * @~
 */
typedef struct fp64_batch_parameters_s {
    fp64_word_t rows; /**< @~japanese 今回の処理が対象とする行数。
                       * @~english Number of rows processed in the current dispatch.
                       * @~
                       */
    fp64_word_t columns; /**< @~japanese 出力行列の列数。
                          * @~english Number of columns in the output matrix.
                          * @~
                          */
    fp64_word_t inner; /**< @~japanese 内積の項数。
                        * @~english Number of terms in each dot product.
                        * @~
                        */
    fp64_word_t row_begin; /**< @~japanese 行列Aにおける現在の行のまとまりの開始行番号。
                            * @~english Starting row index in matrix A for the current row batch.
                            * @~
                            */
    fp64_word_t precision_a; /**< @~japanese 行列Aの整数幅。
                              * @~english Integer width of matrix A.
                              * @~
                              */
    fp64_word_t precision_b; /**< @~japanese 行列Bの整数幅。
                              * @~english Integer width of matrix B.
                              * @~
                              */
    fp64_word_t total_rows; /**< @~japanese 行列A全体の行数。Strassen法で上半分と下半分のブロックを対応付けるために使う。
                             * @~english Total number of rows in matrix A, used in Strassen's algorithm to pair
                             * rows from the top and bottom halves.
                             * @~
                             */
    fp64_word_t strassen; /**< @~japanese Strassen法で2x2ブロックに分割して処理する場合は1、通常の行列積では0。
                           * @~english One when partitioning into 2x2 blocks for Strassen multiplication,
                           * otherwise zero.
                           * @~
                           */
    fp64_word_t shifted; /**< @~japanese 内積方向の指数の調整量を適用する場合は1、適用しない場合は0。
                          * @~english One when applying inner-dimension exponent adjustments, otherwise zero.
                          * @~
                          */
} fp64_batch_parameters_t;

/**
 * @~japanese
 * @brief 非負の32ビット整数を表現するために必要な二進数の桁数を求める。
 * @param[in] value 調べる符号なし32ビット整数。
 * @return valueが0の場合は0、それ以外の場合は最上位の1のビット位置（0始まり）に1を加えた値。
 * @~english
 * @brief Compute the number of binary digits needed to represent an unsigned 32-bit integer.
 * @param[in] value Unsigned 32-bit integer to examine.
 * @return Zero when value is 0; otherwise 1 plus the zero-based position of the most significant set bit.
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
 * @brief 有限のFP64値の絶対値を囲む2のべき乗の指数（1 + floor(log2(|x|))）を求める。
 * @param[in] bits 有限のFP64値のビット列。
 * @return 非ゼロ値の場合はREADME.md「計算の定義」で定めた指数、ゼロの場合はFP64_ZERO_EXPONENT。
 * @pre bitsはNaNまたは無限大ではないこと。
 * @~english
 * @brief Compute the power-of-two exponent (1 + floor(log2(|x|))) bounding the magnitude of a finite FP64
 * value.
 * @param[in] bits Bit pattern of a finite FP64 value.
 * @return For a nonzero value, the exponent defined in README.en.md, section "Numerical semantics"; for zero,
 * FP64_ZERO_EXPONENT.
 * @pre bits must not be a NaN or infinity.
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
 * @brief 2語（64ビット）の符号なし整数を右へビットシフトする。
 * @param[in] value 下位語と上位語の組で表された整数。
 * @param[in] shift 右へずらすビット数。
 * @return 下位ビットを切り捨てた結果。shiftが64以上の場合は0。
 * @~english
 * @brief Right-shift an unsigned two-word (64-bit) integer.
 * @param[in] value Integer stored as low and high 32-bit words.
 * @param[in] shift Number of bits to shift right.
 * @return Result with low bits truncated, or zero when shift is at least 64.
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
 * @brief 2語（64ビット）の符号なし整数を左へビットシフトする。
 * @param[in] value 下位語と上位語の組で表された整数。
 * @param[in] shift 左へずらすビット数（64未満）。
 * @return 左へずらした結果。
 * @pre shiftは64未満であり、シフト後の値が64ビットに収まること。
 * @~english
 * @brief Left-shift an unsigned two-word (64-bit) integer.
 * @param[in] value Integer stored as low and high 32-bit words.
 * @param[in] shift Number of bits to shift left (must be less than 64).
 * @return Left-shifted 64-bit result.
 * @pre shift must be less than 64, and the shifted value must fit in 64 bits.
 * @~
 */
static inline fp64_bits_t fp64_shift_left(fp64_bits_t value, fp64_word_t shift) {
    if (shift >= 32) return (fp64_bits_t){0, value.low << (shift - 32)};
    if (shift == 0) return value;
    return (fp64_bits_t){value.low << shift, (value.high << shift) | (value.low >> (32 - shift))};
}

/**
 * @~japanese
 * @brief 逆数の近似値との上位32ビット乗算を使って、32ビット符号なし整数を法で割った余りを求める。
 * @param[in] value 割られる32ビット符号なし整数。
 * @param[in] modulus 法と、その法に対する逆数近似値の組。
 * @return 0以上modulus.value未満の余り。
 * @~english
 * @brief Compute the remainder of an unsigned 32-bit integer modulo a small modulus using high-half
 * multiplication by a precomputed reciprocal.
 * @param[in] value Unsigned 32-bit dividend.
 * @param[in] modulus Modulus and its precomputed reciprocal constants.
 * @return Remainder in the range [0, modulus.value).
 * @~
 */
static inline fp64_word_t fp64_word_mod(fp64_word_t value, fp64_modulus_t modulus) {
#ifdef __METAL_VERSION__
    fp64_word_t quotient = metal::mulhi(value, modulus.reciprocal);
#else
    fp64_word_t quotient = (fp64_word_t)(((uint64_t)(value) * modulus.reciprocal) >> 32);
#endif
    // reciprocal は floor(2^32 / m) であるため、近似商 quotient は真の商と等しいか1だけ小さい。
    // したがって、value - quotient * m から m を高々1回引くだけで正確な余りが得られる。
    fp64_word_t remainder = value - quotient * modulus.value;
    return remainder >= modulus.value ? remainder - modulus.value : remainder;
}

/**
 * @~japanese
 * @brief 整数化した仮数と2のべき乗の積を法で割り、符号付きINT8に収まる対称な余りへ変換する。
 * @param[in] mantissa 整数化で取り出した仮数の絶対値（最大53ビット）。
 * @param[in] negative 元の要素が負である場合はtrue。
 * @param[in] powers 仮数の下位語と上位語に掛ける2のべき乗の余り（fp64_crt_plan_tのpowersの1要素）。
 * @param[in] modulus 256以下の法と剰余計算用の定数。
 * @return [-floor(m / 2), ceil(m / 2) - 1] の範囲にある対称な余り（符号付きINT8に収まる）。
 * @pre mantissaの上位語（mantissa.high）は21ビット以下であること。
 * @~english
 * @brief Reduce the product of an integer-converted significand and a power of two modulo a modulus into a
 * symmetric residue that fits in signed INT8.
 * @param[in] mantissa Magnitude of the significand retained by integer conversion (at most 53 bits).
 * @param[in] negative true if the original element is negative.
 * @param[in] powers Packed power-of-two residues multiplying the low and high significand words (one entry from
 * powers in fp64_crt_plan_t).
 * @param[in] modulus Modulus at most 256 and its reduction constants.
 * @return Symmetric residue in [-floor(m / 2), ceil(m / 2) - 1], which fits in signed INT8.
 * @pre The high word of mantissa (mantissa.high) must use at most 21 bits.
 * @~
 */
static inline int fp64_signed_residue(fp64_bits_t mantissa, bool negative, fp64_word_t powers, fp64_modulus_t modulus) {
    // 仮数の上位語は最大21ビット、下位語の余りと各べき乗の余りは255以下であるため、
    // 2つの積の和は32ビット符号なし整数でオーバーフローしない。
    fp64_word_t remainder = fp64_word_mod(mantissa.high * (powers >> 8)
                                         + fp64_word_mod(mantissa.low, modulus) * (powers & 255u), modulus);
    if (negative && remainder != 0) remainder = modulus.value - remainder;
    return remainder >= (modulus.value + 1) / 2 ? (int)(remainder) - (int)(modulus.value) : (int)(remainder);
}

/**
 * @~japanese
 * @brief 2つの多倍長整数の大小を比較する。
 * @param[in] left 左辺の多倍長整数。
 * @param[in] right 右辺の多倍長整数。
 * @param[in] limbs 比較する桁数。
 * @return left < right の場合は-1、等しい場合は0、left > right の場合は1。
 * @~english
 * @brief Compare two multi-precision integers.
 * @param[in] left Left multi-precision integer.
 * @param[in] right Right multi-precision integer.
 * @param[in] limbs Number of digits to compare.
 * @return -1 if left < right, 0 if equal, and 1 if left > right.
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
 * @brief 2つの多倍長整数の非負の差（left - right）を求める。
 * @param[in] left 引かれる多倍長整数。
 * @param[in] right 引く多倍長整数。
 * @param[in] limbs 計算する桁数。
 * @return leftからrightを引いた多倍長整数。
 * @pre leftはright以上であること。
 * @~english
 * @brief Compute the nonnegative difference (left - right) of two multi-precision integers.
 * @param[in] left Minuend.
 * @param[in] right Subtrahend.
 * @param[in] limbs Number of digits to process.
 * @return Multi-precision integer equal to left minus right.
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
 * @brief 多倍長整数valueに、別の多倍長整数basisと小さい整数multiplierの積を加算する。
 * @param[in,out] value 加算先の多倍長整数。
 * @param[in] basis 被乗数の多倍長整数。
 * @param[in] multiplier 255以下の乗数。
 * @param[in] limbs 計算する桁数。
 * @pre 加算後の結果がlimbs桁に収まること。
 * @~english
 * @brief Add the product of a multi-precision integer basis and a small integer multiplier to value.
 * @param[in,out] value Destination multi-precision integer.
 * @param[in] basis Multiplicand multi-precision integer.
 * @param[in] multiplier Multiplier (at most 255).
 * @param[in] limbs Number of digits to process.
 * @pre The sum must fit in limbs digits.
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
 * @brief 多倍長整数を256以下の法で割った余りを求める。
 * @param[in] value 割られる多倍長整数。
 * @param[in] modulus 256以下の法と剰余計算用の定数。
 * @param[in] limbs 有効な桁数。
 * @return 0以上modulus.value未満の余り。
 * @~english
 * @brief Compute the remainder of a multi-precision integer modulo a modulus at most 256.
 * @param[in] value Multi-precision dividend.
 * @param[in] modulus Modulus at most 256 and its reduction constants.
 * @param[in] limbs Number of active digits.
 * @return Remainder in the range [0, modulus.value).
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
 * @brief 多倍長整数を表現するために必要な二進数の桁数を求める。
 * @param[in] value 調べる多倍長整数。
 * @param[in] limbs 有効な桁数。
 * @return 値が0の場合は0、それ以外の場合は最上位の1のビット位置（0始まり）に1を加えた値。
 * @~english
 * @brief Compute the number of binary digits needed to represent a multi-precision integer.
 * @param[in] value Multi-precision integer to examine.
 * @param[in] limbs Number of active digits.
 * @return Zero when value is 0; otherwise 1 plus the zero-based position of the most significant set bit.
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
 * @brief 多倍長整数の指定したビット位置から連続する32ビットを取り出す。
 * @param[in] value 対象の多倍長整数。
 * @param[in] start 取り出す最下位ビットの位置（0始まり）。
 * @param[in] limbs 有効な桁数。
 * @return startから始まる32ビットの値（有効な桁の範囲を超える上位ビットは0）。
 * @~english
 * @brief Extract 32 contiguous bits starting at a specified bit position in a multi-precision integer.
 * @param[in] value Multi-precision integer to read.
 * @param[in] start Zero-based bit position of the least significant bit to extract.
 * @param[in] limbs Number of active digits.
 * @return 32-bit window starting at start, with bits beyond the active digits filled with zero.
 * @~
 */
FP64_BIG_FUNCTION
static inline fp64_word_t fp64_big_window(FP64_BIG_VALUE value, fp64_word_t start, fp64_word_t limbs) {
    fp64_word_t digit = start / FP64_DIGIT_BITS, offset = start % FP64_DIGIT_BITS;
    if (digit >= limbs) return 0;
    // 1桁は18ビットであるため、任意のビット位置から始まる32ビットは最大3つの連続する桁にまたがる。
    fp64_word_t result = value.digits[digit] >> offset;
    if (digit + 1 < limbs) result |= value.digits[digit + 1] << (FP64_DIGIT_BITS - offset);
    if (offset > 2 * FP64_DIGIT_BITS - 32 && digit + 2 < limbs)
        result |= value.digits[digit + 2] << (2 * FP64_DIGIT_BITS - offset);
    return result;
}

/**
 * @~japanese
 * @brief 多倍長整数において、指定したビット位置より下位に1のビットが存在するかを調べる。
 * @param[in] value 対象の多倍長整数。
 * @param[in] end 調べる範囲の上限のビット位置（この位置のビット自体は含めない）。
 * @param[in] limbs 有効な桁数。
 * @return 0番目からend - 1番目までのビットに1が1つでもあればtrue、すべて0であればfalse。
 * @~english
 * @brief Check whether any bit strictly below a specified bit position is nonzero in a multi-precision
 * integer.
 * @param[in] value Multi-precision integer to examine.
 * @param[in] end Exclusive upper bit position of the range to check.
 * @param[in] limbs Number of active digits.
 * @return true if any bit in positions [0, end - 1] is 1, otherwise false.
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
 * @brief 多倍長整数と2のべき乗の積（(-1)^negative * magnitude * 2^scale）を、FP64のビット列へ直接丸める。
 * @param[in] magnitude 絶対値を表す多倍長整数。
 * @param[in] negative 値が負である場合はtrue。
 * @param[in] scale 掛ける2のべき乗の指数。
 * @param[in] limbs 有効な桁数。
 * @return README.md「計算の定義」に従って最近接偶数丸めしたFP64のビット列。
 * @~english
 * @brief Round the scaled multi-precision integer ((-1)^negative * magnitude * 2^scale) directly to an FP64
 * bit pattern.
 * @param[in] magnitude Multi-precision integer representing the magnitude.
 * @param[in] negative true if the value is negative.
 * @param[in] scale Base-2 exponent scaling the magnitude.
 * @param[in] limbs Number of active digits.
 * @return FP64 bit pattern rounded to nearest, ties to even, following README.en.md, section "Numerical
 * semantics".
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
    // 丸めの加算によって仮数が54ビット（ビット53が1）に繰り上がった場合は、1ビット右へずらして指数を1増やす。
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
