#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "arithmetic.h"
#include "fp64_fma.h"

using namespace metal;
using namespace mpp::tensor_ops;

/**
 * @~japanese
 * @brief 行のまとまり内の行番号を、入力行列Aにおける行番号へ変換する。
 * @param[in] parameters 行列の寸法と現在処理している行のまとまりの情報。
 * @param[in] row 行のまとまり内の行番号（0始まり）。
 * @return 入力行列Aにおける行番号。
 * @~english
 * @brief Map a row index within the current row batch to a row index in input matrix A.
 * @param[in] parameters Matrix dimensions and current row-batch parameters.
 * @param[in] row Zero-based row index within the row batch.
 * @return Row index in input matrix A.
 * @~
 */
static inline uint fp64_input_row(constant fp64_batch_parameters_t& parameters, uint row) {
    if (!parameters.strassen) return parameters.row_begin + row;
    // Strassen法では、行のまとまりの前半に入力Aの上半分の行を、後半に下半分の対応する行を割り当てる。
    uint half_rows = parameters.rows / 2;
    return parameters.row_begin + (row < half_rows ? row : row - half_rows + parameters.total_rows / 2);
}

/**
 * @~japanese
 * @brief 有限のFP64値の絶対値において、最下位の非ゼロビットが位置する2のべき乗の指数を求める。
 * @param[in] bits 有限のFP64値のビット列。
 * @return 非ゼロ値の場合は最下位非ゼロビットの指数、ゼロの場合はINT_MAX（2147483647）。
 * @~english
 * @brief Compute the base-2 exponent of the least significant nonzero bit in the magnitude of a finite FP64
 * value.
 * @param[in] bits Bit pattern of a finite FP64 value.
 * @return Exponent of the least significant nonzero bit for a nonzero value, or INT_MAX (2147483647) for zero.
 * @~
 */
static inline int fp64_lowest_bit(fp64_bits_t bits) {
    uint exponent = (bits.high >> 20) & 2047u;
    uint high = bits.high & 0xfffffu;
    if (exponent != 0) high |= 0x100000u;
    int scale = exponent != 0 ? int(exponent) - 1075 : -1074;
    // x & (0u - x) で最下位の1のビットだけを取り出し、そのビット位置から2のべき乗の指数を求める。
    if (bits.low != 0) return scale + int(fp64_word_bit_length(bits.low & (0u - bits.low))) - 1;
    if (high != 0) return scale + 32 + int(fp64_word_bit_length(high & (0u - high))) - 1;
    return 2147483647;
}

/**
 * @~japanese
 * @brief 行列の各行を走査し、最大値を囲む指数、最下位の非ゼロビットの最小指数、および非有限値の有無を求める。
 * @tparam shifted 内積方向の指数の調整量を適用する場合はtrue。
 * @param[in] input 行優先で並ぶFP64のビット列。
 * @param[out] scales 各行の最大値を囲む指数、最下位ビットの最小指数、および非有限値の有無の出力先。
 * @param[in] parameters 行列の寸法。
 * @param[in] inner_shifts 内積方向の指数の調整量。shiftedがtrueの場合だけ参照する。
 * @param[in] group 担当する行番号。
 * @param[in] lane SIMDグループ内のスレッド番号（0～31）。
 * @pre shiftedがtrueの場合、inputは有限値だけを含む行列Aであること。
 * @~english
 * @brief Scan each row of a matrix to find the exponent bounding the maximum magnitude, the minimum exponent of
 * the least significant nonzero bit, and whether any nonfinite value is present.
 * @tparam shifted true when applying inner-dimension exponent adjustments.
 * @param[in] input Row-major FP64 bit patterns.
 * @param[out] scales Per-row output storing the bounding exponent, minimum least-significant-bit exponent, and
 * nonfinite flag.
 * @param[in] parameters Matrix dimensions.
 * @param[in] inner_shifts Inner-dimension exponent adjustments, read only when shifted is true.
 * @param[in] group Assigned row index.
 * @param[in] lane Thread index within the SIMD group (0 to 31).
 * @pre When shifted is true, input must be matrix A and contain only finite values.
 * @~
 */
template <bool shifted>
kernel void find_row_scales(device const fp64_bits_t* input [[buffer(0)]],
                        device fp64_scale_t* scales [[buffer(1)]],
                        constant fp64_batch_parameters_t& parameters [[buffer(2)]],
                        device const int* inner_shifts [[buffer(3)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    int maximum = FP64_ZERO_EXPONENT;
    int minimum = 2147483647;
    uint nonfinite = 0;
    for (uint index = lane; index < parameters.inner; index += 32) {
        ulong offset = ulong(fp64_input_row(parameters, group)) * parameters.inner + index;
        fp64_bits_t bits = input[offset];
        if (!shifted && ((bits.high >> 20) & 2047u) == 2047u) nonfinite = 1;
        else {
            int exponent = fp64_exponent(bits), lowest = fp64_lowest_bit(bits);
            if constexpr (shifted) {
                if (exponent != FP64_ZERO_EXPONENT) {
                    exponent += inner_shifts[index];
                    lowest += inner_shifts[index];
                }
            }
            maximum = max(maximum, exponent);
            minimum = min(minimum, lowest);
        }
    }
    maximum = simd_max(maximum);
    minimum = simd_min(minimum);
    nonfinite = simd_max(nonfinite);
    if (lane == 0) {
        bool zero = maximum == FP64_ZERO_EXPONENT;
        scales[group] = {zero ? 0 : maximum, minimum, nonfinite};
    }
}

template [[host_name("find_row_scales")]] kernel void find_row_scales<false>(device const fp64_bits_t*, device fp64_scale_t*, constant fp64_batch_parameters_t&, device const int*, uint, uint);
template [[host_name("find_shifted_row_scales")]] kernel void find_row_scales<true>(device const fp64_bits_t*, device fp64_scale_t*, constant fp64_batch_parameters_t&, device const int*, uint, uint);

/**
 * @~japanese
 * @brief 隣接する32列をまとめて走査し、各列における最大値を囲む指数、最下位の非ゼロビットの最小指数、および非有限値の有無を求める。
 * @tparam shifted 内積方向の指数の調整量を適用する場合はtrue。
 * @param[in] input 行優先で並ぶFP64のビット列。
 * @param[out] scales 各列の最大値を囲む指数、最下位ビットの最小指数、および非有限値の有無の出力先。
 * @param[in] parameters 行列の寸法。
 * @param[in] inner_shifts 内積方向の指数の調整量。shiftedがtrueの場合だけ参照する。
 * @param[in] group 担当する32列のまとまりの番号。
 * @param[in] lane SIMDグループ内のスレッド番号（0～31、担当する列のオフセットに対応する）。
 * @param[in] subgroup 内積方向の走査を分担するSIMDグループの番号。
 * @pre shiftedがtrueの場合、inputは有限値だけを含む行列Bであること。
 * @~english
 * @brief Scan batches of 32 adjacent columns in parallel to find each column's bounding exponent, minimum
 * least-significant-bit exponent, and whether any nonfinite value is present.
 * @tparam shifted true when applying inner-dimension exponent adjustments.
 * @param[in] input Row-major FP64 bit patterns.
 * @param[out] scales Per-column output storing the bounding exponent, minimum least-significant-bit exponent,
 * and nonfinite flag.
 * @param[in] parameters Matrix dimensions.
 * @param[in] inner_shifts Inner-dimension exponent adjustments, read only when shifted is true.
 * @param[in] group Index of the assigned 32-column batch.
 * @param[in] lane Thread index within the SIMD group (0 to 31, corresponding to the column offset).
 * @param[in] subgroup Index of the SIMD group sharing the scan along the inner dimension.
 * @pre When shifted is true, input must be matrix B and contain only finite values.
 * @~
 */
template <bool shifted>
kernel void find_column_scales(device const fp64_bits_t* input [[buffer(0)]],
                               device fp64_scale_t* scales [[buffer(1)]],
                               constant fp64_batch_parameters_t& parameters [[buffer(2)]],
                               device const int* inner_shifts [[buffer(3)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint subgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup int maxima[FP64_COLUMN_SIMD_GROUPS][32];
    threadgroup int minima[FP64_COLUMN_SIMD_GROUPS][32];
    threadgroup uint specials[FP64_COLUMN_SIMD_GROUPS][32];
    uint column = group * 32 + lane;
    int maximum = FP64_ZERO_EXPONENT;
    int minimum = 2147483647;
    uint nonfinite = 0;
    if (column < parameters.columns) {
        for (uint row = subgroup; row < parameters.inner; row += FP64_COLUMN_SIMD_GROUPS) {
            fp64_bits_t bits = input[ulong(row) * parameters.columns + column];
            if (!shifted && ((bits.high >> 20) & 2047u) == 2047u) nonfinite = 1;
            else {
                int exponent = fp64_exponent(bits), lowest = fp64_lowest_bit(bits);
                if constexpr (shifted) {
                    if (exponent != FP64_ZERO_EXPONENT) {
                        exponent -= inner_shifts[row];
                        lowest -= inner_shifts[row];
                    }
                }
                maximum = max(maximum, exponent);
                minimum = min(minimum, lowest);
            }
        }
    }
    maxima[subgroup][lane] = maximum;
    minima[subgroup][lane] = minimum;
    specials[subgroup][lane] = nonfinite;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (subgroup == 0 && column < parameters.columns) {
        #pragma unroll
        for (uint index = 1; index < FP64_COLUMN_SIMD_GROUPS; ++index) {
            maximum = max(maximum, maxima[index][lane]);
            minimum = min(minimum, minima[index][lane]);
            nonfinite |= specials[index][lane];
        }
        bool zero = maximum == FP64_ZERO_EXPONENT;
        scales[column] = {zero ? 0 : maximum, minimum, nonfinite};
    }
}

template [[host_name("find_column_scales")]] kernel void find_column_scales<false>(device const fp64_bits_t*, device fp64_scale_t*, constant fp64_batch_parameters_t&, device const int*, uint, uint, uint);
template [[host_name("find_shifted_column_scales")]] kernel void find_column_scales<true>(device const fp64_bits_t*, device fp64_scale_t*, constant fp64_batch_parameters_t&, device const int*, uint, uint, uint);

/**
 * @~japanese
 * @brief 行列Aの各列と行列Bの対応する各行の最大指数から、積A_{ik} * B_{kj}を変えずに指数差を縮める調整量を求める。
 * @param[in] a 行列Aの列ごとの解析結果。
 * @param[in] b 行列Bの行ごとの解析結果。
 * @param[out] inner_shifts 行列Aの第k列へ加算し、行列Bの第k行から減算する指数の調整量。
 * @param[in] parameters 行列全体の寸法。
 * @param[in] position 内積方向の添字kに対応する2次元グリッド上のスレッド位置。
 * @pre 入力行列AとBのすべての要素が有限値であること。
 * @~english
 * @brief Compute product-preserving exponent adjustments from the maximum exponents of each column of A and the
 * corresponding row of B.
 * @param[in] a Per-column analysis of matrix A.
 * @param[in] b Per-row analysis of matrix B.
 * @param[out] inner_shifts Exponent adjustment added to column k of A and subtracted from row k of B.
 * @param[in] parameters Dimensions of the full matrices.
 * @param[in] position Two-dimensional thread position corresponding to inner-dimension index k.
 * @pre All elements of input matrices A and B must be finite.
 * @~
 */
kernel void find_inner_shifts(device const fp64_scale_t* a [[buffer(0)]],
                              device const fp64_scale_t* b [[buffer(1)]],
                              device int* inner_shifts [[buffer(2)]],
                              constant fp64_batch_parameters_t& parameters [[buffer(3)]],
                              uint2 position [[thread_position_in_grid]]) {
    ulong index = ulong(position.y) * min(parameters.inner, 65536u) + position.x;
    if (index >= parameters.inner) return;
    int shift = 0;
    if (a[index].lowest_exponent != 2147483647 && b[index].lowest_exponent != 2147483647) {
        int difference = b[index].exponent - a[index].exponent;
        // 負の整数に対する / 2 は0方向へ丸められるため、負の奇数のときだけ1を引いて floor((b - a) / 2) に揃える。
        shift = difference / 2 - int(difference < 0 && difference % 2 != 0);
    }
    inner_shifts[index] = shift;
}

/**
 * @~japanese
 * @brief 行ごとおよび列ごとの解析結果を集計し、行列全体を切り捨てずに整数化できる整数幅と非有限値の有無を求める。
 * @param[in] a 行列Aの行ごとの解析結果。
 * @param[in] b 行列Bの列ごとの解析結果。
 * @param[out] analysis 行列全体の整数幅p_A, p_Bと非有限値の有無の出力先。
 * @param[in] parameters 行列全体の寸法。
 * @param[in] lane SIMDグループ内のスレッド番号（0～31）。
 * @~english
 * @brief Aggregate the per-row and per-column analyses to determine the integer widths p_A and p_B needed for
 * lossless integer conversion of the full matrices and whether any nonfinite value is present.
 * @param[in] a Per-row analysis of matrix A.
 * @param[in] b Per-column analysis of matrix B.
 * @param[out] analysis Output structure receiving the required integer widths and nonfinite flag.
 * @param[in] parameters Dimensions of the full matrices.
 * @param[in] lane Thread index within the SIMD group (0 to 31).
 * @~
 */
kernel void analyse_inputs(device const fp64_scale_t* a [[buffer(0)]],
                           device const fp64_scale_t* b [[buffer(1)]],
                           device fp64_input_analysis_t* analysis [[buffer(2)]],
                           constant fp64_batch_parameters_t& parameters [[buffer(3)]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint pa = 1, pb = 1, nonfinite = 0;
    for (uint row = lane; row < parameters.rows; row += 32) {
        pa = max(pa, uint(max(1, a[row].exponent - a[row].lowest_exponent)));
        nonfinite |= a[row].nonfinite;
    }
    for (uint column = lane; column < parameters.columns; column += 32) {
        pb = max(pb, uint(max(1, b[column].exponent - b[column].lowest_exponent)));
        nonfinite |= b[column].nonfinite;
    }
    pa = simd_max(pa);
    pb = simd_max(pb);
    nonfinite = simd_max(nonfinite);
    if (lane == 0) *analysis = {pa, pb, nonfinite};
}

/**
 * @~japanese
 * @brief 整数化した1つの行または列のすべての要素に共通する、下位に続くゼロのビット数の下限（最大8）を求める。
 * @param[in] scale 行または列の解析結果。
 * @param[in] precision 整数化に使用する整数幅。
 * @return 下位に続くゼロのビット数の下限（0～8）。行または列の全要素がゼロの場合は8。
 * @~english
 * @brief Compute a lower bound (capped at 8) on the number of trailing zero bits shared by all
 * integer-converted elements in a row or column.
 * @param[in] scale Analysis result for the row or column.
 * @param[in] precision Integer width used for conversion.
 * @return Lower bound on trailing zero bits (0 to 8), or 8 if all elements in the row or column are zero.
 * @~
 */
static inline uint fp64_trailing_bits(fp64_scale_t scale, uint precision) {
    return scale.lowest_exponent == 2147483647 ? 8u
         : uint(clamp(scale.lowest_exponent + int(precision) - scale.exponent, 0, 8));
}

/**
 * @~japanese
 * @brief 整数化のために桁位置を揃えた仮数と、2のべき乗の余りの表を参照するシフト量。
 * @~english
 * @brief Significand aligned for integer conversion paired with the shift amount indexing the power-of-two
 * residue table.
 * @~
 */
struct fp64_quantized_input_s {
    fp64_bits_t mantissa; /**< @~japanese 整数化で残る仮数の絶対値（最大53ビット）。
                           * @~english Magnitude of the significand retained after alignment (at most 53 bits).
                           * @~
                           */
    uint shift; /**< @~japanese 仮数に掛ける2のべき乗の指数（powers配列の添字）。
                 * @~english Power-of-two exponent multiplying the significand (index into the powers table).
                 * @~
                 */
    bool negative; /**< @~japanese 元の要素が負である場合はtrue。
                    * @~english true if the original element is negative.
                    * @~
                    */
};

/**
 * @~japanese
 * @brief 有限のFP64値を、法に依存しない整数化済みの仮数と2のべき乗のシフト量へ分解する。
 * @tparam shifted 内積方向の指数の調整量を適用する場合はtrue。
 * @param[in] bits 入力要素のFP64ビット列。
 * @param[in] precision 行または列に適用する整数幅。
 * @param[in] scale 行または列の最大値を囲む指数から、この要素に対する内積方向の調整量を引いた基準指数。
 * @return 法に依存しない整数化済みの仮数、シフト量、および符号。
 * @pre bitsは有限値であり、precisionはその行または列のすべての非ゼロビットを保持できる幅であること。
 * @~english
 * @brief Decompose a finite FP64 value into a modulus-independent aligned significand and power-of-two shift.
 * @tparam shifted true when applying inner-dimension exponent adjustments.
 * @param[in] bits FP64 bit pattern of the input element.
 * @param[in] precision Integer width applied to the row or column.
 * @param[in] scale Row or column bounding exponent minus the inner-dimension adjustment for this element.
 * @return Modulus-independent aligned significand, shift amount, and sign.
 * @pre bits must be finite, and precision must be wide enough to retain all nonzero bits in the row or column.
 * @~
 */
template <bool shifted>
static inline fp64_quantized_input_s fp64_quantize(fp64_bits_t bits, uint precision, int scale) {
    if constexpr (shifted) {
        // 値がゼロの要素には有効な指数がないため、他の非ゼロ要素から決まった調整量によって範囲外のシフト量が作られないよう先に0を返す。
        if ((bits.low | (bits.high & 0x7fffffffu)) == 0) return {{0, 0}, 0, false};
    }
    uint raw_exponent = (bits.high >> 20) & 2047u;
    fp64_bits_t mantissa = {bits.low, bits.high & 0xfffffu};
    int exponent = -1074;
    if (raw_exponent != 0) {
        mantissa.high |= 0x100000u;
        exponent = int(raw_exponent) - 1075;
    }
    int shift = exponent + int(precision) - scale;
    if (shift < 0) mantissa = fp64_shift_right(mantissa, uint(-shift));
    return {mantissa, uint(max(shift, 0)), (bits.high >> 31) != 0};
}

/**
 * @~japanese
 * @brief 入力行列の各要素を整数化し、使用するすべての法に対する符号付きINT8の余りを生成する。
 * @tparam shifted 内積方向の指数の調整量を適用する場合はtrue。
 * @param[in] input 入力行列のFP64ビット列。
 * @param[in] scales 行または列ごとの解析結果。
 * @param[out] output 法ごとに連続して並ぶ符号付きINT8の余り配列。
 * @param[in] parameters 行列の寸法と整数幅。
 * @param[in] plan 使用する法と2のべき乗の余りの表。
 * @param[in] column_mode 行列Bを処理する場合は1、行列Aを処理する場合は0。
 * @param[in] inner_shifts 内積方向の指数の調整量。shiftedがtrueの場合だけ参照する。
 * @param[in] position 処理する要素に対応する2次元グリッド上のスレッド位置。
 * @~english
 * @brief Convert each element of an input matrix to integers and generate signed INT8 residues for all active
 * moduli.
 * @tparam shifted true when applying inner-dimension exponent adjustments.
 * @param[in] input FP64 bit patterns of the input matrix.
 * @param[in] scales Per-row or per-column analysis results.
 * @param[out] output Output array of signed INT8 residues stored contiguously per modulus.
 * @param[in] parameters Matrix dimensions and integer widths.
 * @param[in] plan Active moduli and power-of-two residue tables.
 * @param[in] column_mode One when processing matrix B, zero when processing matrix A.
 * @param[in] inner_shifts Inner-dimension exponent adjustments, read only when shifted is true.
 * @param[in] position Two-dimensional thread position corresponding to the element being processed.
 * @~
 */
template <bool shifted>
kernel void make_residues(device const fp64_bits_t* input [[buffer(0)]],
                          device const fp64_scale_t* scales [[buffer(1)]],
                          device int8_t* output [[buffer(2)]],
                          constant fp64_batch_parameters_t& parameters [[buffer(3)]],
                          constant fp64_crt_plan_t& plan [[buffer(4)]],
                          constant uint& column_mode [[buffer(5)]],
                          device const int* inner_shifts [[buffer(6)]],
                          uint2 position [[thread_position_in_grid]]) {
    ulong count = column_mode ? ulong(parameters.inner) * parameters.columns
                             : ulong(parameters.rows) * parameters.inner;
    ulong index = ulong(position.y) * min(count, ulong(65536)) + position.x;
    if (index >= count) return;
    uint scale_index = column_mode ? uint(index % parameters.columns) : uint(index / parameters.inner);
    ulong input_index = column_mode ? index
                                   : ulong(fp64_input_row(parameters, scale_index)) * parameters.inner + index % parameters.inner;
    uint precision = column_mode ? parameters.precision_b : parameters.precision_a;
    if (!column_mode) scale_index = fp64_input_row(parameters, scale_index);
    int adjustment = 0;
    if constexpr (shifted) {
        adjustment = inner_shifts[column_mode ? index / parameters.columns : index % parameters.inner];
        if (column_mode) adjustment = -adjustment;
    }
    auto value = fp64_quantize<shifted>(input[input_index], precision, scales[scale_index].exponent - adjustment);
    for (uint t = 0; t < plan.count; ++t) {
        output[ulong(t) * count + index] = int8_t(fp64_signed_residue(value.mantissa, value.negative,
                                                                   plan.powers[t][value.shift], plan.moduli[t]));
    }
}

template [[host_name("make_residues")]] kernel void make_residues<false>(device const fp64_bits_t*, device const fp64_scale_t*, device int8_t*, constant fp64_batch_parameters_t&, constant fp64_crt_plan_t&, constant uint&, device const int*, uint2);
template [[host_name("make_shifted_residues")]] kernel void make_residues<true>(device const fp64_bits_t*, device const fp64_scale_t*, device int8_t*, constant fp64_batch_parameters_t&, constant fp64_crt_plan_t&, constant uint&, device const int*, uint2);

/**
 * @~japanese
 * @brief 2つの対称な余りの和または差を、再び同じ対称な余りの範囲へ戻す。
 * @param[in] value 2つの対称な余りの和または差。
 * @param[in] modulus 256以下の法。
 * @return [-floor(m / 2), ceil(m / 2) - 1] の範囲にある符号付きINT8の余り。
 * @~english
 * @brief Reduce the sum or difference of two symmetric residues back into the symmetric INT8 residue range.
 * @param[in] value Sum or difference of two symmetric residues.
 * @param[in] modulus Modulus at most 256.
 * @return Signed INT8 symmetric residue in [-floor(m / 2), ceil(m / 2) - 1].
 * @~
 */
static inline int8_t fp64_balance(int value, uint modulus) {
    int upper = int((modulus + 1) / 2);
    if (value >= upper) value -= int(modulus);
    if (value < upper - int(modulus)) value += int(modulus);
    return int8_t(value);
}

/**
 * @~japanese
 * @brief 2x2に分割した4つの入力ブロックから、Strassen法の7つのブロック積に入力する和と差の余りを直接生成する。
 * @tparam columns 行列Bのブロックを処理する場合はtrue、行列Aのブロックを処理する場合はfalse。
 * @tparam shifted 内積方向の指数の調整量を適用する場合はtrue。
 * @param[in] input 入力行列のFP64ビット列。
 * @param[in] scales 行または列ごとの解析結果。
 * @param[out] output 法および7つの演算ごとに連続して並ぶ符号付きINT8の余り配列。
 * @param[in] parameters 行列の寸法と整数幅。
 * @param[in] plan 使用する法と2のべき乗の余りの表。
 * @param[in] inner_shifts 内積方向の指数の調整量。shiftedがtrueの場合だけ参照する。
 * @param[in] position 分割後の1ブロック内の要素に対応する2次元グリッド上のスレッド位置。
 * @~english
 * @brief Directly generate the signed INT8 residue sums and differences for the seven Strassen block products
 * from four 2x2 input blocks.
 * @tparam columns true when processing blocks of matrix B, false when processing blocks of matrix A.
 * @tparam shifted true when applying inner-dimension exponent adjustments.
 * @param[in] input FP64 bit patterns of the input matrix.
 * @param[in] scales Per-row or per-column analysis results.
 * @param[out] output Output array of signed INT8 block sums and differences arranged by modulus and operation.
 * @param[in] parameters Matrix dimensions and integer widths.
 * @param[in] plan Active moduli and power-of-two residue tables.
 * @param[in] inner_shifts Inner-dimension exponent adjustments, read only when shifted is true.
 * @param[in] position Two-dimensional thread position corresponding to an element within one quarter-block.
 * @~
 */
template <bool columns, bool shifted>
kernel void strassen_operands(device const fp64_bits_t* input [[buffer(0)]],
                              device const fp64_scale_t* scales [[buffer(1)]],
                              device int8_t* output [[buffer(2)]],
                              constant fp64_batch_parameters_t& parameters [[buffer(3)]],
                              constant fp64_crt_plan_t& plan [[buffer(4)]],
                              device const int* inner_shifts [[buffer(5)]],
                              uint2 position [[thread_position_in_grid]]) {
    uint width = columns ? parameters.columns : parameters.inner;
    uint height = columns ? parameters.inner : parameters.rows;
    ulong count = ulong(width / 2) * (height / 2);
    ulong index = ulong(position.y) * min(count, ulong(65536)) + position.x;
    if (index >= count) return;
    uint row = uint(index / (width / 2)), column = uint(index % (width / 2));
    ulong offset = ulong(columns ? row : fp64_input_row(parameters, row)) * width + column;
    ulong lower = ulong(columns ? row + height / 2 : fp64_input_row(parameters, row + height / 2)) * width + column;
    uint first_scale = columns ? column : fp64_input_row(parameters, row);
    uint second_scale = columns ? column + width / 2 : fp64_input_row(parameters, row + height / 2);
    uint precision = columns ? parameters.precision_b : parameters.precision_a;
    int first_adjustment = 0, second_adjustment = 0;
    if constexpr (shifted) {
        first_adjustment = inner_shifts[columns ? row : column];
        second_adjustment = inner_shifts[columns ? row + height / 2 : column + width / 2];
        if constexpr (columns) {
            first_adjustment = -first_adjustment;
            second_adjustment = -second_adjustment;
        }
    }
    auto first_value = fp64_quantize<shifted>(input[offset], precision, scales[first_scale].exponent - first_adjustment);
    auto second_value = fp64_quantize<shifted>(input[offset + width / 2], precision,
                                             scales[columns ? second_scale : first_scale].exponent
                                             - (columns ? first_adjustment : second_adjustment));
    auto third_value = fp64_quantize<shifted>(input[lower], precision,
                                            scales[columns ? first_scale : second_scale].exponent
                                            - (columns ? second_adjustment : first_adjustment));
    auto fourth_value = fp64_quantize<shifted>(input[lower + width / 2], precision, scales[second_scale].exponent - second_adjustment);
    for (uint t = 0; t < plan.count; ++t) {
        device int8_t* destination = output + ulong(t) * 7 * count + index;
        fp64_modulus_t divisor = plan.moduli[t];
        int first = fp64_signed_residue(first_value.mantissa, first_value.negative, plan.powers[t][first_value.shift], divisor);
        int second = fp64_signed_residue(second_value.mantissa, second_value.negative, plan.powers[t][second_value.shift], divisor);
        int third = fp64_signed_residue(third_value.mantissa, third_value.negative, plan.powers[t][third_value.shift], divisor);
        int fourth = fp64_signed_residue(fourth_value.mantissa, fourth_value.negative, plan.powers[t][fourth_value.shift], divisor);
        uint modulus = plan.moduli[t].value;
        destination[0 * count] = fp64_balance(first + fourth, modulus);
        if constexpr (columns) {
            destination[1 * count] = int8_t(first);
            destination[2 * count] = fp64_balance(second - fourth, modulus);
            destination[3 * count] = fp64_balance(third - first, modulus);
            destination[4 * count] = int8_t(fourth);
            destination[5 * count] = fp64_balance(first + second, modulus);
            destination[6 * count] = fp64_balance(third + fourth, modulus);
        } else {
            destination[1 * count] = fp64_balance(third + fourth, modulus);
            destination[2 * count] = int8_t(first);
            destination[3 * count] = int8_t(fourth);
            destination[4 * count] = fp64_balance(first + second, modulus);
            destination[5 * count] = fp64_balance(third - first, modulus);
            destination[6 * count] = fp64_balance(second - fourth, modulus);
        }
    }
}

template [[host_name("strassen_operands_a")]] kernel void strassen_operands<false, false>(device const fp64_bits_t*, device const fp64_scale_t*, device int8_t*, constant fp64_batch_parameters_t&, constant fp64_crt_plan_t&, device const int*, uint2);
template [[host_name("strassen_operands_b")]] kernel void strassen_operands<true, false>(device const fp64_bits_t*, device const fp64_scale_t*, device int8_t*, constant fp64_batch_parameters_t&, constant fp64_crt_plan_t&, device const int*, uint2);
template [[host_name("strassen_shifted_operands_a")]] kernel void strassen_operands<false, true>(device const fp64_bits_t*, device const fp64_scale_t*, device int8_t*, constant fp64_batch_parameters_t&, constant fp64_crt_plan_t&, device const int*, uint2);
template [[host_name("strassen_shifted_operands_b")]] kernel void strassen_operands<true, true>(device const fp64_bits_t*, device const fp64_scale_t*, device int8_t*, constant fp64_batch_parameters_t&, constant fp64_crt_plan_t&, device const int*, uint2);

/**
 * @~japanese
 * @brief INT8の余りの行列積をタイルごとに計算し、法で割った非負の余り（0以上m未満）を保存する。
 * @tparam single_chunk 内積の項数がFP64_K_CHUNK（1024）以下であり、1回の行列積で厳密な内積が求まる場合はtrue。
 * @param[in] a 法（Strassen法では法と演算）ごとに並ぶ行列Aの余り。
 * @param[in] b 法（Strassen法では法と演算）ごとに並ぶ行列Bの余り。
 * @param[out] output 法（Strassen法では法と演算）ごとに並ぶ、0以上m未満の出力の余り。
 * @param[in] parameters 行列の寸法と整数幅。
 * @param[in] plan 使用する法と剰余計算用の定数。
 * @param[in] scales_a 行列Aの行ごとの解析結果（法256での下位ゼロビット数の判定に使用する）。
 * @param[in] scales_b 行列Bの列ごとの解析結果（法256での下位ゼロビット数の判定に使用する）。
 * @param[in] group 担当する出力タイル（列タイル、行タイル、法または演算の番号）。
 * @param[in] lane SIMDグループ内のスレッド番号（0～31）。
 * @param[in] thread_index スレッドグループ内の通し番号。
 * @~english
 * @brief Compute a tile of an INT8 modular matrix product and store nonnegative residues in [0, m).
 * @tparam single_chunk true when the dot-product length is at most FP64_K_CHUNK (1024) so that a single matrix
 * multiplication preserves exact integer sums.
 * @param[in] a INT8 residues of A arranged by modulus (and by Strassen operation when active).
 * @param[in] b INT8 residues of B arranged by modulus (and by Strassen operation when active).
 * @param[out] output Nonnegative output residues in [0, m) arranged by modulus (and Strassen operation).
 * @param[in] parameters Matrix dimensions and integer widths.
 * @param[in] plan Active moduli and reduction constants.
 * @param[in] scales_a Per-row analysis of A, used to check trailing zero bits for modulus 256.
 * @param[in] scales_b Per-column analysis of B, used to check trailing zero bits for modulus 256.
 * @param[in] group Assigned output tile (column tile, row tile, modulus or operation index).
 * @param[in] lane Thread index within the SIMD group (0 to 31).
 * @param[in] thread_index Linear thread index within the threadgroup.
 * @~
 */
template <bool single_chunk>
kernel void residue_matmul(device int8_t* a [[buffer(0)]],
                           device int8_t* b [[buffer(1)]],
                           device uchar* output [[buffer(2)]],
                           constant fp64_batch_parameters_t& parameters [[buffer(3)]],
                           constant fp64_crt_plan_t& plan [[buffer(4)]],
                           device const fp64_scale_t* scales_a [[buffer(5)]],
                           device const fp64_scale_t* scales_b [[buffer(6)]],
                           uint3 group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint thread_index [[thread_index_in_threadgroup]]) {
    uint row = group.y * FP64_TILE_ROWS, column = group.x * FP64_TILE_COLUMNS;
    fp64_modulus_t modulus = plan.moduli[parameters.strassen ? group.z / 7 : group.z];
    if (modulus.value == 256) {
        // 法256では、タイル内のAの行とBの列における下位ゼロビット数の下限の和が8以上であれば、
        // 内積のすべての項が 2^8 = 256 の倍数となるため、行列積の実行を省いて0を書き込む。
        uint trailing_a = 8, trailing_b = 8;
        for (uint index = lane; index < FP64_TILE_ROWS; index += 32) {
            uint input_row = row + index;
            if (input_row < parameters.rows) {
                trailing_a = min(trailing_a, fp64_trailing_bits(scales_a[parameters.row_begin + input_row], parameters.precision_a));
                if (parameters.strassen)
                    trailing_a = min(trailing_a, fp64_trailing_bits(scales_a[parameters.row_begin + input_row + parameters.total_rows / 2], parameters.precision_a));
            }
        }
        for (uint index = lane; index < FP64_TILE_COLUMNS; index += 32) {
            uint input_column = column + index;
            if (input_column < parameters.columns) {
                trailing_b = min(trailing_b, fp64_trailing_bits(scales_b[input_column], parameters.precision_b));
                if (parameters.strassen)
                    trailing_b = min(trailing_b, fp64_trailing_bits(scales_b[input_column + parameters.columns], parameters.precision_b));
            }
        }
        if (simd_min(trailing_a) + simd_min(trailing_b) >= 8) {
            for (uint index = thread_index; index < FP64_TILE_ROWS * FP64_TILE_COLUMNS; index += 32 * FP64_SIMD_GROUPS) {
                uint output_row = row + index / FP64_TILE_COLUMNS;
                uint output_column = column + index % FP64_TILE_COLUMNS;
                if (output_row < parameters.rows && output_column < parameters.columns)
                    output[(ulong(group.z) * parameters.rows + output_row) * parameters.columns + output_column] = 0;
            }
            return;
        }
    }
    constexpr auto descriptor = matmul2d_descriptor(FP64_TILE_ROWS, FP64_TILE_COLUMNS, int(dynamic_extent), false, false, false);
    matmul2d<descriptor, execution_simdgroups<FP64_SIMD_GROUPS>> operation;
    device int8_t* a_base = a + ulong(group.z) * parameters.rows * parameters.inner;
    device int8_t* b_base = b + ulong(group.z) * parameters.inner * parameters.columns;
    auto a_tensor = tensor(a_base, dextents<int, 2>{int(parameters.inner), int(parameters.rows)}, array<int, 2>{1, int(parameters.inner)});
    auto b_tensor = tensor(b_base, dextents<int, 2>{int(parameters.columns), int(parameters.inner)}, array<int, 2>{1, int(parameters.columns)});
    auto accumulated = operation.get_destination_cooperative_tensor<decltype(a_tensor), decltype(b_tensor), int>();
    #pragma unroll
    for (uint i = 0; i < accumulated.get_capacity(); ++i) {
        if (accumulated.is_valid_element(i)) accumulated[i] = 0;
    }
    if constexpr (single_chunk) {
        if (row + FP64_TILE_ROWS <= parameters.rows && column + FP64_TILE_COLUMNS <= parameters.columns && parameters.inner == FP64_K_CHUNK) {
            auto full_a = a_tensor.slice<FP64_K_CHUNK, FP64_TILE_ROWS>(0, int(row));
            auto full_b = b_tensor.slice<FP64_TILE_COLUMNS, FP64_K_CHUNK>(int(column), 0);
            operation.run(full_a, full_b, accumulated);
        } else {
            auto chunk_a = a_tensor.slice(0, int(row));
            auto chunk_b = b_tensor.slice(int(column), 0);
            operation.run(chunk_a, chunk_b, accumulated);
        }
    } else {
        for (uint begin = 0; begin < parameters.inner; begin += FP64_K_CHUNK) {
            uint length = min(FP64_K_CHUNK, parameters.inner - begin);
            auto chunk_a = tensor(a_base + begin, dextents<int, 2>{int(length), int(parameters.rows)}, array<int, 2>{1, int(parameters.inner)}).slice(0, int(row));
            auto chunk_b = tensor(b_base + ulong(begin) * parameters.columns, dextents<int, 2>{int(parameters.columns), int(length)}, array<int, 2>{1, int(parameters.columns)}).slice(int(column), 0);
            auto partial = operation.get_destination_cooperative_tensor<decltype(chunk_a), decltype(chunk_b), int>();
            #pragma unroll
            for (uint i = 0; i < partial.get_capacity(); ++i) {
                if (partial.is_valid_element(i)) partial[i] = 0;
            }
            if (row + FP64_TILE_ROWS <= parameters.rows && column + FP64_TILE_COLUMNS <= parameters.columns && length == FP64_K_CHUNK) {
                auto full_a = a_tensor.slice<FP64_K_CHUNK, FP64_TILE_ROWS>(int(begin), int(row));
                auto full_b = b_tensor.slice<FP64_TILE_COLUMNS, FP64_K_CHUNK>(int(column), int(begin));
                operation.run(full_a, full_b, partial);
            } else {
                operation.run(chunk_a, chunk_b, partial);
            }
            #pragma unroll
            for (uint i = 0; i < accumulated.get_capacity(); ++i) {
                if (accumulated.is_valid_element(i)) {
                    accumulated[i] += partial[i];
                }
            }
            // 符号付き32ビット整数のオーバーフローを防ぐため、65536項ごとに法で割った余りへ戻す。
            if ((begin + length) % FP64_K_ACCUMULATE == 0 && begin + length < parameters.inner) {
                #pragma unroll
                for (uint i = 0; i < accumulated.get_capacity(); ++i) {
                    if (accumulated.is_valid_element(i)) {
                        uint remainder = fp64_word_mod(uint(abs(accumulated[i])), modulus);
                        if (accumulated[i] < 0 && remainder != 0) remainder = modulus.value - remainder;
                        accumulated[i] = int(remainder);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (uint i = 0; i < accumulated.get_capacity(); ++i) {
        if (accumulated.is_valid_element(i)) {
            auto coordinates = accumulated.get_multidimensional_index(i);
            uint output_column = column + uint(coordinates[0]);
            uint output_row = row + uint(coordinates[1]);
            if (output_column < parameters.columns && output_row < parameters.rows) {
                uint remainder = fp64_word_mod(uint(abs(accumulated[i])), modulus);
                if (accumulated[i] < 0 && remainder != 0) remainder = modulus.value - remainder;
                output[(ulong(group.z) * parameters.rows + output_row) * parameters.columns + output_column] = uchar(remainder);
            }
        }
    }
}

template [[host_name("residue_matmul_chunk")]] kernel void residue_matmul<true>(device int8_t*, device int8_t*, device uchar*, constant fp64_batch_parameters_t&, constant fp64_crt_plan_t&, device const fp64_scale_t*, device const fp64_scale_t*, uint3, uint, uint);
template [[host_name("residue_matmul_accumulate")]] kernel void residue_matmul<false>(device int8_t*, device int8_t*, device uchar*, constant fp64_batch_parameters_t&, constant fp64_crt_plan_t&, device const fp64_scale_t*, device const fp64_scale_t*, uint3, uint, uint);

/**
 * @~japanese
 * @brief Strassen法の7つのブロック積の余りを加減算し、2x2の4つの出力ブロックの余りを組み立てる。
 * @param[in] residues 法および7つの演算ごとに並ぶブロック積の余り。
 * @param[out] output 法ごとに並ぶ出力行列の余り。
 * @param[in] parameters 行列の寸法。
 * @param[in] plan 使用する法と剰余計算用の定数。
 * @param[in] position 出力ブロック内の要素の2次元位置（x, y）と、4つの出力ブロックの番号（z = 0..3）。
 * @~english
 * @brief Combine the residues of the seven Strassen block products into the residues of the four 2x2 output
 * blocks.
 * @param[in] residues Residues of the seven block products arranged by modulus and operation.
 * @param[out] output Output matrix residues arranged by modulus.
 * @param[in] parameters Matrix dimensions.
 * @param[in] plan Active moduli and reduction constants.
 * @param[in] position Two-dimensional element position within an output block (x, y) and output block index
 * (z = 0..3).
 * @~
 */
kernel void strassen_combine(device const uchar* residues [[buffer(0)]],
                             device uchar* output [[buffer(1)]],
                             constant fp64_batch_parameters_t& parameters [[buffer(2)]],
                             constant fp64_crt_plan_t& plan [[buffer(3)]],
                             uint3 position [[thread_position_in_grid]]) {
    uint half_columns = parameters.columns / 2, half_rows = parameters.rows / 2;
    ulong block_count = ulong(half_rows) * half_columns;
    ulong index = ulong(position.y) * min(block_count, ulong(65536)) + position.x;
    if (index >= block_count) return;
    ulong output_index = (index / half_columns + (position.z / 2) * half_rows) * parameters.columns
                       + index % half_columns + (position.z % 2) * half_columns;
    ulong count = ulong(parameters.rows) * parameters.columns;
    for (uint t = 0; t < plan.count; ++t) {
        device const uchar* blocks = residues + ulong(t) * 7 * block_count + index;
        int combined;
        switch (position.z) {
            case 0: combined = int(blocks[0 * block_count]) + blocks[3 * block_count]
                               - blocks[4 * block_count] + blocks[6 * block_count]; break;
            case 1: combined = int(blocks[2 * block_count]) + blocks[4 * block_count]; break;
            case 2: combined = int(blocks[1 * block_count]) + blocks[3 * block_count]; break;
            default: combined = int(blocks[0 * block_count]) - blocks[1 * block_count]
                                + blocks[2 * block_count] + blocks[5 * block_count]; break;
        }
        fp64_modulus_t modulus = plan.moduli[t];
        // combined には最大1個のブロック積が減算されるため、2 * m を足して非負にしてから余りを求める。
        output[ulong(t) * count + output_index] = uchar(fp64_word_mod(uint(combined + int(2 * modulus.value)), modulus));
    }
}

/**
 * @~japanese
 * @brief 各出力要素について法ごとの余りからCRTで厳密な内積を復元し、FP64へ丸めたビット列を書き出す。
 * @tparam limb_count 各スレッドが保持する多倍長整数の桁数。
 * @param[in] residues 法ごとに並ぶ出力の余り。
 * @param[in] scales_a 行列Aの行ごとの解析結果。
 * @param[in] scales_b 行列Bの列ごとの解析結果。
 * @param[out] output 出力行列のFP64ビット列。
 * @param[in] parameters 行列の寸法と整数幅。
 * @param[in] plan CRTの復元係数と法の積。
 * @param[in] position 行のまとまり内の出力要素に対応する2次元グリッド上のスレッド位置。
 * @~english
 * @brief Reconstruct the exact dot product for each output element from its modular residues using CRT and
 * write the rounded FP64 bit pattern.
 * @tparam limb_count Number of multi-precision integer digits held by each thread.
 * @param[in] residues Output residues arranged by modulus.
 * @param[in] scales_a Per-row analysis of matrix A.
 * @param[in] scales_b Per-column analysis of matrix B.
 * @param[out] output Output FP64 bit patterns.
 * @param[in] parameters Matrix dimensions and integer widths.
 * @param[in] plan CRT reconstruction coefficients and modulus product.
 * @param[in] position Two-dimensional thread position corresponding to an output element within the row batch.
 * @~
 */
template <uint limb_count>
kernel void reconstruct(device const uchar* residues [[buffer(0)]],
                        device const fp64_scale_t* scales_a [[buffer(1)]],
                        device const fp64_scale_t* scales_b [[buffer(2)]],
                        device fp64_bits_t* output [[buffer(3)]],
                        constant fp64_batch_parameters_t& parameters [[buffer(4)]],
                        constant fp64_crt_plan_t& plan [[buffer(5)]],
                        uint2 position [[thread_position_in_grid]]) {
    ulong count = ulong(parameters.rows) * parameters.columns;
    ulong index = ulong(position.y) * min(count, ulong(65536)) + position.x;
    if (index >= count) return;
    fp64_big_uint_s<limb_count> sums = {}, value = {}, multiple = {}, product = {}, half_product = {};
    float quotient = 0;
    for (uint t = 0; t < plan.count; ++t) {
        uint residue = residues[ulong(t) * count + index];
        quotient += float(residue) * plan.ratios[t];
        #pragma unroll
        for (uint digit = 0; digit < limb_count; ++digit)
            sums.digits[digit] += residue * plan.coefficients[t].digits[digit];
    }
    uint multiplier = uint(quotient + 0.5f), carry = 0, product_carry = 0;
    #pragma unroll
    for (uint digit = 0; digit < limb_count; ++digit) {
        uint total = sums.digits[digit] + carry;
        value.digits[digit] = total & FP64_DIGIT_MASK;
        carry = total >> FP64_DIGIT_BITS;
        total = multiplier * plan.product.digits[digit] + product_carry;
        multiple.digits[digit] = total & FP64_DIGIT_MASK;
        product_carry = total >> FP64_DIGIT_BITS;
        product.digits[digit] = plan.product.digits[digit];
        half_product.digits[digit] = plan.half_product.digits[digit];
    }
    // FP32の比から求めた近似商 multiplier の誤差は 0.1 未満であるため、差 |value - multiple| は必ず P 未満になる。
    // その差が P / 2 を超える場合は P を1回引いて対称な範囲 [-P / 2, P / 2) の絶対値と符号を確定する。
    bool negative = fp64_big_compare(value, multiple, limb_count) < 0;
    value = negative ? fp64_big_subtract(multiple, value, limb_count)
                     : fp64_big_subtract(value, multiple, limb_count);
    if (fp64_big_compare(value, half_product, limb_count) > 0) {
        value = fp64_big_subtract(product, value, limb_count);
        negative = !negative;
    }
    int scale = scales_a[fp64_input_row(parameters, uint(index / parameters.columns))].exponent + scales_b[index % parameters.columns].exponent
              - int(parameters.precision_a) - int(parameters.precision_b);
    ulong output_index = ulong(fp64_input_row(parameters, uint(index / parameters.columns))) * parameters.columns
                       + index % parameters.columns;
    output[output_index] = fp64_pack_fp64(value, negative, scale, limb_count);
}

/**
 * @~japanese
 * @brief 指定した多倍長整数の桁数でCRT復元用カーネルのテンプレートを実体化する補助マクロ。
 * @param[in] name Metalライブラリー内のカーネル名の接尾辞。
 * @param[in] size 各スレッドが保持する多倍長整数の桁数。
 * @~english
 * @brief Helper macro instantiating the CRT reconstruction kernel template for a specified digit capacity.
 * @param[in] name Kernel name suffix in the Metal library.
 * @param[in] size Number of multi-precision integer digits held by each thread.
 * @~
 */
#define FP64_RECONSTRUCTION_KERNEL(name, size) \
    template [[host_name("reconstruct_" #name)]] kernel void reconstruct<size>(device const uchar*, device const fp64_scale_t*, device const fp64_scale_t*, device fp64_bits_t*, constant fp64_batch_parameters_t&, constant fp64_crt_plan_t&, uint2);
FP64_RECONSTRUCTION_KERNELS(FP64_RECONSTRUCTION_KERNEL)
#undef FP64_RECONSTRUCTION_KERNEL

/**
 * @~japanese
 * @brief スレッドグループの共有メモリーに分解済みのタイルを読み込み、GPUの整数演算によるFP64の積和演算で行列積を求める。
 * @tparam finite_inputs 行列AとBの両方が有限値だけを含むと分かっている場合はtrue。
 * @param[in] a 行列AのFP64ビット列。
 * @param[in] b 行列BのFP64ビット列。
 * @param[out] output 出力行列のFP64ビット列。
 * @param[in] parameters 行列全体の寸法。
 * @param[in] group 担当する16x16の出力タイル（列タイル、行タイル）。
 * @param[in] local 16x16タイル内のスレッド位置（列オフセット、行オフセット）。
 * @~english
 * @brief Load unpacked input tiles into threadgroup shared memory and compute the matrix product using
 * integer-based FP64 fused multiply-add on the GPU.
 * @tparam finite_inputs true when both input matrices A and B are known to contain only finite values.
 * @param[in] a FP64 bit patterns of matrix A.
 * @param[in] b FP64 bit patterns of matrix B.
 * @param[out] output Output FP64 bit patterns.
 * @param[in] parameters Dimensions of the full matrices.
 * @param[in] group Assigned 16x16 output tile (column tile, row tile).
 * @param[in] local Thread coordinates within the 16x16 tile (column offset, row offset).
 * @~
 */
template <bool finite_inputs>
kernel void floating_matmul(device const fp64_bits_t* a [[buffer(0)]],
                            device const fp64_bits_t* b [[buffer(1)]],
                            device fp64_bits_t* output [[buffer(2)]],
                            constant fp64_batch_parameters_t& parameters [[buffer(3)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint2 local [[thread_position_in_threadgroup]]) {
    threadgroup fp64_fma_operand_t tile_a[16 * 32], tile_b[32 * 16];
    uint row = group.y * 16 + local.y, column = group.x * 16 + local.x;
    uint thread_index = local.y * 16 + local.x;
    fp64_bits_t sum = {0, 0};
    for (uint begin = 0; begin < parameters.inner;) {
        uint length = min(32u, parameters.inner - begin);
        for (uint index = thread_index; index < 512; index += 256) {
            uint ar = group.y * 16 + index / 32, ak = index % 32;
            uint bk = index / 16, bc = group.x * 16 + index % 16;
            tile_a[index] = fp64_unpack_operand(ar < parameters.rows && ak < length ? a[ulong(ar) * parameters.inner + begin + ak] : fp64_bits_t{0, 0});
            tile_b[index] = fp64_unpack_operand(bc < parameters.columns && bk < length ? b[ulong(begin + bk) * parameters.columns + bc] : fp64_bits_t{0, 0});
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = 0; index < length; ++index)
            sum = fp64_fused_multiply_add(tile_a[local.y * 32 + index], tile_b[index * 16 + local.x], sum, finite_inputs);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        begin += length;
    }
    if (row < parameters.rows && column < parameters.columns) output[ulong(row) * parameters.columns + column] = sum;
}

template [[host_name("floating_matmul_finite")]] kernel void floating_matmul<true>(device const fp64_bits_t*, device const fp64_bits_t*, device fp64_bits_t*, constant fp64_batch_parameters_t&, uint2, uint2);
template [[host_name("floating_matmul_general")]] kernel void floating_matmul<false>(device const fp64_bits_t*, device const fp64_bits_t*, device fp64_bits_t*, constant fp64_batch_parameters_t&, uint2, uint2);
