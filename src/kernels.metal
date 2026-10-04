#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "arithmetic.h"

using namespace metal;
using namespace mpp::tensor_ops;

/**
 * @brief 行のまとまりの位置を、入力Aの行へ対応させる。
 * @param[in] parameters 行列の寸法と処理するブロック。
 * @param[in] row 行のまとまりの中の位置。
 * @return 入力Aにおける行の位置。
 */
static inline uint fp64_input_row(constant fp64_batch_parameters_t& parameters, uint row) {
    if (!parameters.strassen) return parameters.row_begin + row;
    uint half_rows = parameters.rows / 2;
    return parameters.row_begin + (row < half_rows ? row : row - half_rows + parameters.total_rows / 2);
}

/**
 * @brief FP64の絶対値に含まれる、最下位の非ゼロのビットの指数を求める。
 * @param[in] bits 有限のFP64のビット列。
 * @return 非ゼロ値ではビットの指数。ゼロでは符号付き32ビット整数の最大値。
 */
static inline int fp64_lowest_bit(fp64_bits_t bits) {
    uint exponent = (bits.high >> 20) & 2047u;
    uint high = bits.high & 0xfffffu;
    if (exponent != 0) high |= 0x100000u;
    int scale = exponent != 0 ? int(exponent) - 1075 : -1074;
    if (bits.low != 0) return scale + int(fp64_word_bit_length(bits.low & (0u - bits.low))) - 1;
    if (high != 0) return scale + 32 + int(fp64_word_bit_length(high & (0u - high))) - 1;
    return 2147483647;
}

/**
 * @brief Aの行に含まれる最大値の指数を求める。
 * @param[in] input FP64のビット列。
 * @param[out] scales 各行の指数。
 * @param[in] parameters 行列の寸法。
 * @param[in] group 担当する行。
 * @param[in] lane SIMDグループ内のスレッド位置。
 */
kernel void find_row_scales(device const fp64_bits_t* input [[buffer(0)]],
                        device fp64_scale_t* scales [[buffer(1)]],
                        constant fp64_batch_parameters_t& parameters [[buffer(2)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    int maximum = FP64_ZERO_EXPONENT;
    int minimum = 2147483647;
    for (uint index = lane; index < parameters.inner; index += 32) {
        ulong offset = ulong(fp64_input_row(parameters, group)) * parameters.inner + index;
        fp64_bits_t bits = input[offset];
        maximum = max(maximum, fp64_exponent(bits));
        minimum = min(minimum, fp64_lowest_bit(bits));
    }
    maximum = simd_max(maximum);
    minimum = simd_min(minimum);
    if (lane == 0) {
        bool zero = maximum == FP64_ZERO_EXPONENT;
        scales[group] = {zero ? 0 : maximum, zero || minimum + int(parameters.precision_a) - maximum >= 8};
    }
}

/**
 * @brief Bの隣接する列を同時に読み、各列の最大値の指数を求める。
 * @param[in] input FP64のビット列。
 * @param[out] scales 各列の指数。
 * @param[in] parameters 行列の寸法。
 * @param[in] group 担当する32列のまとまり。
 * @param[in] lane SIMDグループ内の列位置。
 * @param[in] subgroup 内積方向を分担するSIMDグループの位置。
 */
kernel void find_column_scales(device const fp64_bits_t* input [[buffer(0)]],
                               device fp64_scale_t* scales [[buffer(1)]],
                               constant fp64_batch_parameters_t& parameters [[buffer(2)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint subgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup int maxima[FP64_COLUMN_SIMD_GROUPS][32];
    threadgroup int minima[FP64_COLUMN_SIMD_GROUPS][32];
    uint column = group * 32 + lane;
    int maximum = FP64_ZERO_EXPONENT;
    int minimum = 2147483647;
    if (column < parameters.columns) {
        for (uint row = subgroup; row < parameters.inner; row += FP64_COLUMN_SIMD_GROUPS) {
            fp64_bits_t bits = input[ulong(row) * parameters.columns + column];
            maximum = max(maximum, fp64_exponent(bits));
            minimum = min(minimum, fp64_lowest_bit(bits));
        }
    }
    maxima[subgroup][lane] = maximum;
    minima[subgroup][lane] = minimum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (subgroup == 0 && column < parameters.columns) {
        #pragma unroll
        for (uint index = 1; index < FP64_COLUMN_SIMD_GROUPS; ++index) {
            maximum = max(maximum, maxima[index][lane]);
            minimum = min(minimum, minima[index][lane]);
        }
        bool zero = maximum == FP64_ZERO_EXPONENT;
        scales[column] = {zero ? 0 : maximum, zero || minimum + int(parameters.precision_b) - maximum >= 8};
    }
}

/** @brief 整数化で残す仮数と、剰余の係数を参照する指数。 */
struct fp64_quantized_input_s {
    fp64_bits_t mantissa; /**< 切り捨てた仮数の絶対値。 */
    uint shift; /**< 仮数に掛ける2のべき乗の指数。 */
    bool negative; /**< 入力の符号。 */
};

/**
 * @brief FP64の入力を、整数化で残す仮数と指数へ分解する。
 * @param[in] bits 入力のビット列。
 * @param[in] precision 整数幅。
 * @param[in] scale 行または列の最大値の指数。
 * @return 法に依存しない整数化済みの仮数と指数。
 */
static inline fp64_quantized_input_s fp64_quantize(fp64_bits_t bits, uint precision, int scale) {
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
 * @brief 入力の各要素からすべての法の余りを生成する。
 * @param[in] input FP64のビット列。
 * @param[in] scales 行または列の指数。
 * @param[out] output 法ごとに連続したINT8の行列。
 * @param[in] parameters 行列の寸法と整数幅。
 * @param[in] plan 使用する法。
 * @param[in] column_mode Bを処理する場合は1、Aを処理する場合は0。
 * @param[in] position 対象要素の二次元の位置。
 */
kernel void make_residues(device const fp64_bits_t* input [[buffer(0)]],
                          device const fp64_scale_t* scales [[buffer(1)]],
                          device int8_t* output [[buffer(2)]],
                          constant fp64_batch_parameters_t& parameters [[buffer(3)]],
                          constant fp64_crt_plan_t& plan [[buffer(4)]],
                          constant uint& column_mode [[buffer(5)]],
                          uint2 position [[thread_position_in_grid]]) {
    ulong count = column_mode ? ulong(parameters.inner) * parameters.columns
                             : ulong(parameters.rows) * parameters.inner;
    ulong index = ulong(position.y) * min(count, ulong(65536)) + position.x;
    if (index >= count) return;
    uint scale_index = column_mode ? uint(index % parameters.columns) : uint(index / parameters.inner);
    ulong input_index = column_mode ? index
                                   : ulong(fp64_input_row(parameters, scale_index)) * parameters.inner + index % parameters.inner;
    uint precision = column_mode ? parameters.precision_b : parameters.precision_a;
    auto value = fp64_quantize(input[input_index], precision, scales[scale_index].exponent);
    for (uint t = 0; t < plan.count; ++t) {
        output[ulong(t) * count + index] = int8_t(fp64_signed_residue(value.mantissa, value.negative,
                                                                   plan.powers[t][value.shift], plan.moduli[t]));
    }
}

/**
 * @brief 対称な余り二つの和または差を、同じ範囲へ戻す。
 * @param[in] value 対称な余り二つの和または差。
 * @param[in] modulus 256以下の法。
 * @return INT8に収まる対称な余り。
 */
static inline int8_t fp64_balance(int value, uint modulus) {
    int upper = int((modulus + 1) / 2);
    if (value >= upper) value -= int(modulus);
    if (value < upper - int(modulus)) value += int(modulus);
    return int8_t(value);
}

/**
 * @brief 四つの入力ブロックから、Strassen法の七つの演算の入力を生成する。
 * @tparam columns Bのブロックを処理する場合はtrue。
 * @param[in] input FP64のビット列。
 * @param[in] scales 行または列の最大値の指数。
 * @param[out] output 法と演算ごとに並ぶ、ブロックの和と差。
 * @param[in] parameters 行列の寸法。
 * @param[in] plan 使用する法。
 * @param[in] position ブロック内の二次元の位置。
 */
template <bool columns>
kernel void strassen_operands(device const fp64_bits_t* input [[buffer(0)]],
                              device const fp64_scale_t* scales [[buffer(1)]],
                              device int8_t* output [[buffer(2)]],
                              constant fp64_batch_parameters_t& parameters [[buffer(3)]],
                              constant fp64_crt_plan_t& plan [[buffer(4)]],
                              uint2 position [[thread_position_in_grid]]) {
    uint width = columns ? parameters.columns : parameters.inner;
    uint height = columns ? parameters.inner : parameters.rows;
    ulong count = ulong(width / 2) * (height / 2);
    ulong index = ulong(position.y) * min(count, ulong(65536)) + position.x;
    if (index >= count) return;
    uint row = uint(index / (width / 2)), column = uint(index % (width / 2));
    ulong offset = ulong(columns ? row : fp64_input_row(parameters, row)) * width + column;
    ulong lower = ulong(columns ? row + height / 2 : fp64_input_row(parameters, row + height / 2)) * width + column;
    uint first_scale = columns ? column : row;
    uint second_scale = columns ? column + width / 2 : row + height / 2;
    uint precision = columns ? parameters.precision_b : parameters.precision_a;
    auto first_value = fp64_quantize(input[offset], precision, scales[first_scale].exponent);
    auto second_value = fp64_quantize(input[offset + width / 2], precision, scales[columns ? second_scale : first_scale].exponent);
    auto third_value = fp64_quantize(input[lower], precision, scales[columns ? first_scale : second_scale].exponent);
    auto fourth_value = fp64_quantize(input[lower + width / 2], precision, scales[second_scale].exponent);
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

template [[host_name("strassen_operands_a")]] kernel void strassen_operands<false>(device const fp64_bits_t*, device const fp64_scale_t*, device int8_t*, constant fp64_batch_parameters_t&, constant fp64_crt_plan_t&, uint2);
template [[host_name("strassen_operands_b")]] kernel void strassen_operands<true>(device const fp64_bits_t*, device const fp64_scale_t*, device int8_t*, constant fp64_batch_parameters_t&, constant fp64_crt_plan_t&, uint2);

/**
 * @brief INT8の行列積を計算し、法ごとの余りだけを保存する。
 * @tparam single_chunk 内積全体の整数精度を一回の行列積で保てる場合はtrue。
 * @param[in] a 法ごとに並ぶAの余り。
 * @param[in] b 法ごとに並ぶBの余り。
 * @param[out] output 法ごとに並ぶ、0以上の出力の余り。
 * @param[in] parameters 行列の寸法。
 * @param[in] plan 使用する法。
 * @param[in] scales_a Aの行の指数と、法256の余りの情報。
 * @param[in] scales_b Bの列の指数と、法256の余りの情報。
 * @param[in] group 列、行、法の順で表す担当タイル。
 * @param[in] lane SIMDグループ内のスレッド位置。
 * @param[in] thread_index スレッドグループ内の位置。
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
        bool zero_a = true, zero_b = true;
        for (uint index = lane; index < FP64_TILE_ROWS; index += 32) {
            uint input_row = row + index;
            if (input_row < parameters.rows) {
                zero_a = zero_a && scales_a[input_row].zero_mod_256;
                if (parameters.strassen) zero_a = zero_a && scales_a[input_row + parameters.rows].zero_mod_256;
            }
        }
        for (uint index = lane; index < FP64_TILE_COLUMNS; index += 32) {
            uint input_column = column + index;
            if (input_column < parameters.columns) {
                zero_b = zero_b && scales_b[input_column].zero_mod_256;
                if (parameters.strassen) zero_b = zero_b && scales_b[input_column + parameters.columns].zero_mod_256;
            }
        }
        if (simd_all(zero_a) || simd_all(zero_b)) {
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
 * @brief 七つのブロック積から、四つの出力ブロックの余りを組み立てる。
 * @param[in] residues 法と演算ごとに並ぶ七つの積の余り。
 * @param[out] output 法ごとに並ぶ出力の余り。
 * @param[in] parameters 行列の寸法。
 * @param[in] plan 使用する法。
 * @param[in] position ブロックの中の二次元の位置と、出力ブロックの番号。
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
        output[ulong(t) * count + output_index] = uchar(fp64_word_mod(uint(combined + int(2 * modulus.value)), modulus));
    }
}

/**
 * @brief 出力要素ごとにCRTを復元し、FP64のビット列を保存する。
 * @tparam limb_count スレッドが保持する整数の桁数。
 * @param[in] residues 法ごとに並ぶ出力の余り。
 * @param[in] scales_a Aの行の指数。
 * @param[in] scales_b Bの列の指数。
 * @param[out] output FP64のビット列。
 * @param[in] parameters 行列の寸法と整数幅。
 * @param[in] plan CRTの係数。
 * @param[in] position 行のまとまりの中の二次元の出力位置。
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
    bool negative = fp64_big_compare(value, multiple, limb_count) < 0;
    value = negative ? fp64_big_subtract(multiple, value, limb_count)
                     : fp64_big_subtract(value, multiple, limb_count);
    if (fp64_big_compare(value, half_product, limb_count) > 0) {
        value = fp64_big_subtract(product, value, limb_count);
        negative = !negative;
    }
    int scale = scales_a[index / parameters.columns].exponent + scales_b[index % parameters.columns].exponent
              - int(parameters.precision_a) - int(parameters.precision_b);
    ulong output_index = ulong(fp64_input_row(parameters, uint(index / parameters.columns))) * parameters.columns
                       + index % parameters.columns;
    output[output_index] = fp64_pack_fp64(value, negative, scale, limb_count);
}

/**
 * @brief 整数配列の容量を指定して、Metalの復元カーネルを宣言する。
 * @param[in] name Metalで使用する名前の接尾辞。
 * @param[in] size スレッドが保持する整数の桁数。
 */
#define FP64_RECONSTRUCTION_KERNEL(name, size) \
    template [[host_name("reconstruct_" #name)]] kernel void reconstruct<size>(device const uchar*, device const fp64_scale_t*, device const fp64_scale_t*, device fp64_bits_t*, constant fp64_batch_parameters_t&, constant fp64_crt_plan_t&, uint2);
FP64_RECONSTRUCTION_KERNELS(FP64_RECONSTRUCTION_KERNEL)
#undef FP64_RECONSTRUCTION_KERNEL
