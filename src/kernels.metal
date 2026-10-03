#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "arithmetic.h"

using namespace metal;
using namespace mpp::tensor_ops;

/**
 * @brief Aの行またはBの列に含まれる最大値の指数を求める。
 * @param[in] input FP64のビット列。
 * @param[out] scales 各行または各列の指数。
 * @param[in] parameters 行列の寸法。
 * @param[in] column_mode Bの列を処理する場合は1、Aの行を処理する場合は0。
 * @param[in] group 担当する行または列。
 * @param[in] lane SIMDグループ内のスレッド位置。
 */
kernel void find_scales(device const fp64_bits_t* input [[buffer(0)]],
                        device int* scales [[buffer(1)]],
                        constant fp64_batch_parameters_t& parameters [[buffer(2)]],
                        constant uint& column_mode [[buffer(3)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    int maximum = FP64_ZERO_EXPONENT;
    for (uint index = lane; index < parameters.inner; index += 32) {
        ulong offset = column_mode ? ulong(index) * parameters.columns + group
                                  : ulong(parameters.row_begin + group) * parameters.inner + index;
        maximum = max(maximum, fp64_exponent(input[offset]));
    }
    maximum = simd_max(maximum);
    if (lane == 0) scales[group] = maximum == FP64_ZERO_EXPONENT ? 0 : maximum;
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
                          device const int* scales [[buffer(1)]],
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
    fp64_bits_t bits = input[index + (column_mode ? 0 : ulong(parameters.row_begin) * parameters.inner)];
    uint precision = column_mode ? parameters.precision_b : parameters.precision_a;
    uint raw_exponent = (bits.high >> 20) & 2047u;
    fp64_bits_t mantissa = {bits.low, bits.high & 0xfffffu};
    int exponent = -1074;
    if (raw_exponent != 0) {
        mantissa.high |= 0x100000u;
        exponent = int(raw_exponent) - 1075;
    }
    int shift = exponent + int(precision) - scales[scale_index];
    if (shift < 0) mantissa = fp64_shift_right(mantissa, uint(-shift));
    for (uint t = 0; t < plan.count; ++t) {
        output[ulong(t) * count + index] = int8_t(fp64_signed_residue(mantissa, (bits.high >> 31) != 0,
                                                               plan.powers[t][uint(max(shift, 0))], plan.moduli[t]));
    }
}

/**
 * @brief INT8の行列積を計算し、法ごとの余りだけを保存する。
 * @param[in] a 法ごとに並ぶAの余り。
 * @param[in] b 法ごとに並ぶBの余り。
 * @param[out] output 法ごとに並ぶ、0以上の出力の余り。
 * @param[in] parameters 行列の寸法。
 * @param[in] plan 使用する法。
 * @param[in] group 列、行、法の順で表す担当タイル。
 */
kernel void residue_matmul(device int8_t* a [[buffer(0)]],
                           device int8_t* b [[buffer(1)]],
                           device uchar* output [[buffer(2)]],
                           constant fp64_batch_parameters_t& parameters [[buffer(3)]],
                           constant fp64_crt_plan_t& plan [[buffer(4)]],
                           uint3 group [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = matmul2d_descriptor(FP64_TILE_ROWS, FP64_TILE_COLUMNS, int(dynamic_extent), false, false, false);
    matmul2d<descriptor, execution_simdgroups<FP64_SIMD_GROUPS>> operation;
    uint row = group.y * FP64_TILE_ROWS, column = group.x * FP64_TILE_COLUMNS;
    device int8_t* a_base = a + ulong(group.z) * parameters.rows * parameters.inner;
    device int8_t* b_base = b + ulong(group.z) * parameters.inner * parameters.columns;
    auto a_tensor = tensor(a_base, dextents<int, 2>{int(parameters.inner), int(parameters.rows)}, array<int, 2>{1, int(parameters.inner)});
    auto b_tensor = tensor(b_base, dextents<int, 2>{int(parameters.columns), int(parameters.inner)}, array<int, 2>{1, int(parameters.columns)});
    auto accumulated = operation.get_destination_cooperative_tensor<decltype(a_tensor), decltype(b_tensor), int>();
    #pragma unroll
    for (uint i = 0; i < accumulated.get_capacity(); ++i) {
        if (accumulated.is_valid_element(i)) accumulated[i] = 0;
    }
    fp64_modulus_t modulus = plan.moduli[group.z];
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
        if ((begin + length) % FP64_K_ACCUMULATE == 0 || begin + length == parameters.inner) {
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
    #pragma unroll
    for (uint i = 0; i < accumulated.get_capacity(); ++i) {
        if (accumulated.is_valid_element(i)) {
            auto coordinates = accumulated.get_multidimensional_index(i);
            uint output_column = column + uint(coordinates[0]);
            uint output_row = row + uint(coordinates[1]);
            if (output_column < parameters.columns && output_row < parameters.rows)
                output[(ulong(group.z) * parameters.rows + output_row) * parameters.columns + output_column] = uchar(accumulated[i]);
        }
    }
}

/**
 * @brief 出力要素ごとにCRTを復元し、FP64のビット列を保存する。
 * @param[in] residues 法ごとに並ぶ出力の余り。
 * @param[in] scales_a Aの行の指数。
 * @param[in] scales_b Bの列の指数。
 * @param[out] output FP64のビット列。
 * @param[in] parameters 行列の寸法と整数幅。
 * @param[in] plan CRTの係数。
 * @param[in] position 行のまとまりの中の二次元の出力位置。
 */
kernel void reconstruct(device const uchar* residues [[buffer(0)]],
                        device const int* scales_a [[buffer(1)]],
                        device const int* scales_b [[buffer(2)]],
                        device fp64_bits_t* output [[buffer(3)]],
                        constant fp64_batch_parameters_t& parameters [[buffer(4)]],
                        constant fp64_crt_plan_t& plan [[buffer(5)]],
                        uint2 position [[thread_position_in_grid]]) {
    ulong count = ulong(parameters.rows) * parameters.columns;
    ulong index = ulong(position.y) * min(count, ulong(65536)) + position.x;
    if (index >= count) return;
    fp64_big_uint_t value = {};
    for (uint t = 0; t < plan.count; ++t) {
        fp64_crt_step(&value, plan.prefixes[t], plan.moduli[t], plan.inverses[t], residues[ulong(t) * count + index], plan.stage_limbs[t]);
    }
    bool negative = fp64_big_compare(value, plan.half_product, plan.limbs) > 0;
    if (negative) value = fp64_big_subtract(plan.product, value, plan.limbs);
    int scale = scales_a[index / parameters.columns] + scales_b[index % parameters.columns]
              - int(parameters.precision_a) - int(parameters.precision_b);
    output[ulong(parameters.row_begin) * parameters.columns + index] = fp64_pack_fp64(value, negative, scale, plan.limbs);
}
