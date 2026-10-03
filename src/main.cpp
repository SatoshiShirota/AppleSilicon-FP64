#include "apple_fp64/matmul.hpp"

#include <Accelerate/Accelerate.h>
#include <mach-o/dyld.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>

namespace {

using apple_fp64::measurement_s;
using apple_fp64::multiplier_c;
using apple_fp64::options_s;

/**
 * @brief コマンド引数を32ビットの非負整数へ変換する。
 * @param[in] text 十進数の引数。
 * @return 非負整数。
 * @exception std::invalid_argument 引数が整数の条件を満たさない。
 */
std::uint32_t parse_number(std::string_view text) {
    std::uint32_t value = 0;
    auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument("非負の十進整数を指定してください: " + std::string(text));
    return value;
}

/**
 * @brief 行列のバイト数を算出する。
 * @param[in] rows 行数。
 * @param[in] columns 列数。
 * @return FP64行列のバイト数。
 * @exception std::invalid_argument 寸法またはサイズが表現範囲を超える。
 */
std::size_t matrix_bytes(std::uint32_t rows, std::uint32_t columns) {
    if (rows > INT_MAX || columns > INT_MAX)
        throw std::invalid_argument("行列の次元は符号付き32ビット整数の範囲で指定してください。");
    std::uint64_t count = std::uint64_t(rows) * columns;
    if (count > std::uint64_t(std::numeric_limits<std::streamsize>::max()) / sizeof(double))
        throw std::invalid_argument("行列ファイルのサイズが表現範囲を超えます。");
    return std::size_t(count) * sizeof(double);
}

/**
 * @brief 有限のFP64値を、行優先のバイナリーファイルから読む。
 * @param[in] path 入力ファイル。
 * @param[in] rows 行数。
 * @param[in] columns 列数。
 * @return ファイルの全要素。
 * @exception std::invalid_argument ファイルの長さが一致しないか、非有限値を含む。
 * @exception std::runtime_error ファイルを読めない。
 */
std::vector<double> read_matrix(const std::filesystem::path& path, std::uint32_t rows, std::uint32_t columns) {
    std::size_t bytes = matrix_bytes(rows, columns);
    if (std::filesystem::file_size(path) != bytes)
        throw std::invalid_argument("入力ファイルの長さが行列の寸法と一致しません: " + path.string());
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("入力ファイルを開けません: " + path.string());
    std::vector<double> values(bytes / sizeof(double));
    if (bytes != 0 && !file.read(reinterpret_cast<char*>(values.data()), std::streamsize(bytes)))
        throw std::runtime_error("入力ファイルを読み込めません: " + path.string());
    for (double value : values) {
        if (!std::isfinite(value)) throw std::invalid_argument("入力ファイルは有限のFP64値だけを含む必要があります: " + path.string());
    }
    return values;
}

/**
 * @brief 実行ファイルと同じディレクトリーにあるMetalライブラリーを求める。
 * @return fp64.metallibの絶対パス。
 * @exception std::runtime_error 実行ファイルの位置を取得できない。
 */
std::filesystem::path library_path() {
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> path(size);
    if (_NSGetExecutablePath(path.data(), &size) != 0)
        throw std::runtime_error("実行ファイルの位置を取得できません。");
    return std::filesystem::canonical(path.data()).parent_path() / "fp64.metallib";
}

/**
 * @brief 指定されたファイルの行列積を計算する。
 * @param[in] argc 引数の個数。
 * @param[in] argv コマンド引数。
 * @return 正常終了の場合は0。
 * @exception std::invalid_argument 引数が不正。
 * @exception std::runtime_error ファイルまたは計算の処理に失敗する。
 */
int multiply_files(int argc, char** argv) {
    if (argc != 10 && argc != 11)
        throw std::invalid_argument("使い方: fp64_metal multiply M N K p_A p_B A.bin B.bin C.bin [行のまとまりの大きさ]");
    std::uint32_t m = parse_number(argv[2]), n = parse_number(argv[3]), k = parse_number(argv[4]);
    options_s options;
    options.precision_a = parse_number(argv[5]);
    options.precision_b = parse_number(argv[6]);
    if (argc == 11) options.batch_rows = parse_number(argv[10]);
    matrix_bytes(m, n);
    auto a = read_matrix(argv[7], m, k);
    auto b = read_matrix(argv[8], k, n);
    multiplier_c multiplier(library_path());
    auto result = multiplier.multiply(a, b, m, n, k, options);
    std::ofstream file(argv[9], std::ios::binary);
    if (!file) throw std::runtime_error("出力ファイルを開けません。");
    if (!result.values.empty()) file.write(reinterpret_cast<const char*>(result.values.data()), std::streamsize(result.values.size() * sizeof(double)));
    file.close();
    if (!file) throw std::runtime_error("出力ファイルへ書き込めません。");
    std::cout << "デバイス: " << multiplier.device_name() << '\n'
              << "法の個数: " << result.measurement.modulus_count << '\n'
              << "全体の実時間: " << result.measurement.total_seconds * 1000 << " ms\n";
    return 0;
}

/**
 * @brief 時間の系列の中央値を求める。
 * @param[in] samples 全試行の値。
 * @return ソートした値の中央。偶数個の場合は中央の二値の平均。
 * @pre samplesは空ではない。
 */
double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    std::size_t middle = samples.size() / 2;
    return samples.size() % 2 ? samples[middle] : (samples[middle - 1] + samples[middle]) / 2;
}

/** @brief 性能比較の一方式について保持する全試行の測定値。 */
struct benchmark_series_s {
    const char* name; /**< 方式名。 */
    std::vector<double> total; /**< 試行ごとの全体の実時間。 */
    std::vector<double> prepare; /**< 試行ごとの入力変換の時間。 */
    std::vector<double> product; /**< 試行ごとのGPUの行列積の時間。 */
    std::vector<double> reconstruct; /**< 試行ごとの復元の時間。 */
    std::vector<double> wait; /**< 試行ごとのCPUの待ち時間。 */
    measurement_s latest; /**< 最後の試行の資源量。 */
};

/**
 * @brief GPU完結版とAccelerateの性能を、同じ入力と試行回数で比較する。
 * @param[in] argc 引数の個数。
 * @param[in] argv コマンド引数。
 * @return 正常終了の場合は0。
 * @exception std::invalid_argument 寸法または試行回数が不正。
 * @exception std::runtime_error 計算に失敗する。
 */
int benchmark(int argc, char** argv) {
    if (argc > 5) throw std::invalid_argument("使い方: fp64_metal benchmark [行列の次数=512] [試行回数=5] [整数幅=60]");
    std::uint32_t n = argc > 2 ? parse_number(argv[2]) : 512;
    std::uint32_t trials = argc > 3 ? parse_number(argv[3]) : 5;
    options_s options;
    if (argc > 4) options.precision_a = options.precision_b = parse_number(argv[4]);
    if (n == 0 || trials == 0) throw std::invalid_argument("行列の次数と試行回数は正の値で指定してください。");
    std::size_t count = matrix_bytes(n, n) / sizeof(double);
    std::vector<double> a(count), b(count), reference(count), gpu(count);
    std::mt19937_64 random(123);
    std::uniform_real_distribution<double> distribution(-1, 1);
    for (std::size_t index = 0; index < count; ++index) {
        a[index] = distribution(random);
        b[index] = distribution(random);
    }
    auto initialization_start = std::chrono::steady_clock::now();
    multiplier_c multiplier(library_path());
    double initialization = std::chrono::duration<double>(std::chrono::steady_clock::now() - initialization_start).count();
    std::array<benchmark_series_s, 2> series = {};
    series[0].name = "Accelerate";
    series[1].name = "GPU完結版";
    auto run = [&](std::size_t method, bool record) {
        measurement_s measurement;
        if (method == 0) {
            auto start = std::chrono::steady_clock::now();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, int(n), int(n), int(n), 1,
                        a.data(), int(n), b.data(), int(n), 0, reference.data(), int(n));
            measurement.total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        } else {
            auto result = multiplier.multiply(a, b, n, n, n, options);
            measurement = result.measurement;
            gpu = std::move(result.values);
        }
        if (record) {
            auto& current = series[method];
            current.total.push_back(measurement.total_seconds);
            current.prepare.push_back(measurement.prepare_seconds);
            current.product.push_back(measurement.product_seconds);
            current.reconstruct.push_back(measurement.reconstruct_seconds);
            current.wait.push_back(measurement.wait_seconds);
            current.latest = measurement;
        }
    };
    for (std::size_t method = 0; method < series.size(); ++method) run(method, false);
    for (std::uint32_t trial = 0; trial < trials; ++trial) {
        for (std::size_t offset = 0; offset < series.size(); ++offset) run((trial + offset) % series.size(), true);
    }
    double maximum_error = 0;
    for (std::size_t index = 0; index < count; ++index) {
        maximum_error = std::max(maximum_error, std::abs(gpu[index] - reference[index]));
    }
    std::cout << "デバイス: " << multiplier.device_name() << '\n'
              << "行列: " << n << " × " << n << "、整数幅: " << options.precision_a << "、試行回数: " << trials << '\n'
              << "Metalの初期化: " << initialization * 1000 << " ms\n"
              << "全試行でAとBの変換を含めています。方式ごとに一回の準備実行を除外しています。\n"
              << "全体の実時間の中央値、最小値、最大値を示します。\n"
              << std::fixed << std::setprecision(3);
    double baseline = median(series[0].total);
    for (const auto& current : series) {
        double total = median(current.total);
        std::cout << current.name << ": " << total * 1000 << " ms ("
                  << *std::min_element(current.total.begin(), current.total.end()) * 1000 << " ～ "
                  << *std::max_element(current.total.begin(), current.total.end()) * 1000 << " ms)、"
                  << 2.0 * n * n * n / total / 1e9 << " GFLOP/s、Accelerateとの速度比 " << baseline / total << '\n';
        if (current.latest.modulus_count != 0) {
            std::cout << "  入力変換 " << median(current.prepare) * 1000 << " ms、GPUの行列積 " << median(current.product) * 1000
                      << " ms、復元 " << median(current.reconstruct) * 1000 << " ms、CPUの待ち時間 " << median(current.wait) * 1000 << " ms\n"
                      << "  法 " << current.latest.modulus_count << " 個、Metalの作業領域 " << current.latest.workspace_bytes / 1048576.0 << " MiB\n";
        }
    }
    std::cout << std::scientific << "Accelerateとの最大絶対差: " << maximum_error << '\n';
    return 0;
}

/**
 * @brief 指定されたコマンドを実行する。
 * @param[in] argc 引数の個数。
 * @param[in] argv コマンド引数。
 * @return 正常終了では0。
 * @exception std::invalid_argument コマンドが不正。
 */
int run_command(int argc, char** argv) {
    if (argc >= 2 && std::string_view(argv[1]) == "multiply") return multiply_files(argc, argv);
    if (argc >= 2 && std::string_view(argv[1]) == "benchmark") return benchmark(argc, argv);
    throw std::invalid_argument("使い方: fp64_metal benchmark [次数] [試行回数] [整数幅]\n"
                                "        fp64_metal multiply M N K p_A p_B A.bin B.bin C.bin [行のまとまりの大きさ]");
}

} // namespace

/**
 * @brief 実験用コマンドの入口。
 * @param[in] argc 引数の個数。
 * @param[in] argv コマンド引数。
 * @return 正常終了では0、入力または実行の失敗では1。
 */
int main(int argc, char** argv) {
    try {
        return run_command(argc, argv);
    } catch (const std::invalid_argument& error) {
        std::cerr << error.what() << '\n';
    } catch (const std::runtime_error& error) {
        std::cerr << error.what() << '\n';
    } catch (const std::bad_alloc&) {
        std::cerr << "CPUのメモリーを確保できません。\n";
    }
    return 1;
}
