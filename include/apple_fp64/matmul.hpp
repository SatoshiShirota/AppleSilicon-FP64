#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace apple_fp64 {

/** @brief 計算の整数幅と、作業領域に保持する行数。 */
struct options_s {
    std::uint32_t precision_a = 60; /**< Aの整数幅。正の値で指定する。 */
    std::uint32_t precision_b = 60; /**< Bの整数幅。正の値で指定する。 */
    std::uint32_t batch_rows = 256; /**< 一度に処理する行数。正の値で指定する。 */
};

/** @brief 一回の積について取得した処理時間と資源量。 */
struct measurement_s {
    double prepare_seconds = 0; /**< 入力変換に費やしたGPUの実行時間。 */
    double product_seconds = 0; /**< 余りの行列積に費やしたGPUの実行時間。 */
    double reconstruct_seconds = 0; /**< 復元に費やしたGPUの実行時間。 */
    double wait_seconds = 0; /**< CPUがGPUの完了を待った実時間。 */
    double total_seconds = 0; /**< 作業領域の確保、転送、同期を含む実時間。 */
    std::size_t workspace_bytes = 0; /**< 計算に使用したMetalバッファーの容量の合計。 */
    std::uint32_t modulus_count = 0; /**< 使用した法の個数。 */
};

/** @brief 行優先の出力と、その計算についての測定値。 */
struct result_s {
    std::vector<double> values; /**< M行N列の連続した出力。 */
    measurement_s measurement; /**< 計算に費やした時間と資源量。 */
};

/** @brief Metalのデバイス、実行待ち行列とパイプラインを保持する実装。 */
struct implementation_s;

/**
 * @brief Metalパイプラインと作業領域を再利用して行列積を実行する。
 * @note 同じインスタンスへの呼び出しを並行して行ってはならない。
 */
class multiplier_c {
public:
    /**
     * @brief コンパイル済みのMetalライブラリーを読み込む。
     * @param[in] library_path CMakeで生成したfp64.metallibのパス。
     * @exception std::runtime_error デバイス、ライブラリー、パイプラインを作成できない。
     */
    explicit multiplier_c(const std::filesystem::path& library_path);
    /** @brief 所有するMetalの資源を解放する。 */
    ~multiplier_c();
    multiplier_c(const multiplier_c&) = delete;
    multiplier_c& operator=(const multiplier_c&) = delete;

    /**
     * @brief 二つのFP64行列の積を求める。
     * @param[in] a M行K列の連続した入力。呼び出しが完了するまで変更してはならない。
     * @param[in] b K行N列の連続した入力。呼び出しが完了するまで変更してはならない。
     * @param[in] m Aと出力の行数。
     * @param[in] n Bと出力の列数。
     * @param[in] k 内積の項数。
     * @param[in] options 整数幅と行のまとまりの大きさ。
     * @return 呼び出し側が所有する出力と測定値。
     * @pre aとbは有限のFP64値だけを含むこと。
     * @note 計算の数値的な意味はREADME.md「計算の定義」で定める。
     * @exception std::invalid_argument 寸法、入力の長さ、設定、法の積が計算条件を満たさない。
     * @exception std::runtime_error バッファーの確保またはMetalの実行に失敗した。
     * @exception std::bad_alloc CPUのメモリー確保に失敗した。
     */
    result_s multiply(std::span<const double> a, std::span<const double> b,
                      std::uint32_t m, std::uint32_t n, std::uint32_t k,
                      options_s options = {});

    /** @brief 使用しているMetalデバイスの名前を返す。 @return デバイス名。 */
    std::string device_name() const;

private:
    std::unique_ptr<implementation_s> implementation; /**< Metalの資源を所有する実装。 */
};

} // namespace apple_fp64
