#include "apple_fp64/matmul.h"
#include "timing.h"

#include <Accelerate/Accelerate.h>
#include <mach-o/dyld.h>

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/**
 * @~japanese
 * @brief コマンドライン引数の文字列を32ビットの非負整数に変換する。
 * @param[in] text 10進数で表された引数の文字列。
 * @param[out] value 変換後の非負整数。
 * @return すべての文字を変換できた場合はtrue。
 * @~english
 * @brief Convert a command-line argument string to a non-negative 32-bit integer.
 * @param[in] text Argument string in decimal notation.
 * @param[out] value Converted non-negative integer.
 * @return true if every character was converted.
 * @~
 */
static bool fp64_parse_number(const char *text, uint32_t *value)
{
    uint32_t parsed = 0;
    if (*text == '\0') goto invalid;
    for (const char *position = text; *position != '\0'; ++position) {
        if (*position < '0' || *position > '9') goto invalid;
        uint32_t digit = (uint32_t)(*position - '0');
        if (parsed > (UINT32_MAX - digit) / 10) goto invalid;
        parsed = parsed * 10 + digit;
    }
    *value = parsed;
    return true;
invalid:
    fprintf(stderr, "非負の十進整数を指定してください: %s\n", text);
    return false;
}

/**
 * @~japanese
 * @brief 倍精度の行列を格納するバイナリーファイルのバイト数を求める。
 * @param[in] rows 行数。
 * @param[in] columns 列数。
 * @param[out] bytes 行列全体のバイト数。
 * @return 行列の寸法とファイルサイズが表現可能な範囲内であればtrue。
 * @~english
 * @brief Compute the byte size of a binary file storing an FP64 matrix.
 * @param[in] rows Number of rows.
 * @param[in] columns Number of columns.
 * @param[out] bytes Total byte size of the matrix.
 * @return true if the matrix dimensions and file size are within representable bounds.
 * @~
 */
static bool fp64_matrix_bytes(uint32_t rows, uint32_t columns, size_t *bytes)
{
    if (rows > INT_MAX || columns > INT_MAX) {
        fprintf(stderr, "行列の次元は符号付き32ビット整数の範囲で指定してください。\n");
        return false;
    }
    uint64_t count = (uint64_t)rows * columns;
    if (count > (uint64_t)INT64_MAX / sizeof(double)) {
        fprintf(stderr, "行列ファイルのサイズが表現範囲を超えます。\n");
        return false;
    }
    *bytes = (size_t)count * sizeof(double);
    return true;
}

/**
 * @~japanese
 * @brief 行優先で並んだ倍精度のバイナリーファイルから行列を読み込む。
 * @param[in] path 入力ファイルのパス。
 * @param[in] rows 行数。
 * @param[in] columns 列数。
 * @param[out] values 読み込んだ要素の配列。空の行列ではNULL。呼び出し側がfreeで解放する。
 * @param[out] count 読み込んだ要素の個数。
 * @return ファイルを読み込めた場合はtrue。失敗した場合は理由を標準エラー出力へ書き出す。
 * @~english
 * @brief Read a matrix from a row-major FP64 binary file.
 * @param[in] path Path to the input file.
 * @param[in] rows Number of rows.
 * @param[in] columns Number of columns.
 * @param[out] values Array of loaded elements (NULL for an empty matrix). The caller releases it with free.
 * @param[out] count Number of loaded elements.
 * @return true if the file was read; writes the failure reason to standard error on failure.
 * @~
 */
static bool fp64_read_matrix(const char *path, uint32_t rows, uint32_t columns,
                             double **values, size_t *count)
{
    *values = NULL;
    *count = 0;
    size_t bytes;
    if (!fp64_matrix_bytes(rows, columns, &bytes)) return false;
    struct stat information;
    if (stat(path, &information) != 0) {
        fprintf(stderr, "入力ファイルの情報を取得できません: %s\n", path);
        return false;
    }
    if ((uint64_t)information.st_size != bytes) {
        fprintf(stderr, "入力ファイルの長さが行列の寸法と一致しません: %s\n", path);
        return false;
    }
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "入力ファイルを開けません: %s\n", path);
        return false;
    }
    double *data = bytes != 0 ? malloc(bytes) : NULL;
    bool succeeded = false;
    if (bytes != 0 && data == NULL) {
        fprintf(stderr, "CPUのメモリーを確保できません。\n");
        goto finish;
    }
    if (bytes != 0 && fread(data, 1, bytes, file) != bytes) {
        fprintf(stderr, "入力ファイルを読み込めません: %s\n", path);
        goto finish;
    }
    *values = data;
    *count = bytes / sizeof(double);
    succeeded = true;
finish:
    fclose(file);
    if (!succeeded) free(data);
    return succeeded;
}

/**
 * @~japanese
 * @brief 実行ファイルと同じディレクトリーにあるMetalライブラリー（fp64.metallib）のパスを求める。
 * @return 呼び出し側がfreeで解放するパス文字列。取得できなかった場合はNULL。
 * @~english
 * @brief Resolve the path to the Metal library (fp64.metallib) in the same directory as the executable.
 * @return Path string to be released by the caller with free, or NULL if resolution failed.
 * @~
 */
static char *fp64_library_path(void)
{
    uint32_t size = 0;
    _NSGetExecutablePath(NULL, &size);
    char *path = malloc(size);
    if (path == NULL) {
        fprintf(stderr, "CPUのメモリーを確保できません。\n");
        return NULL;
    }
    if (_NSGetExecutablePath(path, &size) != 0) {
        free(path);
        fprintf(stderr, "実行ファイルの位置を取得できません。\n");
        return NULL;
    }
    char *canonical = realpath(path, NULL);
    free(path);
    if (canonical == NULL) {
        perror("実行ファイルのパスを解決できません");
        return NULL;
    }
    char *slash = strrchr(canonical, '/');
    assert(slash != NULL);
    size_t prefix = (size_t)(slash + 1 - canonical);
    char *library = realloc(canonical, prefix + sizeof("fp64.metallib"));
    if (library == NULL) {
        free(canonical);
        fprintf(stderr, "CPUのメモリーを確保できません。\n");
        return NULL;
    }
    memcpy(library + prefix, "fp64.metallib", sizeof("fp64.metallib"));
    return library;
}

/**
 * @~japanese
 * @brief ライブラリーのエラーメッセージを標準エラー出力へ書き出し、その文字列を解放する。
 * @param[in,out] error 失敗した操作のエラー情報。
 * @pre error->messageがNULLになるのは、CPUのメモリー確保に失敗した場合に限る。
 * @~english
 * @brief Write the library error message to standard error and release the message string.
 * @param[in,out] error Error information from the failed operation.
 * @pre A NULL error->message occurs only when CPU memory allocation fails.
 * @~
 */
static void fp64_print_error(apple_fp64_error_t *error)
{
    fprintf(stderr, "%s\n", error->message != NULL ? error->message : "CPUのメモリーを確保できません。");
    apple_fp64_error_destroy(error);
}

/**
 * @~japanese
 * @brief コマンドラインツールで使用するMetalの計算器を作成する。
 * @param[out] multiplier 作成された計算器を受け取るポインター。
 * @return 作成に成功した場合はtrue。
 * @~english
 * @brief Create the Metal multiplier used by the command-line tool.
 * @param[out] multiplier Pointer receiving the created multiplier.
 * @return true if creation succeeded.
 * @~
 */
static bool fp64_create_multiplier(apple_fp64_multiplier_t **multiplier)
{
    char *path = fp64_library_path();
    if (path == NULL) return false;
    apple_fp64_error_t error = {0};
    apple_fp64_status_t status = apple_fp64_multiplier_create(path, multiplier, &error);
    free(path);
    if (status != APPLE_FP64_SUCCESS) fp64_print_error(&error);
    return status == APPLE_FP64_SUCCESS;
}

/**
 * @~japanese
 * @brief 指定されたファイルから行列AとBを読み込み、積Cを計算してファイルへ書き出す。
 * @param[in] argc 引数の個数。
 * @param[in] argv コマンドライン引数。
 * @return 正常終了時は0、入力の不備や実行の失敗時は1。
 * @~english
 * @brief Read matrices A and B from the specified files, compute their product C, and write it to a file.
 * @param[in] argc Number of arguments.
 * @param[in] argv Command-line arguments.
 * @return Zero on success, one on invalid input or execution failure.
 * @~
 */
static int fp64_multiply_files(int argc, char **argv)
{
    if (argc != 8 && argc != 9) {
        fprintf(stderr, "使い方: fp64_metal multiply M N K A.bin B.bin C.bin [行のまとまりの大きさ]\n");
        return 1;
    }
    uint32_t m, n, k;
    apple_fp64_options_t options = apple_fp64_default_options();
    if (!fp64_parse_number(argv[2], &m) || !fp64_parse_number(argv[3], &n)
        || !fp64_parse_number(argv[4], &k)
        || (argc == 9 && !fp64_parse_number(argv[8], &options.batch_rows))) return 1;
    size_t output_bytes;
    if (!fp64_matrix_bytes(m, n, &output_bytes)) return 1;
    double *a = NULL, *b = NULL;
    size_t a_count, b_count;
    apple_fp64_multiplier_t *multiplier = NULL;
    apple_fp64_result_t result = {0};
    apple_fp64_error_t error = {0};
    int exit_code = 1;
    if (!fp64_read_matrix(argv[5], m, k, &a, &a_count)
        || !fp64_read_matrix(argv[6], k, n, &b, &b_count)
        || !fp64_create_multiplier(&multiplier)) goto finish;
    apple_fp64_status_t status = apple_fp64_multiply(multiplier, a, a_count, b, b_count,
                                                    m, n, k, options, &result, &error);
    if (status != APPLE_FP64_SUCCESS) {
        fp64_print_error(&error);
        goto finish;
    }
    FILE *file = fopen(argv[7], "wb");
    if (file == NULL) {
        fprintf(stderr, "出力ファイルを開けません。\n");
        goto finish;
    }
    bool written = output_bytes == 0 || fwrite(result.values, 1, output_bytes, file) == output_bytes;
    if (fclose(file) != 0) written = false;
    if (!written) {
        fprintf(stderr, "出力ファイルへ書き込めません。\n");
        goto finish;
    }
    printf("デバイス: %s\n法の個数: %u\n全体の実時間: %g ms\n",
           apple_fp64_device_name(multiplier), result.measurement.modulus_count,
           result.measurement.total_seconds * 1000);
    exit_code = 0;
finish:
    apple_fp64_result_destroy(&result);
    apple_fp64_multiplier_destroy(multiplier);
    free(a);
    free(b);
    return exit_code;
}

/**
 * @~japanese
 * @brief ベンチマークの入力行列を再現するためのmt19937_64乱数生成器の状態。
 * @~english
 * @brief State of the mt19937_64 generator used to produce reproducible benchmark input matrices.
 * @~
 */
typedef struct fp64_random_s {
    uint64_t state[312]; /**< @~japanese メルセンヌ・ツイスターの状態配列。
                          * @~english State array of the Mersenne Twister.
                          * @~
                          */
    size_t position; /**< @~japanese 次に取り出す要素の位置。312に達したら状態配列を更新する。
                      * @~english Index of the next element to extract; state is twisted when it reaches 312.
                      * @~
                      */
} fp64_random_t;

/**
 * @~japanese
 * @brief ベンチマーク用の乱数生成器を種で初期化する。
 * @param[out] random 初期化する乱数生成器の状態。
 * @param[in] seed 乱数の種。
 * @~english
 * @brief Seed and initialize the random number generator for benchmarking.
 * @param[out] random Generator state to initialize.
 * @param[in] seed Random seed.
 * @~
 */
static void fp64_random_initialize(fp64_random_t *random, uint64_t seed)
{
    random->state[0] = seed;
    for (size_t index = 1; index < 312; ++index)
        random->state[index] = UINT64_C(6364136223846793005)
                            * (random->state[index - 1] ^ (random->state[index - 1] >> 62)) + index;
    random->position = 312;
}

/**
 * @~japanese
 * @brief mt19937_64を用いて区間[-1, 1)の一様乱数を倍精度で生成する。
 * @param[in,out] random 乱数生成器の状態。
 * @return [-1, 1)の範囲にある行列要素の値。
 * @~english
 * @brief Generate a uniform FP64 random number in the interval [-1, 1) using mt19937_64.
 * @param[in,out] random Generator state.
 * @return Matrix element value in [-1, 1).
 * @~
 */
static double fp64_random_value(fp64_random_t *random)
{
    if (random->position == 312) {
        for (size_t index = 0; index < 312; ++index) {
            uint64_t joined = (random->state[index] & UINT64_C(0xffffffff80000000))
                            | (random->state[(index + 1) % 312] & UINT64_C(0x7fffffff));
            random->state[index] = random->state[(index + 156) % 312] ^ (joined >> 1)
                                ^ ((joined & 1) ? UINT64_C(0xb5026f5aa96619e9) : 0);
        }
        random->position = 0;
    }
    uint64_t value = random->state[random->position++];
    value ^= (value >> 29) & UINT64_C(0x5555555555555555);
    value ^= (value << 17) & UINT64_C(0x71d67fffeda60000);
    value ^= (value << 37) & UINT64_C(0xfff7eee000000000);
    value ^= value >> 43;
    double unit = (double)value * 0x1p-64;
    if (unit >= 1) unit = nextafter(1, 0);
    return unit * 2 - 1;
}

/**
 * @~japanese
 * @brief qsort用に2つの測定値（double）を比較する。
 * @param[in] left 左側の値へのポインター。
 * @param[in] right 右側の値へのポインター。
 * @return 左が小さければ負、等しければ0、大きければ正。
 * @~english
 * @brief Compare two double measurements for qsort.
 * @param[in] left Pointer to the left value.
 * @param[in] right Pointer to the right value.
 * @return Negative if left is smaller, zero if equal, positive if greater.
 * @~
 */
static int fp64_compare_samples(const void *left, const void *right)
{
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

/**
 * @~japanese
 * @brief 測定値の配列を昇順に並べ替え、中央値を求める。
 * @param[in,out] samples 測定値の配列。関数内で昇順に並べ替えられる。
 * @param[in] count 試行回数（1以上）。
 * @return 要素数が偶数のときは中央2つの平均、奇数のときは中央の値。
 * @~english
 * @brief Sort an array of measurements in ascending order and return its median.
 * @param[in,out] samples Array of measurements, sorted in place in ascending order.
 * @param[in] count Number of trials (at least 1).
 * @return Mean of the two middle elements when count is even, or the middle element when odd.
 * @~
 */
static double fp64_median(double *samples, size_t count)
{
    qsort(samples, count, sizeof(*samples), fp64_compare_samples);
    size_t middle = count / 2;
    return count % 2 ? samples[middle] : (samples[middle - 1] + samples[middle]) / 2;
}

/**
 * @~japanese
 * @brief ベンチマークで1つの計算方式について全試行の測定値を保持する構造体。
 * @~english
 * @brief Structure holding measurements across all trials for one computation method in the benchmark.
 * @~
 */
typedef struct fp64_benchmark_series_s {
    const char *name; /**< @~japanese 方式の名前。
                       * @~english Name of the method.
                       * @~
                       */
    double *total; /**< @~japanese 試行ごとの全体の実時間。
                    * @~english Total wall-clock time for each trial.
                    * @~
                    */
    double *prepare; /**< @~japanese 試行ごとの入力変換時間。
                      * @~english Input conversion time for each trial.
                      * @~
                      */
    double *product; /**< @~japanese 試行ごとのGPUによる行列積の計算時間。
                      * @~english GPU matrix multiplication time for each trial.
                      * @~
                      */
    double *reconstruct; /**< @~japanese 試行ごとのCRTによる復元時間。
                          * @~english CRT reconstruction time for each trial.
                          * @~
                          */
    double *wait; /**< @~japanese 試行ごとのCPUの待ち時間。
                   * @~english CPU wait time for each trial.
                   * @~
                   */
    apple_fp64_measurement_t latest; /**< @~japanese 最後の試行で記録された資源の使用量。
                                      * @~english Resource usage recorded in the final trial.
                                      * @~
                                      */
} fp64_benchmark_series_t;

/**
 * @~japanese
 * @brief 1つの測定区間の中で同じ方式を指定回数だけ反復実行し、行列積1回あたりの平均時間を求める。
 * @param[in] method 0ならAccelerate、1ならAppleSilicon-FP64。
 * @param[in] repetitions 区間内での反復回数（1以上）。
 * @param[in,out] multiplier GPUの計算器。
 * @param[in] a n行n列の入力行列A。
 * @param[in] b n行n列の入力行列B。
 * @param[out] reference Accelerateの出力を受け取るn行n列の配列。
 * @param[in] n 正方行列の次数。
 * @param[in] options GPU計算の設定。
 * @param[out] measurement 各段階の平均時間と、最後の反復における資源の使用量。
 * @param[out] error 実行に失敗した場合のエラー情報。
 * @return すべての反復に成功した場合はtrue。
 * @note 時間の測定範囲と反復の方法はREADME.mdの「コマンドの使用方法」で定める。
 * @~english
 * @brief Run one method for the specified number of repetitions within a measurement interval and compute the mean time per matrix product.
 * @param[in] method 0 for Accelerate, 1 for AppleSilicon-FP64.
 * @param[in] repetitions Number of repetitions within the interval (at least 1).
 * @param[in,out] multiplier GPU multiplier.
 * @param[in] a n-by-n input matrix A.
 * @param[in] b n-by-n input matrix B.
 * @param[out] reference n-by-n array receiving the Accelerate output.
 * @param[in] n Order of the square matrices.
 * @param[in] options GPU computation options.
 * @param[out] measurement Mean time for each stage and resource usage from the final repetition.
 * @param[out] error Error information on execution failure.
 * @return true if all repetitions succeeded.
 * @note Timing boundaries and repetition rules are defined in README.en.md, section "Command-line usage".
 * @~
 */
static bool fp64_benchmark_interval(size_t method, uint32_t repetitions,
                                     apple_fp64_multiplier_t *multiplier,
                                     const double *a, const double *b, double *reference,
                                     uint32_t n, apple_fp64_options_t options,
                                     apple_fp64_measurement_t *measurement, apple_fp64_error_t *error)
{
    *measurement = (apple_fp64_measurement_t){0};
    double start = fp64_monotonic_seconds();
    if (method == 0) {
        for (uint32_t iteration = 0; iteration < repetitions; ++iteration)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)n, (int)n, (int)n,
                        1, a, (int)n, b, (int)n, 0, reference, (int)n);
    } else {
        size_t count = (size_t)n * n;
        for (uint32_t iteration = 0; iteration < repetitions; ++iteration) {
            apple_fp64_result_t result = {0};
            apple_fp64_status_t status = apple_fp64_multiply(multiplier, a, count, b, count,
                                                            n, n, n, options, &result, error);
            if (status != APPLE_FP64_SUCCESS) return false;
            measurement->prepare_seconds += result.measurement.prepare_seconds;
            measurement->product_seconds += result.measurement.product_seconds;
            measurement->reconstruct_seconds += result.measurement.reconstruct_seconds;
            measurement->wait_seconds += result.measurement.wait_seconds;
            measurement->workspace_bytes = result.measurement.workspace_bytes;
            measurement->modulus_count = result.measurement.modulus_count;
            apple_fp64_result_destroy(&result);
        }
    }
    measurement->total_seconds = (fp64_monotonic_seconds() - start) / repetitions;
    measurement->prepare_seconds /= repetitions;
    measurement->product_seconds /= repetitions;
    measurement->reconstruct_seconds /= repetitions;
    measurement->wait_seconds /= repetitions;
    return true;
}

/**
 * @~japanese
 * @brief 同じ入力行列と試行回数を使って、AppleSilicon-FP64とAccelerateの実行速度と誤差を比較する。
 * @param[in] argc 引数の個数。
 * @param[in] argv コマンドライン引数。
 * @return 正常終了時は0、入力の不備や実行の失敗時は1。
 * @~english
 * @brief Compare execution speed and numerical difference between AppleSilicon-FP64 and Accelerate using identical inputs and trial counts.
 * @param[in] argc Number of arguments.
 * @param[in] argv Command-line arguments.
 * @return Zero on success, one on invalid input or execution failure.
 * @~
 */
static int fp64_benchmark(int argc, char **argv)
{
    if (argc > 4) {
        fprintf(stderr, "使い方: fp64_metal benchmark [行列の次数=512] [試行回数=5]\n");
        return 1;
    }
    uint32_t n = 512, trials = 5;
    apple_fp64_options_t options = apple_fp64_default_options();
    if ((argc > 2 && !fp64_parse_number(argv[2], &n))
        || (argc > 3 && !fp64_parse_number(argv[3], &trials))) return 1;
    if (n == 0 || trials == 0) {
        fprintf(stderr, "行列の次数と試行回数は正の値で指定してください。\n");
        return 1;
    }
    size_t bytes;
    if (!fp64_matrix_bytes(n, n, &bytes)) return 1;
    double *a = malloc(bytes), *b = malloc(bytes), *reference = malloc(bytes);
    double *samples = calloc((size_t)trials * 16, sizeof(double));
    apple_fp64_multiplier_t *multiplier = NULL;
    apple_fp64_result_t gpu = {0};
    apple_fp64_error_t error = {0};
    int exit_code = 1;
    if (a == NULL || b == NULL || reference == NULL || samples == NULL) {
        fprintf(stderr, "CPUのメモリーを確保できません。\n");
        goto finish;
    }
    size_t count = bytes / sizeof(double);
    fp64_random_t random;
    fp64_random_initialize(&random, 123);
    for (size_t index = 0; index < count; ++index) {
        a[index] = fp64_random_value(&random);
        b[index] = fp64_random_value(&random);
    }
    double initialization_start = fp64_monotonic_seconds();
    if (!fp64_create_multiplier(&multiplier)) goto finish;
    double initialization = fp64_monotonic_seconds() - initialization_start;
    fp64_benchmark_series_t series[2] = {{.name = "Accelerate"}, {.name = "AppleSilicon-FP64"}};
    for (size_t method = 0; method < 2; ++method) {
        series[method].total = samples + (size_t)trials * 5 * method;
        series[method].prepare = series[method].total + trials;
        series[method].product = series[method].prepare + trials;
        series[method].reconstruct = series[method].product + trials;
        series[method].wait = series[method].reconstruct + trials;
    }
    double *intervals = samples + (size_t)trials * 10;
    double *ratios = intervals + (size_t)trials * 4;
    double *deviations = ratios + trials;
    const double warmup_seconds = 0.25; /**< @~japanese 各方式で準備実行を続ける最小の累計時間（秒）。
                                         * @~english Minimum cumulative warm-up duration for each method, in seconds.
                                         * @~
                                         */
    const double interval_seconds = 0.05; /**< @~japanese 1区間あたりの反復回数を決めるための目標時間（秒）。
                                           * @~english Target duration per interval used to choose the repetition count, in seconds.
                                           * @~
                                           */
    uint32_t repetitions[2] = {1, 1};
    double warmed[2] = {0, 0};
    for (size_t round = 0; round < 2 || warmed[0] < warmup_seconds || warmed[1] < warmup_seconds; ++round) {
        for (size_t method = 0; method < 2; ++method) {
            apple_fp64_measurement_t measurement = {0};
            if (!fp64_benchmark_interval(method, repetitions[method], multiplier, a, b, reference,
                                          n, options, &measurement, &error)) {
                fp64_print_error(&error);
                goto finish;
            }
            warmed[method] += measurement.total_seconds * repetitions[method];
            repetitions[method] = (uint32_t)fmin(UINT32_MAX, ceil(interval_seconds / measurement.total_seconds));
        }
    }
    for (uint32_t trial = 0; trial < trials; ++trial) {
        for (size_t interval = 0; interval < 4; ++interval) {
            size_t method = (trial + (interval == 1 || interval == 2)) % 2;
            apple_fp64_measurement_t measurement = {0};
            if (!fp64_benchmark_interval(method, repetitions[method], multiplier, a, b, reference,
                                          n, options, &measurement, &error)) {
                fp64_print_error(&error);
                goto finish;
            }
            intervals[(size_t)trial * 4 + interval] = measurement.total_seconds;
            fp64_benchmark_series_t *current = &series[method];
            current->total[trial] += measurement.total_seconds / 2;
            current->prepare[trial] += measurement.prepare_seconds / 2;
            current->product[trial] += measurement.product_seconds / 2;
            current->reconstruct[trial] += measurement.reconstruct_seconds / 2;
            current->wait[trial] += measurement.wait_seconds / 2;
            current->latest = measurement;
        }
        ratios[trial] = series[0].total[trial] / series[1].total[trial];
    }
    if (apple_fp64_multiply(multiplier, a, count, b, count, n, n, n, options, &gpu, &error)
        != APPLE_FP64_SUCCESS) {
        fp64_print_error(&error);
        goto finish;
    }
    double maximum_error = 0;
    for (size_t index = 0; index < count; ++index)
        maximum_error = fmax(maximum_error, fabs(gpu.values[index] - reference[index]));
    printf("デバイス: %s\n行列: %u × %u、試行回数: %u\nMetalの初期化: %g ms\n",
           apple_fp64_device_name(multiplier), n, n, trials, initialization * 1000);
    printf("準備実行: 各方式を累計%.3f秒以上、2区間以上。区間の目標時間: %.3f秒。\n",
           warmup_seconds, interval_seconds);
    for (size_t method = 0; method < 2; ++method)
        printf("%sの準備実行: %.3f秒、一区間の反復回数: %u\n",
               series[method].name, warmed[method], repetitions[method]);
    printf("AppleSilicon-FP64の全反復でAとBの変換と戻り値の解放を含めています。\n"
           "各区間の値は一回当たりの実時間です。実行した順に示します。\n");
    for (uint32_t trial = 0; trial < trials; ++trial) {
        printf("試行 %u:", trial + 1);
        for (size_t interval = 0; interval < 4; ++interval) {
            size_t method = (trial + (interval == 1 || interval == 2)) % 2;
            printf(" %s %.6f ms%s", series[method].name,
                   intervals[(size_t)trial * 4 + interval] * 1000, interval < 3 ? " →" : "");
        }
        printf("、速度比 %.6f\n", ratios[trial]);
        printf("  入力変換 %.6f ms、GPUの行列積 %.6f ms、復元 %.6f ms、CPUの待ち時間 %.6f ms\n",
               series[1].prepare[trial] * 1000, series[1].product[trial] * 1000,
               series[1].reconstruct[trial] * 1000, series[1].wait[trial] * 1000);
    }
    double ratio = fp64_median(ratios, trials);
    printf("同一試行の速度比（Accelerate / AppleSilicon-FP64）: %.6f (%.6f ～ %.6f)\n",
           ratio, ratios[0], ratios[trials - 1]);
    printf("全体の実時間の中央値、最小値、最大値と、中央値からの絶対偏差の中央値を示します。\n");
    for (size_t method = 0; method < 2; ++method) {
        fp64_benchmark_series_t *current = &series[method];
        double total = fp64_median(current->total, trials);
        for (uint32_t trial = 0; trial < trials; ++trial)
            deviations[trial] = fabs(current->total[trial] - total);
        double deviation = fp64_median(deviations, trials);
        printf("%s: %.6f ms (%.6f ～ %.6f ms)、%.3f GFLOP/s、絶対偏差の中央値 %.3f%%\n",
               current->name, total * 1000, current->total[0] * 1000, current->total[trials - 1] * 1000,
               2.0 * n * n * n / total / 1e9, deviation / total * 100);
        if (method == 1) {
            printf("  入力変換 %.3f ms、GPUの行列積 %.3f ms、復元 %.3f ms、CPUの待ち時間 %.3f ms\n"
                   "  法 %u 個、Metalの作業領域 %.3f MiB\n",
                   fp64_median(current->prepare, trials) * 1000, fp64_median(current->product, trials) * 1000,
                   fp64_median(current->reconstruct, trials) * 1000, fp64_median(current->wait, trials) * 1000,
                   current->latest.modulus_count, current->latest.workspace_bytes / 1048576.0);
        }
    }
    printf("Accelerateとの最大絶対差: %.3e\n", maximum_error);
    exit_code = 0;
finish:
    apple_fp64_result_destroy(&gpu);
    apple_fp64_multiplier_destroy(multiplier);
    free(a);
    free(b);
    free(reference);
    free(samples);
    return exit_code;
}

/**
 * @~japanese
 * @brief コマンドラインツール（fp64_metal）のエントリーポイント。
 * @param[in] argc 引数の個数。
 * @param[in] argv コマンドライン引数。
 * @return 正常終了時は0、入力の不備や実行の失敗時は1。
 * @~english
 * @brief Entry point of the fp64_metal command-line tool.
 * @param[in] argc Number of arguments.
 * @param[in] argv Command-line arguments.
 * @return Zero on success, one on invalid input or execution failure.
 * @~
 */
int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "multiply") == 0) return fp64_multiply_files(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "benchmark") == 0) return fp64_benchmark(argc, argv);
    fprintf(stderr, "使い方: fp64_metal benchmark [次数] [試行回数]\n"
                    "        fp64_metal multiply M N K A.bin B.bin C.bin [行のまとまりの大きさ]\n");
    return 1;
}
