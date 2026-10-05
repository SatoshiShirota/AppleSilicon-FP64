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
 * @brief コマンド引数を32ビットの非負整数へ変換する。
 * @param[in] text 十進数の引数。
 * @param[out] value 非負整数。
 * @return 全文字を変換できた場合はtrue。
 * @~english
 * @brief Convert a command-line argument to a nonnegative 32-bit integer.
 * @param[in] text Decimal argument.
 * @param[out] value Nonnegative integer.
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
 * @brief FP64行列のファイルのバイト数を算出する。
 * @param[in] rows 行数。
 * @param[in] columns 列数。
 * @param[out] bytes 行列のバイト数。
 * @return 寸法とファイルのサイズを表現できる場合はtrue。
 * @~english
 * @brief Compute the size in bytes of an FP64 matrix file.
 * @param[in] rows Number of rows.
 * @param[in] columns Number of columns.
 * @param[out] bytes Matrix size in bytes.
 * @return true if the dimensions and file size are representable.
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
 * @brief FP64のビット列を、行優先のバイナリーファイルから読む。
 * @param[in] path 入力ファイル。
 * @param[in] rows 行数。
 * @param[in] columns 列数。
 * @param[out] values 呼び出し側がfreeで解放する配列。空の場合はNULL。
 * @param[out] count 要素数。
 * @return 入力を読み込めた場合はtrue。失敗理由を標準エラー出力へ書く。
 * @~english
 * @brief Read FP64 bit patterns from a row-major binary file.
 * @param[in] path Input file.
 * @param[in] rows Number of rows.
 * @param[in] columns Number of columns.
 * @param[out] values Array to be released by the caller with free. NULL when empty.
 * @param[out] count Number of elements.
 * @return true if the input was read. Write the failure reason to standard error.
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
 * @brief 実行ファイルと同じディレクトリーにあるMetalライブラリーのパスを求める。
 * @return 呼び出し側がfreeで解放するパス。取得失敗時はNULL。
 * @~english
 * @brief Find the path of the Metal library in the same directory as the executable.
 * @return Path to be released by the caller with free. NULL if the path could not be obtained.
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
 * @brief ライブラリーの診断を標準エラー出力へ書き、メッセージを解放する。
 * @param[in,out] error 失敗した操作の診断。
 * @pre 診断がNULLである失敗はCPUのメモリー確保の失敗だけであること。
 * @~english
 * @brief Write the library diagnostic to standard error and release the message.
 * @param[in,out] error Diagnostic for the failed operation.
 * @pre A NULL diagnostic must indicate only a CPU memory allocation failure.
 * @~
 */
static void fp64_print_error(apple_fp64_error_t *error)
{
    fprintf(stderr, "%s\n", error->message != NULL ? error->message : "CPUのメモリーを確保できません。");
    apple_fp64_error_destroy(error);
}

/**
 * @~japanese
 * @brief コマンドが使用するMetalの計算器を作成する。
 * @param[out] multiplier 成功時の計算器。
 * @return 作成できた場合はtrue。
 * @~english
 * @brief Create the Metal multiplier used by the command-line tool.
 * @param[out] multiplier Multiplier on success.
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
 * @brief 指定されたファイルの行列積を計算する。
 * @param[in] argc 引数の個数。
 * @param[in] argv コマンド引数。
 * @return 正常終了の場合は0、入力または実行の失敗では1。
 * @~english
 * @brief Compute the matrix product from the specified files.
 * @param[in] argc Number of arguments.
 * @param[in] argv Command-line arguments.
 * @return Zero on success, one on input or execution failure.
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
 * @brief mt19937_64の状態。性能測定の入力を再現する。
 * @~english
 * @brief State of mt19937_64 used to reproduce benchmark inputs.
 * @~
 */
typedef struct fp64_random_s {
    uint64_t state[312]; /**< @~japanese Mersenne Twisterの状態の語。
                          * @~english Words of the Mersenne Twister state.
                          * @~
                          */
    size_t position; /**< @~japanese 次に取り出す語の位置。312では状態を更新する。
                      * @~english Position of the next word to extract. Update the state at 312.
                      * @~
                      */
} fp64_random_t;

/**
 * @~japanese
 * @brief 性能測定用の乱数列を初期化する。
 * @param[out] random 初期化する状態。
 * @param[in] seed 乱数の種。
 * @~english
 * @brief Initialize the random sequence used for benchmarking.
 * @param[out] random State to initialize.
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
 * @brief mt19937_64から、区間[-1,1)のFP64の入力値を求める。
 * @param[in,out] random 乱数の状態。
 * @return 一様分布の行列の要素。
 * @~english
 * @brief Generate an FP64 input value in the interval [-1,1) from mt19937_64.
 * @param[in,out] random Random state.
 * @return Uniformly distributed matrix element.
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
 * @brief 二つの測定値をqsortで比較する。
 * @param[in] left 左のdoubleへのポインター。
 * @param[in] right 右のdoubleへのポインター。
 * @return 左が小さい場合は負、等しい場合は0、大きい場合は正。
 * @~english
 * @brief Compare two measurements for qsort.
 * @param[in] left Pointer to the left double.
 * @param[in] right Pointer to the right double.
 * @return Negative if the left value is smaller, zero if equal, and positive if greater.
 * @~
 */
static int fp64_compare_samples(const void *left, const void *right)
{
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

/**
 * @~japanese
 * @brief 全試行の測定値をソートし、中央値を求める。
 * @param[in,out] samples 測定値。昇順に並べ替える。
 * @param[in] count 正の試行回数。
 * @return 偶数個の場合は中央の二値の平均、それ以外は中央の値。
 * @~english
 * @brief Sort measurements from all trials and compute their median.
 * @param[in,out] samples Measurements. Sorted in ascending order.
 * @param[in] count Positive number of trials.
 * @return Mean of the two middle values for an even count, otherwise the middle value.
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
 * @brief 性能比較の一方式について保持する全試行の測定値。
 * @~english
 * @brief Measurements from all trials retained for one method in the performance comparison.
 * @~
 */
typedef struct fp64_benchmark_series_s {
    const char *name; /**< @~japanese 方式名。
                       * @~english Method name.
                       * @~
                       */
    double *total; /**< @~japanese 試行ごとの全体の実時間。
                    * @~english Total wall-clock time for each trial.
                    * @~
                    */
    double *prepare; /**< @~japanese 試行ごとの入力変換の時間。
                      * @~english Input conversion time for each trial.
                      * @~
                      */
    double *product; /**< @~japanese 試行ごとのGPUの行列積の時間。
                      * @~english GPU matrix multiplication time for each trial.
                      * @~
                      */
    double *reconstruct; /**< @~japanese 試行ごとの復元の時間。
                          * @~english Reconstruction time for each trial.
                          * @~
                          */
    double *wait; /**< @~japanese 試行ごとのCPUの待ち時間。
                   * @~english CPU wait time for each trial.
                   * @~
                   */
    apple_fp64_measurement_t latest; /**< @~japanese 最後の試行の資源量。
                                      * @~english Resource usage from the last trial.
                                      * @~
                                      */
} fp64_benchmark_series_t;

/**
 * @~japanese
 * @brief 同じ方式を連続して実行し、行列積一回当たりの時間を求める。
 * @param[in] method Accelerateは0、AppleSilicon-FP64は1。
 * @param[in] repetitions 区間内の反復回数。正の値。
 * @param[in,out] multiplier GPUの計算器。
 * @param[in] a n行n列の入力。
 * @param[in] b n行n列の入力。
 * @param[out] reference Accelerateの出力を保持するn行n列の配列。
 * @param[in] n 行列の次数。
 * @param[in] options GPUでの計算の設定。
 * @param[out] measurement 各処理の平均時間と最後の反復の資源量。
 * @param[out] error 実行に失敗した場合の診断。
 * @return 全反復に成功した場合はtrue。
 * @note 時間の範囲と反復の意味はREADME.md「コマンドの使用方法」で定める。
 * @~english
 * @brief Run the same method repeatedly and measure the time per matrix product.
 * @param[in] method Zero for Accelerate, one for AppleSilicon-FP64.
 * @param[in] repetitions Positive number of repetitions within the interval.
 * @param[in,out] multiplier GPU multiplier.
 * @param[in] a n-by-n input.
 * @param[in] b n-by-n input.
 * @param[out] reference n-by-n array storing the Accelerate output.
 * @param[in] n Matrix order.
 * @param[in] options GPU computation options.
 * @param[out] measurement Mean time for each stage and resource usage from the last repetition.
 * @param[out] error Diagnostic on execution failure.
 * @return true if every repetition succeeded.
 * @note Timing boundaries and repetition semantics are defined in README.en.md, section "Command-line usage".
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
 * @brief AppleSilicon-FP64とAccelerateの性能を、同じ入力と試行回数で比較する。
 * @param[in] argc 引数の個数。
 * @param[in] argv コマンド引数。
 * @return 正常終了の場合は0、入力または実行の失敗では1。
 * @~english
 * @brief Compare AppleSilicon-FP64 and Accelerate performance using the same inputs and trial counts.
 * @param[in] argc Number of arguments.
 * @param[in] argv Command-line arguments.
 * @return Zero on success, one on input or execution failure.
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
    const double warmup_seconds = 0.25; /**< @~japanese 各方式に割り当てる準備実行の最小時間。
                                         * @~english Minimum warm-up duration allocated to each method.
                                         * @~
                                         */
    const double interval_seconds = 0.05; /**< @~japanese 反復回数を決めるための区間の目標時間。
                                           * @~english Target interval duration used to choose the repetition
                                           * count.
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
 * @brief 実験用コマンドの入口。
 * @param[in] argc 引数の個数。
 * @param[in] argv コマンド引数。
 * @return 正常終了では0、入力または実行の失敗では1。
 * @~english
 * @brief Entry point of the experimental command-line tool.
 * @param[in] argc Number of arguments.
 * @param[in] argv Command-line arguments.
 * @return Zero on success, one on input or execution failure.
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
