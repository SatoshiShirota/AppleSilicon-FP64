# AppleSilicon-FP64

Apple SiliconのMetalで、FP64形式の実数密行列積を計算するライブラリーと実験用コマンドである。公開APIとコマンドはC、Metalの資源管理と実行指示はObjective-Cで実装する。C++からも同じCのヘッダーと関数を利用できる。尾崎スキームIIの整数剰余による行列積を使い、値に依存する演算をすべてGPUで実行する。

## 用語

| 用語 | この文書での意味 |
|---|---|
| 法 | 整数の余りを求めるときの除数。 |
| CRT | 中国剰余定理による復元。複数の法に対する余りから、範囲内の整数を一意に求める。 |
| 整数幅 | 行または列の最大値を基準として、整数化で残す二進数の桁数。FP64の仮数の桁数とは異なる。 |

## 計算の定義

入力は有限のFP64値を持つ、行優先の連続した行列\(A\in\mathbb{R}^{M\times K}\)と\(B\in\mathbb{R}^{K\times N}\)である。係数付きの積、転置、NaN、無限大の入力を含むBLAS互換の操作は扱わない。空の行列と\(K=0\)を許容する。

行列\(A\)の行ごとに\(h_i=1+\lfloor\log_2\max_k|A_{ik}|\rfloor\)、行列\(B\)の列ごとに同じ定義で\(g_j\)を決める。全要素がゼロの場合の指数は0とする。指数の取得にはFP64のビット列を使う。

整数幅を\(p_A,p_B\)とすると、計算する整数は次の式で定まる。\(\operatorname{trunc}\)はゼロ方向への切り捨てである。

\[
Q^A_{ik}=\operatorname{trunc}(A_{ik}2^{p_A-h_i}),\qquad
Q^B_{kj}=\operatorname{trunc}(B_{kj}2^{p_B-g_j})
\]

出力は整数行列積を復元した後、FP64形式へ最近接偶数丸めした値である。非正規化数は最終的な刻みに直接丸める。範囲を超える値は符号付きの無限大になる。厳密にゼロとなる積は正のゼロになる。負の値が丸めによってゼロになる場合は負のゼロになる。

\[
\widehat C_{ij}=\operatorname{RN}_{64}\left((Q^A Q^B)_{ij}2^{h_i+g_j-p_A-p_B}\right)
\]

この定義は、元の入力を無条件に損失なく保持する保証や、`cblas_dgemm`とのビット単位の一致を意味しない。たとえば\(A=[1,-1,2^{-80}]\)、\(B=[1,1,1]^T\)では、整数幅が60のときに小さい項が消える。必要な整数幅は入力の指数の分布と、許容する誤差によって決まる。

## GPUでの計算

GPUで使う演算を次の表に示す。

| 演算 | 実行の方法 |
|---|---|
| 行と列の指数の取得 | FP64のビット列から整数演算で求める。 |
| INT8の余りへの変換 | 仮数と指数から整数化した値の余りを求める。 |
| INT8入力・INT32蓄積による行列積 | Metalの`mpp::tensor_ops::matmul2d`を使う。 |
| CRTとFP64形式への丸め | 整数配列から結果のビット列を組み立てる。 |

法には256と、256未満の相異なる奇素数を大きい順に使う。法の積\(P\)が\(P>2K2^{p_A+p_B}\)を満たすまで法を選ぶ。利用できる法の積では要求された整数幅を支えられない場合、計算を拒否する。整数幅を自動的に減らす処理は行わない。

余りは符号付きINT8に収める。内積は各部分の絶対値の総和が\(2^{24}\)以下になる長さに分割する。この範囲では、FP32の中間値を使う実行でも整数を正確に表せる。部分内積はINT32でまとめて加算し、オーバーフローしない長さごとに余りを求める。INT32の出力型だけを根拠に、長い内積全体の整数精度を仮定しない。整数化の指数は分割によって変えない。CRTは基数\(2^{24}\)の整数配列で行う。GPUの復元に必要な積、加算、除算は32ビット整数の範囲に収まる。

行列積は複数のSIMDグループでタイルを分担する。行、列、内積の範囲がタイル全体を覆う場合は、コンパイル時に決まるテンソルの寸法を使う。行列の端では実際の寸法を使う。

入力変換とCRTでは、法ごとに除算用の係数を一度作成する。余りは乗算の上位32ビットと一回の補正から求める。入力変換では、仮数の切り捨てを法の数にかかわらず一度だけ行う。2のべき乗の余りは定数の表から取得する。CRTで使用する整数配列の桁数は、取り込んだ法の積に応じて増やす。

\(B\)の余りは一度生成して保持する。\(A\)と出力は行のまとまりごとに処理する。入力の64ビットは二つの32ビット整数として読む。CPUは定数の作成と実行指示を担当し、出力を読む前にGPUの完了を待つ。

## 実行環境と資源

Apple Silicon、macOS 26以降、Metal Shading Language 4をコンパイルできるXcode、CMakeが必要である。CPU側の実装にはC11とObjective-CのARCを使う。GPUカーネルはMetal Shading Languageで記述する。検証にはPython 3とC++コンパイラーを使う。外部の数値計算ライブラリーは実装に必要ない。性能比較にはmacOSのAccelerateを使う。

行列の各次元はMetalのテンソルが表現できる符号付き32ビット整数の範囲とする。作業領域は使用する法の数と行列の形状から確保する。失敗は戻り値で分類し、診断メッセージを返す。資源不足やMetalの実行失敗をCPU計算へ置き換える処理は行わない。

## ビルドと検証

XcodeのMetal Toolchainが未導入の場合、先にAppleの追加コンポーネントを導入する。

```sh
xcodebuild -downloadComponent MetalToolchain
```

次のコマンドでライブラリー、Metalカーネル、実験用コマンドをビルドする。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

検証ではPython標準ライブラリーの`Fraction`と任意長整数を使う。入力の整数化、厳密な内積、FP64への最終丸めを独立に計算し、GPUの出力の全ビットと比較する。通常値の丸め、非正規化数、オーバーフロー、桁の打ち消し、INT32の内積の分割を検証する。Cからは同じ計算器で寸法、入力、整数幅を変えた積と失敗時の診断を検証する。C++からもヘッダーを読み込み、ライブラリーをリンクして積を検証する。

ライブラリーとコマンドだけをビルドする場合は、CMakeに`-DBUILD_TESTING=OFF`を指定する。この構成ではC++コンパイラーとPythonを使わない。

## コマンドの使用方法

正方行列について、Accelerateの`cblas_dgemm`とGPU完結版の性能を比較する。引数は行列の次数、測定の試行回数、両入力の整数幅の順である。

```sh
./build/fp64_metal benchmark 512 5 60
```

初期化と方式ごとの一回の準備実行を、反復測定から除く。全試行で入力変換を行う。全体の時間は作業領域の確保、入力と出力のコピー、同期を含む。処理時間には中央値と最小値、最大値を示す。方式の実行順序は試行ごとに巡回させる。

入力変換、行列積、復元の時間はGPUの実行時間である。CPUの待ち時間はGPUの実行時間と重なるため、内訳の合計は全体の実時間と一致しない。`Accelerateとの最大絶対差`は性能測定で使った入力についての差であり、精度の保証ではない。

任意の行列は、行優先のFP64値を並べたバイナリーファイルで入力する。ファイルには見出しを付けない。各要素はリトルエンディアンのIEEE 754 binary64形式である。引数は\(M,N,K\)、\(p_A,p_B\)、二つの入力ファイル、出力ファイルの順である。最後の引数で行のまとまりの大きさを指定できる。

```sh
./build/fp64_metal multiply 128 96 64 60 60 A.bin B.bin C.bin 256
```

ファイルの長さが寸法と一致しない場合や、NaNまたは無限大を含む場合、コマンドは理由を標準エラー出力へ書き、終了コード1で終了する。計算が正常に完了するまで、出力ファイルは開かない。

コマンドのファイルの読み込みと有限値の検査はCPUで行う。ライブラリーでは、有限値の前提に従う入力を受け取り、行列の値をCPUで走査せずに計算する。

## CとC++からの使用方法

公開する型、入出力の条件、所有権と失敗時の動作は`include/apple_fp64/matmul.h`で定義する。ヘッダーはC++で読み込む場合に`extern "C"`を適用する。ライブラリーを利用するためのC++のラッパーは必要ない。

`apple_fp64_multiplier_t`はMetalのパイプラインと作業領域を保持するため、複数の積について同じ計算器を再利用できる。作業領域の容量が不足する場合だけ、より大きい領域を確保する。

次の例はCとC++の両方でコンパイルできる。

```c
#include <apple_fp64/matmul.h>
#include <stdio.h>

int main(void)
{
    const double a[] = {1, 2, 3, 4};
    const double b[] = {5, 6, 7, 8};
    apple_fp64_multiplier_t *multiplier = NULL;
    apple_fp64_result_t result = {0};
    apple_fp64_error_t error = {0};
    apple_fp64_status_t status = apple_fp64_multiplier_create("build/fp64.metallib", &multiplier, &error);
    if (status == APPLE_FP64_SUCCESS) {
        status = apple_fp64_multiply(multiplier, a, 4, b, 4, 2, 2, 2,
                                     apple_fp64_default_options(), &result, &error);
    }
    if (status == APPLE_FP64_SUCCESS) {
        printf("%g %g %g %g\n", result.values[0], result.values[1], result.values[2], result.values[3]);
    } else {
        fprintf(stderr, "%s\n", error.message != NULL ? error.message : "CPUのメモリーを確保できません。");
    }
    apple_fp64_result_destroy(&result);
    apple_fp64_error_destroy(&error);
    apple_fp64_multiplier_destroy(multiplier);
    return status == APPLE_FP64_SUCCESS ? 0 : 1;
}
```

CMakeの`apple_fp64`ターゲットをリンクする。実行時には、ビルドで生成された`fp64.metallib`も配置する。

## 参考資料

数値計算の背景は[尾崎スキームIIの原論文](https://arxiv.org/abs/2504.08009)にある。整数によるCRT復元とFP64のビット列の組み立ては、Apple Silicon向けの構成である。Metalでの行列積には[AppleのMetal 4の解説](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4)と、Xcode SDKの`MPPTensorOpsMatMul2d.h`で定義された整数型の組み合わせを使う。
