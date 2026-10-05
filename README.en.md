# AppleSilicon-FP64

[日本語](README.md) | English

AppleSilicon-FP64 is a library and an experimental command-line tool for computing dense real matrix products in FP64 format using Metal on Apple Silicon. The public API and command-line tool are implemented in C. Metal resource management and dispatch are implemented in Objective-C. C++ callers can use the same C header and functions. Depending on the input values, the library selects either matrix multiplication using integer residues or FP64 fused multiply-add using integer arithmetic. All numerical operations that depend on the input values run on the GPU.

## Glossary

| Term | Meaning in this document |
|---|---|
| Modulus | The divisor used to compute an integer remainder. |
| CRT | Reconstruction using the Chinese remainder theorem. Remainders modulo several moduli uniquely determine an integer within a given range. |
| Integer width | The number of binary digits required to represent the inputs as integers without loss, using the maximum value in each row or column as the reference. This is different from the number of significand bits in FP64. |

## Numerical semantics

The inputs are contiguous, row-major matrices $`A`$ and $`B`$ containing FP64 values. All finite values, subnormal numbers, signed zeros, NaNs, and infinities are supported. Empty matrices and $`K=0`$ are allowed. BLAS-compatible operations with scaling coefficients or transposition are outside the scope of the library.

For each row of matrix $`A`$, the library determines $`h_i=1+\lfloor\log_2\max_k|A_{ik}|\rfloor`$. For each column of matrix $`B`$, it determines $`g_j`$ using the same definition. The exponent is 0 when all elements are zero. Exponents are obtained from the FP64 bit patterns. NaNs and infinities are excluded from this analysis.

For finite nonzero values, let $`\ell_i`$ and $`t_j`$ be the smallest exponents of the least significant nonzero bits in each row and column, respectively. The required width is 1 when all elements are zero. The integer widths needed to preserve the inputs without loss are determined as follows:

$$
p_A=\max_i\max(1,h_i-\ell_i),\qquad
p_B=\max_j\max(1,g_j-t_j)
$$

If all inputs are finite, the library checks whether the product of the available moduli can cover the original integer widths and the dot-product length. If it cannot, the library adjusts exponents along the inner dimension and recomputes the integer widths.

The adjustment is determined from the maximum absolute values in column $`k`$ of A and row $`k`$ of B. If either consists entirely of zeros, $`d_k=0`$. Otherwise, the library uses the following exponents and adjustment:

$$
\alpha_k=1+\left\lfloor\log_2\max_i|A_{ik}|\right\rfloor,\qquad
\beta_k=1+\left\lfloor\log_2\max_j|B_{kj}|\right\rfloor
$$

$$
d_k=\left\lfloor\frac{\beta_k-\alpha_k}{2}\right\rfloor,\qquad
\widetilde A_{ik}=A_{ik}2^{d_k},\qquad
\widetilde B_{kj}=B_{kj}2^{-d_k}
$$

Each product satisfies $`\widetilde A_{ik}\widetilde B_{kj}=A_{ik}B_{kj}`$. The same procedure is applied to the adjusted values to determine the row and column exponents and integer widths. The adjusted values are not rounded to FP64.

If the required range can be covered for either the original inputs or the inputs with adjusted exponents, the library uses INT8 matrix multiplication and CRT. Conversion to integers does not truncate any information. Each output is the exact dot product rounded to FP64 using round-to-nearest, ties-to-even. An output whose exact value is zero is positive zero.

$$
\widehat C_{ij}=\mathrm{RN}_{64}\left(\sum_{k=0}^{K-1} A_{ik}B_{kj}\right)
$$

If the product of the moduli cannot cover the required integer range even after exponent adjustment, or if the inputs contain NaNs or infinities, the library uses the original inputs and performs FP64 fused multiply-add using integer arithmetic on the GPU. Each dot product starts from $`s_0=+0`$ and proceeds in input order along the inner dimension. Each fused multiply-add rounds the product and addition together once, using round-to-nearest, ties-to-even.

$$
s_{k+1}=\mathrm{RN}_{64}(A_{ik}B_{kj}+s_k),\qquad
\widehat C_{ij}=s_K
$$

Both methods round subnormal results directly to the FP64 subnormal spacing. Overflow produces a signed infinity. A negative nonzero value that rounds to zero produces negative zero. A fused multiply-add involving a NaN, multiplication of zero by infinity, or addition of infinities with opposite signs returns a quiet NaN with the bit pattern `0x7ff8000000000000`. NaN payloads and CPU floating-point exception flags are not propagated.

The number and order of rounding operations differ between the two methods. The library does not guarantee bitwise agreement with `cblas_dgemm`, or rounding of the exact dot product only once for every possible input. Inputs are not truncated in advance, even when their exponents span a wide range.

## GPU computation

The following table describes the operations performed on the GPU.

| Operation | Implementation |
|---|---|
| Input analysis | Inspect the exponents and least significant bits of FP64 bit patterns to determine the required integer widths and whether nonfinite values are present. |
| Conversion to INT8 residues | Compute the residues of the integer representations from the significands and exponents. |
| Matrix multiplication with INT8 inputs and INT32 accumulation | Use Metal's `mpp::tensor_ops::matmul2d`. |
| CRT and rounding to FP64 format | Construct the output bit patterns from integer arrays. |
| FP64 fused multiply-add | Represent the significand product and addend as integers, add them, and round to an FP64 bit pattern. |

Analysis of the rows of A and the columns of B is encoded in the same Metal compute command encoder and runs concurrently. After both analyses complete, synchronization makes their memory writes visible to subsequent operations. The integer widths and the presence of nonfinite values are then aggregated.

When exponent adjustment is needed, analysis results for the columns of A and the rows of B are stored in the same analysis buffers, and the GPU computes the adjustments. Once the adjustments have been computed, the row and column analysis results are updated using the adjusted exponents. Computing the adjustments and repeating the analysis are combined into one additional Metal submission. The input bit patterns are preserved; integer exponents are added or subtracted during analysis and residue conversion. Strassen's algorithm also uses the adjustments along the inner dimension that correspond to each block.

### Matrix multiplication using integer residues

The moduli are pairwise coprime integers no greater than 256, used in descending order. Composite moduli allow multiple prime factors to be handled by a single matrix multiplication. Moduli are selected until their product $`P`$ satisfies $`P>2K2^{p_A+p_B}`$. Method selection and rounding semantics are defined in “Numerical semantics.”

Residues are represented in signed INT8. Each dot product is split into segments for which the sum of the absolute values of all terms is at most $`2^{24}`$. Within this range, integers can be represented exactly even by an execution path that uses FP32 intermediate values. Partial dot products are added in INT32, and residues are computed at intervals that prevent overflow. The INT32 output type alone is not taken as evidence that an entire long dot product is computed with exact integer arithmetic. Splitting does not change the exponents used for conversion to integers.

For CRT, the library constructs $`W_t=(P/m_t)((P/m_t)^{-1}\bmod m_t)`$ for each modulus. Let $`r_t`$ be the residue for each output. The library computes $`Y=\sum_t r_t W_t`$ using an integer array in radix $`2^{18}`$. The sum of digit products and the carry for each digit fit in a 32-bit unsigned integer. Each digit's sum is independent of the other digits.

The ratio $`Y/P`$ is approximated in FP32 from the coefficient ratios and residues, and a nearby integer $`q`$ is selected. There are at most 49 moduli, and each residue is at most 255. Within this range, the absolute error from rounding the ratios and adding them is less than 0.1. The exact integer difference $`Y-qP`$ therefore has an absolute value less than $`P`$. If its absolute value exceeds $`P/2`$, a single addition or subtraction of $`P`$ determines the signed result. FP32 is used to select the candidate; reconstruction of the output value and final rounding use integer arithmetic.

Multiple SIMD groups share the matrix multiplication tiles. When the row, column, and inner-dimension ranges cover a complete tile, compile-time tensor dimensions are used. At matrix edges, the actual dimensions are used.

When an entire dot product fits within the range that preserves exact integer results in a single multiplication, the library uses a pipeline without an array of partial dot products or an accumulation step. For long dot products, the final residue calculation and output write occur in the same iteration.

A lower bound on the number of trailing zero bits shared by all integer representations is derived from the input analysis and the selected integer widths. For a block product modulo 256, the minimum of these bounds is determined over the relevant rows of A and columns of B. If the two minima sum to at least 8, every term in the dot product is a multiple of 256, so the multiplication is skipped and zero is written. With Strassen's algorithm, the minimum includes both blocks used to form each sum or difference.

For large matrix products using integer residues, the library applies one level of Strassen's algorithm. It partitions each matrix into four blocks and computes the result with seven block products instead of eight. Block sums and differences are also computed using integer arithmetic modulo each modulus, so the definitions of integer conversion and final rounding do not change. This method is used when M and N are at least 1024, M is even, N is a multiple of 128, K is a multiple of 2048, and each row batch contains at least 128 rows. Equal numbers of rows from the top and bottom halves are processed together, keeping the total number of retained rows within the configured limit. Dedicated kernels assemble the four output blocks before CRT reconstruction.

CRT pipelines are created for different integer-array capacities. The library selects a capacity that can hold the required digits, so GPU threads do not carry unnecessary high-order digits.

Input conversion computes a division coefficient once for each modulus. Residues are obtained using the high 32 bits of a multiplication and one correction. Each input is decomposed into a significand and exponent only once, regardless of the number of moduli. When lower input bits are shifted to the right, all discarded bits are zero. The residues of powers of two used for the upper and lower parts of the significand are obtained from a single table. When obtaining exponents for B, adjacent threads read adjacent columns, and multiple SIMD groups share the inner dimension.

Reconstruction and input-conversion coefficients are reused when the dot-product length and the integer widths of both inputs remain the same. Input values are converted on the GPU on every call.

The matrix-multiplication input for $`B`$ is generated once and retained. $`A`$ and the output are processed in row batches. For Strassen's algorithm, conversion from FP64 to integers and the block sums and differences are computed in the same kernel, without storing intermediate residue matrices.

### FP64 fused multiply-add using integer arithmetic

The product of two FP64 significands is computed as an integer of up to 106 bits. Its exponent is aligned with the addend, and addition or subtraction uses a 128-bit integer. Bits shifted below the integer range are represented by a flag in the least significant bit indicating whether any of them were nonzero. This information is used for final rounding. The product alone is never converted to FP64 before rounding the fused result.

The product and addend are added if their signs agree and subtracted otherwise. For subtraction, the borrow from the highest digit determines the sign. The borrow is expanded into a bit mask to obtain the absolute difference without branching on the sign. Addition and subtraction use the same final-rounding procedure.

Significand multiplication is computed directly from 32-bit partial products and carries. Exponent alignment separates 32-bit and 64-bit shifts to avoid indexing an array at variable positions. Final rounding retains two bits immediately below the retained significand. One determines whether the value is at a rounding midpoint, and the other also records whether any lower bits are nonzero. Significand extraction and the rounding decision use the same exponent-alignment result. The rounding increment, either 0 or 1, is computed with bit operations, and integer comparisons determine carries.

When input analysis establishes that both matrices are finite, the library uses a pipeline compiled for finite inputs. This avoids checking the multiplicands for NaNs and infinities in every fused multiply-add. Overflow during accumulation and rounding of subnormal results are still handled regardless of the input range.

Threadgroups share the output rows and columns. Inputs are decomposed into signed significands and exponents when loaded into shared memory. Multiple outputs reuse the decomposed values, avoiding repeated decomposition of the same multiplicands in each fused multiply-add. Input order along the inner dimension is preserved. The final batch includes only terms that actually exist. Residue matrices and CRT workspace are not used.

### CPU dispatch

The 64 bits of each input are read as two 32-bit integers. For each analysis, the CPU receives a 12-byte result containing the integer widths and the presence of nonfinite values computed by the GPU. The CPU uses this result to select the compute pipeline and workspace. When CRT is used, the CPU constructs reconstruction coefficients that do not depend on the input values. It does not scan matrix values or compute dot products. GPU completion is awaited before reading the output.

## Requirements and resources

Apple Silicon, macOS 26 or later, an Xcode installation that can compile Metal Shading Language 4, and CMake are required. The minimum CMake version is defined by `cmake_minimum_required` in [CMakeLists.txt](CMakeLists.txt). The CPU implementation uses C11 and Objective-C with ARC. GPU kernels are written in Metal Shading Language. Validation requires Python 3 and a C++ compiler. No external numerical library is required by the implementation. Performance comparisons use macOS Accelerate.

Each matrix dimension must fit in the signed 32-bit integer range supported by Metal tensors. Inputs, outputs, and row and column analysis results are retained for the full matrices. Exponent adjustment additionally retains one 32-bit integer per term in the inner dimension. Analysis arrays are expanded and reused with sufficient capacity for both the rows and columns of each input. The capacity of CRT intermediate arrays depends on the number of moduli and the row batch size. For FP64 fused multiply-add, the row batch setting does not affect workspace size. Failures are classified by return status and accompanied by diagnostic messages. Resource shortages and Metal execution failures do not trigger CPU computation.

## Build and validation

If Xcode's Metal Toolchain is not installed, download Apple's additional component first.

```sh
xcodebuild -downloadComponent MetalToolchain
```

Build the library, Metal kernels, and experimental command-line tool with the following commands:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Validation uses `Fraction` from the Python standard library. For inputs that use CRT, the exact dot product and final rounding are computed independently, whether or not exponents are adjusted. For inputs that remain outside the CRT range after adjustment, and for inputs containing nonfinite values, the exact rational result of each fused multiply-add is rounded to FP64. Every bit of the GPU output is compared with the expected value. Validation covers rounding of normal values, subnormal numbers, overflow, cancellation, NaNs, infinities, and splitting INT32 dot products. Inputs with different numbers of trailing zero bits across rows and columns are also checked.

Exponent-adjustment tests cover terms lost by rounding after each fused multiply-add and cancellation of products outside the FP64 range. They also verify preservation of input information and the sign of zero in the final output when adjusted values have absolute magnitudes below the smallest positive FP64 subnormal number.

FP64 fused multiply-add using integer arithmetic is compared with CPU `fma` for 20,000 cases covering the full exponent range and cancellation. C tests reuse the same multiplier with different dimensions and input ranges, and check failure diagnostics. Strassen tests compare rectangular products against exact integer expectations, with different numbers of trailing zero bits in the top, bottom, left, and right blocks. The same products are also checked for blocks with different exponents along the inner dimension. A C++ test includes the header, links the library, and verifies a product.

Benchmark validation checks the execution order of intervals. It independently aggregates times, speed ratios, and absolute deviations from the reported interval values, then compares them with the displayed summaries.

Library integration is validated from independent C and C++ projects. Both source integration and an installed package moved to a different location are checked for public-header inclusion, linking, Metal-file discovery, and matrix multiplication. Source integration also verifies that enabling `BUILD_TESTING` in the parent does not add this library's command-line tool or tests.

To build only the library and command-line tool, pass `-DBUILD_TESTING=OFF` to CMake. This configuration does not use a C++ compiler or Python.

To build only the library, also pass `-DAPPLE_FP64_BUILD_TOOLS=OFF`. `APPLE_FP64_BUILD_TOOLS` defaults to `ON` for a standalone build and `OFF` when included in another project. Tests are built only in a standalone build with `BUILD_TESTING` enabled. The command-line tool required by those tests is also built in that case.

## Command-line usage

For square matrices, the benchmark compares Accelerate's `cblas_dgemm` with AppleSilicon-FP64. The arguments are the matrix order followed by the number of measurement trials.

```sh
./build/fp64_metal benchmark 512 5
```

Initialization and warmup are excluded from repeated measurements. Warmup alternates between the two methods until each has accumulated at least 0.25 seconds of execution and completed at least two intervals. Within each interval, the same input matrices are multiplied repeatedly. The most recent warmup time is used to determine a repetition count for each method that makes an interval last approximately 0.05 seconds. If a single multiplication takes longer, the repetition count is one. Repetition counts are fixed after warmup and are not adjusted in response to measurements.

Each trial measures both methods twice. The interval order is shown below.

| Trial | First interval | Second interval | Third interval | Fourth interval |
|---|---|---|---|---|
| Odd-numbered | Accelerate | AppleSilicon-FP64 | AppleSilicon-FP64 | Accelerate |
| Even-numbered | AppleSilicon-FP64 | Accelerate | Accelerate | AppleSilicon-FP64 |

Both methods time each complete interval using a monotonic clock on the calling side. Dividing by the repetition count gives the wall-clock time per matrix product. AppleSilicon-FP64 includes the time needed to release the result of each repetition. Input analysis and conversion run on every repetition. Workspace and output allocation, input copies, and synchronization are also included. Accelerate reuses a preallocated output array. The GPU writes directly into the output storage returned to the caller.

For each trial, the two interval times are averaged separately for each method. The trial's speed ratio is the Accelerate average divided by the AppleSilicon-FP64 average. Median, minimum, and maximum values are reported for each method's time and for the trial speed ratios. For each method's time, the median absolute deviation from the median is also reported as a percentage of the median time. Individual interval values and their execution order are printed. No outliers are excluded.

Input-conversion time is GPU execution time and includes input analysis. Matrix-multiplication and reconstruction times are also GPU execution times. CPU wait time overlaps GPU execution, so the sum of the individual timings does not equal the overall wall-clock time. The output labeled `Accelerateとの最大絶対差` (“maximum absolute difference from Accelerate”) is the difference for the benchmark inputs, not an accuracy guarantee.

Arbitrary matrices can be supplied as binary files containing row-major FP64 values, without a header. Each element is stored in little-endian IEEE 754 binary64 format. The arguments are $`M,N,K`$, the two input files, and the output file, in that order. An optional final argument specifies the row batch size.

```sh
./build/fp64_metal multiply 128 96 64 A.bin B.bin C.bin 256
```

If a file's length does not match its dimensions, the command writes the reason to standard error and exits with status 1. The output file is not opened until computation has completed successfully.

The CPU reads the command-line tool's input files. NaNs and infinities are accepted. Their numerical treatment is defined in “Numerical semantics.”

## Benchmark results and performance

The following table lists the conditions used for measurements with the `benchmark` command. “Command-line usage” describes which operations are included in the measurements and which are excluded.

| Item | Condition |
|---|---|
| Chip | Apple M3 Max with a 16-core CPU and a 40-core GPU. |
| Memory | 128 GB. |
| OS | macOS 27.0 (26A428). |
| Development environment and build | Xcode 27.0 (27A266a), Release. |
| Power | AC power. |
| Runtime settings | Environment variables controlling Accelerate's thread count and Metal validation were unset. |
| Concurrent activity | OS background tasks and desktop applications were running. |
| Input | FP64 square matrices generated from a uniform distribution over [-1, 1), with random seed 123. |
| Integer widths | Determined automatically from the inputs. |
| Row batch size | 256 rows. |

For each matrix order, three sets of seven trials are run in separate processes. The tables aggregate the trial values defined in “Command-line usage.” The representative value is the median of all 21 trials. Parentheses contain the minimum and maximum over all 21 trials. No outliers are excluded. Time ranges describe trial averages, not the latency range of individual calls. The measurements can be reproduced with the following commands:

```sh
for series in 1 2 3; do
    for n in 128 256 512 1024 2048 4096 8192; do
        ./build/fp64_metal benchmark "$n" 7
    done
done
```

Metal initialization took 28.0–46.3 ms in measurements with the ordinary inputs.

The following table shows overall wall-clock times and AppleSilicon-FP64's effective throughput. Speed ratios are also aggregated from individual trial ratios, so they generally differ from the ratio of the median times. A ratio greater than 1 means AppleSilicon-FP64 is faster. Effective GFLOP/s is calculated from the median overall wall-clock time, using $`2N^3`$ as the operation count for a square matrix product of order $`N`$.

| Order | Accelerate (ms) | AppleSilicon-FP64 (ms) | Speed ratio | AppleSilicon-FP64 effective GFLOP/s |
|---:|---:|---:|---:|---:|
| 128 | 0.0132 (0.0130–0.0137) | 0.5675 (0.4999–0.6400) | 0.023 (0.020–0.026) | 7.4 |
| 256 | 0.0896 (0.0868–0.0927) | 0.7198 (0.6399–0.9237) | 0.123 (0.098–0.137) | 46.6 |
| 512 | 0.3840 (0.3810–0.4005) | 1.3473 (1.2334–1.5448) | 0.288 (0.247–0.309) | 199.2 |
| 1024 | 2.9435 (2.8911–3.0256) | 4.6256 (4.4305–4.8140) | 0.640 (0.602–0.667) | 464.3 |
| 2048 | 23.6977 (23.5078–24.4395) | 24.9748 (24.6720–25.7605) | 0.950 (0.927–0.983) | 687.9 |
| 4096 | 192.5540 (188.0395–196.9010) | 173.3290 (170.8985–175.3040) | 1.112 (1.092–1.132) | 792.9 |
| 8192 | 1534.8580 (1514.8270–1583.9300) | 1337.1440 (1300.7025–1369.6780) | 1.152 (1.119–1.171) | 822.3 |

The following table shows timing variability across all 21 trials. The median absolute deviation is the percentage defined in “Command-line usage.” Because it is less affected by a small number of large fluctuations, it should be considered together with the minimum and maximum times.

| Order | Accelerate median absolute deviation (%) | AppleSilicon-FP64 median absolute deviation (%) |
|---:|---:|---:|
| 128 | 0.711 | 5.788 |
| 256 | 1.143 | 5.085 |
| 512 | 0.611 | 5.229 |
| 1024 | 0.611 | 1.976 |
| 2048 | 0.472 | 0.594 |
| 4096 | 0.703 | 0.607 |
| 8192 | 0.434 | 1.152 |

The following table breaks down GPU execution time and resource usage. Each component time uses the same aggregation method as the overall wall-clock time. Metal workspace is the sum of the capacities of the Metal buffers used for computation. CPU input arrays and other allocations consume additional memory.

| Order | Input conversion (ms) | GPU matrix multiplication (ms) | Reconstruction (ms) | Modulus count | Metal workspace (MiB) |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.063 | 0.030 | 0.018 | 15 | 1.119 |
| 256 | 0.131 | 0.086 | 0.030 | 15 | 4.357 |
| 512 | 0.387 | 0.423 | 0.125 | 15 | 13.550 |
| 1024 | 1.223 | 2.652 | 0.465 | 15 | 46.562 |
| 2048 | 2.727 | 19.121 | 2.699 | 16 | 244.085 |
| 4096 | 7.584 | 152.796 | 10.663 | 16 | 904.132 |
| 8192 | 38.097 | 1219.881 | 41.564 | 16 | 3472.226 |

Across all measured cases with ordinary inputs, the maximum absolute difference from Accelerate was `1.492e-12`.

### Inputs with a wide exponent range

Matrices generated from the same random values as the ordinary inputs are transformed so that exponents alternate along the inner dimension. For even $`k`$, column $`k`$ of A is multiplied by $`2^{-500}`$ and row $`k`$ of B by $`2^{500}`$. For odd $`k`$, the factors are reversed. The exponent changes cancel in each product, widening the input exponent range without changing the magnitude of the result.

To examine the effect of signs, matrices of order 512 and above are also measured after taking the absolute value of every random element and applying the same exponent transformation.

These inputs are passed to the public C API and measured using the same warmup, repetition-count selection, interval order, and aggregation method as in “Command-line usage.” Each input condition is measured in three sets of seven trials. The following table gives the medians and ranges over all 21 trials. Exponent adjustment selected CRT for every input. GPU matrix-multiplication time covers only the multiplication itself, excluding input conversion and reconstruction.

| Order | Input signs | Accelerate (ms) | AppleSilicon-FP64 (ms) | GPU matrix multiplication (ms) | Modulus count | AppleSilicon-FP64 median absolute deviation (%) | Metal workspace (MiB) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | Mixed positive and negative | 0.000520 (0.000518–0.000549) | 0.749960 (0.646867–0.846608) | 0.018 | 15 | 6.181 | 0.114 |
| 64 | Mixed positive and negative | 0.002129 (0.002116–0.002548) | 0.804655 (0.664664–0.874804) | 0.025 | 15 | 3.935 | 0.310 |
| 128 | Mixed positive and negative | 0.013319 (0.013270–0.014881) | 0.836678 (0.726169–0.886571) | 0.027 | 15 | 1.874 | 1.120 |
| 512 | Mixed positive and negative | 0.383215 (0.380586–0.389694) | 1.786138 (1.610000–2.011351) | 0.431 | 15 | 4.709 | 13.552 |
| 512 | Nonnegative | 0.383350 (0.379599–0.387721) | 1.729391 (1.450906–1.993316) | 0.432 | 15 | 6.203 | 13.552 |
| 1024 | Mixed positive and negative | 2.966559 (2.936794–3.204647) | 5.261150 (5.129150–5.457150) | 2.659 | 15 | 1.396 | 46.566 |
| 1024 | Nonnegative | 2.951970 (2.895000–3.147028) | 5.241000 (4.943091–5.470100) | 2.658 | 15 | 1.964 | 46.566 |
| 2048 | Mixed positive and negative | 23.728500 (23.554333–24.348334) | 26.109500 (25.694000–27.047750) | 19.154 | 16 | 0.431 | 244.093 |
| 2048 | Nonnegative | 23.599500 (23.347333–24.261834) | 26.144250 (25.449250–26.840500) | 19.106 | 16 | 0.731 | 244.093 |
| 4096 | Mixed positive and negative | 194.341000 (189.743000–213.096500) | 176.196500 (175.222500–177.843500) | 152.970 | 16 | 0.295 | 904.148 |
| 4096 | Nonnegative | 191.328000 (187.696000–196.124500) | 175.896500 (173.647500–177.203000) | 152.514 | 16 | 0.428 | 904.148 |
| 8192 | Mixed positive and negative | 1538.281500 (1511.738500–1694.330500) | 1356.759500 (1307.479000–1777.079000) | 1225.263 | 16 | 1.966 | 3472.257 |
| 8192 | Nonnegative | 1534.375000 (1519.494000–1541.803500) | 1342.934500 (1311.778500–1390.592000) | 1224.072 | 16 | 1.198 | 3472.257 |

The maximum absolute differences from Accelerate for these measured inputs were `1.492e-12` with mixed signs and `2.501e-11` with nonnegative values. These values do not guarantee accuracy for other inputs or dimensions. The numerical conditions are defined in “Numerical semantics.”

A matrix of order 1024 is also measured with exponents alternating along both the row and column dimensions, as an example where exponent adjustment does not reduce the integer widths. Row and column indices start at 0. When the sum of the indices is even, elements of A are multiplied by $`2^{-500}`$ and elements of B by $`2^{500}`$; when it is odd, the factors are reversed. The same random inputs with mixed signs are used. Each column of A and each row of B then contains both large and small values.

The following table shows the results from three sets of seven trials under the same conditions. Input analysis includes exponent adjustment and repeated analysis. The modulus count is 0, indicating that FP64 fused multiply-add is used.

| Order | Accelerate (ms) | AppleSilicon-FP64 (ms) | Input analysis (ms) | GPU fused multiply-add (ms) | AppleSilicon-FP64 median absolute deviation (%) | Metal workspace (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 2.959889 (2.899500–3.062088) | 49.129750 (48.067250–50.277250) | 0.353 | 47.206 | 0.913 | 24.027 |

The maximum absolute difference from Accelerate for this input was 0.

### Discussion

#### Matrix size and comparison results

With ordinary inputs, Accelerate is faster in all 21 trials for matrices up to order 2048. AppleSilicon-FP64 is faster in all 21 trials at orders 4096 and 8192.

Doubling the order increases the operation count of conventional square matrix multiplication by a factor of eight. AppleSilicon-FP64's overall wall-clock time increases by approximately 6.94 times from order 2048 to 4096, and by approximately 7.71 times from order 4096 to 8192. The measured cases at order 2048 and above use Strassen's algorithm as described in “Matrix multiplication using integer residues.” However, the number of moduli also changes with the dimensions, so measurements at different dimensions cannot establish the speedup attributable to Strassen's algorithm alone.

The tables compare overall wall-clock times when API calls reuse a multiplier. Creating a multiplier for every call adds initialization cost. Accelerate uses a preallocated output array, so the two APIs require different work. The operations included in the measurements are defined in “Command-line usage.”

#### Input analysis and small matrices

To determine integer widths that preserve the inputs without loss, the GPU analyzes the inputs, and the CPU receives the results before selecting a computation method. This synchronization occurs on every API call. Omitting input analysis could lose small terms when matrices with different exponent ranges are passed to the same multiplier.

At order 128, the sum of the representative GPU component times is approximately 19% of the representative overall wall-clock time. When the amount of computation is small, dispatch, synchronization, and output allocation are likely to account for a substantial fraction of the cost. However, the representative values do not come from the same trial, so their difference from the overall time cannot be treated as CPU time.

#### Large matrices and memory

At order 8192, matrix multiplication using residues accounts for approximately 94% of the sum of the representative GPU component times. Performance for large matrices strongly depends on how efficiently matrix products are computed for the required moduli. Because integer widths are determined from the inputs, the number of required moduli depends on the input values.

Metal workspace increases by approximately 3.84 times from order 4096 to 8192. FP64 input and output buffers and the intermediate array for B are required for the full matrices. Reducing the row batch size therefore does not reduce total capacity in the same proportion. Changing the row batch size also affects whether Strassen's algorithm is used. Both time and capacity need to be measured when choosing this setting.

#### Accuracy and scope of measurement

The measured ordinary inputs and the inputs whose exponents cancel along the inner dimension use CRT, without truncation during conversion to integers. Differences from Accelerate arise from different rounding orders and counts in the dot products. Repeated measurements with the same inputs do not increase the variety of inputs checked for accuracy. Numerical validation compares outputs with independent reference values as described in “Build and validation.”

Inputs that remain outside the CRT range after exponent adjustment use FP64 fused multiply-add. This method rounds after each fused multiply-add, so cancellation can produce large relative errors. Small absolute differences measured with ordinary inputs cannot be used as an accuracy guarantee over a wide exponent range. The numerical semantics are defined in “Numerical semantics.”

#### Performance over a wide exponent range

Performance is not determined solely by the overall input exponent range. When changes in the exponents of A and B cancel along the inner dimension, adjustment can reduce the widths sufficiently for lossless integer conversion, allowing INT8 matrix multiplication and CRT. The measured inputs use the same number of moduli as the ordinary inputs, and processing times for large matrices are also close to those for ordinary inputs.

At orders 4096 and 8192 with canceling exponents, the median speed ratios from paired trials show that AppleSilicon-FP64 is approximately 1.09–1.14 times faster than Accelerate. However, some trials at order 8192 with mixed signs were slower than Accelerate. Performance estimates need to consider the reported ranges as well as the medians.

Exponent adjustment requires additional input analysis and synchronization for the CPU to receive the analysis results. For small matrices, this cost is large relative to the amount of computation. If adjustment cannot bring the inputs within the CRT range, FP64 fused multiply-add is performed after paying the analysis cost, and exponent adjustment provides no speedup.

CRT requires residue matrices and reconstruction workspace. The size of the exponent-adjustment array alone does not determine memory use for inputs with a wide exponent range. Both the selected computation method and the modulus count derived from the inputs need to be considered.

These measurements apply to the stated environment and inputs. Performance on other GPUs or inputs needs to be measured under the corresponding conditions.

## Using the library from C and C++

Public types, input and output conditions, ownership, and failure behavior are defined in `include/apple_fp64/matmul.h`. The header applies `extern "C"` when included from C++. No C++ wrapper is required to use the library.

Doxygen comments contain Japanese and English descriptions. Set `OUTPUT_LANGUAGE` to `Japanese` or `English` to output the selected language using [Doxygen's language filter](https://www.doxygen.nl/manual/commands.html#cmdtilde).

An `apple_fp64_multiplier_t` retains Metal pipelines and workspace, allowing the same multiplier to be reused for multiple products. Larger workspace is allocated only when existing capacity is insufficient.

The following example compiles as either C or C++:

```c
#include <apple_fp64/matmul.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s /path/to/fp64.metallib\n", argv[0]);
        return 1;
    }
    const double a[] = {1, 2, 3, 4};
    const double b[] = {5, 6, 7, 8};
    apple_fp64_multiplier_t *multiplier = NULL;
    apple_fp64_result_t result = {0};
    apple_fp64_error_t error = {0};
    apple_fp64_status_t status = apple_fp64_multiplier_create(argv[1], &multiplier, &error);
    if (status == APPLE_FP64_SUCCESS) {
        status = apple_fp64_multiply(multiplier, a, 4, b, 4, 2, 2, 2,
                                     apple_fp64_default_options(), &result, &error);
    }
    if (status == APPLE_FP64_SUCCESS) {
        printf("%g %g %g %g\n", result.values[0], result.values[1], result.values[2], result.values[3]);
    } else {
        fprintf(stderr, "%s\n", error.message != NULL ? error.message : "Could not allocate CPU memory.");
    }
    apple_fp64_result_destroy(&result);
    apple_fp64_error_destroy(&error);
    apple_fp64_multiplier_destroy(multiplier);
    return status == APPLE_FP64_SUCCESS ? 0 : 1;
}
```

### Integrating from source

Add the source directory to the consuming project's `CMakeLists.txt`:

```cmake
add_subdirectory(path/to/AppleSilicon-FP64 apple-fp64)
```

By default, this builds only the library and the required Metal kernels. The consuming project does not need a C++ compiler or Python to build the library. A consuming application written in C++ still needs a C++ compiler.

### Installing and using the package

Install the library built with the procedure in “Build and validation” into the chosen directory:

```sh
cmake --install build --prefix /path/to/apple-fp64
```

The installation contains the static library, public header, Metal file, CMake package configuration, and LICENSE. The experimental command-line tool and tests are not installed.

Find the package in the consuming project's `CMakeLists.txt`:

```cmake
find_package(AppleSiliconFP64 CONFIG REQUIRED)
```

Add the installation directory to the search path when configuring that project:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/apple-fp64
```

The entire installation directory can be moved without breaking the package. After moving it, specify its new location in the search path.

### Linking and deploying the Metal file

For either integration method, link `AppleSiliconFP64::apple_fp64`. The target propagates the public-header search path and the Metal and Foundation link settings to the consuming project. `AppleSiliconFP64_METALLIB` contains the absolute path to the corresponding `fp64.metallib`. When integrating from source, this file is generated during the build.

For a command-line application, the following example copies the Metal file into the executable's directory:

```cmake
add_executable(my_app main.c)
target_link_libraries(my_app PRIVATE AppleSiliconFP64::apple_fp64)
add_custom_command(TARGET my_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${AppleSiliconFP64_METALLIB}"
        "$<TARGET_FILE_DIR:my_app>/fp64.metallib"
    VERBATIM)
```

For a macOS application, include the Metal file as a resource in the application bundle. At runtime, obtain its deployed path and pass it to `apple_fp64_multiplier_create`. The library does not search for the file automatically. Use a path that does not depend on the current working directory.

## License

Usage terms are defined in [LICENSE](LICENSE).

## References

The numerical background is described in the [original paper on Ozaki Scheme II](https://arxiv.org/abs/2504.08009). Integer-based CRT reconstruction and construction of FP64 bit patterns are adapted for Apple Silicon. Matrix multiplication in Metal uses [Apple's description of Metal 4](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4) and the integer type combinations defined in `MPPTensorOpsMatMul2d.h` in the Xcode SDK.
