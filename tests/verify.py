"""公開コマンドの数値結果を、整数と有理数による独立した計算で検証する。"""

import math
import random
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass
class Case:
    """行列の入力と、検証する数値条件を保持する。

    Attributes:
        name: 検証する条件の名称。
        m: 出力の行数。
        n: 出力の列数。
        k: 内積の項数。
        a: 行優先の入力A。
        b: 行優先の入力B。
        pa: Aの整数幅。
        pb: Bの整数幅。
        batch: 一度に処理する行数。
    """

    name: str
    m: int
    n: int
    k: int
    a: list[float]
    b: list[float]
    pa: int = 60
    pb: int = 60
    batch: int = 256


def scale_exponent(values):
    """最大絶対値を囲む2のべき乗の指数を、整数から求める。

    Args:
        values: 一行または一列の有限値。

    Returns:
        最大値の指数。全要素がゼロの場合は0。
    """
    maximum = max((abs(Fraction(value)) for value in values), default=Fraction(0))
    if maximum == 0:
        return 0
    # FP64の分母は2のべき乗なので、この差が最大値を囲む指数になる。
    return maximum.numerator.bit_length() - (maximum.denominator.bit_length() - 1)


def quantize(value, scale, precision):
    """有理数から、ゼロ方向に切り捨てた整数を求める。

    Args:
        value: 有限の入力値。
        scale: 最大値から求めた指数。
        precision: 整数幅。

    Returns:
        定義に従って整数化した値。
    """
    number = Fraction(value)
    exponent = precision - scale
    number *= Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2**-exponent)
    return int(number)


def exact_result(case):
    """CRTを使わず、厳密な整数の内積と最終丸めで期待値を求める。

    Args:
        case: 検証する入力と整数幅。

    Returns:
        FP64の期待値のビット列。
    """
    ha = [scale_exponent(case.a[i * case.k : (i + 1) * case.k]) for i in range(case.m)]
    gb = [scale_exponent(case.b[j :: case.n]) for j in range(case.n)] if case.n else []
    qa = [quantize(value, ha[index // case.k], case.pa) for index, value in enumerate(case.a)]
    qb = [quantize(value, gb[index % case.n], case.pb) for index, value in enumerate(case.b)]
    output = bytearray()
    for i in range(case.m):
        for j in range(case.n):
            integer = sum(qa[i * case.k + t] * qb[t * case.n + j] for t in range(case.k))
            exponent = ha[i] + gb[j] - case.pa - case.pb
            exact = Fraction(integer * 2**exponent) if exponent >= 0 else Fraction(integer, 2**-exponent)
            try:
                rounded = float(exact)
            except OverflowError:
                rounded = -math.inf if integer < 0 else math.inf
            output.extend(struct.pack("<d", rounded))
    return bytes(output)


def matrix_cases():
    """数値的な意味と処理が変わる境界について、入力を作成する。

    Returns:
        検証する入力の列。
    """
    tiny = 2.0**-1074
    minimum = 2.0**-1022
    largest = float.fromhex("0x1.fffffffffffffp1023")
    cases = [
        Case("一要素の負の積", 1, 1, 1, [-1.25], [3.5]),
        Case("ゼロの行と列", 2, 2, 2, [0.0, -0.0, 2, -3], [0, 4, -0.0, 5]),
        Case("空の出力の行", 0, 3, 2, [], [1.0] * 6),
        Case("空の出力の列", 3, 0, 2, [1.0] * 6, []),
        Case("空の内積は正のゼロ", 2, 3, 0, [], []),
        Case("保持する仮数が偶数の中間値", 1, 1, 2, [1.0, 2.0**-53], [1, 1], 54, 1),
        Case("保持する仮数が奇数の中間値", 1, 1, 2, [math.nextafter(1, 2), 2.0**-53], [1, 1], 54, 1),
        Case("中間値より下位の非ゼロの桁", 1, 1, 3, [1, 2.0**-53, 2.0**-100], [1, 1, 1], 101, 1),
        Case("仮数の繰り上がり", 1, 1, 2, [math.nextafter(2, 1), 2.0**-53], [1, 1], 54, 1),
        Case("最小の非正規化数", 1, 1, 1, [tiny], [1], 1, 1),
        Case("最小の非正規化数の半分", 1, 1, 1, [minimum], [2.0**-53], 1, 1),
        Case("負の値のゼロへの丸め", 1, 1, 1, [-minimum], [2.0**-53], 1, 1),
        Case("非正規化数へ直接丸める", 1, 1, 2, [minimum, tiny], [2.0**-53, 2.0**-53], 53, 1),
        Case("非正規化数から正規化数への境界", 1, 1, 2, [minimum, -tiny], [1, 0.5], 53, 2),
        Case("最大の有限値", 1, 1, 1, [largest], [1], 53, 1),
        Case("正負のオーバーフロー", 2, 1, 1, [largest, -largest], [2], 53, 1),
        Case("打ち消しで残る項を整数化が捨てる", 1, 1, 3, [1, -1, 2.0**-80], [1, 1, 1], 60, 60),
        Case("打ち消しで残る項を整数幅の増加で保持する", 1, 1, 3, [1, -1, 2.0**-80], [1, 1, 1], 81, 1),
        Case("大きな積の厳密な打ち消し", 1, 1, 2, [largest, -largest], [2, 2], 53, 1),
        Case("利用できる全法を使う整数幅", 1, 1, 1, [-math.pi], [math.e], 170, 170),
        Case("INT32の内積を分割する長さ", 1, 1, 131073, [0.5] * 131073, [0.5] * 131073, 8, 8),
    ]
    random_source = random.Random(17)
    for name, m, n, k, pa, pb, batch, spread in [
        ("タイルの端にある長方形", 35, 37, 33, 60, 60, 256, 3),
        ("作業領域の再利用と端の行", 11, 7, 19, 60, 60, 3, 4),
        ("二次元の入力の処理範囲", 2, 257, 257, 60, 60, 1, 2),
        ("異なる整数幅", 5, 9, 13, 53, 80, 2, 15),
        ("狭い整数幅による切り捨て", 4, 3, 7, 5, 7, 3, 9),
        ("広い指数の分布", 3, 4, 8, 60, 60, 2, 1000),
    ]:
        a = [math.ldexp(random_source.uniform(-1, 1), random_source.randint(-spread, spread)) for _ in range(m * k)]
        b = [math.ldexp(random_source.uniform(-1, 1), random_source.randint(-spread, spread)) for _ in range(k * n)]
        cases.append(Case(name, m, n, k, a, b, pa, pb, batch))
    return cases


def run_case(executable, directory, case):
    """公開コマンドの出力を、独立した期待値の全ビットと比較する。

    Args:
        executable: ビルド済みコマンド。
        directory: 一時ファイルの格納先。
        case: 検証する入力。
    """
    a_path, b_path, c_path = (directory / name for name in ("a.bin", "b.bin", "c.bin"))
    a_path.write_bytes(struct.pack(f"<{len(case.a)}d", *case.a))
    b_path.write_bytes(struct.pack(f"<{len(case.b)}d", *case.b))
    expected = exact_result(case)
    command = [str(executable), "multiply", str(case.m), str(case.n), str(case.k), str(case.pa), str(case.pb), str(a_path), str(b_path), str(c_path), str(case.batch)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if process.returncode:
        raise AssertionError(f"{case.name}: {process.stderr}")
    actual = c_path.read_bytes()
    if actual != expected:
        for index in range(len(expected) // 8):
            observed, wanted = actual[index * 8 : (index + 1) * 8], expected[index * 8 : (index + 1) * 8]
            if observed != wanted:
                raise AssertionError(f"{case.name}, 要素{index}: {observed.hex()} != {wanted.hex()}")
        raise AssertionError(f"{case.name}: 出力の長さが一致しません。")
    print(f"成功: {case.name}", flush=True)


def rejected_inputs(executable, directory):
    """入力境界で定義された拒否動作を確認する。

    Args:
        executable: ビルド済みコマンド。
        directory: 一時ファイルの格納先。
    """
    a_path, b_path, c_path = (directory / name for name in ("a.bin", "b.bin", "c.bin"))
    for name, a_bytes, precision, batch in [
        ("行列ファイルの長さ", b"", 60, 256),
        ("非有限のファイル入力", struct.pack("<d", math.nan), 60, 256),
        ("法の積を超える整数幅", struct.pack("<d", 1), 171, 256),
        ("ゼロの整数幅", struct.pack("<d", 1), 0, 256),
        ("ゼロの行のまとまりの大きさ", struct.pack("<d", 1), 60, 0),
    ]:
        a_path.write_bytes(a_bytes)
        b_path.write_bytes(struct.pack("<d", 1))
        c_path.unlink(missing_ok=True)
        command = [str(executable), "multiply", "1", "1", "1", str(precision), str(precision), str(a_path), str(b_path), str(c_path), str(batch)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if process.returncode == 0 or not process.stderr.strip() or c_path.exists():
            raise AssertionError(f"拒否できませんでした: {name}")
        print(f"成功: {name}の拒否", flush=True)


def main():
    """一時ディレクトリーで数値検証と入力境界の検証を実行する。"""
    executable = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as directory:
        for case in matrix_cases():
            run_case(executable, Path(directory), case)
        rejected_inputs(executable, Path(directory))


if __name__ == "__main__":
    main()
