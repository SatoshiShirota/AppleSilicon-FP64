"""公開コマンドの数値結果を、有理数による独立した計算で検証する。"""

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
        batch: 一度に処理する行数。
        fused: 広い指数範囲または非有限値を含み、各積和演算で丸める入力。
    """

    name: str
    m: int
    n: int
    k: int
    a: list[float]
    b: list[float]
    batch: int = 256
    fused: bool = False


def round_fraction(value):
    """厳密な有理数をFP64へ最近接偶数丸めする。

    Args:
        value: 丸める値。

    Returns:
        丸めた値。表現範囲を超える場合は符号に従う無限大。
    """
    try:
        return float(value)
    except OverflowError:
        return -math.inf if value < 0 else math.inf


def fused_result(a, b, c):
    """浮動小数点の乗算を使わず、積和演算の期待値を求める。

    Args:
        a: 第一の乗数。
        b: 第二の乗数。
        c: 加数。

    Returns:
        厳密な積と加数の和を一回だけ丸めた値。
    """
    if any(math.isnan(value) for value in (a, b, c)):
        return math.nan
    negative = math.copysign(1, a) != math.copysign(1, b)
    if math.isinf(a) or math.isinf(b):
        if a == 0 or b == 0 or (math.isinf(c) and negative != (c < 0)):
            return math.nan
        return -math.inf if negative else math.inf
    if math.isinf(c):
        return c
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        return -0.0 if (a == 0 or b == 0) and c == 0 and negative and math.copysign(1, c) < 0 else 0.0
    return round_fraction(exact)


def exact_result(case):
    """定義された内積の丸め方で、期待値の全ビットを求める。

    Args:
        case: 検証する入力。

    Returns:
        FP64の期待値のビット列。
    """
    output = bytearray()
    for i in range(case.m):
        for j in range(case.n):
            if case.fused:
                rounded = 0.0
                for t in range(case.k):
                    rounded = fused_result(case.a[i * case.k + t], case.b[t * case.n + j], rounded)
            else:
                exact = sum((Fraction(case.a[i * case.k + t]) * Fraction(case.b[t * case.n + j])
                             for t in range(case.k)), Fraction(0))
                rounded = round_fraction(exact)
            output.extend(struct.pack("<Q", 0x7ff8000000000000) if math.isnan(rounded) else struct.pack("<d", rounded))
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
        Case("保持する仮数が偶数の中間値", 1, 1, 2, [1.0, 2.0**-53], [1, 1]),
        Case("保持する仮数が奇数の中間値", 1, 1, 2, [math.nextafter(1, 2), 2.0**-53], [1, 1]),
        Case("中間値より下位の非ゼロの桁", 1, 1, 3, [1, 2.0**-53, 2.0**-100], [1, 1, 1]),
        Case("仮数の繰り上がり", 1, 1, 2, [math.nextafter(2, 1), 2.0**-53], [1, 1]),
        Case("最小の非正規化数", 1, 1, 1, [tiny], [1]),
        Case("最小の非正規化数の半分", 1, 1, 1, [minimum], [2.0**-53]),
        Case("負の値のゼロへの丸め", 1, 1, 1, [-minimum], [2.0**-53]),
        Case("非正規化数へ直接丸める", 1, 1, 2, [minimum, tiny], [2.0**-53, 2.0**-53]),
        Case("非正規化数から正規化数への境界", 1, 1, 2, [minimum, -tiny], [1, 0.5]),
        Case("最大の有限値", 1, 1, 1, [largest], [1]),
        Case("正負のオーバーフロー", 2, 1, 1, [largest, -largest], [2]),
        Case("打ち消しで残る小さい項", 1, 1, 3, [1, -1, 2.0**-80], [1, 1, 1]),
        Case("指数差が大きい乗数の積", 1, 1, 2, [1, 2.0**-80], [0, 2.0**80]),
        Case("大きな積の厳密な打ち消し", 1, 1, 2, [largest, -largest], [2, 2]),
        Case("CRTで表せる広い整数の積", 1, 1, 2, [1, 2.0**-168], [1, 2.0**-168]),
        Case("CRTの範囲を超える整数の積", 1, 1, 2, [1, 2.0**-169], [1, 2.0**-169], fused=True),
        Case("全有限範囲の小さい項を保持する", 1, 1, 2, [largest, tiny], [0, 1], fused=True),
        Case("指数が逆向きの全有限範囲の積", 1, 1, 2, [largest, tiny], [tiny, largest], fused=True),
        Case("積和演算は積を途中で丸めない", 1, 1, 3,
             [largest, -1, 1 + 2.0**-27], [0, 1, 1 - 2.0**-27], fused=True),
        Case("積和演算の中間値は偶数へ丸める", 3, 1, 3,
             [largest, 1, 2.0**-53,
              largest, math.nextafter(1, 2), 2.0**-53,
              largest, math.nextafter(2, 1), 2.0**-53], [0, 1, 1], fused=True),
        Case("積の範囲を超えた値を加数が打ち消す", 1, 1, 3,
             [tiny, -largest, largest], [0, 1, 1.5], fused=True),
        Case("積和演算の負のアンダーフロー", 1, 1, 2, [largest, -tiny], [0, 0.5], fused=True),
        Case("積和演算の正規化数への繰り上がり", 1, 1, 3,
             [largest, minimum, -tiny], [0, 1, 0.5], fused=True),
        Case("NaNの静寂化と正規化", 2, 2, 1,
             [struct.unpack("<d", struct.pack("<Q", 0xfff0000000000001))[0], 1], [1, math.nan], fused=True),
        Case("正負の無限大とゼロの積", 2, 3, 1, [math.inf, -math.inf], [1, -1, -0.0], fused=True),
        Case("反対符号の無限大の和", 1, 1, 2, [math.inf, -math.inf], [1, 1], fused=True),
        Case("無限大の和と符号付きゼロ", 2, 2, 2, [math.inf, 1, -0.0, -0.0], [1, -0.0, 1, 0.0], fused=True),
        Case("INT32の内積を分割する長さ", 1, 1, 131073, [0.5] * 131073, [0.5] * 131073),
    ]
    a = [0.5, -0.5, 0.5] * 64 + [0.5, -0.25, 0.5]
    b = [0.5] * 129 + [0.5] * 64 + [0.25] * 64 + [17 / 32 + 2.0**-40] + [0.5] * 129
    cases.append(Case("下位ゼロビット数が異なる行と列の積", 65, 129, 3, a, b))
    random_source = random.Random(17)
    for name, m, n, k, batch, spread, fused in [
        ("タイルの端にある長方形", 35, 37, 33, 256, 3, False),
        ("作業領域の再利用と端の行", 11, 7, 19, 3, 4, False),
        ("二次元の入力の処理範囲", 2, 257, 257, 1, 2, False),
        ("行と列で異なる指数の幅", 5, 9, 13, 2, 15, False),
        ("広い指数の分布", 17, 19, 65, 2, 1000, True),
    ]:
        a = [math.ldexp(random_source.uniform(-1, 1), random_source.randint(-spread, spread)) for _ in range(m * k)]
        b = [math.ldexp(random_source.uniform(-1, 1), random_source.randint(-spread, spread)) for _ in range(k * n)]
        cases.append(Case(name, m, n, k, a, b, batch, fused))
    a = [1.0] * (17 * 33)
    b = [1.0] * (33 * 19)
    a[-1], b[-1] = tiny, largest
    cases.append(Case("末尾の行と列を含む積和演算", 17, 19, 33, a, b, 1, True))
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
    command = [str(executable), "multiply", str(case.m), str(case.n), str(case.k), str(a_path), str(b_path), str(c_path), str(case.batch)]
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
    for name, a_bytes, batch in [
        ("行列ファイルの長さ", b"", 256),
        ("ゼロの行のまとまりの大きさ", struct.pack("<d", 1), 0),
    ]:
        a_path.write_bytes(a_bytes)
        b_path.write_bytes(struct.pack("<d", 1))
        c_path.unlink(missing_ok=True)
        command = [str(executable), "multiply", "1", "1", "1", str(a_path), str(b_path), str(c_path), str(batch)]
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
