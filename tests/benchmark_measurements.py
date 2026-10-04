"""ベンチマークの実行順序と、区間の実測値から得る集計を検証する。"""

import math
import re
import statistics
import subprocess
import sys


def main() -> None:
    """二試行の出力を独立に集計し、表示された時間と速度比を照合する。"""
    output = subprocess.run(
        [sys.argv[1], "benchmark", "128", "2", "60"],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout
    names = ("Accelerate", "AppleSilicon-FP64")
    accelerate, metal = names
    orders = ((accelerate, metal, metal, accelerate), (metal, accelerate, accelerate, metal))
    times = {name: [] for name in names}
    ratios = []
    trials = re.findall(r"^試行 (\d+): (.+)、速度比 ([\d.]+)$", output, re.MULTILINE)
    assert len(trials) == len(orders), output
    for expected_index, ((index, values, reported_ratio), order) in enumerate(zip(trials, orders), 1):
        assert int(index) == expected_index, output
        intervals = re.findall(r"(Accelerate|AppleSilicon-FP64) ([\d.]+) ms", values)
        assert tuple(name for name, _ in intervals) == order, output
        for name in names:
            samples = [float(value) for method, value in intervals if method == name]
            assert len(samples) == 2 and all(value > 0 for value in samples), output
            times[name].append(statistics.mean(samples))
        numerator, denominator = (times[name][-1] for name in names)
        ratio = numerator / denominator
        # 区間の表示はミリ秒の小数点以下六桁へ丸められている。
        tolerance = 0.0000005 * (1 + ratio) / denominator + 0.0000005
        assert math.isclose(float(reported_ratio), ratio, abs_tol=tolerance), output
        ratios.append(float(reported_ratio))

    for name in names:
        warmup = re.search(rf"^{re.escape(name)}の準備実行: ([\d.]+)秒、一区間の反復回数: (\d+)$",
                           output, re.MULTILINE)
        assert warmup is not None and int(warmup[2]) > 0, output
        summary = re.search(
            rf"^{re.escape(name)}: ([\d.]+) ms \(([\d.]+) ～ ([\d.]+) ms\)、"
            rf"([\d.]+) GFLOP/s、絶対偏差の中央値 ([\d.]+)%$",
            output, re.MULTILINE,
        )
        assert summary is not None, output
        median = statistics.median(times[name])
        expected = (median, min(times[name]), max(times[name]))
        for actual, value in zip(summary.group(1, 2, 3), expected):
            assert math.isclose(float(actual), value, abs_tol=0.000001), output
        deviation = statistics.median(abs(value - median) for value in times[name]) / median * 100
        tolerance = 0.0001 / median + 0.0005
        assert math.isclose(float(summary[5]), deviation, abs_tol=tolerance), output

    summary = re.search(r"^同一試行の速度比（Accelerate / AppleSilicon-FP64）: "
                        r"([\d.]+) \(([\d.]+) ～ ([\d.]+)\)$", output, re.MULTILINE)
    assert summary is not None, output
    for actual, expected in zip(summary.groups(), (statistics.median(ratios), min(ratios), max(ratios))):
        assert math.isclose(float(actual), expected, abs_tol=0.000001), output
    print("区間の実行順序と、時間・速度比・絶対偏差の集計が一致しました。")


if __name__ == "__main__":
    main()
