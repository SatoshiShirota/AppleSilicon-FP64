"""外部のCおよびC++プロジェクトから、ソース組み込みとパッケージ導入の両方でライブラリーを利用できるか検証する。"""

import subprocess
import sys
import tempfile
from pathlib import Path


def run(arguments: list[str]) -> None:
    """コマンドを実行し、失敗したときは標準出力と標準エラー出力を表示して例外を送出する。

    Args:
        arguments: 実行するコマンドとその引数のリスト。
    """
    completed = subprocess.run(arguments, capture_output=True, text=True, timeout=60)
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        completed.check_returncode()


def main() -> None:
    """CまたはC++のみを有効にした外部プロジェクトを構成し、ライブラリーのリンクと行列積の実行を確認する。"""
    cmake, ctest, source_name, build_name, configuration, generator = sys.argv[1:]
    source = Path(source_name)
    build = Path(build_name)
    with tempfile.TemporaryDirectory(prefix="apple-fp64-consumers-") as directory:
        workspace = Path(directory)
        installed = workspace / "installed package"
        relocated = workspace / "relocated package"
        run([cmake, "--install", str(build), "--prefix", str(installed), "--config", configuration])
        installed.rename(relocated)
        for mode in ("source", "package"):
            for language in ("C", "CXX"):
                consumer_build = workspace / f"{mode}-{language}"
                settings = [f"-DAPPLE_FP64_SOURCE_DIR={source}"] if mode == "source" else [f"-DCMAKE_PREFIX_PATH={relocated}"]
                run([cmake, "-S", str(source / "tests" / "consumer"), "-B", str(consumer_build),
                     "-G", generator, f"-DCMAKE_BUILD_TYPE={configuration}",
                     f"-DAPPLE_FP64_CONSUMER_LANGUAGE={language}", "-DBUILD_TESTING=ON", *settings])
                run([cmake, "--build", str(consumer_build), "--config", configuration])
                if language == "C":
                    cache = (consumer_build / "CMakeCache.txt").read_text()
                    assert "CMAKE_CXX_COMPILER:" not in cache, "Cからの利用にC++コンパイラーが必要になっています。"
                    assert "Python3_EXECUTABLE:" not in cache, "外部からの利用にPythonが必要になっています。"
                run([ctest, "--test-dir", str(consumer_build), "-C", configuration, "--output-on-failure"])
                print(f"{mode} / {language}: 公開ヘッダー、リンク設定、Metalライブラリーと行列積を確認しました。")


if __name__ == "__main__":
    main()
