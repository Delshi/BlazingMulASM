"""Assemble a Windows x64 DLL using LLVM. No C/C++ sources or Windows SDK."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def tool(name: str, llvm_bin: str | None) -> str:
    suffix = ".exe" if os.name == "nt" else ""
    candidates = []
    if llvm_bin:
        candidates.append(Path(llvm_bin) / (name + suffix))
    if os.environ.get("LLVM_BIN"):
        candidates.append(Path(os.environ["LLVM_BIN"]) / (name + suffix))
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    candidates.append(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) /
                      "LLVM" / "bin" / (name + suffix))
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f"{name} not found. Install LLVM or pass --llvm-bin PATH.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-bin")
    args = parser.parse_args()
    clang, linker = tool("clang", args.llvm_bin), tool("lld-link", args.llvm_bin)
    (ROOT / "build").mkdir(exist_ok=True)
    commands = [[clang, "--target=x86_64-pc-windows-msvc", "-c", "-x", "assembler",
                 "vector_ops.S", "-o", "build/vector_ops.obj"],
                [linker, "/dll", "/noentry", "/nodefaultlib", "/machine:x64",
                 "/dynamicbase", "/nxcompat", "/timestamp:0", "/def:vector_ops.def",
                 "/out:vector_ops.dll", "/implib:build/vector_ops.lib", "build/vector_ops.obj"]]
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    info = dict(clang=subprocess.check_output([clang, "--version"], text=True),
                linker=subprocess.check_output([linker, "--version"], text=True),
                commands=commands)
    info["sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in ("vector_ops.S", "vector_ops.def", "vector_ops.dll")}
    (ROOT / "build_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("Built:", ROOT / "vector_ops.dll")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Build failed: {exc}") from exc
