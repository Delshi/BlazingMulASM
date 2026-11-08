"""Windows x64 assembly benchmark. Python 3.10+, standard library only."""
from __future__ import annotations

import argparse
import array
import ctypes as ct
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
F32P = ct.POINTER(ct.c_float)
NAMES = ("dot_scalar", "dot_avx", "dot_fma", "had_scalar", "had_avx",
         "had_unroll", "outer_scalar", "outer_avx", "outer_unroll")
REQUIRED = (0, 1, 3, 0, 1, 1, 0, 1, 1)


class Batch(ct.Structure):
    _fields_ = [("kind", ct.c_uint64), ("repeats", ct.c_uint64),
                ("a", ct.c_void_p), ("b", ct.c_void_p),
                ("dst", ct.c_void_p), ("n", ct.c_uint64),
                ("m", ct.c_uint64)]


class Buffer:
    """Owned, 64-byte-aligned float32 storage with 64-byte guard regions."""
    def __init__(self, n: int, values=None, shift: int = 0):
        if n < 0 or shift not in (0, 4):
            raise ValueError("Invalid buffer size/alignment shift")
        self.n = n
        self.raw = ct.create_string_buffer(max(n, 1) * 4 + 256)
        self.address = ((ct.addressof(self.raw) + 127) & ~63) + shift
        self.view = (ct.c_float * n).from_address(self.address)
        self.ptr = ct.cast(self.address, F32P)
        self.poison()
        if values is not None:
            data = array.array("f", values)
            if data.itemsize != 4 or len(data) != n:
                raise ValueError("Expected exactly n binary32 values")
            ct.memmove(self.address, data.tobytes(), n * 4)

    def poison(self):
        ct.memset(ct.addressof(self.raw), 0xA5, ct.sizeof(self.raw))

    def snapshot(self):
        return ct.string_at(self.address, self.n * 4)

    def check_guards(self):
        expected = b"\xa5" * 64
        if (ct.string_at(self.address - 64, 64) != expected or
                ct.string_at(self.address + 4 * self.n, 64) != expected):
            raise AssertionError("Write outside the output buffer")


def configure(lib):
    """Explicit signatures prevent truncation of 64-bit pointers."""
    lib.cpu_features.argtypes, lib.cpu_features.restype = [], ct.c_uint32
    lib.run_batch.argtypes, lib.run_batch.restype = [ct.POINTER(Batch)], ct.c_float
    for name in NAMES:
        fn = getattr(lib, name)
        if name.startswith("dot"):
            fn.argtypes, fn.restype = [F32P, F32P, ct.c_size_t], ct.c_float
        elif name.startswith("had"):
            fn.argtypes, fn.restype = [F32P, F32P, F32P, ct.c_size_t], None
        else:
            fn.argtypes = [F32P, ct.c_size_t, F32P, ct.c_size_t, F32P]
            fn.restype = None
    return lib


def available(lib, group: int):
    flags = int(lib.cpu_features())
    return [i for i in range(group * 3, group * 3 + 3)
            if flags & REQUIRED[i] == REQUIRED[i]]


def job_for(kind, a, b, out, repeats=1):
    return Batch(kind, repeats, a.address, b.address, out.address, a.n, b.n)


def invoke(lib, kind, a, b, out):
    fn = getattr(lib, NAMES[kind])
    if kind < 3:
        return fn(a.ptr, b.ptr, a.n)
    if kind < 6:
        fn(a.ptr, b.ptr, out.ptr, a.n)
    else:
        fn(a.ptr, a.n, b.ptr, b.n, out.ptr)
    return None


def test_all(lib):
    """Bounded finite data; guards detect writes, not all out-of-bounds reads."""
    rng, checks = random.Random(2026), 0
    lengths = (0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33,
               63, 64, 65, 127, 256, 257, 1023, 1024, 1025, 4097)
    for n in lengths:
        for shift in (0, 4):
            for pattern in ("random", "zero", "cancellation"):
                av = [rng.uniform(-1, 1) for _ in range(n)]
                bv = [rng.uniform(-1, 1) for _ in range(n)]
                if pattern == "zero":
                    av = [0.0] * n
                elif pattern == "cancellation":
                    av = [1.0 if i % 2 else -1.0 for i in range(n)]
                    bv = [0.5] * n
                a, b = Buffer(n, av, shift), Buffer(n, bv, shift)
                original = (a.snapshot(), b.snapshot())
                products = [float(x) * float(y) for x, y in zip(a.view, b.view)]
                ref = math.fsum(products)
                tol = 1e-6 + 2e-6 * math.fsum(map(abs, products))
                expected = [ct.c_float(v).value for v in products]
                for kind in available(lib, 0) + available(lib, 1):
                    out = Buffer(n, shift=shift)
                    for repeats in (None, 3):
                        out.poison()
                        if repeats is None:
                            got = invoke(lib, kind, a, b, out)
                        else:
                            job = job_for(kind, a, b, out, repeats)
                            got = lib.run_batch(ct.byref(job))
                        if kind < 3:
                            if not math.isfinite(got) or abs(got - ref) > tol:
                                raise AssertionError((NAMES[kind], n, got, ref, tol))
                        elif list(out.view) != expected:
                            raise AssertionError((NAMES[kind], n, "wrong elements"))
                        out.check_guards()
                        if (a.snapshot(), b.snapshot()) != original:
                            raise AssertionError("Input changed")
                        checks += 1
    shapes = ((0, 0), (0, 8), (7, 0), (1, 1), (2, 7), (3, 8),
              (7, 9), (16, 17), (33, 65), (64, 64), (5, 257))
    for n, m in shapes:
        for shift in (0, 4):
            a = Buffer(n, (rng.uniform(-1, 1) for _ in range(n)), shift)
            b = Buffer(m, (rng.uniform(-1, 1) for _ in range(m)), shift)
            original = (a.snapshot(), b.snapshot())
            expected = [ct.c_float(x * y).value for x in a.view for y in b.view]
            for kind in available(lib, 2):
                out = Buffer(n * m, shift=shift)
                for repeats in (None, 3):
                    out.poison()
                    if repeats is None:
                        invoke(lib, kind, a, b, out)
                    else:
                        lib.run_batch(ct.byref(job_for(kind, a, b, out, repeats)))
                    if list(out.view) != expected:
                        raise AssertionError((NAMES[kind], n, m, "wrong elements"))
                    out.check_guards()
                    if (a.snapshot(), b.snapshot()) != original:
                        raise AssertionError("Input changed")
                    checks += 1
                before = out.snapshot()
                lib.run_batch(ct.byref(job_for(kind, a, b, out, 0)))
                if out.snapshot() != before:
                    raise AssertionError("Zero repeats modified output")
                checks += 1
    print(f"PASS: {checks} checks; unsupported SIMD variants are skipped.")
    return checks


def demo(lib):
    a, b, out = Buffer(3, [1, 2, 3]), Buffer(3, [4, 5, 6]), Buffer(9)
    print("a =", list(a.view), "b =", list(b.view))
    print("dot =", invoke(lib, available(lib, 0)[-1], a, b, out))
    invoke(lib, available(lib, 1)[-1], a, b, out)
    print("had =", list(out.view[:3]))
    invoke(lib, available(lib, 2)[-1], a, b, out)
    print("outer =", [list(out.view[i:i+3]) for i in range(0, 9, 3)])


def elapsed(lib, job):
    pointer = ct.byref(job)  # prepare before starting the clock
    start = time.perf_counter_ns()
    lib.run_batch(pointer)
    return time.perf_counter_ns() - start


def calibrate(lib, job, target_ns):
    job.repeats = 1
    elapsed(lib, job)  # warm-up and first touch; not retained as a sample
    while True:
        duration = elapsed(lib, job)
        if duration >= target_ns:
            return int(job.repeats)
        next_count = max(job.repeats + 1,
                         int(job.repeats * min(16, target_ns / max(duration, 1) * 1.2)))
        if next_count > 2**31:
            raise RuntimeError("Could not calibrate the batch; lower --target-ms")
        job.repeats = next_count


def environment(lib, args, checks):
    cpu = platform.processor()
    if os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return dict(timestamp=datetime.now().astimezone().isoformat(),
                os=platform.platform(), python=sys.version, bits=ct.sizeof(ct.c_void_p)*8,
                cpu=cpu, logical_cpus=os.cpu_count(), feature_mask=int(lib.cpu_features()),
                dll_sha256=hashlib.sha256((ROOT / "vector_ops.dll").read_bytes()).hexdigest(),
                clock=vars(time.get_clock_info("perf_counter")), seed=2026,
                samples=args.samples, target_ms=args.target_ms, tests_passed=checks,
                quick=args.quick, requested_cpu=args.cpu, alignment_bytes=64,
                timing="Warm repeated buffers; single native batch per Python call; "
                       "includes native call/loop overhead; empty control not subtracted")


def benchmark(lib, args, checks):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = args.out or ROOT / "results" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    rng, records, summary = random.Random(2026), [], []
    sizes = [256, 4096] if args.quick else [32, 256, 4096, 65536, 1048576]
    shapes = [(32, 64), (256, 256)] if args.quick else [
        (8, 8), (32, 64), (128, 128), (256, 256), (1024, 1024), (2048, 2048)]
    cases = [(g, n, n) for g in (0, 1) for n in sizes]
    cases += [(2, n, m) for n, m in shapes]
    (folder / "environment.json").write_text(
        json.dumps(environment(lib, args, checks), indent=2, ensure_ascii=False), encoding="utf-8")
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for group, n, m in cases:
            a = Buffer(n, (rng.uniform(-1, 1) for _ in range(n)))
            b = Buffer(m, (rng.uniform(-1, 1) for _ in range(m)))
            out = Buffer(n * m if group == 2 else n)
            kinds = available(lib, group) + [9 + group]
            jobs = {k: job_for(k, a, b, out) for k in kinds}
            samples = {k: [] for k in kinds}
            for job in jobs.values():
                calibrate(lib, job, args.target_ms * 1e6)
            for sample in range(args.samples):
                order = kinds.copy()
                rng.shuffle(order)
                for kind in order:
                    job = jobs[kind]
                    duration = elapsed(lib, job)
                    value = duration / job.repeats
                    samples[kind].append(value)
                    name = NAMES[kind] if kind < 9 else "empty_control"
                    records.append(dict(operation=("dot", "had", "outer")[group],
                                        n=n, m=m, variant=name, sample=sample + 1,
                                        repeats=int(job.repeats), total_ns=duration,
                                        ns_per_call=value))
            base = statistics.median(samples[group * 3])
            empty = statistics.median(samples[9 + group])
            for kind in kinds:
                vals = samples[kind]
                med = statistics.median(vals)
                name = NAMES[kind] if kind < 9 else "empty_control"
                row = dict(operation=("dot", "had", "outer")[group], n=n, m=m,
                           variant=name, median_ns=med, min_ns=min(vals), max_ns=max(vals),
                           speedup_vs_scalar=base / med if kind < 9 else "",
                           empty_fraction=empty / med if kind < 9 else "")
                summary.append(row)
                print(f"{row['operation']:5} {n:8} x {m:<8} {name:14} {med:12.2f} ns")
    finally:
        if gc_was_enabled:
            gc.enable()
    for filename, rows in (("raw_samples.csv", records), ("summary.csv", summary)):
        with (folder / filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Local Windows measurement report", "",
             "Baseline: scalar assembly. Python is the driver, not a speed baseline.",
             "Empty control is diagnostic and is NOT subtracted. Large empty_fraction "
             "means call/loop overhead matters. Reused buffers are warm, not cold RAM.", "",
             "| Operation | n | m | Variant | Median, ns | Speedup |", "|---|---:|---:|---|---:|---:|"]
    for row in summary:
        speed = row["speedup_vs_scalar"]
        speed = f"{speed:.3f}" if isinstance(speed, float) else "n/a"
        lines.append(f"| {row['operation']} | {row['n']} | {row['m']} | {row['variant']} | "
                     f"{row['median_ns']:.2f} | {speed} |")
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Results:", folder)


def pin_thread(cpu):
    if cpu is None:
        return lambda: None
    if not 0 <= cpu < min(os.cpu_count() or 1, 64):
        raise ValueError("--cpu must be a logical CPU index in the current processor group (0..63)")
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentThread.argtypes, kernel.GetCurrentThread.restype = [], ct.c_void_p
    kernel.SetThreadAffinityMask.argtypes = [ct.c_void_p, ct.c_size_t]
    kernel.SetThreadAffinityMask.restype = ct.c_size_t
    thread = kernel.GetCurrentThread()
    old = kernel.SetThreadAffinityMask(thread, 1 << cpu)
    if not old:
        raise ct.WinError(ct.get_last_error())
    return lambda: kernel.SetThreadAffinityMask(thread, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("test", "demo", "bench"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-ms", type=float, default=30.0)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--out", type=Path, help="New output directory; must not exist")
    args = parser.parse_args()
    if os.name != "nt" or ct.sizeof(ct.c_void_p) != 8:
        parser.error("Use 64-bit CPython on 64-bit Windows (target: Windows 10 x64)")
    if args.samples < 3 or not 1 <= args.target_ms <= 1000:
        parser.error("Use --samples >= 3 and 1 <= --target-ms <= 1000")
    if ct.sizeof(Batch) != 56 or ct.sizeof(ct.c_float) != 4:
        raise RuntimeError("Unexpected ABI layout")
    lib = configure(ct.CDLL(str(ROOT / "vector_ops.dll")))
    flags = int(lib.cpu_features())
    print(f"AVX usable={bool(flags & 1)}, FMA usable={bool(flags & 2)}, AVX2={bool(flags & 4)}")
    restore = pin_thread(args.cpu)
    try:
        if args.mode == "demo":
            demo(lib)
        else:
            checks = test_all(lib)
            if args.mode == "bench":
                benchmark(lib, args, checks)
    finally:
        restore()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
