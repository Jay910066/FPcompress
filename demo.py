#!/usr/bin/env python3
"""
FPcompress Advanced Demo Script
===============================
An interactive, high-performance demo of the FPcompress framework.
Features:
- Auto-probes CPU (g++) and GPU (nvcc) compilers.
- Auto-compiles CPU variants, GPU variants (if CUDA is present), and Legacy FPC (2006).
- Generates high-entropy synthetic double & single-precision scientific datasets.
- Benchmarks compression ratios, encoding speeds, and decoding speeds.
- Verifies exact binary-level lossless data integrity (PASS/FAIL).
- Outputs a gorgeous, detailed results table comparing CPU, GPU, and Legacy algorithms.
- Automatically generates and saves a premium performance scatter plot (Throughput vs. Ratio) using Matplotlib.
"""

import struct
import os
import subprocess
import sys
import math
import time
import random
import shutil

# ─── Configuration ───────────────────────────────────────────────────────────
DEMO_DIR = "demo_data"
DP_FILE = os.path.join(DEMO_DIR, "test_double.bin")   # 64-bit doubles
SP_FILE = os.path.join(DEMO_DIR, "test_single.bin")   # 32-bit floats
NUM_ELEMENTS = 1024 * 1024  # 1M elements (8MB for DP, 4MB for SP)

# ─── Color Palette ───────────────────────────────────────────────────────────
class Color:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"

def run_cmd(cmd, cwd=None):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=False, cwd=cwd)
        stdout_str = ""
        stderr_str = ""
        if res.stdout:
            try:
                stdout_str = res.stdout.decode("utf-8", errors="replace")
            except Exception:
                stdout_str = res.stdout.decode("cp950", errors="replace")
        if res.stderr:
            try:
                stderr_str = res.stderr.decode("utf-8", errors="replace")
            except Exception:
                stderr_str = res.stderr.decode("cp950", errors="replace")
        return res.returncode == 0, stdout_str, stderr_str
    except Exception as e:
        return False, "", str(e)

# ─── Step 1: Detect Environment & Auto-Compile ───────────────────────────────
def check_and_compile():
    print(f"{Color.CYAN}=== Environment Detection & Auto-Compilation ==={Color.END}")
    
    # 0. Active Windows CUDA & MSVC Autodetection
    if os.name == "nt":
        # If nvcc is not directly callable, search the default NVIDIA Computing Toolkit directories
        has_nvcc, _, _ = run_cmd("nvcc --version")
        if not has_nvcc:
            base_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
            if os.path.exists(base_dir):
                found_versions = []
                for entry in os.listdir(base_dir):
                    entry_path = os.path.join(base_dir, entry)
                    bin_dir = os.path.join(entry_path, "bin")
                    if os.path.isdir(entry_path) and os.path.exists(os.path.join(bin_dir, "nvcc.exe")):
                        found_versions.append((entry, bin_dir))
                if found_versions:
                    # Sort version strings semantically (e.g. v12.4 > v12.1 > v11.8)
                    def parse_ver(v_name):
                        num_part = v_name[1:] if v_name.lower().startswith("v") else v_name
                        try:
                            return tuple(int(x) for x in num_part.split("."))
                        except ValueError:
                            return (0,)
                    found_versions.sort(key=lambda x: parse_ver(x[0]))
                    latest_ver, bin_dir = found_versions[-1]
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
                    print(f"  • {Color.YELLOW}[INFO] Autodetected installed CUDA {latest_ver}. Dynamic PATH injected: {bin_dir}{Color.END}")

        # Inject MSVC host compiler path so nvcc can locate cl.exe on Windows
        has_cl, _, _ = run_cmd("cl")
        if not has_cl:
            msvc_bases = [
                r"C:\Program Files\Microsoft Visual Studio",
                r"C:\Program Files (x86)\Microsoft Visual Studio"
            ]
            cl_dirs = []
            for base in msvc_bases:
                if os.path.exists(base):
                    for root, dirs, files in os.walk(base):
                        if "cl.exe" in files:
                            # Prefer 64-bit native tools
                            if "Hostx64\\x64" in root or "HostX64\\x64" in root:
                                cl_dirs.insert(0, root)
                            else:
                                cl_dirs.append(root)
            if cl_dirs:
                selected_cl_dir = cl_dirs[0]
                os.environ["PATH"] = selected_cl_dir + os.pathsep + os.environ["PATH"]
                print(f"  • {Color.YELLOW}[INFO] Autodetected MSVC (cl.exe). Dynamic PATH injected: {selected_cl_dir}{Color.END}")

    # 1. Check for standard CPU compiler (g++ / gcc)
    has_gxx, out, _ = run_cmd("g++ --version")
    if not has_gxx:
        print(f"{Color.RED}[ERROR] g++ not found. Please install a C++17 compiler (GCC/MSYS2) and add it to PATH.{Color.END}")
        sys.exit(1)
    gxx_ver = out.split("\n")[0] if out else "Unknown Version"
    print(f"  • C++ Compiler found: {Color.GREEN}{gxx_ver}{Color.END}")

    # 2. Check for GPU compiler (nvcc)
    has_nvcc, out, _ = run_cmd("nvcc --version")
    nvcc_ver = "Not Available"
    if has_nvcc:
        nvcc_ver = out.split("\n")[3] if len(out.split("\n")) > 3 else "CUDA Present"
        print(f"  • CUDA GPU Compiler found: {Color.GREEN}{nvcc_ver}{Color.END}")
    else:
        print(f"  • CUDA GPU Compiler: {Color.YELLOW}Not Found (GPU benchmarks will be skipped gracefully){Color.END}")

    # 3. Compile Legacy FPC (FPC 2006)
    print("\n  • Compiling Legacy FPC (FPC 2006)...", end="", flush=True)
    fpc_src = "fpc.c"
    fpc_exe = "fpc.exe" if os.name == "nt" else "fpc"
    if not os.path.exists(fpc_src):
        print(f" {Color.RED}FAIL (fpc.c source file missing!){Color.END}")
        sys.exit(1)
    
    # Attempt to compile with gcc, fallback to g++
    compiled, _, _ = run_cmd(f"gcc -O3 {fpc_src} -o {fpc_exe}")
    if not compiled:
        compiled, _, _ = run_cmd(f"g++ -O3 {fpc_src} -o {fpc_exe}")
        
    if compiled:
        print(f" {Color.GREEN}SUCCESS -> {fpc_exe}{Color.END}")
    else:
        print(f" {Color.RED}FAIL (Could not compile fpc.c){Color.END}")
        sys.exit(1)

    # 4. Compile CPU and GPU Compressors
    print("  • Checking FPcompress executables...")
    
    cpu_targets = [
        ("double_src", "speed-cpu-compress", "speed-compressor-double.cpp"),
        ("double_src", "speed-cpu-decompress", "speed-decompressor-double.cpp"),
        ("double_src", "ratio-cpu-compress", "ratio-compressor-double.cpp"),
        ("double_src", "ratio-cpu-decompress", "ratio-decompressor-double.cpp"),
        ("single_src", "speed-cpu-compress", "speed-compressor-single.cpp"),
        ("single_src", "speed-cpu-decompress", "speed-decompressor-single.cpp"),
        ("single_src", "ratio-cpu-compress", "ratio-compressor-single.cpp"),
        ("single_src", "ratio-cpu-decompress", "ratio-decompressor-single.cpp"),
    ]

    gpu_targets = [
        ("double_src", "speed-gpu-compress", "speed-compressor-double.cu"),
        ("double_src", "speed-gpu-decompress", "speed-decompressor-double.cu"),
        ("double_src", "ratio-gpu-compress", "ratio-compressor-double.cu"),
        ("double_src", "ratio-gpu-decompress", "ratio-decompressor-double.cu"),
        ("single_src", "speed-gpu-compress", "speed-compressor-single.cu"),
        ("single_src", "speed-gpu-decompress", "speed-decompressor-single.cu"),
        ("single_src", "ratio-gpu-compress", "ratio-compressor-single.cu"),
        ("single_src", "ratio-gpu-decompress", "ratio-decompressor-single.cu"),
    ]

    # Compile CPU targets if missing
    for folder, name, src in cpu_targets:
        exe_path = os.path.join(folder, f"{name}.exe" if os.name == "nt" else name)
        if not os.path.exists(exe_path):
            print(f"    Compiling CPU {name}...", end="", flush=True)
            ok, _, err = run_cmd(f"g++ -O3 -march=native -fopenmp -I. -std=c++17 -o {exe_path} {os.path.join(folder, src)}")
            if ok:
                print(f" {Color.GREEN}OK{Color.END}")
            else:
                print(f" {Color.RED}FAIL: {err}{Color.END}")

    # Compile GPU targets if nvcc is available and executables are missing
    gpu_compiled_successfully = True if has_nvcc else False
    if has_nvcc:
        for folder, name, src in gpu_targets:
            exe_path = os.path.join(folder, f"{name}.exe" if os.name == "nt" else name)
            if not os.path.exists(exe_path):
                print(f"    Compiling GPU {name}...", end="", flush=True)
                
                # Handle platform-specific host compiler optimizations
                if os.name == "nt":
                    xcompiler_flags = "/O2 /openmp /Zc:preprocessor"
                else:
                    xcompiler_flags = "-O3 -march=native -fopenmp"
                
                # Use architecture sm_75 for Turing GPUs, if compilation fails, try fallback sm_80
                ok, _, err = run_cmd(f"nvcc -O3 -arch=sm_75 -fmad=false -Xcompiler \"{xcompiler_flags}\" -I. -std=c++17 -o {exe_path} {os.path.join(folder, src)}")
                if ok:
                    print(f" {Color.GREEN}OK{Color.END}")
                else:
                    print(f" {Color.YELLOW}FAIL (Try fallback arch sm_80)...{Color.END}", end="")
                    ok, _, err = run_cmd(f"nvcc -O3 -arch=sm_80 -fmad=false -Xcompiler \"{xcompiler_flags}\" -I. -std=c++17 -o {exe_path} {os.path.join(folder, src)}")
                    if ok:
                        print(f" {Color.GREEN}OK{Color.END}")
                    else:
                        print(f" {Color.RED}FAIL: GPU target will be disabled.{Color.END}")
                        gpu_compiled_successfully = False
                        break

    print()
    return has_nvcc and gpu_compiled_successfully, fpc_exe


# ─── Step 2: Generate Test Data ─────────────────────────────────────────────
def generate_test_data():
    os.makedirs(DEMO_DIR, exist_ok=True)
    print(f"{Color.CYAN}=== Step 1: Generating High-Entropy Scientific Datasets ==={Color.END}")
    random.seed(42)

    # 1. Double Precision Data
    print(f"  • Generating {NUM_ELEMENTS:,} DP (64-bit) scientific values ({NUM_ELEMENTS * 8 / 1024 / 1024:.1f} MB)...")
    doubles = []
    for i in range(NUM_ELEMENTS):
        # Mix of patterns: sine wave + small noise + periodic high-energy spikes
        base = math.sin(i * 0.001) * 120.0 + math.cos(i * 0.0037) * 45.0
        noise = random.gauss(0, 0.015)
        spike = random.gauss(0, 150.0) if i % 15000 == 0 else 0.0
        doubles.append(base + noise + spike)

    with open(DP_FILE, "wb") as f:
        f.write(struct.pack(f"<{NUM_ELEMENTS}d", *doubles))
    dp_size = os.path.getsize(DP_FILE)
    print(f"    -> {Color.GREEN}DP File Created:{Color.END} {DP_FILE} ({dp_size:,} bytes)")

    # 2. Single Precision Data
    print(f"  • Generating {NUM_ELEMENTS:,} SP (32-bit) scientific values ({NUM_ELEMENTS * 4 / 1024 / 1024:.1f} MB)...")
    floats = [float(x) for x in doubles]  # Cast to single-precision float
    with open(SP_FILE, "wb") as f:
        f.write(struct.pack(f"<{NUM_ELEMENTS}f", *floats))
    sp_size = os.path.getsize(SP_FILE)
    print(f"    -> {Color.GREEN}SP File Created:{Color.END} {SP_FILE} ({sp_size:,} bytes)")
    print()

    return dp_size, sp_size


# ─── Step 3: Run FPcompress Benchmarks ──────────────────────────────────────
def run_fpcompress_bench(exe, input_file, compressed_file, decompressed_file, mode="compress"):
    """Runs FPcompress executable and parses throughput and ratio."""
    ext = ".exe" if os.name == "nt" else ""
    exe_path = exe if exe.endswith(ext) else exe + ext
    
    if mode == "compress":
        cmd = [exe_path, input_file, compressed_file, "y"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if res.returncode != 0:
                return None, None, res.stderr
            
            # Parse output
            ratio = None
            throughput = None
            for line in res.stdout.split("\n"):
                if "ratio:" in line:
                    parts = line.split()
                    for p in parts:
                        if p.endswith("x"):
                            try:
                                ratio = float(p[:-1])
                            except:
                                pass
                if "encoding throughput:" in line:
                    parts = line.split()
                    for p in parts:
                        try:
                            throughput = float(p)
                        except:
                            pass
            return ratio, throughput, None
        except Exception as e:
            return None, None, str(e)
    else:
        cmd = [exe_path, input_file, decompressed_file, "y"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if res.returncode != 0:
                return None, res.stderr
            
            throughput = None
            for line in res.stdout.split("\n"):
                if "decoding throughput:" in line:
                    parts = line.split()
                    for p in parts:
                        try:
                            throughput = float(p)
                        except:
                            pass
            return throughput, None
        except Exception as e:
            return None, str(e)


# ─── Step 4: Run Legacy FPC Benchmarks ───────────────────────────────────────
def run_legacy_fpc_bench(fpc_exe, input_file, compressed_file, decompressed_file, orig_size, mode="compress"):
    """Runs legacy FPC using elegant Python file I/O redirection to bypass shell differences."""
    exe_path = fpc_exe if (fpc_exe.endswith(".exe") or os.name != "nt") else fpc_exe + ".exe"
    
    if mode == "compress":
        try:
            # FPC table size log2 parameter: 20 is typical for scientific data
            t_start = time.perf_counter()
            with open(input_file, "rb") as fin, open(compressed_file, "wb") as fout:
                res = subprocess.run([exe_path, "20"], stdin=fin, stdout=fout, capture_output=False, timeout=90)
            t_end = time.perf_counter()
            
            if res.returncode != 0:
                return None, None, f"FPC exit code {res.returncode}"
            
            duration = t_end - t_start
            compressed_size = os.path.getsize(compressed_file)
            ratio = orig_size / compressed_size if compressed_size > 0 else 1.0
            throughput = (orig_size / (1024 * 1024 * 1024)) / duration if duration > 0 else 0.0
            return ratio, throughput, None
        except Exception as e:
            return None, None, str(e)
    else:
        try:
            t_start = time.perf_counter()
            with open(input_file, "rb") as fin, open(decompressed_file, "wb") as fout:
                res = subprocess.run([exe_path], stdin=fin, stdout=fout, capture_output=False, timeout=90)
            t_end = time.perf_counter()
            
            if res.returncode != 0:
                return None, f"FPC decompress exit code {res.returncode}"
            
            duration = t_end - t_start
            throughput = (orig_size / (1024 * 1024 * 1024)) / duration if duration > 0 else 0.0
            return throughput, None
        except Exception as e:
            return None, str(e)


# ─── Helper: File Roundtrip Verification ─────────────────────────────────────
def verify_integrity(original, decompressed):
    if not os.path.exists(decompressed):
        return False
    with open(original, "rb") as f1, open(decompressed, "rb") as f2:
        return f1.read() == f2.read()


# ─── Step 5: Matplotlib Interactive Plotting ──────────────────────────────────
def plot_results(results):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"\n{Color.YELLOW}[INFO] Matplotlib/Numpy not found. Skipping scatter plot generation.{Color.END}")
        print("  -> To enable visual charts, install dependencies: `conda install matplotlib numpy` or `pip install matplotlib numpy`\n")
        return

    print(f"\n{Color.CYAN}=== Step 4: Generating Scientific Scatter Plot ==={Color.END}")
    
    # Filter double-precision results for plotting
    dp_results = [r for r in results if r[1] == "DP" and r[4] is not None]
    sp_results = [r for r in results if r[1] == "SP" and r[4] is not None]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=120)
    fig.suptitle("FPcompress Framework: Throughput vs. Compression Ratio (1M Elements)", fontsize=16, fontweight='bold', color='#2c3e50')

    # Color themes
    colors = {
        "CPU": "#3498db",  # Elegant Blue
        "GPU": "#e74c3c",  # Vibrant Red
        "FPC": "#7f8c8d"   # Classic Grey
    }
    
    markers = {
        "speed": "o",   # Circles for speed optimized
        "ratio": "D",   # Diamonds for ratio optimized
        "FPC": "X"      # Cross for legacy
    }

    # Plot helper
    def draw_chart(ax, data_list, title):
        ax.set_title(title, fontsize=13, fontweight='bold', color='#34495e')
        ax.grid(True, linestyle='--', alpha=0.5, color='#bdc3c7')
        ax.set_facecolor('#fdfefe')
        
        for name, prec, orig, comp, ratio, comp_tp, decomp_tp, match in data_list:
            # Determine group
            group = "CPU"
            if "GPU" in name:
                group = "GPU"
            elif "FPC" in name:
                group = "FPC"
                
            style = "ratio" if "ratio" in name.lower() else "speed"
            if group == "FPC":
                style = "FPC"
                
            color = colors[group]
            marker = markers[style]
            
            # Plot Compression point if available
            if comp_tp is not None:
                ax.scatter(comp_tp, ratio, color=color, marker=marker, s=150, edgecolor='black', zorder=5,
                           label=f"{name} (Comp)")
                ax.annotate(name.split()[0], (comp_tp, ratio), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='semibold')
                
            # Plot Decompression point (dashed circle) if available
            if decomp_tp is not None:
                ax.scatter(decomp_tp, ratio, color=color, marker=marker, s=150, facecolors='none', edgecolor=color, 
                           linestyle='--', linewidth=2, zorder=4, label=f"{name} (Decomp)")
                ax.annotate(f"Dec: {decomp_tp:.1f}G", (decomp_tp, ratio), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, alpha=0.7)

        ax.set_xlabel("Throughput (GB/s)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Compression Ratio (higher = better)", fontsize=11, fontweight='bold')
        
        # Remove duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', frameon=True, shadow=True)

    # Left plot: DP
    if dp_results:
        draw_chart(axes[0], dp_results, "Double-Precision (64-bit) Datasets")
    
    # Right plot: SP
    if sp_results:
        draw_chart(axes[1], sp_results, "Single-Precision (32-bit) Datasets")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart_output = "demo_comparison_chart.png"
    plt.savefig(chart_output)
    plt.close()
    
    print(f"  {Color.GREEN}SUCCESS -> High-quality chart saved as:{Color.END} {chart_output}")
    print()


# ─── Main Orchestrator ───────────────────────────────────────────────────────
def main():
    print("=" * 90)
    print(f"  {Color.BOLD}{Color.CYAN}FPcompress Premium Benchmark & Analytics Suite{Color.END}")
    print("=" * 90)
    print()

    has_gpu, fpc_exe = check_and_compile()
    dp_size, sp_size = generate_test_data()

    # Define benchmark list
    # Format: (display_name, precision, input_file, original_size, compress_exe, decompress_exe, is_legacy_fpc)
    benchmarks = [
        # ─── Double Precision (64-bit) ───
        ("DPspeed (CPU)", "DP", DP_FILE, dp_size, "double_src/speed-cpu-compress", "double_src/speed-cpu-decompress", False),
        ("DPratio (CPU)", "DP", DP_FILE, dp_size, "double_src/ratio-cpu-compress", "double_src/ratio-cpu-decompress", False),
        ("FPC (Legacy CPU)", "DP", DP_FILE, dp_size, fpc_exe, fpc_exe, True),
    ]
    
    if has_gpu:
        benchmarks.append(("DPspeed (GPU)", "DP", DP_FILE, dp_size, "double_src/speed-gpu-compress", "double_src/speed-gpu-decompress", False))
        benchmarks.append(("DPratio (GPU)", "DP", DP_FILE, dp_size, "double_src/ratio-gpu-compress", "double_src/ratio-gpu-decompress", False))

    # Add Single Precision benchmarks
    benchmarks.extend([
        # ─── Single Precision (32-bit) ───
        ("SPspeed (CPU)", "SP", SP_FILE, sp_size, "single_src/speed-cpu-compress", "single_src/speed-cpu-decompress", False),
        ("SPratio (CPU)", "SP", SP_FILE, sp_size, "single_src/ratio-cpu-compress", "single_src/ratio-cpu-decompress", False),
    ])

    if has_gpu:
        benchmarks.append(("SPspeed (GPU)", "SP", SP_FILE, sp_size, "single_src/speed-gpu-compress", "single_src/speed-gpu-decompress", False))
        benchmarks.append(("SPratio (GPU)", "SP", SP_FILE, sp_size, "single_src/ratio-gpu-compress", "single_src/ratio-gpu-decompress", False))

    results = []

    print(f"{Color.CYAN}=== Step 2: Executing Performance Benchmarks ==={Color.END}")
    print("-" * 90)

    for label, prec, input_file, orig_size, comp_exe, decomp_exe, is_fpc in benchmarks:
        compressed = input_file + ".comp"
        decompressed = input_file + ".decomp"

        print(f"  Running {Color.BOLD}{label}{Color.END}...")
        
        # 1. Compress
        print(f"    Compressing...", end="", flush=True)
        if is_fpc:
            ratio, comp_tp, err = run_legacy_fpc_bench(comp_exe, input_file, compressed, decompressed, orig_size, "compress")
        else:
            ratio, comp_tp, err = run_fpcompress_bench(comp_exe, input_file, compressed, decompressed, "compress")

        if err:
            print(f" {Color.RED}[ERROR]: {err}{Color.END}")
            results.append((label, prec, orig_size, None, None, None, None, False))
            continue
        comp_size = os.path.getsize(compressed) if os.path.exists(compressed) else 0
        print(f" done ({comp_size / 1024 / 1024:.2f} MB, {ratio:.3f}x)")

        # 2. Decompress
        print(f"    Decompressing...", end="", flush=True)
        if is_fpc:
            decomp_tp, err = run_legacy_fpc_bench(decomp_exe, compressed, compressed, decompressed, orig_size, "decompress")
        else:
            decomp_tp, err = run_fpcompress_bench(decomp_exe, compressed, None, decompressed, "decompress")

        if err:
            print(f" {Color.RED}[ERROR]: {err}{Color.END}")
            results.append((label, prec, orig_size, comp_size, ratio, comp_tp, None, False))
            continue
        print(f" done")

        # 3. Verify Exact Lossless Reconstruction
        print(f"    Verifying Lossless Reconstruction...", end="", flush=True)
        match = verify_integrity(input_file, decompressed)
        if match:
            print(f" {Color.GREEN}PASS{Color.END}")
        else:
            print(f" {Color.RED}FAIL (Data corruption detected!){Color.END}")

        results.append((label, prec, orig_size, comp_size, ratio, comp_tp, decomp_tp, match))

        # Clean up temporary files
        for f in [compressed, decompressed]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        print()

    # ─── Step 6: Beautiful Results Summary Table ─────────────────────────────────
    print(f"{Color.CYAN}=== Step 3: Performance Summary Table ==={Color.END}")
    print("=" * 110)
    print(f"  {Color.BOLD}{'Algorithm':<18} {'Precision':^10} {'Original':>12} {'Compressed':>12} {'Ratio':>10} {'Comp (GB/s)':>14} {'Dec (GB/s)':>14} {'Verify':^8}{Color.END}")
    print("-" * 110)

    for label, prec, orig, comp, ratio, comp_tp, decomp_tp, match in results:
        orig_str = f"{orig / 1024 / 1024:.2f} MB" if orig else "N/A"
        comp_str = f"{comp / 1024 / 1024:.2f} MB" if comp else "N/A"
        ratio_str = f"{ratio:.3f}x" if ratio else "N/A"
        comp_tp_str = f"{comp_tp:.3f}" if comp_tp else "N/A"
        decomp_tp_str = f"{decomp_tp:.3f}" if decomp_tp else "N/A"
        
        if match:
            verify_str = f"{Color.GREEN}PASS{Color.END}"
        else:
            verify_str = f"{Color.RED}FAIL{Color.END}"

        print(f"  {label:<18} {prec:^10} {orig_str:>12} {comp_str:>12} {ratio_str:>10} {comp_tp_str:>14} {decomp_tp_str:>14} {verify_str:^8}")

    print("=" * 110)
    print(f"  {Color.BOLD}Throughput Legend:{Color.END} GB/s = GigaBytes processed per second (higher is faster).")
    print(f"  {Color.BOLD}Ratio Legend:{Color.END} Higher ratio represents better compression. 2.0x means compressed file is 50% of original.")
    print()

    # Generate Matplotlib chart
    plot_results(results)

    # Clean up temp directories/artifacts
    if os.path.exists("test_header.cpp"):
        os.remove("test_header.cpp")
    if os.path.exists("combined-double.txt"):
        os.remove("combined-double.txt")


if __name__ == "__main__":
    main()
