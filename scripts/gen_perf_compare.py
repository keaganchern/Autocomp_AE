#!/usr/bin/env python3
"""Generate a self-contained Zephyr RVV benchmark (benchmarks/perf_compare)
comparing, per kernel, three implementations over a batch-size sweep:

  Upstream — XNNPACK's hand-tuned RVV microkernel (repo-local XNNPACK checkout)
  Sol      — the pre-autocomp kernel in sols/cgo-kernels
  Autocomp — best_candidate_so_far.txt from the prob's autocomp output run

Style mirrors zephyr-chipyard-sw/samples/rvv_bench/src/main.c:
NUM_WARMUP untimed iterations per (kernel, impl, batch) cell, then
NUM_ITERATIONS timed via rdcycle/rdinstret, table + CSV output, no XNNPACK
link — every variant's source is embedded and prefix-renamed.

Usage:
    python3 scripts/gen_perf_compare.py          # writes $PERF_COMPARE_APP/src/main.c

Paths (environment variables, defaults relative to this repo):
    AUTOCOMP_REPO      this repository (sols/, output/, scripts/neon_inlined/)
    XNNPACK_SRC        $AUTOCOMP_REPO/XNNPACK/src          (Upstream and Neon2RVV lanes)
    AUTOCOMP_OUTPUT    $AUTOCOMP_REPO/output                (search runs: Autocomp lane)
    PERF_COMPARE_APP   $AUTOCOMP_REPO/benchmarks/perf_compare  (the Zephyr app: neon2rvv.h,
                       CMakeLists.txt, prj.conf; see its README)
"""

import glob
import os
import re
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("AUTOCOMP_REPO", _HERE)
XNN_SRC = os.environ.get("XNNPACK_SRC", f"{REPO}/XNNPACK/src")
SOLS = f"{REPO}/sols/cgo-kernels"
OUTPUT = os.environ.get("AUTOCOMP_OUTPUT", f"{REPO}/output")
APP = os.environ.get("PERF_COMPARE_APP", f"{REPO}/benchmarks/perf_compare")

# ---------------------------------------------------------------- extraction

_FUNC_DECL_RE = re.compile(
    r'(?:static\s+|inline\s+|__attribute__\S+\s+)*\w[\w\s*]*\s+(\w+)\s*\([^)]*\)[^{;]*\{',
    re.DOTALL,
)


def blank_comments(s: str) -> str:
    def white(m):
        return ''.join('\n' if c == '\n' else ' ' for c in m.group(0))
    s = re.sub(r'/\*.*?\*/', white, s, flags=re.DOTALL)
    s = re.sub(r'//[^\n]*', white, s)
    # Preprocessor lines (incl. backslash continuations): a function-like
    # #define can false-match the function-decl regex and its body braces
    # would corrupt the span scan.
    s = re.sub(r'(?m)^[ \t]*#(?:\\\n|[^\n])*', white, s)
    return s


def function_spans(code: str):
    """Yield (name, start, brace_close) over a comment-blanked scan."""
    scan = blank_comments(code)
    pos = 0
    while pos < len(scan):
        m = _FUNC_DECL_RE.search(scan, pos)
        if not m:
            return
        brace_open = m.end() - 1
        depth, i = 1, brace_open + 1
        while i < len(scan):
            if scan[i] == '{':
                depth += 1
            elif scan[i] == '}':
                depth -= 1
                if depth == 0:
                    yield (m.group(1), m.start(), i)
                    pos = i + 1
                    break
            i += 1
        else:
            return


def top_level_defines(text: str, spans):
    """(name, block) for every #define outside any function span. Candidates
    may carry their own helper macros (incl. multi-line function-like ones);
    these are re-emitted renamed so they can't collide across variants or
    with the bench header."""
    out = []
    for m in re.finditer(r'(?m)^[ \t]*#[ \t]*define[ \t]+(\w+)(?:\\\n|[^\n])*',
                         text):
        if any(s <= m.start() <= e for _, s, e in spans):
            continue
        out.append((m.group(1), m.group(0)))
    return out


def prep_variant(text: str, prefix: str):
    """Extract every function definition and top-level macro, prefix-rename
    all defined names (definitions AND intra-variant uses), and return
    (code, main_name, pack_name). main = the test_rvv function when present
    (candidates may define helpers above it), else the first function not
    named pack_weights; pack = the pack_weights function when present."""
    spans = list(function_spans(text))
    funcs = [(n, text[s:e + 1]) for n, s, e in spans]
    if not funcs:
        return None, None, None
    defines = top_level_defines(text, spans)
    names = [n for n, _ in funcs] + [n for n, _ in defines]
    pieces = [block for _, block in defines]
    pieces.extend(body for _, body in funcs)
    code = "\n\n".join(pieces)
    for n in set(names):
        code = re.sub(rf'\b{re.escape(n)}\b', f'{prefix}_{n}', code)
    fnames = [n for n, _ in funcs]
    main = next((n for n in fnames if n == 'test_rvv'), None) \
        or next((n for n in fnames if n != 'pack_weights'), None)
    pack = 'pack_weights' if 'pack_weights' in fnames else None
    return code, (f'{prefix}_{main}' if main else None), \
        (f'{prefix}_{pack}' if pack else None)


def load_upstream(rel):
    if rel is None:
        return None
    p = rel if rel.startswith('/') else f"{XNN_SRC}/{rel}"
    return open(p).read() if os.path.exists(p) else None


def load_sol(pid):
    pinned = _pinned(pid, SEL.sol_files)
    if pinned is not None:
        return pinned
    hits = sorted(glob.glob(f"{REPO}/sols/{SEL.sol_prob_type}/{pid}_*.c"))
    return open(hits[0]).read() if hits else None


def _pinned(pid, files):
    """A <label or prob id>=<path> override from the command line, if any."""
    for key in (str(pid), *(row[1] for row in KERNELS if row[0] == pid)):
        if key in files:
            return open(files[key]).read()
    return None


def load_autocomp(pid):
    # --autocomp-file <kernel>=<path> pins the Autocomp lane to one kernel file and --autocomp-run
    # to one search run; otherwise the newest run of the prob in --autocomp-prob-type.
    pinned = _pinned(pid, SEL.autocomp_files)
    if pinned is not None:
        return pinned
    if SEL.autocomp_run:
        hits = glob.glob(f"{SEL.autocomp_run}/best_candidate_so_far.*")
    else:
        hits = glob.glob(f"{OUTPUT}/built:saturn-rvv_{SEL.autocomp_prob_type}_{pid}_*/best_candidate_so_far.txt")
    if not hits:
        return None
    best = max(hits, key=os.path.getmtime)
    t = open(best).read()
    return t if t.strip() else None


# ---------------------------------------------------------------- selection
# Which kernels, implementations (lanes) and sections main.c runs. Nothing selected = everything,
# gated by the RUN_* defines below exactly as before.

LANES = ("Upstream", "Sol", "Autocomp", "Neon2RVV")          # slot order of fns[IMPL_COUNT]
SECTIONS = ("main", "neon2rvv", "autovec")        # "<kernel>", "<kernel>-<section>"


class _Selection:
    kernels = set()          # labels and/or prob ids; empty = all
    lanes = set(LANES)
    sections = set(SECTIONS)
    active = False           # any of --kernels/--impls/--sections given
    autocomp_files = {}
    sol_files = {}           # --sol-file <kernel>=<path>: the Sol lane of that kernel
    simde = False            # --simde: the Neon lane through SIMDe instead of neon2rvv.h
    autocomp_run = ""
    autocomp_prob_type = "cgo-kernels"
    sol_prob_type = "cgo-kernels"
    lane_nr = {}             # --lane-nr <kernel>=<lane>:<expr>,...: weight tile width of a lane (GEMM family)


SEL = _Selection()


def parse_selection(argv):
    import argparse
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernels", default=env("PERF_COMPARE_KERNELS", ""),
                    help="comma-separated kernel labels or prob ids, e.g. f32-vadd,7 (default: all)")
    ap.add_argument("--impls", default=env("PERF_COMPARE_IMPLS", ""),
                    help=f"comma-separated lanes from {','.join(LANES)} (default: all)")
    ap.add_argument("--sections", default=env("PERF_COMPARE_SECTIONS", ""),
                    help=f"comma-separated sections from {','.join(SECTIONS)} (default: the RUN_* defines; "
                         "with --kernels/--impls: all sections of the selected kernels)")
    ap.add_argument("--autocomp-file", action="append", default=[], metavar="KERNEL=PATH",
                    help="time PATH in the Autocomp lane of KERNEL (label or prob id); repeatable")
    ap.add_argument("--sol-file", action="append", default=[], metavar="KERNEL=PATH",
                    help="time PATH in the Sol lane of KERNEL (label or prob id); repeatable")
    ap.add_argument("--simde", action="store_true",
                    help="compile the Neon lane's kernels through SIMDe (simde/arm/neon.h, native aliases) "
                         "instead of neon2rvv.h; its sections and lane are labeled SIMDe. Build with "
                         "-DSIMDE_INCLUDE=<simde checkout> (the app adds -mrvv-vector-bits=zvl)")
    ap.add_argument("--autocomp-run", default=env("PERF_COMPARE_AUTOCOMP_RUN", ""),
                    help="Autocomp lane from this search output directory (one kernel)")
    ap.add_argument("--autocomp-prob-type", default=env("PERF_COMPARE_AUTOCOMP_PROB_TYPE", "cgo-kernels"),
                    help="prob type of the search runs searched under output/ (default cgo-kernels)")
    ap.add_argument("--sol-prob-type", default=env("PERF_COMPARE_SOL_PROB_TYPE", "cgo-kernels"),
                    help="sols/<prob type> holding the Sol lane (default cgo-kernels)")
    ap.add_argument("--lane-nr", action="append", default=[], metavar="KERNEL=LANE:EXPR[,LANE:EXPR]",
                    help="GEMM-family kernels: the weight tile width (NR) the lane's kernel reads, a C expression, "
                         "e.g. f32-gemm-minmax=Autocomp:__riscv_vsetvlmax_e32m8() for a kernel that tiles by its "
                         "own vl (default: the VLEN 512 kernels' 8, or 16 for qc4w); repeatable")
    a = ap.parse_args(argv)
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    SEL.kernels = set(split(a.kernels))
    if a.impls:
        bad = set(split(a.impls)) - set(LANES)
        if bad:
            ap.error(f"unknown --impls {sorted(bad)}; choose from {LANES}")
        SEL.lanes = set(split(a.impls))
    if a.sections:
        bad = set(split(a.sections)) - set(SECTIONS)
        if bad:
            ap.error(f"unknown --sections {sorted(bad)}; choose from {SECTIONS}")
        SEL.sections = set(split(a.sections))
    SEL.active = bool(a.kernels or a.impls or a.sections)
    for spec in a.autocomp_file:
        key, _, path = spec.partition("=")
        if not path or not os.path.exists(path):
            ap.error(f"--autocomp-file wants KERNEL=PATH to an existing file, got {spec!r}")
        SEL.autocomp_files[key] = path
    for spec in a.sol_file:
        key, _, path = spec.partition("=")
        if not path or not os.path.exists(path):
            ap.error(f"--sol-file wants KERNEL=PATH to an existing file, got {spec!r}")
        SEL.sol_files[key] = path
    for spec in a.lane_nr:
        key, _, lanes_ = spec.partition("=")
        for item in split(lanes_):
            lane, _, expr = item.partition(":")
            if lane not in LANES or not expr:
                ap.error(f"--lane-nr wants KERNEL=LANE:EXPR with LANE from {LANES}, got {spec!r}")
            SEL.lane_nr.setdefault(key, {})[lane] = expr
    SEL.simde = a.simde
    SEL.autocomp_run = a.autocomp_run
    SEL.autocomp_prob_type = a.autocomp_prob_type
    SEL.sol_prob_type = a.sol_prob_type


def _split_section(name):
    for s in SECTIONS[1:]:
        if name.endswith("-" + s):
            return name[: -len(s) - 1], s
    return name, "main"


def _kernel_selected(label):
    if not SEL.kernels:
        return True
    pids = {str(row[0]) for row in KERNELS if row[1] == label}
    return label in SEL.kernels or bool(pids & SEL.kernels)


_TABLE_RE = re.compile(r"(fns\[IMPL_COUNT\] = \{ )(.*?)( \};)")
_PK_RE = re.compile(r"(pk\[IMPL_COUNT\] = \{ )(.*?)( \};)")


def select_call(call):
    """A runner call restricted to the selection, or None to drop it."""
    # run_<family>("label", ...) for 1-D kernels, bench_<kind>("label", ...) for structured ones
    m = re.search(r'\b(?:run|bench)_\w+\("([^"]+)"', call)
    if not m:
        sys.exit(f"selection: cannot find the section label in a call:\n{call}")
    label, section = _split_section(m.group(1))
    if section not in SEL.sections or not _kernel_selected(label):
        return None
    keep = [lane in SEL.lanes for lane in LANES]

    def mask(mt):
        slots = [s.strip() for s in mt.group(2).split(",")]
        if len(slots) != len(LANES):
            return mt.group(0)
        return mt.group(1) + ", ".join(s if k else "NULL" for s, k in zip(slots, keep)) + mt.group(3)

    call = _PK_RE.sub(mask, _TABLE_RE.sub(mask, call))
    t = _TABLE_RE.search(call)
    if t and all(s.strip() == "NULL" for s in t.group(2).split(",")):
        return None
    return call


_BLOCK_RE = re.compile(r"/\* =+ (?:prob \S+ |struct )?(\S+) — ([^=]+?) =+ \*/")
_BLOCK_LANE = {"up": "Upstream", "sol": "Sol", "ac": "Autocomp", "neon2rvv": "Neon2RVV", "nv": "Neon2RVV"}


def select_block(block):
    """Keep an embedded kernel variant only if a selected section can call it."""
    m = _BLOCK_RE.search(block)
    if not m:                                   # shared preludes (the autovec one)
        return "autovec" in SEL.sections
    label, tag = _split_section(m.group(1))[0], m.group(2).split()[0]
    if not _kernel_selected(label):
        return False
    if tag in _BLOCK_LANE:
        return _BLOCK_LANE[tag] in SEL.lanes
    return tag in SEL.sections and "Autocomp" in SEL.lanes   # autovec: Autocomp lane


# ---------------------------------------------------------------- kernel table
# family: unary (batch,in,out,params) | binary (batch,a,b,out,params)
#         rsum (batch,in,acc*,params) | rdsum (rows,ch,in,stride,zero,acc*,params)
# in/out ctype only affect buffer casts; batch is bytes==elements (1B inputs).

KERNELS = [
    # pid, label,            family,  out_kind,   upstream rel path,                         params expr
    # vclamp main lane = LMUL-maximal u8v (best at VLEN=512 from batch 1024 up;
    # the u4v config pick stays as a probe row — it wins only at batch 256).
    (0,  "u8-vclamp",        "unary",  "KIND_U8",  "u8-vclamp/gen/u8-vclamp-rvv-u8v.c",        "&p_u8_minmax"),
    (1,  "s8-vclamp",        "unary",  "KIND_S8",  "s8-vclamp/gen/s8-vclamp-rvv-u8v.c",        "&p_s8_minmax"),
    (2,  "qu8-vadd",         "binary", "KIND_U8",  "qu8-vadd/gen/qu8-vadd-minmax-rvv-u2v.c",   "&p_qu8_add"),
    (3,  "qu8-rsum",         "rsum",   "KIND_I32", "qu8-rsum/gen/qu8-rsum-rvv-u2v.c",          "&p_qs8_rsum"),
    (24, "qu8-rdsum",        "rdsum",  "KIND_I32", "qu8-rdsum/gen/qu8-rdsum-7p7x-rvv-u2v.c",   "&p_qs8_rsum"),
    (18, "qu8-f32-vcvt",     "unary",  "KIND_F32", "qu8-f32-vcvt/gen/qu8-f32-vcvt-rvv-u2v.c",  "&p_qu8_f32_cvt"),
    (6,  "qs8-vmul",         "binary", "KIND_S8",  "qs8-vmul/gen/qs8-vmul-minmax-f32-rvv-u2v.c", "&p_qs8_mul"),
    (5,  "qs8-vlrelu",       "unary",  "KIND_S8",  "qs8-vlrelu/gen/qs8-vlrelu-rvv-u2v.c",      "&p_qs8_lrelu"),
    (4,  "qs8-vcvt",         "unary",  "KIND_S8",  "qs8-vcvt/gen/qs8-vcvt-rvv-u4v.c",          "&p_qs8_cvt"),
    (16, "qs8-vadd",         "binary", "KIND_S8",  "qs8-vadd/gen/qs8-vadd-minmax-rvv-u2v.c",   "&p_qs8_add"),
    (17, "qs8-rsum",         "rsum",   "KIND_I32", "qs8-rsum/gen/qs8-rsum-rvv-u2v.c",          "&p_qs8_rsum"),
    (23, "qs8-rdsum",        "rdsum",  "KIND_I32", "qs8-rdsum/gen/qs8-rdsum-7p7x-rvv-u2v.c",   "&p_qs8_rsum"),
    (20, "qs8-f32-vcvt",     "unary",  "KIND_F32", "qs8-f32-vcvt/gen/qs8-f32-vcvt-rvv-u2v.c",  "&p_qs8_f32_cvt"),
    # ---- f32 elementwise (probs 7-15, 19, 21, 22). Upstream = the RVV config
    # pick (the NEON files these sols were translated from can't run here).
    # batch is BYTES for all of these; runners pass n*4.
    (7,  "f32-vadd",         "f32bin", "FILL_PM1", "f32-vbinary/gen/f32-vadd-rvv-u8v.c",       "&p_f32_default"),
    (19, "f32-vsub",         "f32bin", "FILL_PM1", "f32-vbinary/gen/f32-vsub-rvv-u8v.c",       "&p_f32_default"),
    (13, "f32-vmul",         "f32bin", "FILL_PM1", "f32-vbinary/gen/f32-vmul-rvv-u8v.c",       "&p_f32_default"),
    (14, "f32-vmulc",        "f32bin", "FILL_PM1", "f32-vbinary/gen/f32-vmulc-rvv-u8v.c",      "&p_f32_default"),
    (12, "f32-vmin",         "f32bin", "FILL_PM1", "f32-vbinary/gen/f32-vmin-rvv-u8v.c",       "&p_f32_default"),
    (11, "f32-vmax",         "f32bin", "FILL_PM1", "f32-vbinary/gen/f32-vmax-rvv-u8v.c",       "&p_f32_default"),
    (9,  "f32-vdiv",         "f32bin", "FILL_DIV", "f32-vbinary/gen/f32-vdiv-rvv-u8v.c",       "&p_f32_default"),
    # vlrelu/vrndne/vcmul: the main Upstream lane is the LMUL-MAXIMAL shipped
    # variant — the best candidate at VLEN=512 (long chimes suit Saturn) —
    # which is LARGER than XNNPACK's config pick for these three. The config
    # picks are kept as upstream-only extra rows below so a run measures both.
    (10, "f32-vlrelu",       "f32un:FILL_PM1:KIND_F32", None, "f32-vlrelu/gen/f32-vlrelu-rvv-u8v.c", "&p_f32_lrelu"),
    (15, "f32-vrndne",       "f32un:FILL_PM1:KIND_F32", None, "f32-vrnd/gen/f32-vrndne-rvv-u8v.c",   "&p_f32_default"),
    (22, "f32-vsqrt",        "f32un:FILL_ABS:KIND_F32", None, "f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-u8v.c", "&p_f32_default"),
    # f32->f16: upstream lane = rvvfp16arith u8v (LMUL-maximal, like the other
    # f32 mains). Saturn has Zvfh; the build adds CONFIG_RISCV_ISA_EXT_ZVFH=y
    # (riscv_vector.conf) so the f16 vector intrinsics compile. Spike runs
    # need zvfh in --isa as well.
    (21, "f32-f16-vcvt",     "f32un:FILL_PM1:KIND_U16", None,
     "f32-f16-vcvt/gen/f32-f16-vcvt-rvvfp16arith-u8v.c",                       "&p_f32_default"),
    # vcmul u8v cannot exist (4 input streams x m8 exceeds the register file);
    # u4v is the shipped maximum.
    (8,  "f32-vcmul",        "f32cmul", None,      "f32-vcmul/gen/f32-vcmul-rvv-u4v.c",        "&p_f32_default"),
    # Upstream-only extra rows (no sol/autocomp counterpart): the OTHER
    # interesting upstream variant per kernel — the config pick where the main
    # lane is LMUL-maximal, or the larger variant where the main lane is the
    # config pick (vclamp). pid=None disables sol/ac loading.
    (None, "u8-vclamp-u4v",  "unary",  "KIND_U8",  "u8-vclamp/gen/u8-vclamp-rvv-u4v.c",        "&p_u8_minmax"),
    (None, "s8-vclamp-u4v",  "unary",  "KIND_S8",  "s8-vclamp/gen/s8-vclamp-rvv-u4v.c",        "&p_s8_minmax"),
    (None, "f32-vlrelu-u4v", "f32un:FILL_PM1:KIND_F32", None, "f32-vlrelu/gen/f32-vlrelu-rvv-u4v.c", "&p_f32_lrelu"),
    (None, "f32-vrndne-u4v", "f32un:FILL_PM1:KIND_F32", None, "f32-vrnd/gen/f32-vrndne-rvv-u4v.c",   "&p_f32_default"),
    (None, "f32-vcmul-u2v",  "f32cmul", None,      "f32-vcmul/gen/f32-vcmul-rvv-u2v.c",        "&p_f32_default"),
]

HEADER = r'''/*
 * AUTO-GENERATED by scripts/gen_perf_compare.py — do not edit by hand.
 *
 * RVV benchmark: per kernel, three implementations over a batch sweep:
 *   Upstream — XNNPACK hand-tuned RVV microkernel
 *   Sol      — pre-autocomp kernel (sols/cgo-kernels)
 *   Autocomp — best autocomp candidate (output/<run>/best_candidate_so_far.txt)
 * Measures cycles/instructions per element via rdcycle/rdinstret with a
 * per-(kernel,impl,batch) warmup, mirroring samples/rvv_bench.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <riscv_vector.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>

/* NEON intrinsics implemented on RVV (fixed 128-bit semantics) — provides
 * the Neon2RVV lane: upstream NEON kernel sources compiled mechanically. */
#include "neon2rvv.h"

static inline uint64_t rdcycle(void)   { uint64_t v; __asm__ volatile("rdcycle %0"   : "=r"(v)); return v; }
static inline uint64_t rdinstret(void) { uint64_t v; __asm__ volatile("rdinstret %0" : "=r"(v)); return v; }
static inline void fence(void)         { __asm__ volatile("fence" ::: "memory"); }

/* ---- XNNPACK helper macros (no library link; everything is embedded) ---- */
#define XNN_UNPREDICTABLE(x) (x)
#define XNN_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#define XNN_LIKELY(x)   (__builtin_expect(!!(x), 1))
#define XNN_OOB_READS
#define XNN_LOG2_SIZEOF_INT8_T  0
#define XNN_LOG2_SIZEOF_UINT8_T 0
#define XNN_LOG2_SIZEOF_FLOAT   2
#define XNN_LOG2_SIZEOF_INT32_T 2
#define XNN_FORCE_REALIZATION(x) /* compiler-barrier hint; no-op on riscv */
#define XNN_ALIGN(alignment) __attribute__((__aligned__(alignment)))
typedef uint16_t xnn_float16;    /* f16 outputs handled as raw bit patterns */
/* xnnpack/math.h difference-or-zero */
static inline size_t doz(size_t a, size_t b) { return a >= b ? a - b : 0; }

/* ---- param struct layouts (match xnnpack/microparams.h) ---- */
struct xnn_u8_minmax_params  { struct { uint32_t min, max; } scalar; };
struct xnn_s8_minmax_params  { struct { int32_t  min, max; } scalar; };
struct xnn_qs8_rsum_params   { char _; };
struct xnn_qs8_cvt_params    { struct { int16_t input_zero_point; int32_t multiplier; int16_t output_zero_point; } scalar; };
struct xnn_qs8_lrelu_params  { struct { int32_t input_zero_point; int32_t positive_multiplier; int32_t negative_multiplier; int32_t output_zero_point; } scalar; };
struct xnn_qu8_add_minmax_params {
    struct {
        uint8_t a_zero_point; uint8_t b_zero_point; int32_t bias;
        int32_t a_multiplier; int32_t b_multiplier; int32_t shift;
        int16_t output_zero_point; uint8_t output_min; uint8_t output_max;
    } scalar;
};
struct xnn_qs8_add_minmax_params {
    struct {
        int8_t a_zero_point; int8_t b_zero_point; int32_t bias;
        int32_t a_multiplier; int32_t b_multiplier; int32_t shift;
        int16_t output_zero_point; int8_t output_min; int8_t output_max;
    } scalar;
};
union xnn_qs8_mul_minmax_params {
    struct {
        int8_t a_zero_point; int8_t b_zero_point; float scale;
        int16_t output_zero_point; int8_t output_min; int8_t output_max;
    } scalar;
};
struct xnn_qs8_f32_cvt_params { struct { int32_t zero_point; float scale; } scalar; };
struct xnn_qu8_f32_cvt_params { struct { int32_t zero_point; float scale; } scalar; };
struct xnn_f32_default_params { char _; };
struct xnn_f32_lrelu_params   { struct { float slope; } scalar; };

/* ---- param values (mirror harnesses/cgo-kernels test<N>.c) ---- */
static const struct xnn_u8_minmax_params p_u8_minmax = { .scalar = { .min = 16, .max = 240 } };
static const struct xnn_s8_minmax_params p_s8_minmax = { .scalar = { .min = -100, .max = 100 } };
static const struct xnn_qs8_rsum_params  p_qs8_rsum  = { 0 };
static const struct xnn_qs8_cvt_params   p_qs8_cvt   = { .scalar = { .input_zero_point = 0, .multiplier = 0x40000000, .output_zero_point = 0 } };
/* lrelu multipliers are Q·.8 in BOTH generations: the sol computes
 * ((x-zp)*128*mult)>>15 and upstream ((x-zp)*mult+128)>>8 — slope = mult/256.
 * 256 = slope 1.0, 26 ≈ slope 0.1. (Q1.30-style values overflow upstream's
 * i32 macc and truncate in the sol's int16 path.) */
static const struct xnn_qs8_lrelu_params p_qs8_lrelu = { .scalar = { .input_zero_point = 0, .positive_multiplier = 256, .negative_multiplier = 26, .output_zero_point = 0 } };
/* Upstream vadd reads ONLY .bias (expects -(a_zp*a_mult + b_zp*b_mult) baked
 * in, per XNNPACK's init fn) and ignores the zp fields; the sol reads the zp
 * fields and ignores .bias. Setting both keeps every impl on the same math. */
static const struct xnn_qu8_add_minmax_params p_qu8_add = { .scalar = {
    .a_zero_point = 128, .b_zero_point = 128,
    .bias = -((128 + 128) * (1 << 20)),
    .a_multiplier = 1 << 20, .b_multiplier = 1 << 20, .shift = 21,
    .output_zero_point = 128, .output_min = 16, .output_max = 240 } };
static const struct xnn_qs8_add_minmax_params p_qs8_add = { .scalar = {
    .a_zero_point = 0, .b_zero_point = 0, .bias = 0,
    .a_multiplier = 1 << 20, .b_multiplier = 1 << 20, .shift = 21,
    .output_zero_point = 0, .output_min = -112, .output_max = 112 } };
static const union xnn_qs8_mul_minmax_params p_qs8_mul = { .scalar = {
    .a_zero_point = 0, .b_zero_point = 0, .scale = 1.0f / 128.0f,
    .output_zero_point = 0, .output_min = -100, .output_max = 100 } };
static const struct xnn_qs8_f32_cvt_params p_qs8_f32_cvt = { .scalar = { .zero_point = 0,   .scale = 1.0f / 128.0f } };
static const struct xnn_qu8_f32_cvt_params p_qu8_f32_cvt = { .scalar = { .zero_point = 128, .scale = 1.0f / 128.0f } };
static const struct xnn_f32_default_params p_f32_default = { 0 };
static const struct xnn_f32_lrelu_params   p_f32_lrelu   = { .scalar = { .slope = 0.1f } };

/* ---- bench configuration ---- */
#ifndef NUM_ITERATIONS
#define NUM_ITERATIONS 1000
#endif
/* Timing only by default (PERF_NO_SANITY=1): no correctness checks, no untimed reference run and no output
 * comparison in any section. -DEXTRA_CFLAGS=-DPERF_NO_SANITY=0 adds them, for a correctness check on Spike;
 * never time such a build. */
/* PERF_CSV_ONLY=1: print only the "CSV:" line of each cell (no section banner, no table row). On FireSim the run
 * time is the console output (about 35 ms of wall clock per character through HTIF), not the kernels. */
#ifndef PERF_CSV_ONLY
#define PERF_CSV_ONLY 0
#endif
#ifndef PERF_NO_SANITY
#define PERF_NO_SANITY 1
#endif
#ifndef NUM_WARMUP
#define NUM_WARMUP     100
#endif

/* Section gates — all code is compiled either way; set ALL to 1 for the
 * final full all-kernels run.
 *   RUN_QUANTIZED:   the 13 quantized sections (u8/s8/qu8/qs8) — measured.
 *   RUN_ELEMENTWISE: the 12 f32 elementwise mains + their 5 variant probes
 *                    — measured.
 *   RUN_STRUCTURED:  the 11 structured-kernel mains (transposec, pools,
 *                    dwconv, conv-hwc2chw, gemm/igemm/qc4w, packw)
 *                    — measured.
 *   RUN_NEON2RVV:    every *-neon2rvv section (25 elementwise + 7
 *                    structured) — not measured yet.
 *   RUN_AUTOVEC:     the 35 *-autovec sections (XNNPACK scalar C through GCC's
 *                    RVV auto-vectorizer) — not measured yet.
 * Current run set: NEON2RVV + AUTOVEC (vlen=128 baseline run). Each gate can be set from the
 * build instead: west build ... -- -DEXTRA_CFLAGS="-DRUN_QUANTIZED=1 -DRUN_ELEMENTWISE=1 ..." */
#ifndef RUN_QUANTIZED
#define RUN_QUANTIZED 0
#endif
#ifndef RUN_ELEMENTWISE
#define RUN_ELEMENTWISE 0
#endif
#ifndef RUN_STRUCTURED
#define RUN_STRUCTURED 0
#endif
#ifndef RUN_NEON2RVV
#define RUN_NEON2RVV 1
#endif
#ifndef RUN_AUTOVEC
#define RUN_AUTOVEC 1
#endif

/* Some param tables go unreferenced when RUN_QUANTIZED=0. */
#pragma GCC diagnostic ignored "-Wunused-const-variable"

/* Elementwise / reduce sweep: batch in elements (1-byte inputs => bytes). */
static const size_t batch_sizes[] = { 256, 1024, 4096, 16384, 65536 };
#define NUM_BATCH (sizeof(batch_sizes) / sizeof(batch_sizes[0]))
#define MAX_BATCH 65536

/* rdsum sweep: rows varies, channels fixed. elements = rows * channels. */
#define RDSUM_CHANNELS 128
static const size_t rdsum_rows[] = { 16, 32, 64, 128, 256 };
#define NUM_RDSUM (sizeof(rdsum_rows) / sizeof(rdsum_rows[0]))
#define MAX_RDSUM_ROWS 256

/* ---- buffers (static; inputs re-randomized per batch size) ---- */
static uint8_t buf_a  [MAX_BATCH]      __attribute__((aligned(64)));
static uint8_t buf_b  [MAX_BATCH]      __attribute__((aligned(64)));
static uint8_t buf_out[MAX_BATCH * 4]  __attribute__((aligned(64)));
static uint8_t buf_gold[MAX_BATCH * 4] __attribute__((aligned(64)));  /* reference output for sanity */
static uint8_t zero_row[RDSUM_CHANNELS] __attribute__((aligned(64)));  /* stays zero */

/* f32-family buffers; 2x for vcmul's planar complex layout (2n floats). */
static float fbuf_a   [2 * MAX_BATCH] __attribute__((aligned(64)));
static float fbuf_b   [2 * MAX_BATCH] __attribute__((aligned(64)));
static float fbuf_out [2 * MAX_BATCH] __attribute__((aligned(64)));
static float fbuf_gold[2 * MAX_BATCH] __attribute__((aligned(64)));

static uint32_t rand_state = 12345;
static void reset_rand(void) { rand_state = 12345; }
static uint8_t rand_byte(void) {
    rand_state = rand_state * 1103515245 + 12345;
    return (uint8_t)(rand_state >> 16);
}
static void fill_random(uint8_t* p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = rand_byte(); }

/* ---- impl slots ---- */
#define IMPL_COUNT 4
static const char* impl_names[IMPL_COUNT] = { "Upstream", "Sol", "Autocomp", "Neon2RVV" };

/* ---- output sanity check: tolerant compare of each impl vs Upstream ----
 * Not bit-exact on purpose: implementations may reorder math (different
 * rounding sequence in fp32-rescale paths, reassociated reductions), so:
 *   int8/uint8 outputs : |diff| <= 1 quantization step
 *   int32 reductions   : |diff| <= 4
 *   f32 outputs        : |diff| <= 1e-4 + 1e-4*|upstream|
 */
typedef enum { KIND_U8, KIND_S8, KIND_I32, KIND_F32, KIND_U16, KIND_U32 } out_kind_t;

static unsigned g_sanity_fail = 0;
static void check_against_gold(const char* kernel, const char* impl, size_t batch,
                               const void* got_v, const void* gold_v,
                               size_t n_elems, out_kind_t kind, const char* refname) {
    double max_diff = 0.0;
    for (size_t i = 0; i < n_elems; i++) {
        double got, want, tol;
        switch (kind) {
        case KIND_U8:
            got = ((const uint8_t*)got_v)[i];  want = ((const uint8_t*)gold_v)[i];
            tol = 1.0; break;
        case KIND_S8:
            got = ((const int8_t*)got_v)[i];   want = ((const int8_t*)gold_v)[i];
            tol = 1.0; break;
        case KIND_I32:
            got = ((const int32_t*)got_v)[i];  want = ((const int32_t*)gold_v)[i];
            tol = 4.0; break;
        case KIND_U16:
            /* f16 outputs compared on the raw bit pattern: same-sign finite
             * halfs are monotonic in bits, so tol=1 ~= 1 ulp. */
            got = ((const uint16_t*)got_v)[i]; want = ((const uint16_t*)gold_v)[i];
            tol = 1.0; break;
        case KIND_U32:
            /* exact compare (transpose/packw move bits; argmax indices) */
            got = ((const uint32_t*)got_v)[i]; want = ((const uint32_t*)gold_v)[i];
            tol = 0.5; break;
        default: /* KIND_F32 */
            got = ((const float*)got_v)[i];    want = ((const float*)gold_v)[i];
            tol = 1e-4 + 1e-4 * (want < 0 ? -want : want); break;
        }
        double d = got - want; if (d < 0) d = -d;
        if (d > max_diff) max_diff = d;
        if (d > tol) {
            printf("  SANITY %-9s batch=%6zu MISMATCH @i=%zu got=%g want=%g tol=%g  [%s vs %s]\n",
                   impl, batch, i, got, want, tol, kernel, refname);
            g_sanity_fail++;
            return;
        }
    }
    printf("  SANITY %-9s batch=%6zu OK (%zu elems, max|diff|=%g vs %s)\n",
           impl, batch, n_elems, max_diff, refname);
}

/* Sanity vs first available impl (Upstream when present; Sol for sections
 * without an upstream lane, e.g. f32-f16-vcvt and the *-neon2rvv sections). */
static int sanity_ref(void* fns_v[IMPL_COUNT]) {
    for (int k = 0; k < IMPL_COUNT; k++) if (fns_v[k]) return k;
    return -1;
}

static void print_header(const char* title) {
    if (PERF_CSV_ONLY) return;
    printf("\n################################################################\n");
    printf("# %s\n", title);
    printf("################################################################\n");
    printf("  %-10s %8s %15s %15s %12s %12s\n",
           "Impl", "batch", "TotalCycles", "TotalInstrs", "Cyc/Elem", "Ins/Elem");
}

static void report(const char* kernel, const char* impl, size_t batch,
                   uint64_t tc, uint64_t ti, size_t elems) {
    double cpe = (double)tc / ((double)NUM_ITERATIONS * (double)elems);
    double ipe = (double)ti / ((double)NUM_ITERATIONS * (double)elems);
    if (!PERF_CSV_ONLY)
        printf("  %-10s %8zu %15llu %15llu %12.4f %12.4f\n",
               impl, batch, (unsigned long long)tc, (unsigned long long)ti, cpe, ipe);
    printf("CSV: %s,%s,%zu,%llu,%llu,%.6f,%.6f\n",
           kernel, impl, batch, (unsigned long long)tc, (unsigned long long)ti, cpe, ipe);
}

/* ---- family runners ---- */
typedef void (*unary_fn)(size_t batch, const void* in, void* out, const void* params);
typedef void (*binary_fn)(size_t batch, const void* a, const void* b, void* out, const void* params);
typedef void (*rsum_fn)(size_t batch, const void* in, void* acc, const void* params);
typedef void (*rdsum_fn)(size_t rows, size_t channels, const void* in, size_t in_stride,
                         const void* zero, void* acc, const void* params);

#define TIME_CELL(KERNEL, IMPL, BATCH, ELEMS, CALL) do {                       \
    for (size_t w_ = 0; w_ < NUM_WARMUP; w_++) { CALL; }                       \
    fence();                                                                   \
    uint64_t c0_ = rdcycle(), i0_ = rdinstret();                               \
    for (size_t it_ = 0; it_ < NUM_ITERATIONS; it_++) { CALL; }                \
    fence();                                                                   \
    uint64_t c1_ = rdcycle(), i1_ = rdinstret();                               \
    report(KERNEL, IMPL, BATCH, c1_ - c0_, i1_ - i0_, ELEMS);                  \
} while (0)

static void run_unary(const char* kernel, unary_fn fns[IMPL_COUNT], const void* params,
                      out_kind_t out_kind) {
    print_header(kernel);
    for (size_t bi = 0; bi < NUM_BATCH; bi++) {
        size_t n = batch_sizes[bi];
        reset_rand();
        fill_random(buf_a, n);
        /* sanity: each impl's output vs the reference impl, same inputs */
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            fns[ref](n, buf_a, buf_gold, params);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                fns[k](n, buf_a, buf_out, params);
                check_against_gold(kernel, impl_names[k], n, buf_out, buf_gold, n, out_kind, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(kernel, impl_names[k], n, n, fns[k](n, buf_a, buf_out, params));
        }
    }
}

static void run_binary(const char* kernel, binary_fn fns[IMPL_COUNT], const void* params,
                       out_kind_t out_kind) {
    print_header(kernel);
    for (size_t bi = 0; bi < NUM_BATCH; bi++) {
        size_t n = batch_sizes[bi];
        reset_rand();
        fill_random(buf_a, n);
        fill_random(buf_b, n);
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            fns[ref](n, buf_a, buf_b, buf_gold, params);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                fns[k](n, buf_a, buf_b, buf_out, params);
                check_against_gold(kernel, impl_names[k], n, buf_out, buf_gold, n, out_kind, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(kernel, impl_names[k], n, n, fns[k](n, buf_a, buf_b, buf_out, params));
        }
    }
}

static void run_rsum(const char* kernel, rsum_fn fns[IMPL_COUNT], const void* params) {
    print_header(kernel);
    for (size_t bi = 0; bi < NUM_BATCH; bi++) {
        size_t n = batch_sizes[bi];
        reset_rand();
        fill_random(buf_a, n);
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            memset(buf_gold, 0, sizeof(uint32_t));   /* rsum accumulates: zero before the check call */
            fns[ref](n, buf_a, buf_gold, params);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                memset(buf_out, 0, sizeof(uint32_t));
                fns[k](n, buf_a, buf_out, params);
                check_against_gold(kernel, impl_names[k], n, buf_out, buf_gold, 1, KIND_I32, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            memset(buf_out, 0, sizeof(uint32_t));  /* accumulator wraps over iters; timing-only */
            TIME_CELL(kernel, impl_names[k], n, n, fns[k](n, buf_a, buf_out, params));
        }
    }
}

static void run_rdsum(const char* kernel, rdsum_fn fns[IMPL_COUNT], const void* params) {
    print_header(kernel);
    for (size_t bi = 0; bi < NUM_RDSUM; bi++) {
        size_t rows = rdsum_rows[bi];
        size_t elems = rows * RDSUM_CHANNELS;
        reset_rand();
        fill_random(buf_a, elems);
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            memset(buf_gold, 0, RDSUM_CHANNELS * sizeof(uint32_t));
            fns[ref](rows, RDSUM_CHANNELS, buf_a, RDSUM_CHANNELS, zero_row, buf_gold, params);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                memset(buf_out, 0, RDSUM_CHANNELS * sizeof(uint32_t));
                fns[k](rows, RDSUM_CHANNELS, buf_a, RDSUM_CHANNELS, zero_row, buf_out, params);
                check_against_gold(kernel, impl_names[k], rows, buf_out, buf_gold, RDSUM_CHANNELS, KIND_I32, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            memset(buf_out, 0, RDSUM_CHANNELS * sizeof(uint32_t));
            TIME_CELL(kernel, impl_names[k], rows, elems,
                      fns[k](rows, RDSUM_CHANNELS, buf_a, RDSUM_CHANNELS, zero_row, buf_out, params));
        }
    }
}

/* ---- f32 families: batch argument is in BYTES (kernels assert %4==0) ---- */
typedef void (*f32bin_fn)(size_t batch, const float* a, const float* b, float* out, const void* params);
typedef void (*f32un_fn)(size_t batch, const float* in, void* out, const void* params);

/* input fill policies */
#define FILL_PM1 0   /* both inputs in [-1, 1] */
#define FILL_DIV 1   /* divisor bounded to +/-[0.5, 1.5] (no inf/nan) */
#define FILL_ABS 2   /* input non-negative (vsqrt) */

static float rand_f32(void) {
    rand_state = rand_state * 1103515245 + 12345;
    return ((float)((rand_state >> 16) & 0xFFFF) / 65536.0f) * 2.0f - 1.0f;
}
static void fill_f32(float* p, size_t n, int mode) {
    for (size_t i = 0; i < n; i++) {
        float v = rand_f32();
        if (mode == FILL_DIV) v = (v < 0 ? -0.5f : 0.5f) + v;   /* +/-[0.5, 1.5] */
        if (mode == FILL_ABS) v = v < 0 ? -v : v;
        p[i] = v;
    }
}

static void run_f32bin(const char* kernel, f32bin_fn fns[IMPL_COUNT], const void* params,
                       int fill_mode) {
    print_header(kernel);
    for (size_t bi = 0; bi < NUM_BATCH; bi++) {
        size_t n = batch_sizes[bi];
        reset_rand();
        fill_f32(fbuf_a, n, FILL_PM1);
        fill_f32(fbuf_b, n, fill_mode);
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            fns[ref](n * 4, fbuf_a, fbuf_b, fbuf_gold, params);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                fns[k](n * 4, fbuf_a, fbuf_b, fbuf_out, params);
                check_against_gold(kernel, impl_names[k], n, fbuf_out, fbuf_gold,
                                   n, KIND_F32, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(kernel, impl_names[k], n, n, fns[k](n * 4, fbuf_a, fbuf_b, fbuf_out, params));
        }
    }
}

static void run_f32un(const char* kernel, f32un_fn fns[IMPL_COUNT], const void* params,
                      int fill_mode, out_kind_t out_kind) {
    print_header(kernel);
    for (size_t bi = 0; bi < NUM_BATCH; bi++) {
        size_t n = batch_sizes[bi];
        reset_rand();
        fill_f32(fbuf_a, n, fill_mode);
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            fns[ref](n * 4, fbuf_a, fbuf_gold, params);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                fns[k](n * 4, fbuf_a, fbuf_out, params);
                check_against_gold(kernel, impl_names[k], n, fbuf_out, fbuf_gold,
                                   n, out_kind, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(kernel, impl_names[k], n, n, fns[k](n * 4, fbuf_a, fbuf_out, params));
        }
    }
}

/* vcmul: batch = n_complex * sizeof(float); inputs/outputs are 2n floats in
 * planar layout [re[0..n), im[0..n)]. Score per FLOAT written (2n). */
static void run_f32cmul(const char* kernel, f32bin_fn fns[IMPL_COUNT], const void* params) {
    print_header(kernel);
    for (size_t bi = 0; bi < NUM_BATCH; bi++) {
        size_t n = batch_sizes[bi];
        reset_rand();
        fill_f32(fbuf_a, 2 * n, FILL_PM1);
        fill_f32(fbuf_b, 2 * n, FILL_PM1);
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            fns[ref](n * 4, fbuf_a, fbuf_b, fbuf_gold, params);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                fns[k](n * 4, fbuf_a, fbuf_b, fbuf_out, params);
                check_against_gold(kernel, impl_names[k], n, fbuf_out, fbuf_gold,
                                   2 * n, KIND_F32, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(kernel, impl_names[k], n, 2 * n, fns[k](n * 4, fbuf_a, fbuf_b, fbuf_out, params));
        }
    }
}
'''

MAIN_TOP = r'''
int main(void) {
    printf("\n################################################################\n");
    printf("#  RVV PERF COMPARE: Upstream XNNPACK vs Sol vs Autocomp       #\n");
    printf("################################################################\n");
    printf("Board: %s\n", CONFIG_BOARD_TARGET);
    printf("Iterations per cell: %d   Warmup: %d\n", NUM_ITERATIONS, NUM_WARMUP);
'''


# Sections already measured on FireSim — gated behind RUN_QUANTIZED in the
# generated main.c (code still compiled; only the runner calls are skipped).
MEASURED_QUANTIZED_PIDS = {0, 1, 2, 3, 4, 5, 6, 16, 17, 18, 20, 23, 24}

# Neon2RVV lane: upstream NEON kernel source per prob, compiled mechanically
# via neon2rvv.h (fixed 128-bit semantics). Paths relative to XNN_SRC.
NEON_SOURCES = {
    0:  "u8-vclamp/u8-vclamp-neon-u64.c",
    1:  "s8-vclamp/s8-vclamp-neon-u64.c",
    2:  "qu8-vadd/gen/qu8-vadd-minmax-neon-ld64-u8.c",
    3:  "qu8-rsum/gen/qu8-rsum-neon-u16.c",
    24: "qu8-rdsum/gen/qu8-rdsum-7p7x-neon-u32.c",
    18: "qu8-f32-vcvt/gen/qu8-f32-vcvt-neon-u8.c",
    6:  "qs8-vmul/gen/qs8-vmul-minmax-fp32-neon-ld64-u8.c",
    5:  "qs8-vlrelu/gen/qs8-vlrelu-neon-u8.c",
    4:  "qs8-vcvt/gen/qs8-vcvt-neon-u8.c",
    16: "qs8-vadd/gen/qs8-vadd-minmax-neon-ld64-u16.c",
    17: "qs8-rsum/gen/qs8-rsum-neon-u16.c",
    23: "qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-neon-u16.c",
    20: "qs8-f32-vcvt/gen/qs8-f32-vcvt-neon-u8.c",
    7:  "f32-vbinary/gen/f32-vadd-neon-u4.c",
    19: "f32-vbinary/gen/f32-vsub-neon-u4.c",
    13: "f32-vbinary/gen/f32-vmul-neon-u4.c",
    14: "f32-vbinary/gen/f32-vmulc-neon-u4.c",
    12: "f32-vbinary/gen/f32-vmin-neon-u4.c",
    11: "f32-vbinary/gen/f32-vmax-neon-u4.c",
    9:  "f32-vbinary/gen/f32-vdiv-aarch64-neon-u4.c",
    10: "f32-vlrelu/gen/f32-vlrelu-neon-u4.c",
    15: "f32-vrnd/gen/f32-vrndne-neon-u4.c",
    # 22 (f32-vsqrt): both upstream NEON variants are written against
    # XNNPACK's xnn_simd abstraction layer; the file below is that kernel
    # with every abstraction call hand-inlined to its raw-NEON definition.
    22: f"{REPO}/scripts/neon_inlined/f32-vsqrt-neon-inlined.c",
    21: "f32-f16-vcvt/gen/f32-f16-vcvt-neon-u8.c",
    8:  "f32-vcmul/gen/f32-vcmul-neon-u4.c",
}

# Auto-vectorization baseline: XNNPACK's scalar C kernels, compiled with GCC's
# RVV auto-vectorizer (each variant block is wrapped in
# #pragma GCC optimize("tree-vectorize","tree-loop-vectorize") to re-enable the
# vectorizer that Zephyr disables globally with -fno-tree-vectorize). u1 (no
# hand-unroll) variants = the truest "scalar C", cleanest loop for the
# vectorizer. Emitted as separate "<label>-autovec" sections, gated by
# RUN_AUTOVEC, with the autovec kernel in the Autocomp slot and Upstream as the
# in-section sanity reference. Same per-lane weight retiling as the other lanes.
# row_info (elementwise/f16) pids:
AUTOVEC_SOURCES = {
    0:  "u8-vclamp/u8-vclamp-scalar-u4.c",
    1:  "s8-vclamp/s8-vclamp-scalar-u4.c",
    2:  "qu8-vadd/gen/qu8-vadd-minmax-scalar-u1.c",
    3:  "qu8-rsum/gen/qu8-rsum-scalar-u1.c",
    24: "qu8-rdsum/gen/qu8-rdsum-scalar.c",
    18: "qu8-f32-vcvt/gen/qu8-f32-vcvt-scalar-u1.c",
    6:  "qs8-vmul/gen/qs8-vmul-minmax-fp32-scalar-u1.c",
    5:  "qs8-vlrelu/gen/qs8-vlrelu-scalar-andxor-u1.c",
    # 4 qs8-vcvt: PERF-ONLY (see AUTOVEC_NO_SANITY). The scalar variant reads
    # `multiplier` as int32 (fixed-point + shift) while the harness reference
    # (RVV vsmul) reads it as int16 Q15 — incompatible requant conventions, so
    # outputs differ (param 0x40000000 => all-zeros in the Q15 reference vs
    # saturate-to-127 in the int32 scalar). Bit-exactness is impossible here,
    # but the load/multiply/shift/clamp/store work is identical, so the cycle
    # count is a valid autovec perf measure. Emitted with a NULL sanity
    # reference; pair with the upstream qs8-vcvt CSV row for the ratio.
    4:  "qs8-vcvt/gen/qs8-vcvt-scalar-u1.c",
    16: "qs8-vadd/gen/qs8-vadd-minmax-scalar-u1.c",
    17: "qs8-rsum/gen/qs8-rsum-scalar-u1.c",
    23: "qs8-rdsum/gen/qs8-rdsum-minmax-fp32-scalar-u1-acc1.c",
    20: "qs8-f32-vcvt/gen/qs8-f32-vcvt-scalar-u1.c",
    7:  "f32-vbinary/gen/f32-vadd-scalar-u1.c",
    19: "f32-vbinary/gen/f32-vsub-scalar-u1.c",
    13: "f32-vbinary/gen/f32-vmul-scalar-u1.c",
    14: "f32-vbinary/gen/f32-vmulc-scalar-u1.c",
    12: "f32-vbinary/gen/f32-vmin-scalar-u1.c",
    11: "f32-vbinary/gen/f32-vmax-scalar-u1.c",
    9:  "f32-vbinary/gen/f32-vdiv-scalar-u1.c",
    10: "f32-vlrelu/gen/f32-vlrelu-scalar-u1.c",
    15: "f32-vrnd/gen/f32-vrndne-scalar-libm-u1.c",
    22: "f32-vsqrt/gen/f32-vsqrt-scalar-sqrt.c",
    21: "f32-f16-vcvt/gen/f32-f16-vcvt-scalar-bitcast-u1.c",
    8:  "f32-vcmul/gen/f32-vcmul-scalar-u1.c",
}
# pids emitted perf-only: NULL sanity reference (output legitimately differs
# from the harness reference; the cycle count is still a valid perf measure).
AUTOVEC_NO_SANITY = {4}
# Targeted-rerun filter: when AV_ONLY is set (comma-separated base labels),
# only those autovec runner CALLS are emitted (all function defs still compile,
# so nothing else changes). Lets a rerun capture a few kernels without the full
# ~3.5h autovec sweep, e.g. AV_ONLY=qs8-vcvt,qd8-f32-qc4w-gemm-minmax,x32-packw-gemm-goi
AUTOVEC_ONLY = set(filter(None, os.environ.get("AV_ONLY", "").split(",")))
# structured kernels: label -> (path, call template). autovec fn in slot 2
# (Autocomp), upstream slot 0 as sanity ref. nrs_ slot 2 = scalar kernel's NR.
AUTOVEC_STRUCT = {
    "x32-transposec": ("x32-transposec/gen/x32-transposec-1x4-scalar-float.c",
        'bench_transposec("{label}", fns);'),
    "f32-argmaxpool": ("f32-argmaxpool/f32-argmaxpool-9p8x-scalar-c1.c",
        'bench_argmaxpool("{label}", fns, 25);'),
    "f32-maxpool": ("f32-maxpool/gen/f32-maxpool-9p-minmax-scalar-u1.c",
        'bench_maxpool("{label}", fns, 9, (const void*)&p_f32_minmax_wide);'),
    "f32-dwconv-minmax": ("f32-dwconv/gen/f32-dwconv-3p1c-minmax-scalar.c",
        'packfn_t pk[IMPL_COUNT] = {{ NULL, NULL, av_dwconv_pack_cr1, NULL }};\n'
        '      bench_dwconv("{label}", fns, pk, (const void*)&p_f32_minmax_wide);'),
    "f32-dwconv2d-chw": ("f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-scalar-1x1.c",
        'packfn_t pk[IMPL_COUNT] = {{ NULL, NULL, NULL, NULL }};\n'
        '      bench_dwconv2d("{label}", fns, pk, (const void*)&p_f32_minmax_wide);'),
    "f32-conv-hwc2chw": ("f32-conv-hwc2chw/f32-conv-hwc2chw-3x3s2p1c3x4-scalar-1x1.c",
        'packfn_t pk[IMPL_COUNT] = {{ NULL, NULL, NULL, NULL }};\n'
        '      bench_hwc2chw("{label}", fns, pk, (const void*)&p_f32_minmax_wide);'),
    "f32-gemm-minmax": ("f32-gemm/gen/f32-gemm-1x4-minmax-scalar.c",
        'packfn_t pk[IMPL_COUNT] = {{ NULL, NULL, NULL, NULL }};\n'
        '      size_t mrs_[IMPL_COUNT] = {{ 7, 4, 1, 1 }};\n'
        '      size_t nrs_[IMPL_COUNT] = {{ 0, 8, 4, 8 }};\n'
        '      bench_gemm("{label}", fns, pk, mrs_, nrs_, (const void*)&p_f32_minmax_wide);'),
    "f32-igemm-minmax": ("f32-igemm/gen/f32-igemm-1x4-minmax-scalar.c",
        'packfn_t pk[IMPL_COUNT] = {{ NULL, NULL, NULL, NULL }};\n'
        '      size_t nrs_[IMPL_COUNT] = {{ 0, 8, 4, 8 }};\n'
        '      bench_igemm("{label}", fns, pk, nrs_, (const void*)&p_f32_minmax_wide);'),
    "qd8-f32-qc4w-gemm-minmax": ("qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4-minmax-scalar.c",
        'packfn_t pk[IMPL_COUNT] = {{ NULL, NULL, NULL, NULL }};\n'
        '      size_t nrs_[IMPL_COUNT] = {{ 0, 16, 4, 16 }};\n'
        '      bench_qc4w("{label}", fns, pk, nrs_);'),
    "x32-packw-gemm-goi": ("x32-packw/gen/x32-packw-x2-gemm-goi-scalar-float-u4.c",
        'bench_packw("{label}", fns);'),
}

# ------------------------- structured kernels (probs 25-33 + transposec) ----
# Pools/convs/gemms/transpose/packw. Unlike the elementwise families these
# need bespoke runners: several have PER-LANE weight layouts (sol/autocomp
# pack nr=8 / oc4 / m4-block tiles; the upstream RVV variants expect
# vsetvlmax-wide tiles), so each bench builds canonical weights, retiles them
# per lane, and applies the sol/autocomp pack_weights functions where the
# candidates define them (slots 1/2; run outside the timed region, exactly
# like HARNESS_MAIN_MULTI_PACK).

# Autocomp transposec lane: final candidate of the xnnpack-f32 prob-1 run
# (nested CodeCandidate repr — the LAST code='''...''' block is the newest).
TR_CAND = (f"{REPO}/autocomp/backend/xnnpack/tmp_dir/output/"
           "built:saturn-rvv_xnnpack-f32_1_beam_iters8_firesim_SaturnHardwareConfig_"
           "vlen=512_dlen=256_mini-3.1-pro-preview_1-flash-lite-preview_"
           "do0.25_p4_c2_b4_score1_ms1_fgisa1_ex0.25/candidates-iter-8/candidate_0.txt")

NEON_INLINED = f"{REPO}/scripts/neon_inlined"

def load_candidate_repr(path):
    """Extract the newest kernel from a CodeCandidate repr dump."""
    if not os.path.exists(path):
        return None
    parts = open(path).read().split("code='''")
    if len(parts) < 2:
        return None
    return parts[-1].split("'''", 1)[0]


# label, C fn-pointer type, sol/ac prob id (None -> overrides), upstream path,
# neon path (None -> no NEON lane), uses pack slots, bench call template,
# overrides dict (sol_path / ac_path).
STRUCT_KERNELS = [
    ("x32-transposec", "tr_fn", None,
     "x32-transposec/gen/x32-transposec-4x4-rvv.c",
     f"{NEON_INLINED}/x32-transposec-2x2-multi-multi-zip-neon-inlined.c",
     False, 'bench_transposec("{label}", fns);',
     dict(sol_path=f"{NEON_INLINED}/x32-transposec-sol-2x2-neon-inlined.c",
          ac_path=TR_CAND)),
    # Upstream ships only the multi-pass 9p8x argmaxpool, which asserts
    # pooling_elements > 9 — it cannot run the harness's 3x3 (pe=9) config,
    # so the harness-faithful section has no upstream lane (Sol is the
    # sanity ref) and a second pe=25 (5x5) section adds the upstream
    # comparison (sol/ac have generic pass loops and handle any pe).
    ("f32-argmaxpool", "ap_fn", 25,
     None,
     None, False, 'bench_argmaxpool("{label}", fns, 9);', {}),
    ("f32-argmaxpool-pe25", "ap_fn", 25,
     "f32-argmaxpool/f32-argmaxpool-9p8x-rvv-u1v.c",
     None, False, 'bench_argmaxpool("{label}", fns, 25);', {}),
    ("f32-maxpool", "mp_fn", 26,
     "f32-maxpool/gen/f32-maxpool-9p-minmax-rvv-u2v.c",
     None, False,
     'bench_maxpool("{label}", fns, 9, (const void*)&p_f32_minmax_wide);', {}),
    ("f32-dwconv-minmax", "dw_fn", 27,
     "f32-dwconv/gen/f32-dwconv-3p8vc-minmax-rvv.c",
     "f32-dwconv/gen/f32-dwconv-3p4c-minmax-neonfma-acc2.c",
     True, 'bench_dwconv("{label}", fns, pk, (const void*)&p_f32_minmax_wide);', {}),
    ("f32-dwconv2d-chw", "c2_fn", 28,
     "f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-rvv-7x1v.c",
     "f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-aarch64-neonfma-3x4.c",
     True, 'bench_dwconv2d("{label}", fns, pk, (const void*)&p_f32_minmax_wide);', {}),
    ("f32-conv-hwc2chw", "hw_fn", 29,
     "f32-conv-hwc2chw/f32-conv-hwc2chw-3x3s2p1c3x2v-rvv-2x2.c",
     "f32-conv-hwc2chw/f32-conv-hwc2chw-3x3s2p1c3x4-aarch64-neonfma-2x2.c",
     True, 'bench_hwc2chw("{label}", fns, pk, (const void*)&p_f32_minmax_wide);', {}),
    ("f32-gemm-minmax", "gm_fn", 30,
     "f32-gemm/gen/f32-gemm-7x4v-minmax-rvv.c",
     "f32-gemm/gen/f32-gemm-1x8-minmax-neonfma-dup-ld64.c",
     True, 'size_t mrs_[IMPL_COUNT] = {{ 7, 4, 4, 1 }};\n'
           '      size_t nrs_[IMPL_COUNT] = {{ 0, 8, 8, 8 }};\n'
           '      bench_gemm("{label}", fns, pk, mrs_, nrs_, (const void*)&p_f32_minmax_wide);', {}),
    ("f32-igemm-minmax", "ig_fn", 31,
     "f32-igemm/gen/f32-igemm-1x4v-minmax-rvv.c",
     "f32-igemm/gen/f32-igemm-1x8-minmax-neonfma-dup-ld64.c",
     True, 'size_t nrs_[IMPL_COUNT] = {{ 0, 8, 8, 8 }};\n'
           '      bench_igemm("{label}", fns, pk, nrs_, (const void*)&p_f32_minmax_wide);', {}),
    # The user-listed NEON file for qd8 is a qc8w IGEMM — a different kernel
    # (algorithm + quantization), so it cannot serve as a comparable lane.
    ("qd8-f32-qc4w-gemm-minmax", "qc_fn", 32,
     "qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4v-minmax-rvv.c",
     None, True, 'size_t nrs_[IMPL_COUNT] = {{ 0, 16, 16, 16 }};\n'
           '      bench_qc4w("{label}", fns, pk, nrs_);', {}),
    ("x32-packw-gemm-goi", "pw_fn", 33,
     "x32-packw/gen/x32-packw-x4v-gemm-goi-rvv-u8.c",
     f"{NEON_INLINED}/x32-packw-x2-gemm-goi-neon-ld2lane-inlined.c",
     False, 'bench_packw("{label}", fns);', {}),
]

STRUCT_C = r'''
/* ===================================================================== */
/* ========== structured kernels: pools/convs/gemms/transpose ========== */
/* ===================================================================== */
#include <string.h>

#define round_down_po2(n, q) ((n) & ~((q) - 1))
#ifndef XNN_UNREACHABLE
#define XNN_UNREACHABLE __builtin_unreachable()
#endif
static inline size_t round_up_po2(size_t n, size_t q) { return (n + q - 1) & ~(q - 1); }

/* NEON zip intrinsics that neon2rvv.h leaves unimplemented (it only lists
 * them in comments). Used by the rewritten kernels in scripts/neon_inlined/
 * and by the conv-hwc2chw aarch64 source. */
static inline float32x4_t vzip1q_f32(float32x4_t a, float32x4_t b) {
    vuint32m1_t vid  = __riscv_vid_v_u32m1(4);
    vuint32m1_t half = __riscv_vsrl_vx_u32m1(vid, 1, 4);
    vbool32_t   odd  = __riscv_vmsne_vx_u32m1_b32(__riscv_vand_vx_u32m1(vid, 1, 4), 0, 4);
    return __riscv_vmerge_vvm_f32m1(__riscv_vrgather_vv_f32m1(a, half, 4),
                                    __riscv_vrgather_vv_f32m1(b, half, 4), odd, 4);
}
static inline float32x4_t vzip2q_f32(float32x4_t a, float32x4_t b) {
    vuint32m1_t vid  = __riscv_vid_v_u32m1(4);
    vuint32m1_t half = __riscv_vadd_vx_u32m1(__riscv_vsrl_vx_u32m1(vid, 1, 4), 2, 4);
    vbool32_t   odd  = __riscv_vmsne_vx_u32m1_b32(__riscv_vand_vx_u32m1(vid, 1, 4), 0, 4);
    return __riscv_vmerge_vvm_f32m1(__riscv_vrgather_vv_f32m1(a, half, 4),
                                    __riscv_vrgather_vv_f32m1(b, half, 4), odd, 4);
}
static inline uint32x2_t vzip1_u32(uint32x2_t a, uint32x2_t b) {
    vbool32_t odd = __riscv_vmsne_vx_u32m1_b32(
        __riscv_vand_vx_u32m1(__riscv_vid_v_u32m1(2), 1, 2), 0, 2);
    return __riscv_vmerge_vvm_u32m1(__riscv_vrgather_vx_u32m1(a, 0, 2),
                                    __riscv_vrgather_vx_u32m1(b, 0, 2), odd, 2);
}
static inline uint32x2_t vzip2_u32(uint32x2_t a, uint32x2_t b) {
    vbool32_t odd = __riscv_vmsne_vx_u32m1_b32(
        __riscv_vand_vx_u32m1(__riscv_vid_v_u32m1(2), 1, 2), 0, 2);
    return __riscv_vmerge_vvm_u32m1(__riscv_vrgather_vx_u32m1(a, 1, 2),
                                    __riscv_vrgather_vx_u32m1(b, 1, 2), odd, 2);
}

/* param structs (layouts match xnnpack/microparams.h) */
struct xnn_f32_minmax_params { struct { float min; float max; } scalar; };
struct xnn_f32_qc4w_minmax_params { struct { float min; float max; int32_t kernel_zero_point; } scalar; };
struct xnn_qd8_quantization_params { int32_t zero_point; float inv_scale; };
static const struct xnn_f32_minmax_params p_f32_minmax_wide =
    { .scalar = { -__builtin_inff(), __builtin_inff() } };
static const struct xnn_f32_qc4w_minmax_params p_qc4w_minmax =
    { .scalar = { -__builtin_inff(), __builtin_inff(), 8 } };
static const struct xnn_qd8_quantization_params p_qd8_quant[1] =
    { { .zero_point = 3, .inv_scale = 0.5f } };

/* candidate pack_weights contract (HARNESS_MAIN_MULTI_PACK): canonical
 * weights in, lane layout out; units are floats or 4-byte words. */
typedef void (*packfn_t)(const void* src, size_t src_n, void* dst, size_t dst_cap);

static float rand_f1s(void) { return ((int)rand_byte() - 128) / 128.0f; }
static void fillv_f32(float* p, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) p[i] = rand_f1s() * scale;
}

/* Retile a [n_rows × columns] tiled-f32 weight blob (row 0 = bias, then k
 * rows; column tiles of nr_src lanes) into nr_dst-wide tiles, zero-padding
 * the tail tile. Layout per tile: n_rows consecutive rows of nr lanes. */
static void retile_f32(const float* src, float* dst, size_t n_cols,
                       size_t nr_src, size_t nr_dst, size_t n_rows) {
    size_t n_tiles_d = (n_cols + nr_dst - 1) / nr_dst;
    for (size_t t = 0; t < n_tiles_d; t++)
        for (size_t r = 0; r < n_rows; r++)
            for (size_t l = 0; l < nr_dst; l++) {
                size_t j = t * nr_dst + l;
                dst[(t * n_rows + r) * nr_dst + l] = (j < n_cols)
                    ? src[((j / nr_src) * n_rows + r) * nr_src + (j % nr_src)]
                    : 0.0f;
            }
}

/* ---------------- x32-transposec ---------------- */
typedef void (*tr_fn)(const uint32_t*, uint32_t*, size_t, size_t, size_t, size_t);
static const size_t tr_sizes[] = { 128, 256, 512 };
#define TR_MAX 512
static uint32_t tr_in  [TR_MAX * TR_MAX] __attribute__((aligned(64)));
static uint32_t tr_out [TR_MAX * TR_MAX] __attribute__((aligned(64)));
static uint32_t tr_gold[TR_MAX * TR_MAX] __attribute__((aligned(64)));

static void bench_transposec(const char* label, tr_fn fns[IMPL_COUNT]) {
    print_header(label);
    for (size_t si = 0; si < 3; si++) {
        size_t sz = tr_sizes[si], elems = sz * sz;
        reset_rand();
        fill_random((uint8_t*)tr_in, elems * 4);
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            memset(tr_gold, 0, elems * 4);
            fns[ref](tr_in, tr_gold, sz * 4, sz * 4, sz, sz);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                memset(tr_out, 0, elems * 4);
                fns[k](tr_in, tr_out, sz * 4, sz * 4, sz, sz);
                check_against_gold(label, impl_names[k], elems, tr_out, tr_gold,
                                   elems, KIND_U32, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(label, impl_names[k], elems, elems,
                      fns[k](tr_in, tr_out, sz * 4, sz * 4, sz, sz));
        }
    }
}

/* ---------------- pools (maxpool / argmaxpool), 3x3 -> 9 elements -------- */
#define PL_MAX_KE 25   /* 3x3 (harness) and 5x5 (argmaxpool upstream needs >9) */
static const size_t pool_cfgs[][2] = { {64, 32}, {256, 64} };  /* {pixels, channels} */
static float pl_in[256 * PL_MAX_KE * 64] __attribute__((aligned(64)));
static const float* pl_ind[256 * PL_MAX_KE];
static float    pl_out [256 * 64] __attribute__((aligned(64)));
static float    pl_gold[256 * 64] __attribute__((aligned(64)));
static uint32_t pl_idx [256 * 64] __attribute__((aligned(64)));
static uint32_t pl_idxg[256 * 64] __attribute__((aligned(64)));

static void pool_init(size_t pixels, size_t channels, size_t ke) {
    reset_rand();
    fillv_f32(pl_in, pixels * ke * channels, 1.0f);
    for (size_t p = 0; p < pixels; p++)
        for (size_t k = 0; k < ke; k++)
            pl_ind[p * ke + k] = &pl_in[(p * ke + k) * channels];
}

typedef void (*mp_fn)(size_t, size_t, size_t, const float**, size_t, size_t,
                      float*, size_t, size_t, const void*);
static void bench_maxpool(const char* label, mp_fn fns[IMPL_COUNT], size_t ke, const void* params) {
    print_header(label);
    for (size_t si = 0; si < 2; si++) {
        size_t pixels = pool_cfgs[si][0], channels = pool_cfgs[si][1];
        size_t elems = pixels * channels;
        size_t in_inc = ke * sizeof(const float*), out_inc = channels * sizeof(float);
        pool_init(pixels, channels, ke);
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            fns[ref](pixels, ke, channels, pl_ind, 0, 0, pl_gold, in_inc, out_inc, params);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                fns[k](pixels, ke, channels, pl_ind, 0, 0, pl_out, in_inc, out_inc, params);
                check_against_gold(label, impl_names[k], elems, pl_out, pl_gold,
                                   elems, KIND_F32, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(label, impl_names[k], elems, elems,
                      fns[k](pixels, ke, channels, pl_ind, 0, 0, pl_out, in_inc, out_inc, params));
        }
    }
}

typedef void (*ap_fn)(size_t, size_t, size_t, const float**, size_t, size_t,
                      float*, uint32_t*, size_t, size_t, size_t);
static void bench_argmaxpool(const char* label, ap_fn fns[IMPL_COUNT], size_t ke) {
    print_header(label);
    for (size_t si = 0; si < 2; si++) {
        size_t pixels = pool_cfgs[si][0], channels = pool_cfgs[si][1];
        size_t elems = pixels * channels;
        size_t in_inc = ke * sizeof(const float*);
        size_t out_inc = channels * sizeof(float), idx_inc = channels * sizeof(uint32_t);
        pool_init(pixels, channels, ke);
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            memset(pl_idxg, 0, elems * 4);
            fns[ref](pixels, ke, channels, pl_ind, 0, 0, pl_gold, pl_idxg, in_inc, out_inc, idx_inc);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                memset(pl_idx, 0, elems * 4);
                fns[k](pixels, ke, channels, pl_ind, 0, 0, pl_out, pl_idx, in_inc, out_inc, idx_inc);
                check_against_gold(label, impl_names[k], elems, pl_out, pl_gold,
                                   elems, KIND_F32, impl_names[ref]);
                check_against_gold(label, impl_names[k], elems, pl_idx, pl_idxg,
                                   elems, KIND_U32, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(label, impl_names[k], elems, elems,
                      fns[k](pixels, ke, channels, pl_ind, 0, 0, pl_out, pl_idx, in_inc, out_inc, idx_inc));
        }
    }
}

/* ---------------- f32-dwconv-minmax (3-tap, indirection) ---------------- */
/* Canonical (sol/ac) weights: contiguous e32m4 vl-blocks of
 * {bias[vl], k0[vl], k1[vl], k2[vl]}. Upstream 3p8vc: vlmax(e32m8)-strided
 * zero-padded blocks. NEON 3p4c: 4-lane tiles. output_increment = 0 (all
 * lanes advance the output by `channels` internally) -> dense c*w output. */
typedef void (*dw_fn)(size_t, size_t, const float**, const float*, float*,
                      intptr_t, size_t, size_t, size_t, const float*, const void*);
static const size_t dw_cfgs[][2] = { {32, 112}, {128, 56} };  /* {channels, width} */
#define DW_MAX_CH 128
#define DW_MAX_W  112
static float dw_in[DW_MAX_W * 3 * DW_MAX_CH] __attribute__((aligned(64)));
static const float* dw_ind[DW_MAX_W * 3];
static float dw_bias[DW_MAX_CH];
static float dw_k[3 * DW_MAX_CH];
static float dw_w_canon[4 * DW_MAX_CH] __attribute__((aligned(64)));
static float dw_w_up[2048] __attribute__((aligned(64)));
static float dw_w_nv[4 * DW_MAX_CH] __attribute__((aligned(64)));
static float dw_w_lane_buf[2][16 * DW_MAX_CH] __attribute__((aligned(64)));
static float dw_zero[DW_MAX_CH] __attribute__((aligned(64)));  /* stays zero */
static float dw_out [DW_MAX_CH * DW_MAX_W] __attribute__((aligned(64)));
static float dw_gold[DW_MAX_CH * DW_MAX_W] __attribute__((aligned(64)));

static void dw_pack_canon(const float* bias, const float* k, size_t c, float* dst) {
    size_t off = 0;
    for (size_t base = 0; base < c; ) {
        size_t vl = __riscv_vsetvl_e32m4(c - base);
        for (size_t i = 0; i < vl; i++) dst[off + i] = bias[base + i];
        off += vl;
        for (int t = 0; t < 3; t++) {
            for (size_t i = 0; i < vl; i++) dst[off + i] = k[t * c + base + i];
            off += vl;
        }
        base += vl;
    }
}
static void dw_pack_8vc(const float* bias, const float* k, size_t c, float* dst) {
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    size_t off = 0;
    for (size_t base = 0; base < c; base += vlmax) {
        size_t vl = (c - base < vlmax) ? (c - base) : vlmax;
        for (size_t i = 0; i < vlmax; i++) dst[off + i] = (i < vl) ? bias[base + i] : 0.0f;
        off += vlmax;
        for (int t = 0; t < 3; t++) {
            for (size_t i = 0; i < vlmax; i++) dst[off + i] = (i < vl) ? k[t * c + base + i] : 0.0f;
            off += vlmax;
        }
    }
}
static void dw_pack_4c(const float* bias, const float* k, size_t c, float* dst) {
    size_t off = 0;
    for (size_t base = 0; base < c; base += 4) {   /* configs keep c % 4 == 0 */
        for (size_t i = 0; i < 4; i++) dst[off++] = bias[base + i];
        for (int t = 0; t < 3; t++)
            for (size_t i = 0; i < 4; i++) dst[off++] = k[t * c + base + i];
    }
}

/* Autovec dwconv (scalar 3p1c) wants per-channel {bias,k0,k1,k2} (cr=1).
 * Reverse the canonical e32m4 vl-block layout (dw_pack_canon: contiguous
 * blocks of {bias[vl],k0[vl],k1[vl],k2[vl]}) into that. packfn_t contract:
 * src = canonical (4*c floats), dst = cr1 buffer. */
static void av_dwconv_pack_cr1(const void* src_v, size_t src_n,
                               void* dst_v, size_t dst_cap) {
    const float* src = (const float*)src_v;
    float* dst = (float*)dst_v;
    (void)dst_cap;
    size_t c = src_n / 4;
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    size_t off = 0;
    for (size_t base = 0; base < c; base += vlmax) {
        size_t vl = (c - base < vlmax) ? (c - base) : vlmax;
        for (size_t i = 0; i < vl; i++) {
            size_t ch = base + i;
            dst[4 * ch + 0] = src[off + 0 * vl + i];
            dst[4 * ch + 1] = src[off + 1 * vl + i];
            dst[4 * ch + 2] = src[off + 2 * vl + i];
            dst[4 * ch + 3] = src[off + 3 * vl + i];
        }
        off += 4 * vl;
    }
}

static void bench_dwconv(const char* label, dw_fn fns[IMPL_COUNT],
                         packfn_t pk[IMPL_COUNT], const void* params) {
    print_header(label);
    for (size_t si = 0; si < 2; si++) {
        size_t c = dw_cfgs[si][0], w = dw_cfgs[si][1];
        size_t elems = c * w, canon_n = 4 * c;
        reset_rand();
        fillv_f32(dw_in, w * 3 * c, 1.0f);
        fillv_f32(dw_bias, c, 0.5f);
        fillv_f32(dw_k, 3 * c, 0.5f);
        for (size_t col = 0; col < w; col++)
            for (size_t t = 0; t < 3; t++)
                dw_ind[col * 3 + t] = &dw_in[(col * 3 + t) * c];
        dw_pack_canon(dw_bias, dw_k, c, dw_w_canon);
        dw_pack_8vc(dw_bias, dw_k, c, dw_w_up);
        dw_pack_4c(dw_bias, dw_k, c, dw_w_nv);
        const float* w_lane[IMPL_COUNT] = { dw_w_up, dw_w_lane_buf[0], dw_w_lane_buf[1], dw_w_nv };
        for (int k = 1; k <= 2; k++) {
            if (!fns[k]) continue;
            if (pk[k]) pk[k](dw_w_canon, canon_n, dw_w_lane_buf[k - 1], 4 * canon_n);
            else memcpy(dw_w_lane_buf[k - 1], dw_w_canon, canon_n * sizeof(float));
        }
        intptr_t in_stride = (intptr_t)(3 * sizeof(const float*));
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            fns[ref](c, w, dw_ind, w_lane[ref], dw_gold, in_stride, 0, 0, 0, dw_zero, params);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                fns[k](c, w, dw_ind, w_lane[k], dw_out, in_stride, 0, 0, 0, dw_zero, params);
                check_against_gold(label, impl_names[k], elems, dw_out, dw_gold,
                                   elems, KIND_F32, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(label, impl_names[k], elems, elems,
                      fns[k](c, w, dw_ind, w_lane[k], dw_out, in_stride, 0, 0, 0, dw_zero, params));
        }
    }
}

/* ---------------- f32-dwconv2d-chw 3x3p1 (single plane) ---------------- */
/* All lanes read the same 10-float layout (bias + 9 taps); padded to 12 for
 * the NEON variant's OOB vld1q at weights+8. */
typedef void (*c2_fn)(size_t, size_t, const float*, const float*, const float*,
                      float*, uint32_t, const void*);
static const size_t c2_cfgs[][2] = { {112, 112}, {56, 56} };  /* {h, w elems} */
static float c2_in[112 * 112] __attribute__((aligned(64)));
static float c2_w_canon[12] __attribute__((aligned(64)));
static float c2_w_lane_buf[2][48] __attribute__((aligned(64)));
static float c2_zero[112 + 16] __attribute__((aligned(64)));   /* stays zero */
static float c2_out [112 * 112] __attribute__((aligned(64)));
static float c2_gold[112 * 112] __attribute__((aligned(64)));

static void bench_dwconv2d(const char* label, c2_fn fns[IMPL_COUNT],
                           packfn_t pk[IMPL_COUNT], const void* params) {
    print_header(label);
    for (size_t si = 0; si < 2; si++) {
        size_t h = c2_cfgs[si][0], w = c2_cfgs[si][1];
        size_t elems = h * w;
        reset_rand();
        fillv_f32(c2_in, h * w, 1.0f);
        fillv_f32(c2_w_canon, 10, 0.5f);
        c2_w_canon[10] = c2_w_canon[11] = 0.0f;
        const float* w_lane[IMPL_COUNT] = { c2_w_canon, c2_w_lane_buf[0], c2_w_lane_buf[1], c2_w_canon };
        for (int k = 1; k <= 2; k++) {
            if (!fns[k]) continue;
            memset(c2_w_lane_buf[k - 1], 0, sizeof(c2_w_lane_buf[0]));
            if (pk[k]) pk[k](c2_w_canon, 10, c2_w_lane_buf[k - 1], 48);
            else memcpy(c2_w_lane_buf[k - 1], c2_w_canon, 12 * sizeof(float));
        }
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            fns[ref](h, w * 4, c2_in, w_lane[ref], c2_zero, c2_gold, 1, params);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                fns[k](h, w * 4, c2_in, w_lane[k], c2_zero, c2_out, 1, params);
                check_against_gold(label, impl_names[k], elems, c2_out, c2_gold,
                                   elems, KIND_F32, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(label, impl_names[k], elems, elems,
                      fns[k](h, w * 4, c2_in, w_lane[k], c2_zero, c2_out, 1, params));
        }
    }
}

/* ---------------- f32-conv-hwc2chw 3x3s2p1c3 (first layer) ---------------- */
/* Canonical: oc-tile=4, 28 rows (bias + 27 taps) x 4 lanes per tile.
 * Upstream c3x2v expects oc-tile = vsetvlmax(e32m2), same row order. */
typedef void (*hw_fn)(size_t, size_t, size_t, size_t, const float*, const float*,
                      const float*, float*, size_t, size_t, size_t, size_t, const void*);
#define HW_OC 16
static float hw_in[224 * 224 * 3] __attribute__((aligned(64)));
static float hw_zero[224 * 3] __attribute__((aligned(64)));    /* stays zero */
static float hw_w_canon[448] __attribute__((aligned(64)));
static float hw_w_up[2048] __attribute__((aligned(64)));
static float hw_w_lane_buf[2][4 * 448] __attribute__((aligned(64)));
static float hw_out [HW_OC * 112 * 112] __attribute__((aligned(64)));
static float hw_gold[HW_OC * 112 * 112] __attribute__((aligned(64)));

static void bench_hwc2chw(const char* label, hw_fn fns[IMPL_COUNT],
                          packfn_t pk[IMPL_COUNT], const void* params) {
    print_header(label);
    size_t elems = HW_OC * 112 * 112;
    reset_rand();
    fillv_f32(hw_in, 224 * 224 * 3, 1.0f);
    fillv_f32(hw_w_canon, 448, 0.5f);
    retile_f32(hw_w_canon, hw_w_up, HW_OC, 4, __riscv_vsetvlmax_e32m2(), 28);
    const float* w_lane[IMPL_COUNT] = { hw_w_up, hw_w_lane_buf[0], hw_w_lane_buf[1], hw_w_canon };
    for (int k = 1; k <= 2; k++) {
        if (!fns[k]) continue;
        if (pk[k]) pk[k](hw_w_canon, 448, hw_w_lane_buf[k - 1], 4 * 448);
        else memcpy(hw_w_lane_buf[k - 1], hw_w_canon, 448 * sizeof(float));
    }
    int ref = sanity_ref((void**)fns);
    if (!PERF_NO_SANITY && ref >= 0) {
        fns[ref](224, 224, 0, 112, hw_in, hw_zero, w_lane[ref], hw_gold,
                 1, HW_OC, 112 * 4, 112 * 112 * 4, params);
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k] || k == ref) continue;
            fns[k](224, 224, 0, 112, hw_in, hw_zero, w_lane[k], hw_out,
                   1, HW_OC, 112 * 4, 112 * 112 * 4, params);
            check_against_gold(label, impl_names[k], elems, hw_out, hw_gold,
                               elems, KIND_F32, impl_names[ref]);
        }
    }
    for (int k = 0; k < IMPL_COUNT; k++) {
        if (!fns[k]) continue;
        TIME_CELL(label, impl_names[k], elems, elems,
                  fns[k](224, 224, 0, 112, hw_in, hw_zero, w_lane[k], hw_out,
                         1, HW_OC, 112 * 4, 112 * 112 * 4, params));
    }
}

/* ---------------- f32-gemm-minmax (full layer, per-lane MR/NR) ----------- */
/* M = 28 = lcm(7,4): upstream 7x4v runs 4 calls of mr=7, sol/ac 7 of mr=4,
 * NEON 1x8 28 of mr=1 — each lane tiles M with its own MR and reads weights
 * in its own NR tiling (canonical nr=8; upstream nr=vsetvlmax(e32m4)). */
typedef void (*gm_fn)(size_t, size_t, size_t, const float*, size_t, const float*,
                      float*, size_t, size_t, const void*);
#define GM_M 28
static const size_t gm_cfgs[][2] = { {64, 128}, {256, 64} };  /* {nc, kc_elems} */
static float gm_a[GM_M * 128] __attribute__((aligned(64)));
static float gm_w_canon[16640] __attribute__((aligned(64)));
static float gm_w_up[16640] __attribute__((aligned(64)));
static float gm_w_lane_buf[2][4 * 16640] __attribute__((aligned(64)));
static float gm_out [GM_M * 256] __attribute__((aligned(64)));
static float gm_gold[GM_M * 256] __attribute__((aligned(64)));

static void gm_call(gm_fn fn, size_t lane_mr, size_t nc, size_t kc_e,
                    const float* w, float* cbuf, size_t cn_stride, const void* params) {
    size_t kc_bytes = kc_e * sizeof(float);
    for (size_t m = 0; m < GM_M; m += lane_mr) {
        size_t mr = (GM_M - m < lane_mr) ? (GM_M - m) : lane_mr;
        fn(mr, nc, kc_bytes, gm_a + m * kc_e, kc_bytes, w,
           cbuf + m * nc, nc * sizeof(float), cn_stride, params);
    }
}

static void bench_gemm(const char* label, gm_fn fns[IMPL_COUNT], packfn_t pk[IMPL_COUNT],
                       const size_t mrs[IMPL_COUNT], const size_t nrs[IMPL_COUNT],
                       const void* params) {
    print_header(label);
    for (size_t si = 0; si < 2; si++) {
        size_t nc = gm_cfgs[si][0], kc_e = gm_cfgs[si][1];
        size_t elems = GM_M * nc, canon_n = nc * (kc_e + 1);
        reset_rand();
        fillv_f32(gm_a, GM_M * kc_e, 1.0f);
        fillv_f32(gm_w_canon, canon_n, 0.5f);
        size_t nr_up = __riscv_vsetvlmax_e32m4();
        retile_f32(gm_w_canon, gm_w_up, nc, 8, nr_up, kc_e + 1);
        const float* w_lane[IMPL_COUNT] = { gm_w_up, gm_w_lane_buf[0], gm_w_lane_buf[1], gm_w_canon };
        size_t cn_lane[IMPL_COUNT] = { nr_up * 4, 8 * 4, 8 * 4, 8 * 4 };
        for (int k = 1; k <= 2; k++) {
            if (!fns[k]) continue;
            size_t nrk = nrs[k] ? nrs[k] : nr_up;   /* 0 = LMUL4 vlmax */
            cn_lane[k] = nrk * 4;
            if (pk[k]) pk[k](gm_w_canon, canon_n, gm_w_lane_buf[k - 1], 4 * canon_n);
            else if (nrk != 8) retile_f32(gm_w_canon, gm_w_lane_buf[k - 1], nc, 8, nrk, kc_e + 1);
            else memcpy(gm_w_lane_buf[k - 1], gm_w_canon, canon_n * sizeof(float));
        }
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            gm_call(fns[ref], mrs[ref], nc, kc_e, w_lane[ref], gm_gold, cn_lane[ref], params);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                gm_call(fns[k], mrs[k], nc, kc_e, w_lane[k], gm_out, cn_lane[k], params);
                check_against_gold(label, impl_names[k], elems, gm_out, gm_gold,
                                   elems, KIND_F32, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(label, impl_names[k], elems, elems,
                      gm_call(fns[k], mrs[k], nc, kc_e, w_lane[k], gm_out, cn_lane[k], params));
        }
    }
}

/* ---------------- f32-igemm-minmax (mr=1, ks=9 indirection) -------------- */
typedef void (*ig_fn)(size_t, size_t, size_t, size_t, const float**, const float*,
                      float*, size_t, size_t, size_t, const float*, const void*);
#define IG_KS 9
static const size_t ig_cfgs[][2] = { {128, 64}, {64, 128} };  /* {nc, kc_elems} */
static float ig_rows[IG_KS * 128] __attribute__((aligned(64)));
static float ig_zero[128] __attribute__((aligned(64)));        /* stays zero */
static const float* ig_ind[IG_KS];
static float ig_w_canon[73856] __attribute__((aligned(64)));
static float ig_w_up[73856] __attribute__((aligned(64)));
static float ig_w_lane_buf[2][4 * 73856] __attribute__((aligned(64)));
static float ig_out [128] __attribute__((aligned(64)));
static float ig_gold[128] __attribute__((aligned(64)));

static void bench_igemm(const char* label, ig_fn fns[IMPL_COUNT],
                        packfn_t pk[IMPL_COUNT], const size_t nrs[IMPL_COUNT],
                        const void* params) {
    print_header(label);
    for (size_t si = 0; si < 2; si++) {
        size_t nc = ig_cfgs[si][0], kc_e = ig_cfgs[si][1];
        size_t kc_bytes = kc_e * sizeof(float);
        size_t n_rows = 1 + IG_KS * kc_e, canon_n = nc * n_rows;
        reset_rand();
        fillv_f32(ig_rows, IG_KS * kc_e, 1.0f);
        fillv_f32(ig_w_canon, canon_n, 0.5f);
        for (size_t s = 0; s < IG_KS; s++) ig_ind[s] = &ig_rows[s * kc_e];
        ig_ind[0] = ig_zero;   /* padding path, as in the harness */
        size_t nr_up = __riscv_vsetvlmax_e32m4();
        retile_f32(ig_w_canon, ig_w_up, nc, 8, nr_up, n_rows);
        const float* w_lane[IMPL_COUNT] = { ig_w_up, ig_w_lane_buf[0], ig_w_lane_buf[1], ig_w_canon };
        size_t cn_lane[IMPL_COUNT] = { nr_up * 4, 8 * 4, 8 * 4, 8 * 4 };
        for (int k = 1; k <= 2; k++) {
            if (!fns[k]) continue;
            size_t nrk = nrs[k] ? nrs[k] : nr_up;   /* 0 = LMUL4 vlmax */
            cn_lane[k] = nrk * 4;
            if (pk[k]) pk[k](ig_w_canon, canon_n, ig_w_lane_buf[k - 1], 4 * canon_n);
            else if (nrk != 8) retile_f32(ig_w_canon, ig_w_lane_buf[k - 1], nc, 8, nrk, n_rows);
            else memcpy(ig_w_lane_buf[k - 1], ig_w_canon, canon_n * sizeof(float));
        }
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            fns[ref](1, nc, kc_bytes, IG_KS * sizeof(void*), ig_ind, w_lane[ref],
                     ig_gold, nc * 4, cn_lane[ref], 0, ig_zero, params);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                fns[k](1, nc, kc_bytes, IG_KS * sizeof(void*), ig_ind, w_lane[k],
                       ig_out, nc * 4, cn_lane[k], 0, ig_zero, params);
                check_against_gold(label, impl_names[k], nc, ig_out, ig_gold,
                                   nc, KIND_F32, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(label, impl_names[k], nc, nc,
                      fns[k](1, nc, kc_bytes, IG_KS * sizeof(void*), ig_ind, w_lane[k],
                             ig_out, nc * 4, cn_lane[k], 0, ig_zero, params));
        }
    }
}

/* ---------------- qd8-f32-qc4w-gemm-minmax (mr=1) ---------------- */
/* Mixed-type tile: [i32 ksum[nr] | (kc/2) rows of nr nibble-pair bytes |
 * f32 scale[nr] | f32 bias[nr]]. Canonical nr=16; upstream nr=vsetvlmax(e32m4).
 * Random-but-bounded canonical values; both lanes read the same per-column
 * values through their own tiling, so outputs must agree. */
typedef void (*qc_fn)(size_t, size_t, size_t, const int8_t*, size_t, const void*,
                      float*, size_t, size_t, const void*, const void*);
static const size_t qc_cfgs[][2] = { {128, 256}, {256, 128} };  /* {nc, kc} */
static int8_t  qc_a[256] __attribute__((aligned(64)));
static uint8_t qc_w_canon[4864 * 4] __attribute__((aligned(64)));
static uint8_t qc_w_up[4864 * 4] __attribute__((aligned(64)));
static uint8_t qc_w_lane_buf[2][4 * 4864 * 4] __attribute__((aligned(64)));
static float qc_out [256] __attribute__((aligned(64)));
static float qc_gold[256] __attribute__((aligned(64)));

static size_t qc_tile_bytes(size_t nr, size_t kc) { return nr * 4 + (kc / 2) * nr + nr * 8; }

static void qc_build_canon(size_t nc, size_t kc) {
    size_t kp = kc / 2, tb = qc_tile_bytes(16, kc);
    for (size_t t = 0; t < nc / 16; t++) {
        uint8_t* p = qc_w_canon + t * tb;
        int32_t* ksum = (int32_t*)p;
        for (int l = 0; l < 16; l++) ksum[l] = (int32_t)rand_byte() * 16;
        uint8_t* nib = p + 64;
        for (size_t i = 0; i < kp * 16; i++) nib[i] = rand_byte();
        float* scale = (float*)(p + 64 + kp * 16);
        float* bias  = (float*)(p + 64 + kp * 16 + 64);
        for (int l = 0; l < 16; l++) scale[l] = 0.5f + (float)rand_byte() / 512.0f;
        for (int l = 0; l < 16; l++) bias[l]  = rand_f1s();
    }
}

static void qc_retile(const uint8_t* src, uint8_t* dst, size_t nc, size_t kc,
                      size_t nr_s, size_t nr_d) {
    size_t kp = kc / 2;
    size_t st = qc_tile_bytes(nr_s, kc), dt = qc_tile_bytes(nr_d, kc);
    memset(dst, 0, ((nc + nr_d - 1) / nr_d) * dt);
    for (size_t j = 0; j < nc; j++) {
        const uint8_t* s = src + (j / nr_s) * st;
        uint8_t* d = dst + (j / nr_d) * dt;
        size_t ls = j % nr_s, ld = j % nr_d;
        memcpy(d + ld * 4, s + ls * 4, 4);                              /* ksum */
        for (size_t p = 0; p < kp; p++)                                  /* nibbles */
            d[nr_d * 4 + p * nr_d + ld] = s[nr_s * 4 + p * nr_s + ls];
        memcpy(d + nr_d * 4 + kp * nr_d + ld * 4,                        /* scale */
               s + nr_s * 4 + kp * nr_s + ls * 4, 4);
        memcpy(d + nr_d * 4 + kp * nr_d + nr_d * 4 + ld * 4,             /* bias */
               s + nr_s * 4 + kp * nr_s + nr_s * 4 + ls * 4, 4);
    }
}

static void bench_qc4w(const char* label, qc_fn fns[IMPL_COUNT], packfn_t pk[IMPL_COUNT],
                       const size_t nrs[IMPL_COUNT]) {
    print_header(label);
    for (size_t si = 0; si < 2; si++) {
        size_t nc = qc_cfgs[si][0], kc = qc_cfgs[si][1];
        size_t canon_words = (nc / 16) * qc_tile_bytes(16, kc) / 4;
        reset_rand();
        for (size_t i = 0; i < kc; i++) qc_a[i] = (int8_t)rand_byte();
        qc_build_canon(nc, kc);
        size_t nr_up = __riscv_vsetvlmax_e32m4();
        qc_retile(qc_w_canon, qc_w_up, nc, kc, 16, nr_up);
        const uint8_t* w_lane[IMPL_COUNT] = { qc_w_up, qc_w_lane_buf[0], qc_w_lane_buf[1], NULL };
        size_t cn_lane[IMPL_COUNT] = { nr_up * 4, 16 * 4, 16 * 4, 16 * 4 };
        for (int k = 1; k <= 2; k++) {
            if (!fns[k]) continue;
            size_t nrk = nrs[k] ? nrs[k] : nr_up;   /* 0 = LMUL4 vlmax */
            cn_lane[k] = nrk * 4;
            if (pk[k]) pk[k](qc_w_canon, canon_words, qc_w_lane_buf[k - 1], 4 * canon_words);
            else if (nrk != 16) qc_retile(qc_w_canon, qc_w_lane_buf[k - 1], nc, kc, 16, nrk);
            else memcpy(qc_w_lane_buf[k - 1], qc_w_canon, canon_words * 4);
        }
        int ref = sanity_ref((void**)fns);
        if (!PERF_NO_SANITY && ref >= 0) {
            fns[ref](1, nc, kc, qc_a, kc, w_lane[ref], qc_gold, nc * 4, cn_lane[ref],
                     &p_qc4w_minmax, p_qd8_quant);
            for (int k = 0; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                fns[k](1, nc, kc, qc_a, kc, w_lane[k], qc_out, nc * 4, cn_lane[k],
                       &p_qc4w_minmax, p_qd8_quant);
                check_against_gold(label, impl_names[k], nc, qc_out, qc_gold,
                                   nc, KIND_F32, impl_names[ref]);
            }
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(label, impl_names[k], nc, nc,
                      fns[k](1, nc, kc, qc_a, kc, w_lane[k], qc_out, nc * 4, cn_lane[k],
                             &p_qc4w_minmax, p_qd8_quant));
        }
    }
}

/* ---------------- x32-packw-gemm-goi ---------------- */
/* nr is a CALL argument: upstream x4v hard-asserts nr == vsetvlmax(e32m4);
 * sol/ac/NEON run nr=2. Output layouts therefore differ by design — sanity
 * compares only the nr=2 lanes against each other (bit-exact); the upstream
 * lane is timed only. Per-element denominator = input words g*nc*kc for all
 * lanes so the cyc/elem column stays comparable. */
typedef void (*pw_fn)(size_t, size_t, size_t, size_t, size_t, size_t,
                      const uint32_t*, const uint32_t*, const void*, uint32_t*,
                      size_t, const void*);
static const size_t pw_cfgs[][3] = { {1, 64, 128}, {2, 32, 64} };  /* {g, nc, kc} */
static uint32_t pw_w[8192] __attribute__((aligned(64)));
static uint32_t pw_bias[128] __attribute__((aligned(64)));
static uint32_t pw_out [16640] __attribute__((aligned(64)));
static uint32_t pw_gold[16640] __attribute__((aligned(64)));

static void bench_packw(const char* label, pw_fn fns[IMPL_COUNT]) {
    print_header(label);
    size_t nr_lane[IMPL_COUNT] = { __riscv_vsetvlmax_e32m4(), 2, 2, 2 };
    for (size_t si = 0; si < 2; si++) {
        size_t g = pw_cfgs[si][0], nc = pw_cfgs[si][1], kc = pw_cfgs[si][2];
        size_t in_words = g * nc * kc;
        size_t out2_words = g * (nc / 2) * (2 + 2 * kc);
        reset_rand();
        fill_random((uint8_t*)pw_w, in_words * 4);
        fill_random((uint8_t*)pw_bias, g * nc * 4);
        int ref = -1;   /* sanity ref among the nr=2 lanes only */
        for (int k = 1; k < IMPL_COUNT; k++) if (fns[k]) { ref = k; break; }
        if (!PERF_NO_SANITY && ref >= 0) {
            memset(pw_gold, 0, out2_words * 4);
            fns[ref](g, nc, kc, 2, 1, 1, pw_w, pw_bias, NULL, pw_gold, 0, NULL);
            for (int k = 1; k < IMPL_COUNT; k++) {
                if (!fns[k] || k == ref) continue;
                memset(pw_out, 0, out2_words * 4);
                fns[k](g, nc, kc, 2, 1, 1, pw_w, pw_bias, NULL, pw_out, 0, NULL);
                check_against_gold(label, impl_names[k], in_words, pw_out, pw_gold,
                                   out2_words, KIND_U32, impl_names[ref]);
            }
            if (fns[0])
                printf("  SANITY Upstream  skipped (nr=%zu layout differs by design)\n", nr_lane[0]);
        }
        for (int k = 0; k < IMPL_COUNT; k++) {
            if (!fns[k]) continue;
            TIME_CELL(label, impl_names[k], in_words, in_words,
                      fns[k](g, nc, kc, nr_lane[k], 1, 1, pw_w, pw_bias, NULL, pw_out, 0, NULL));
        }
    }
}
'''


def main():
    parse_selection(sys.argv[1:])
    variant_blocks = []
    runner_calls = []
    gated_calls = []
    gated_ew_calls = []
    gated_nv_calls = []
    gated_struct_calls = []
    gated_av_calls = []
    missing = []
    row_info = {}

    for row_idx, (pid, label, family, out_kind, up_rel, params_expr) in enumerate(KERNELS):
        srcs = {
            "up": load_upstream(up_rel),
            "sol": load_sol(pid) if pid is not None else None,
            "ac": load_autocomp(pid) if pid is not None else None,
        }
        fn_exprs = {}
        for tag, text in srcs.items():
            if text is None:
                fn_exprs[tag] = "NULL"
                if pid is not None:
                    missing.append(f"prob {pid} ({label}): {tag} source missing")
                continue
            code, mainfn, _pk = prep_variant(text, f"{tag}_r{row_idx}")
            if not mainfn:
                fn_exprs[tag] = "NULL"
                missing.append(f"prob {pid} ({label}): {tag} had no function")
                continue
            variant_blocks.append(
                f"/* ======== prob {pid} {label} — {tag} ======== */\n{code}\n")
            fn_exprs[tag] = mainfn

        if family.startswith("f32un"):
            _, fill_mode, kind = family.split(":")
            fam_ty, runner, extra = "f32un_fn", "f32un", f", {fill_mode}, {kind}"
        elif family == "f32bin":
            fam_ty, runner, extra = "f32bin_fn", "f32bin", f", {out_kind}"  # out_kind col = FILL_*
        elif family == "f32cmul":
            fam_ty, runner, extra = "f32bin_fn", "f32cmul", ""
        else:
            fam_ty = {"unary": "unary_fn", "binary": "binary_fn",
                      "rsum": "rsum_fn", "rdsum": "rdsum_fn"}[family]
            runner = family
            extra = f", {out_kind}" if family in ("unary", "binary") else ""
        table = ", ".join(
            (f"({fam_ty}){fn_exprs[t]}" if fn_exprs[t] != "NULL" else "NULL")
            for t in ("up", "sol", "ac")) + ", NULL"
        call = (f'    {{ {fam_ty} fns[IMPL_COUNT] = {{ {table} }};\n'
                f'      run_{runner}("{label}", fns, (const void*){params_expr}{extra}); }}\n')
        (gated_calls if pid in MEASURED_QUANTIZED_PIDS else gated_ew_calls).append(call)
        if pid is not None:
            row_info[pid] = (label, fam_ty, runner, extra, params_expr,
                             fn_exprs["sol"], fn_exprs["up"])

    # ---- separate Neon2RVV sections: "<kernel>-neon2rvv". Each pairs the
    # NEON translation with the kernel's Sol (re-timed) purely so the section
    # has an in-section sanity reference. Main sections above stay 3-lane.
    for pid, rel in NEON_SOURCES.items():
        if pid not in row_info:
            continue
        label, fam_ty, runner, extra, params_expr, sol_expr, _up = row_info[pid]
        text = load_upstream(rel)
        if text is None:
            missing.append(f"prob {pid} ({label}): neon source missing")
            continue
        code, mainfn, _pk = prep_variant(text, f"nv_p{pid}")
        if not mainfn:
            missing.append(f"prob {pid} ({label}): neon source had no function")
            continue
        variant_blocks.append(
            f"/* ======== prob {pid} {label} — neon2rvv ======== */\n{code}\n")
        sol_slot = f"({fam_ty}){sol_expr}" if sol_expr != "NULL" else "NULL"
        table = f"NULL, {sol_slot}, NULL, ({fam_ty}){mainfn}"
        gated_nv_calls.append(
            f'    {{ {fam_ty} fns[IMPL_COUNT] = {{ {table} }};\n'
            f'      run_{runner}("{label}-neon2rvv", fns, (const void*){params_expr}{extra}); }}\n')

    # ---- structured kernels (probs 25-33 + transposec): bespoke runners in
    # STRUCT_C. Main section = Upstream/Sol/Autocomp; where a NEON source
    # exists, a separate "<label>-neon2rvv" section re-times Sol as the
    # in-section sanity reference (same shape as the elementwise sections).
    struct_exprs = {}
    for s_idx, (label, fn_ty, pid, up_rel, neon_rel, has_pk, call_fmt,
                overrides) in enumerate(STRUCT_KERNELS):
        pin = lambda files: next((open(files[k]).read() for k in (label, str(pid)) if k in files), None)
        sol_pin, ac_pin = pin(SEL.sol_files), pin(SEL.autocomp_files)
        lane_nr = next((SEL.lane_nr[k] for k in (label, str(pid)) if k in SEL.lane_nr), None)
        if lane_nr:                                   # --lane-nr: rewrite the nrs_ initializer, one slot per lane
            m_nr = re.search(r'(size_t nrs_\[IMPL_COUNT\] = \{\{)([^}]*)(\}\};)', call_fmt)
            if not m_nr:
                sys.exit(f"--lane-nr: {label} has no per-lane NR")
            vals = [v.strip() for v in m_nr.group(2).split(",")]
            for lane, expr in lane_nr.items():
                vals[LANES.index(lane)] = expr
            call_fmt = call_fmt[:m_nr.start()] + m_nr.group(1) + " " + ", ".join(vals) + " " + m_nr.group(3) + call_fmt[m_nr.end():]
        srcs = {
            "up": load_upstream(up_rel),
            "sol": (sol_pin if sol_pin is not None else open(overrides["sol_path"]).read()
                    if "sol_path" in overrides else load_sol(pid)),
            "ac": (ac_pin if ac_pin is not None else load_candidate_repr(overrides["ac_path"])
                   if "ac_path" in overrides else load_autocomp(pid)),
            "nv": load_upstream(neon_rel) if neon_rel else None,
        }
        fn_exprs, pk_exprs = {}, {}
        for tag, text in srcs.items():
            if text is None:
                fn_exprs[tag], pk_exprs[tag] = "NULL", "NULL"
                if not (tag == "nv" and neon_rel is None) and \
                   not (tag == "up" and up_rel is None):
                    missing.append(f"struct {label}: {tag} source missing")
                continue
            code, mainfn, packfn = prep_variant(text, f"{tag}_x{s_idx}")
            if not mainfn:
                fn_exprs[tag], pk_exprs[tag] = "NULL", "NULL"
                missing.append(f"struct {label}: {tag} had no function")
                continue
            variant_blocks.append(
                f"/* ======== struct {label} — {tag} ======== */\n{code}\n")
            fn_exprs[tag] = mainfn
            pk_exprs[tag] = packfn if packfn else "NULL"

        def slot(tag):
            return f"({fn_ty}){fn_exprs[tag]}" if fn_exprs[tag] != "NULL" else "NULL"

        def pkslot(tag):
            return (f"(packfn_t){pk_exprs[tag]}"
                    if pk_exprs.get(tag, "NULL") != "NULL" else "NULL")

        def emit_struct_call(section_label, tags, bucket):
            table = ", ".join(slot(t) if t else "NULL" for t in tags)
            lines = [f'    {{ {fn_ty} fns[IMPL_COUNT] = {{ {table} }};']
            if has_pk:
                pks = ", ".join(pkslot(t) if t else "NULL" for t in tags)
                lines.append(f'      packfn_t pk[IMPL_COUNT] = {{ {pks} }};')
            lines.append('      ' + call_fmt.format(label=section_label) + ' }')
            bucket.append("\n".join(lines) + "\n")

        emit_struct_call(label, ["up", "sol", "ac", None], gated_struct_calls)
        if fn_exprs.get("nv", "NULL") != "NULL":
            emit_struct_call(f"{label}-neon2rvv", [None, "sol", None, "nv"],
                             gated_nv_calls)
        struct_exprs[label] = dict(fn_exprs, _fn_ty=fn_ty)

    # ---- Auto-vectorization baseline: XNNPACK scalar kernels through GCC's
    # RVV vectorizer. Each block wrapped in a tree-vectorize pragma; autovec
    # fn in the Autocomp slot, Upstream as the in-section sanity reference.
    def av_wrap(code):
        # "O3" is required: Zephyr's effective opt level is -Os, whose cost
        # model rejects auto-vectorization even with the pass enabled. -O3 is
        # also the honest level for an autovec baseline (XNNPACK builds at
        # -O2/-O3; nobody auto-vectorizes at -Os). The hand-written intrinsic
        # lanes are unaffected by opt level (their vector ops are explicit).
        # NOTE: this exact 3-flag list is the empirical optimum. Adding more
        # flags to the optimize pragma (-fno-math-errno, -ffast-math, etc.)
        # paradoxically DE-vectorizes kernels — GCC's optimize-pragma multi-flag
        # handling is fragile. Forcing was tested: no flag set vectorizes the 8
        # hard kernels (vsqrt/vrndne libm, qs8/qu8-rdsum narrowing reductions,
        # transposec permute, argmaxpool index-reduction, dwconv2d-chw &
        # conv-hwc2chw boundary control flow); they are genuine GCC RVV
        # vectorizer-capability limits, which is itself the finding.
        return ('#pragma GCC push_options\n'
                '#pragma GCC optimize("O3","tree-vectorize","tree-loop-vectorize")\n'
                + code +
                '\n#pragma GCC pop_options\n')

    # XNNPACK scalar kernels aren't self-contained (the hand-written RVV/NEON
    # ones are): they pull in math.h bitcast/clamp helpers and, for the
    # *-scalar- variants generated from the SIMD template (vsqrt, maxpool), a
    # size-1 xnn_simd f32 backend. Provide both once, before the autovec blocks.
    AV_PRELUDE = r'''
/* ---- XNNPACK scalar helper prelude (autovec lane only) ---- */
#include <math.h>
#ifndef XNN_DISABLE_UBSAN
#define XNN_DISABLE_UBSAN
#endif
static inline uint32_t float_as_uint32(float f){union{float f;uint32_t u;}v;v.f=f;return v.u;}
static inline float uint32_as_float(uint32_t u){union{uint32_t u;float f;}v;v.u=u;return v.f;}
static inline int32_t math_asr_s32(int32_t x, uint32_t s){return x>>s;}
static inline float math_min_f32(float a,float b){return a<b?a:b;}
static inline float math_max_f32(float a,float b){return a>b?a:b;}
static inline int32_t math_min_s32(int32_t a,int32_t b){return a<b?a:b;}
static inline int32_t math_max_s32(int32_t a,int32_t b){return a>b?a:b;}
static inline uint32_t math_min_u32(uint32_t a,uint32_t b){return a<b?a:b;}
static inline uint32_t math_max_u32(uint32_t a,uint32_t b){return a>b?a:b;}
static inline float math_muladd_f32(float a,float b,float c){return a*b+c;}
/* size-1 scalar xnn_simd f32 backend (vsqrt / maxpool scalar templates) */
typedef float xnn_simd_f32_t;
#define xnn_simd_size_f32 1
#define xnn_simd_bytes_f32 ((size_t)sizeof(float))
static inline xnn_simd_f32_t xnn_loadu_f32(const float* p){return *p;}
static inline void xnn_storeu_f32(float* p, xnn_simd_f32_t v){*p=v;}
static inline xnn_simd_f32_t xnn_set1_f32(float v){return v;}
static inline xnn_simd_f32_t xnn_sqrt_f32(xnn_simd_f32_t v){return sqrtf(v);}
static inline xnn_simd_f32_t xnn_min_f32(xnn_simd_f32_t a, xnn_simd_f32_t b){return a<b?a:b;}
static inline xnn_simd_f32_t xnn_max_f32(xnn_simd_f32_t a, xnn_simd_f32_t b){return a>b?a:b;}
static inline xnn_simd_f32_t xnn_load_tail_f32(const float* p, size_t n){(void)n;return *p;}
static inline xnn_simd_f32_t xnn_load_tail_safe_f32(const float* p, size_t n, float d){(void)n;(void)d;return *p;}
'''
    variant_blocks.append(AV_PRELUDE)

    # elementwise / f16 (row_info pids)
    for pid, rel in AUTOVEC_SOURCES.items():
        if pid not in row_info:
            missing.append(f"autovec pid {pid}: not in row_info")
            continue
        label, fam_ty, runner, extra, params_expr, sol_expr, up_expr = row_info[pid]
        text = load_upstream(rel)
        if text is None:
            missing.append(f"autovec {label}: source missing ({rel})")
            continue
        code, mainfn, _pk = prep_variant(text, f"av_p{pid}")
        if not mainfn:
            missing.append(f"autovec {label}: no function")
            continue
        variant_blocks.append(
            av_wrap(f"/* ======== prob {pid} {label} — autovec ======== */\n{code}\n"))
        if AUTOVEC_ONLY and label not in AUTOVEC_ONLY:
            continue
        if pid in AUTOVEC_NO_SANITY:
            table = f"NULL, NULL, ({fam_ty}){mainfn}, NULL"   # perf-only, no sanity
        else:
            ref_expr, ref_slot = (up_expr, "up") if up_expr != "NULL" else (sol_expr, "sol")
            s0 = f"({fam_ty}){ref_expr}" if ref_expr != "NULL" else "NULL"
            table = (f"{s0}, NULL, ({fam_ty}){mainfn}, NULL" if ref_slot == "up"
                     else f"NULL, {s0}, ({fam_ty}){mainfn}, NULL")
        gated_av_calls.append(
            f'    {{ {fam_ty} fns[IMPL_COUNT] = {{ {table} }};\n'
            f'      run_{runner}("{label}-autovec", fns, (const void*){params_expr}{extra}); }}\n')

    # structured kernels
    for av_idx, (base_label, (rel, call_fmt)) in enumerate(AUTOVEC_STRUCT.items()):
        info = struct_exprs.get(base_label)
        if not info:
            missing.append(f"autovec {base_label}: no struct_exprs")
            continue
        fn_ty = info["_fn_ty"]
        text = load_upstream(rel)
        if text is None:
            missing.append(f"autovec {base_label}: source missing ({rel})")
            continue
        code, mainfn, _pk = prep_variant(text, f"av_s{av_idx}")
        if not mainfn:
            missing.append(f"autovec {base_label}: no function")
            continue
        variant_blocks.append(
            av_wrap(f"/* ======== struct {base_label} — autovec ======== */\n{code}\n"))
        if AUTOVEC_ONLY and base_label not in AUTOVEC_ONLY:
            continue
        up = info.get("up", "NULL")
        up_slot = f"({fn_ty}){up}" if up != "NULL" else "NULL"
        # packw upstream layout differs by design -> need Sol as sanity partner
        if base_label == "x32-packw-gemm-goi":
            sol = info.get("sol", "NULL")
            sol_slot = f"({fn_ty}){sol}" if sol != "NULL" else "NULL"
            table = f"{up_slot}, {sol_slot}, ({fn_ty}){mainfn}, NULL"
        else:
            table = f"{up_slot}, NULL, ({fn_ty}){mainfn}, NULL"
        gated_av_calls.append(
            f'    {{ {fn_ty} fns[IMPL_COUNT] = {{ {table} }};\n'
            f'      ' + call_fmt.format(label=f"{base_label}-autovec") + ' }\n')

    header, struct_c = HEADER, STRUCT_C
    if SEL.active:
        # Only the selected kernels/lanes/sections: filter the calls, drop the variants nothing calls,
        # and turn every RUN_* gate on (what remains is exactly the selection).
        sel = lambda calls: [c for c in (select_call(c) for c in calls) if c]
        gated_calls, gated_ew_calls, gated_struct_calls = map(sel, (gated_calls, gated_ew_calls, gated_struct_calls))
        gated_nv_calls, gated_av_calls, runner_calls = sel(gated_nv_calls), sel(gated_av_calls), sel(runner_calls)
        variant_blocks = [b for b in variant_blocks if select_block(b)]
        header = re.sub(r"(?m)^#define (RUN_\w+) [01][ \t]*$", r"#define \1 1", header)
        if "Neon2RVV" not in SEL.lanes:
            # neon2rvv.h and the NEON zip shims only serve the Neon2RVV lane.
            header = header.replace('#include "neon2rvv.h"', '/* neon2rvv.h: Neon2RVV lane not selected */')
            struct_c = re.sub(r"/\* NEON zip intrinsics that neon2rvv\.h.*?(?=/\* param structs)", "",
                              struct_c, flags=re.S)
        n_calls = sum(map(len, (gated_calls, gated_ew_calls, gated_struct_calls, gated_nv_calls, gated_av_calls,
                                runner_calls)))
        print(f"selection: kernels={sorted(SEL.kernels) or 'all'} lanes={sorted(SEL.lanes)} "
              f"sections={sorted(SEL.sections)} -> {n_calls} sections, {len(variant_blocks)} variants")
        if len(SEL.lanes) == 1:
            print("note: one lane selected, so the in-section sanity check has no other output to compare with",
                  file=sys.stderr)
        if not n_calls:
            sys.exit("selection matches no section")

    if SEL.simde:
        # The Neon lane through SIMDe: its own definitions of every Neon intrinsic (the zip shims included),
        # native aliases for the arm_neon.h names, RVV implementations where SIMDe has them (fixed VLEN).
        header = header.replace('#include "neon2rvv.h"',
                                '#define SIMDE_ENABLE_NATIVE_ALIASES\n#include <simde/arm/neon.h>')
        struct_c = re.sub(r"/\* NEON zip intrinsics that neon2rvv\.h.*?(?=/\* param structs)", "",
                          struct_c, flags=re.S)
        header = header.replace('{ "Upstream", "Sol", "Autocomp", "Neon2RVV" }', '{ "Upstream", "Sol", "Autocomp", "SIMDe" }')
        rel = lambda calls: [c.replace('-neon2rvv"', '-simde"') for c in calls]
        gated_nv_calls = rel(gated_nv_calls)
    out = [header]
    out.append(struct_c)
    out.append("\n/* ================= embedded kernel variants ================= */\n")
    out.extend(variant_blocks)
    out.append(MAIN_TOP)
    out.append("#if RUN_QUANTIZED\n")
    out.append("\n".join(gated_calls))
    out.append("#endif /* RUN_QUANTIZED */\n\n")
    out.append("#if RUN_ELEMENTWISE\n")
    out.append("\n".join(gated_ew_calls))
    out.append("#endif /* RUN_ELEMENTWISE */\n\n")
    out.append("#if RUN_STRUCTURED\n")
    out.append("\n".join(gated_struct_calls))
    out.append("#endif /* RUN_STRUCTURED */\n\n")
    out.append("#if RUN_NEON2RVV\n")
    out.append("\n".join(gated_nv_calls))
    out.append("#endif /* RUN_NEON2RVV */\n\n")
    # GCC's vectorized reduction epilogue for the scalar gemm/igemm emits
    # vfmv.f.s, which deadlocks Saturn RTL at vlen=128/dlen=64 — run those two
    # sections LAST so a hang there can't lose any other section's results.
    _av_risky = [c for c in gated_av_calls
                 if "f32-gemm-minmax-autovec" in c or "f32-igemm-minmax-autovec" in c]
    gated_av_calls = [c for c in gated_av_calls if c not in _av_risky] + _av_risky
    out.append("#if RUN_AUTOVEC\n")
    out.append("\n".join(gated_av_calls))
    out.append("#endif /* RUN_AUTOVEC */\n\n")
    out.append("\n".join(runner_calls))
    out.append(r'''
    printf("\n################################################################\n");
    if (PERF_NO_SANITY) {
        printf("#  SANITY: disabled (PERF_NO_SANITY): timing only              #\n");
    } else if (g_sanity_fail == 0) {
        printf("#  SANITY: all outputs match Upstream                          #\n");
    } else {
        printf("#  SANITY: %u MISMATCH(ES) vs Upstream — see lines above       #\n", g_sanity_fail);
    }
    printf("#  BENCHMARK COMPLETE                                          #\n");
    printf("################################################################\n");
    sys_reboot(SYS_REBOOT_COLD);
    return 0;
}
''')

    os.makedirs(f"{APP}/src", exist_ok=True)
    path = f"{APP}/src/main.c"
    with open(path, "w") as f:
        f.write("".join(out))
    print(f"wrote {path}")
    for m in missing:
        print(f"WARNING: {m}", file=sys.stderr)


if __name__ == "__main__":
    main()
