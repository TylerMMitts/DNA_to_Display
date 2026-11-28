# Memory guard: keeps the pipeline running on a machine that is short of memory, and
# recovers by itself when it runs out.
#
# Two kinds of failure were seen on a 16 GB laptop with a nearly full system drive (so
# Windows could not grow its page file):
#   - a crash: MemoryError, "Unable to allocate", CUDA out of memory, a worker process
#     killed (BrokenProcessPool), "the paging file is too small";
#   - a stall: OpenBLAS failed to allocate its thread buffers ("Memory allocation still
#     failed after 10 retries, giving up") and the analysis thread hung with no error.
# The answer to both is the same: every stage's analysis runs in a child process
# (run_guarded), which is watched. A crash with a memory signature, or no CPU progress
# for STALL_S, ends that child, frees what it held, and the analysis is retried at the
# next memory level (fewer threads, fewer worker processes, smaller GPU batches).
#
#   level 0  normal: 4 math threads, worker processes sized from free memory
#   level 1  reduced: 2 threads, half the workers, half the GPU batch
#   level 2  minimal: 1 thread, no worker processes, GPU batch 2
# The level starts higher by itself when free memory is already low, and the child
# reads it from D2D_MEMORY_LEVEL before numpy is loaded (d2d/__init__.py).
import gc
import os
import subprocess
import sys
import time

LEVELS = 3
STALL_S = 15 * 60            # no CPU progress for this long = stalled
THREADS = {0: '4', 1: '2', 2: '1'}
MEMORY_SIGNS = ('MemoryError', 'Unable to allocate', 'out of memory', 'OutOfMemoryError', 'BrokenProcessPool',
                'Memory allocation still failed', 'paging file is too small', 'std::bad_alloc',
                'cannot allocate memory', 'DefaultCPUAllocator')


def level():
    try:
        return min(max(int(os.environ.get('D2D_MEMORY_LEVEL', '0')), 0), LEVELS - 1)
    except ValueError:
        return 0


def thread_env(lvl):
    t = THREADS[min(lvl, LEVELS - 1)]
    env = {k: t for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['D2D_MEMORY_LEVEL'] = str(lvl)
    return env


def available_gb():
    """(physical, commit) memory available now, in GB. Commit is what Windows can still
    promise (RAM + page file); when it runs out, allocations fail even with RAM free."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        phys = vm.available / 1e9
        if sys.platform == 'win32':
            import ctypes

            class MEM(ctypes.Structure):
                _fields_ = [('len', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total', ctypes.c_ulonglong),
                            ('avail', ctypes.c_ulonglong), ('pt', ctypes.c_ulonglong), ('pa', ctypes.c_ulonglong),
                            ('vt', ctypes.c_ulonglong), ('va', ctypes.c_ulonglong), ('ve', ctypes.c_ulonglong)]
            m = MEM()
            m.len = ctypes.sizeof(MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return phys, m.pa / 1e9
        return phys, (vm.available + psutil.swap_memory().free) / 1e9
    except Exception:  # noqa: BLE001 - unknown platform: assume enough
        return 8.0, 8.0


def start_level():
    """The level to begin at: higher when memory is already short."""
    _, commit = available_gb()
    return max(level(), 2 if commit < 2 else 1 if commit < 5 else 0)


def workers(per_worker_gb, cap=None):
    """Worker processes for a pool: bounded by cores, free memory and the level.
    0 means run in this process."""
    lvl = level()
    if lvl >= 2:
        return 0
    cores = os.cpu_count() or 4
    _, commit = available_gb()
    n = int(max(commit - 2.0, 0) * 0.6 // max(per_worker_gb, 0.05))   # keep 2 GB and 40% back
    n = min(n, cores - 1, cap or cores)
    if lvl == 1:
        n //= 2
    return n if n >= 2 else 0


def gpu_batch(default):
    return {0: default, 1: max(default // 2, 1), 2: min(default, 2)}[level()]


def free():
    gc.collect()
    if 'torch' in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def is_memory_problem(text):
    t = text or ''
    return any(s.lower() in t.lower() for s in MEMORY_SIGNS)


def _cpu_seconds(proc):
    import psutil
    total = 0.0
    try:
        family = [proc] + proc.children(recursive=True)
    except psutil.Error:            # it has just exited
        return float('inf')
    for p in family:
        try:
            c = p.cpu_times()
            total += c.user + c.system
        except psutil.Error:
            pass
    return total


def _kill_tree(proc):
    import psutil
    try:
        family = proc.children(recursive=True) + [proc]
    except psutil.Error:
        return
    for p in family:
        try:
            p.kill()
        except psutil.Error:
            pass


def run_guarded(args, log_path, on_retry=None, cwd=None, env=None):
    """Run a command (a Python module) as a watched child, retrying at lower memory
    levels after a memory crash or a stall. Returns (returncode, log tail); the final
    attempt's code is returned if every level fails."""
    import psutil
    lvl = start_level()
    tail = ''
    for attempt in range(LEVELS):
        lvl = min(lvl, LEVELS - 1)
        child_env = {**os.environ, **(env or {}), **thread_env(lvl)}
        with open(log_path, 'a', encoding='utf-8') as log:
            log.write(f'\n=== {time.strftime("%Y-%m-%d %H:%M:%S")} memory level {lvl}: {" ".join(args)}\n')
            log.flush()
            start = log.tell()
            p = subprocess.Popen(args, cwd=cwd, env=child_env, stdout=log, stderr=subprocess.STDOUT)
            try:
                proc = psutil.Process(p.pid)
            except psutil.Error:        # finished before it could be watched
                proc = None
            last_cpu, last_change, stalled = 0.0, time.time(), False
            while proc is not None and p.poll() is None:
                time.sleep(5)
                cpu = _cpu_seconds(proc)
                if cpu > last_cpu + 1.0:
                    last_cpu, last_change = cpu, time.time()
                elif time.time() - last_change > STALL_S:
                    stalled = True
                    _kill_tree(proc)
                    p.wait()
                    break
            p.wait()
        with open(log_path, encoding='utf-8', errors='replace') as f:
            f.seek(start)
            tail = f.read()[-6000:]          # this attempt only
        if p.returncode == 0 and not stalled:
            return 0, tail
        memory = stalled or is_memory_problem(tail)
        if not memory or lvl >= LEVELS - 1:
            return (p.returncode or 1), tail
        why = (f'no progress for {STALL_S // 60} minutes' if stalled else 'ran out of memory')
        free()
        if on_retry:
            on_retry(why, lvl + 1)
        lvl += 1
    return 1, tail
