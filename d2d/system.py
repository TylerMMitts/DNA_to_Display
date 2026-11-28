# What this computer has, against what training needs.
#
# Read without importing torch (the dashboard server must stay light): the GPU from
# nvidia-smi, memory and battery from the operating system, disk from the workspace's
# drive. The requirements come from the runs that were measured:
#   - LiteVAE at 512 px, batch 8: about 29 GB of GPU memory (IMPC mice on a 32 GB
#     V100); under 30 GB the launcher drops it to batch 4, which needs about 16 GB.
#   - Diffusion 'large' at batch 16 on 64x64 latents fits in 32 GB.
#   - The cluster jobs ran with 8 CPU cores and 64 GB of memory.
#   - IMPC LiteVAE: 80 epochs of 6,666 images took about 16 hours on the V100.
import os
import shutil
import subprocess
import sys

from .project import ROOT

GB = 1e9
CHECKPOINT_GB = 20         # checkpoints, latent cache and previews of one run, roughly

# name, minimum, recommended, unit, why
REQUIREMENTS = [
    ('gpu_memory_gb', 16, 32, 'GB', 'NVIDIA GPU memory. 32 GB trains the image model at batch 8 as tested; '
                                    '16-30 GB runs it at batch 4.'),
    ('ram_gb', 32, 64, 'GB', 'System memory: the image cache and data loading. The cluster jobs used 64 GB.'),
    ('cpu_cores', 4, 8, 'cores', 'Loading and augmenting images; the trainers use 8 workers.'),
    ('disk_free_gb', None, None, 'GB', 'Free space for checkpoints, caches and previews.'),
]


def _gpus():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out:
        try:
            name, total, used = [s.strip() for s in line.rsplit(',', 2)]
            gpus.append({'name': name, 'memory_gb': round(float(total) * 1.048576 / 1000, 1),
                         'used_gb': round(float(used) * 1.048576 / 1000, 1)})
        except ValueError:
            continue
    return gpus


def _ram_gb():
    if sys.platform == 'win32':
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return round(m.ullTotalPhys / GB, 1)
        return None
    try:
        return round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / GB, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _battery():
    """True on a machine with a battery (a laptop): long full-load runs are hard on them."""
    if sys.platform == 'win32':
        import ctypes

        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [('ACLineStatus', ctypes.c_byte), ('BatteryFlag', ctypes.c_byte),
                        ('BatteryLifePercent', ctypes.c_byte), ('SystemStatusFlag', ctypes.c_byte),
                        ('BatteryLifeTime', ctypes.c_ulong), ('BatteryFullLifeTime', ctypes.c_ulong)]
        s = SYSTEM_POWER_STATUS()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return (s.BatteryFlag & 0xFF) not in (128, 255)   # 128 no battery, 255 unknown
        return None
    base = '/sys/class/power_supply'
    if os.path.isdir(base):
        return any(n.startswith('BAT') for n in os.listdir(base))
    return None


def folder_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def specs():
    gpus = _gpus()
    best = max(gpus, key=lambda g: g['memory_gb']) if gpus else None
    disk = shutil.disk_usage(ROOT if ROOT.exists() else ROOT.anchor)
    return {'gpu': best, 'gpus': gpus, 'ram_gb': _ram_gb(), 'cpu_cores': os.cpu_count(),
            'disk_free_gb': round(disk.free / GB, 1), 'battery': _battery(),
            'platform': sys.platform}


def check(dataset_bytes=None):
    """This computer against the requirements. dataset_bytes sets the disk need."""
    s = specs()
    data_gb = (dataset_bytes or 0) / GB
    disk_min = round(data_gb * 2 + CHECKPOINT_GB)          # the data, its prepared copy, the runs
    have = {'gpu_memory_gb': s['gpu']['memory_gb'] if s['gpu'] else 0, 'ram_gb': s['ram_gb'],
            'cpu_cores': s['cpu_cores'], 'disk_free_gb': s['disk_free_gb']}
    rows = []
    for key, lo, hi, unit, why in REQUIREMENTS:
        if key == 'disk_free_gb':
            lo, hi = disk_min, disk_min * 2
        v = have[key]
        level = 'unknown' if v is None else 'ok' if v >= hi else 'min' if v >= lo else 'fail'
        rows.append({'key': key, 'minimum': lo, 'recommended': hi, 'unit': unit, 'have': v,
                     'level': level, 'why': why})
    gpu_text = (f"{s['gpu']['name']} ({s['gpu']['memory_gb']:g} GB)" if s['gpu'] else 'no NVIDIA GPU found')
    rows[0]['have_text'] = gpu_text
    rows.append({'key': 'power', 'minimum': 'desktop or workstation', 'recommended': 'desktop or workstation',
                 'unit': '', 'have': None,
                 'have_text': {True: 'laptop (battery found)', False: 'desktop', None: 'unknown'}[s['battery']],
                 'level': 'min' if s['battery'] else 'ok' if s['battery'] is False else 'unknown',
                 'why': 'Training keeps the GPU at full load for a day or more; many laptops cannot '
                        'supply or cool that for long.'})
    levels = [r['level'] for r in rows]
    if not s['gpu'] or rows[0]['level'] == 'fail' or 'fail' in levels:
        verdict = 'not_recommended'
    elif 'min' in levels:
        verdict = 'possible'
    else:
        verdict = 'ready'
    return {'verdict': verdict, 'rows': rows, 'specs': s,
            'dataset_gb': round(data_gb, 2) if dataset_bytes else None,
            'time_note': 'For scale: the mouse X-ray image model (6,666 training images, 80 epochs) took about '
                         '16 hours on a 32 GB V100, and the diffusion model takes longer. A smaller GPU is '
                         'several times slower.'}
