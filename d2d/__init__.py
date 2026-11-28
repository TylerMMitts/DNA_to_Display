"""DNA to Display: agent-guided preparation, training setup, QC and analysis for any image + genotype dataset."""
import os

# Before numpy loads. OpenBLAS otherwise starts one thread per core, each with its own
# buffers; on a machine short of memory the allocation fails and the work stalls. The
# memory level (resources.py) lowers this further when a stage has to be retried.
_level = os.environ.get('D2D_MEMORY_LEVEL', '0')
_threads = {'0': '4', '1': '2', '2': '1'}.get(_level, '4')
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, _threads)
