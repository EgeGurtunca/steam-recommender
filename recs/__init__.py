# Before numpy loads: OpenBLAS here is built for 24 threads, this laptop has 32, and implicit segfaults on it
# (factors=128). implicit runs its own threads anyway.
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
