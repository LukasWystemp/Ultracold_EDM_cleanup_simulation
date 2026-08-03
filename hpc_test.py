import numpy as np
import scipy.constants as cts
import pylcp
from pylcp.hamiltonians import XFmolecules
from pylcp.common import cart2spherical
from numba import njit
import matplotlib.pyplot as plt
import time
from pylcp.hamiltonians import wigner_3j as _w3j, wigner_6j as _w6j
from multiprocessing import Process, Pool
import os
import traceback
import socket
from itertools import product

P_values = [1, 2, 3]
TB_VALUES = [0.1, 0.5, 1.0]

BASE_ITERATIONS = 20_000_000

def run(params):
    try: 
        import random

        P, tb = params
        t0 = time.time()

        n = int(BASE_ITERATIONS * P)
        inside = 0
        rand = random.random
        for _ in range(n):
            x = rand()
            y = rand()
            if x * x + y * y <= 1.0:
                inside += 1

        pi_est = 4.0 * inside / n
        elapsed = time.time() - t0

        _pi_est = np.linspace(0, pi_est, 25)

        np.save("hpc_test", _pi_est )
        print(f"Finished worker {P}, {tb} in {elapsed} s")
    except Exception:
         return (P, tb, "FAILED:\n" + traceback.format_exc(), time.time() - t0,
                os.getpid())

    return (P, tb, elapsed)


def main():
    N_WORKERS = 4
    combos = list(product(P_values, TB_VALUES))
    print(f"host={socket.gethostname()}  cores={os.cpu_count()}  "
        f"workers={N_WORKERS}  tasks={len(combos)}", flush=True)
 
    t0 = time.time()
    with Pool(processes=N_WORKERS) as pool:
        # imap_unordered yields results as they finish, so you get live
        # progress instead of silence until everything is done.
        for P, tb, elapsed in pool.imap_unordered(run, combos):
            print(f"  done P={P:<4} tb={tb:<5} "
                  f"{elapsed:6.1f}s", flush=True)
 
    print(f"all {len(combos)} tasks finished in {time.time() - t0:.1f}s")

if __name__=="__main__":
    main()