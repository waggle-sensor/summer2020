"""
Helper functions and classes for parallelizing code across GPUs.
Intended for running experiments on Lambda and other compute clusters, not for
production on Sage nodes.
"""

import os
import traceback

import multiprocessing as mp
from multiprocessing.pool import Pool

from yolov3 import utils


def run_parallel(func, args_list, init=True):
    """Run a function multiple times in parallel, with a pool size determined
    by the number of free GPUs.

    func      - function to call
    args_list - a list containing a tuple (or other iterable) of arguments
    init      - specify if this is the first time running paralllel code within the process
    """
    if init:
        try:
            os.environ["MKL_THREADING_LAYER"] = "GNU"
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
    with NoDaemonPool(len(utils.get_free_gpus()) * 10) as pool:
        pool.starmap_async(func, args_list)
        pool.close()
        pool.join()


class ExceptionLogger:
    """Error logger to a file for subprocesses."""

    def __init__(self, callable, out_dir="."):
        self.__callable = callable
        self.out = out_dir

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)
        except Exception:
            with open(f"{self.out}/error.err", "a+") as out:
                out.write(traceback.format_exc())
            raise Exception(traceback.format_exc())
        return result


class NoDaemonProcess(mp.Process):
    """Class to override multithreading errors when calling daemon processes."""

    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NoDaemonPool(Pool):
    """Class to override multithreading errors when pooling daemon processes."""

    Process = NoDaemonProcess

    def starmap_async(self, func, iterable, *args, **kwargs):
        return Pool.starmap_async(
            self, ExceptionLogger(func), iterable, *args, **kwargs
        )
