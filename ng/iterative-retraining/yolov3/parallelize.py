import os
import traceback

import multiprocessing as mp
import multiprocessing.pool as pool

from yolov3 import utils


def run_parallel(func, args_list):
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    mp.set_start_method("spawn")
    with NoDaemonPool(len(utils.get_free_gpus()) * 10) as pool:
        pool.starmap_async(func, args_list)
        pool.close()
        pool.join()


class NoDaemonProcess(mp.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class ExceptionLogger(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)
        except Exception:
            with open("error.err", "a+") as out:
                out.write(traceback.format_exc())
            raise Exception(traceback.format_exc())
        return result


class NoDaemonPool(pool.Pool):
    Process = NoDaemonProcess

    def starmap_async(self, func, iterable, **kwargs):
        return pool.Pool.starmap_async(self, ExceptionLogger(func), iterable, **kwargs)
