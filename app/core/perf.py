from __future__ import annotations

import os


def _detect_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        return int(os.cpu_count() or 4)


def configure_threads(default_cap: int = 8) -> int:
    """
    Авто-настройка потоков под текущую машину/контейнер.
    Управление:
      AE_AUTO_THREADS=0 отключить
      AE_THREAD_CAP=8 ограничение сверху
      AE_NUM_THREADS=4 принудительно
    """
    if str(os.getenv("AE_AUTO_THREADS", "1")).lower() not in {"1", "true", "yes", "on"}:
        return 0

    cpus = _detect_cpus()

    forced = os.getenv("AE_NUM_THREADS")
    if forced and forced.isdigit():
        threads = max(1, int(forced))
    else:
        cap = int(os.getenv("AE_THREAD_CAP", str(default_cap)))
        threads = max(1, min(cpus, cap))

    # Не перетираем, если уже задано окружением
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    except Exception:
        pass

    return threads
