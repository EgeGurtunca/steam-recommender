"""How fast the page is: startup (catalogue, EASE, the ranker) and the time to draw one library, on the demo.

Run: python -m eval.bench_page
"""
import statistics
import time

from recs import web
from recs.data import BUILD_DIR, load_catalog, load_store
from recs.ease import EASE


def main(runs: int = 20) -> None:
    start = time.perf_counter()
    store, catalog = load_store(), load_catalog()
    ranker = web.page_ranker(EASE.load_or_build(BUILD_DIR / "ease"))
    startup = time.perf_counter() - start
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        web.payload(web.DEMO, store, ranker, catalog)
        times.append(time.perf_counter() - t)
    times.sort()
    print(f"startup {startup:.1f} s | one library: median {statistics.median(times) * 1000:.0f} ms, "
          f"slowest of {runs} {times[-1] * 1000:.0f} ms")


if __name__ == "__main__":
    main()
