"""The page's data without the 2 GB of reviews: the four files recs.web reads, about 20 MB in one zip. Unzipped into
the repository's folder, `python -m recs.web` runs with no Kaggle download and no training.

Run: python -m recs.bundle      (needs the full build: data/build/ease/b.npy and sasrec_page.npz)
"""
import zipfile

from recs.data import BUILD_DIR, ROOT
from recs.ease import EASE

OUT = ROOT / "dist" / "demo-data.zip"
FILES = ["games.parquet", "store.parquet", "sasrec_page.npz", "ease/small.npz"]
NOTICE = """Data for steam-recommender's page, derived from two Kaggle datasets:

- Game Recommendations on Steam, by antonkozyriev (CC0: Public Domain)
  https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam
  Used for: games.parquet (titles, release dates, tags), and the EASE and SASRec weights, learned from its
  anonymised reviews. No review or user id is in this bundle.
- 130K Steam Games, by sibamsamanta07 (MIT License), built on the FronkonGames Steam Games Dataset (MIT License)
  https://www.kaggle.com/datasets/sibamsamanta07/130k-steam-games-prices-tags-reviews-ccu
  https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
  Used for: store.parquet (titles, release dates, review counts, tags, genres, price). The MIT License asks
  that its notice be kept with copies: see those pages.

Game titles and tags belong to their owners. Not affiliated with Valve.
"""


def main() -> None:
    EASE.load(BUILD_DIR / "ease").save_small(BUILD_DIR / "ease" / "small.npz")
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(BUILD_DIR / f, f"data/build/{f}")
        z.writestr("data/NOTICE.txt", NOTICE)
    print(f"wrote {OUT}: {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
