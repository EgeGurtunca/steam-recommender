"""The page's data without the 2 GB of reviews: the four files recs.web reads, about 20 MB in one zip. Unzipped into
the repository's folder, `python -m recs.web` runs with no Kaggle download and no training.

Run: python -m recs.bundle      (needs the full build: data/build/ease/b.npy and sasrec_page.npz)
"""
import io
import zipfile

import numpy as np

from recs.data import BUILD_DIR, ROOT
from recs.ease import EASE

OUT = ROOT / "dist" / "demo-data.zip"
FILES = ["games.parquet", "store.parquet", "sasrec_page.npz", "ease/small.npz"]
MIT = """Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE."""
NOTICE = f"""steam-recommender: data for the local page
https://github.com/EgeGurtunca/steam-recommender

What is in this bundle
- data/build/games.parquet: title, release date and tags of 10,816 games
- data/build/store.parquet: title, release date, review counts, tags, genres, median playtime and price of
  130,100 Steam apps
- data/build/ease/small.npz, data/build/sasrec_page.npz: model weights, game-to-game numbers learned from
  anonymised reviews

What is not in it: no user id or other personal data, no review text, no store description, no image (the page
loads game images from Steam when it is opened), no API key. The weights are statistics over many players and
hold nothing about any one of them.

Sources and licenses

1. "Game Recommendations on Steam" by Anton Kozyriev, CC0 1.0 (public domain)
   https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam
   Used for games.parquet and, through its anonymised reviews, the model weights.

2. "130K Steam Games" by SibamSamanta07, MIT License
   https://www.kaggle.com/datasets/sibamsamanta07/130k-steam-games-prices-tags-reviews-ccu
   Used for store.parquet. It appears to be built on "Steam Games Dataset" by Martin Bustos (Fronkon Games),
   MIT License: https://www.kaggle.com/datasets/fronkongames/steam-games-dataset

   Copyright (c) SibamSamanta07
   Copyright (c) Martin Bustos (Fronkon Games)

{MIT}

Steam and the Steam logo are trademarks of Valve Corporation. Game titles and tags belong to their owners and
are used only to name the games. This project is not affiliated with or endorsed by Valve or any game's
publisher.

This bundle is provided as is, without warranty of any kind. If you hold rights to any of this data and want it
removed, open an issue at https://github.com/EgeGurtunca/steam-recommender/issues and it will be taken down.
"""


def without_checks(path) -> bytes:
    """SASRec's weights without check_seq and check_scores: eight training users' histories, kept beside the
    weights only to test the numpy port."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: v for k, v in np.load(path).items() if not k.startswith("check_")})
    return buf.getvalue()


def main() -> None:
    EASE.load(BUILD_DIR / "ease").save_small(BUILD_DIR / "ease" / "small.npz")
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            if f == "sasrec_page.npz":
                z.writestr(f"data/build/{f}", without_checks(BUILD_DIR / f))
            else:
                z.write(BUILD_DIR / f, f"data/build/{f}")
        z.writestr("data/NOTICE.txt", NOTICE)
    print(f"wrote {OUT}: {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
