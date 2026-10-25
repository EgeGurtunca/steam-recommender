# steam-recommender

![tests](https://github.com/EgeGurtunca/steam-recommender/actions/workflows/test.yml/badge.svg)

Give it a Steam library (games and hours played) and it recommends ten games you don't own. I built four
recommenders on 41 million Steam reviews, from "most popular" to a trained matrix factorisation model to a local
LLM reranking the candidates, and measured all of them the same way on users they had never seen.

The short version: the trained models win (EASE best), and the LLM did not make the ranking better while
taking 5,000 times longer. Below is how much, how sure, and why that is still useful to know.

**Stack:** Python 3.12 · DuckDB (SQL over a 2 GB CSV) · numpy / scipy · implicit (ALS) · scikit-learn (TF-IDF) ·
Ollama (`gemma4:12b`) · pytest

## Data

The Kaggle dataset [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)
(snapshot of August 2024): 41.2M reviews from 14.3M anonymised users on 50.9K apps, each with the date, whether
the user recommends the game, and the hours they had played by then. Reviews run from 2010 to the end of 2022.
The raw files are not in this repo.

[recs/data.py](recs/data.py) turns it into training data in SQL, with checks that stop the build on bad data
(negative hours, dates outside the expected range, reviewed apps missing from the game list):

- a **positive** is a review that recommends the game after at least 2 hours of play; one per user and game
- users with fewer than 5 positives and games with fewer than 50 users are dropped, twice over, since each
  pass can starve the other
- result: **1,495,210 users, 10,816 games, 15.8M positives**, built in about 15 seconds

## The four recommenders

All of them implement one function, `recommend(library, k) -> list of app ids`, so the eval (and later a web
page or a bot) can call any of them.

1. **Popularity** ([recs/popularity.py](recs/popularity.py)): the games most training users liked. The floor.
2. **Content** ([recs/content.py](recs/content.py)): TF-IDF over the games' Steam tags, a profile weighted by
   `log1p(hours)` of what you played, cosine similarity. Needs no other users.
3. **ALS** ([recs/als.py](recs/als.py)): implicit-feedback matrix factorisation trained on 1.5M users, with
   confidence growing with hours played. A new user is never trained in: their vector is solved from their
   library against the fixed game vectors, so one library is enough.
4. **EASE** ([recs/ease.py](recs/ease.py)): a game-to-game weight matrix with a closed-form solution, one
   10,816 x 10,816 matrix inverse and no iterations (Steck, 2019). Added after the first round, when I asked
   what more this data could give.
5. **gemma4 reranker** ([recs/rerank.py](recs/rerank.py)): candidates from ALS (and tags), reordered by
   `gemma4:12b` running locally, with one sentence per pick on why. The answer goes through Ollama's structured
   outputs with a JSON schema whose only allowed ids are the candidates, so it **cannot** recommend a game that
   doesn't exist or isn't in the list.

## Evaluation

Before any result existed, I picked 1000 validation and 1000 test users (at least 10 positives each), took them
out of training, and hid 20% of each one's liked games (at least 2). The split is committed in
[eval/split.json](eval/split.json) and the script that wrote it refuses to overwrite it. Every setting (ALS size,
the prompt, which candidates gemma4 sees) was chosen on the validation users; each test number below comes from
a single run, made once, at the end.

The question for every system: are the hidden games in its top 10?

```bash
python -m eval.run_eval --set test --systems popularity,content,als,hybrid --users 1000
```

**Test, 1000 users:**

| system | recall@10 | ndcg@10 | different games recommended | mean popularity rank |
|---|---|---|---|---|
| popularity | 0.048 | 0.031 | 18 | 6 |
| content (tags) | 0.027 | 0.021 | 2928 | 3262 |
| ALS | 0.143 | 0.102 | 648 | 173 |
| **EASE** | **0.172** | **0.140** | 1138 | 303 |
| ALS + content, alternating | 0.100 | 0.075 | 2300 | 1639 |
| content, 2026 tags x popularity | 0.083 | 0.058 | 730 | 206 |

At least one hidden game is in the top 10 for **43% of users with EASE** (37% with ALS, 15% with popularity).
For comparison with papers: ranked against 100 random games instead of the whole catalogue, the protocol many
papers use, EASE's hit rate is 84% (validation). That protocol is known to flatter and even reorder systems
(Krichene and Rendle, 2020), so the full-catalogue numbers are the ones I stand behind.

_Recall@10: hidden games found in the top 10, out of the most that could fit. Mean popularity rank: where the
recommended games sit in the popularity list (1 = most played), so low means a popularity bias._

**The LLM rows, test, the first 100 of the same users** (gemma4 takes about 10 s per user):

| system | recall@10 | ndcg@10 | different games | s / user |
|---|---|---|---|---|
| popularity | 0.055 | 0.038 | 14 | 0 |
| **ALS** | **0.160** | **0.127** | 391 | 0.002 |
| ALS + content, alternating | 0.120 | 0.102 | 664 | 0.002 |
| gemma4 reorders ALS + content (50 candidates) | 0.137 | 0.102 | 619 | 9.9 |
| gemma4 reorders ALS's top 20 | 0.149 | 0.103 | 424 | 9.2 |
| gemma4 alone, no candidates | 0.041 | 0.035 | 196 | 3.5 |

_Same 100 users for every row; at this size one user moves recall by about 0.01, so read the gaps, not the
third decimal._

**How big a gap is real.** Every system is scored on the same users, so each gap can be resampled user by user
(paired bootstrap, 10,000 draws, [eval/intervals.py](eval/intervals.py)). An interval that holds 0 means this
eval cannot tell the two apart:

| comparison (test) | users | recall@10 gap | 95% interval |
|---|---|---|---|
| ALS against EASE | 1000 | -0.030 | -0.042 to -0.017 |
| popularity against ALS | 1000 | -0.095 | -0.110 to -0.080 |
| ALS + content, alternating, against ALS | 1000 | -0.043 | -0.053 to -0.034 |
| gemma4 reorders ALS's top 20, against ALS | 100 | -0.011 | -0.038 to +0.015 |
| gemma4 reorders ALS + content, against that list | 100 | +0.017 | -0.011 to +0.047 |
| gemma4 alone, against ALS | 100 | -0.120 | -0.168 to -0.075 |
| cold start: 2026 tags x popularity, against tags | 125 | +0.016 | +0.000 to +0.040 |
| learned second stage, against EASE (below) | 1000 | +0.001 | -0.003 to +0.006 |

```bash
python -m eval.intervals eval/results/<file>.json ease als    # first system is the base
```

## What I found

- **EASE beats ALS by a fifth** (0.172 against 0.143), with no training loop at all; on validation, weighting
  by hours and fusing it with ALS changed nothing, which is where I stopped. Realistically this data tops out
  around 0.2: some liked games have no trace in the rest of a player's library.
- **EASE's lead grows with the library.** By how many games each test user liked
  (`python -m eval.intervals <file> als ease --by-size`):

  | liked games | users | ALS | EASE | gap, 95% interval |
  |---|---|---|---|---|
  | 10 to 14 | 541 | 0.148 | 0.168 | +0.020 (+0.001 to +0.039) |
  | 15 to 29 | 348 | 0.146 | 0.185 | +0.040 (+0.020 to +0.059) |
  | 30 or more | 111 | 0.106 | 0.152 | +0.046 (+0.023 to +0.071) |

  ALS squeezes a big library into 64 numbers and loses detail; EASE adds up one row per game, so every game
  keeps its say. With a small library the two are close. Every eval user liked at least 10 games, so a library
  of three, which the page does get, is not measured here.
- **ALS is three times better than popularity**, and it recommends 648 different games to 1000 people where
  popularity recommends the same 18 to everyone. Tuning it was cheap: of four settings (64 or 128 factors,
  alpha 1 or 4), 64 factors with alpha 4 found the most hidden games on validation.
- **Tags alone lose to popularity on this metric.** Content recommends from almost the whole catalogue (mean
  popularity rank 3262), but the games people hide are mostly popular ones, so the metric rewards popularity.
  It is a real limit of hold-out evaluation, not only of tag matching.
- **gemma4 did not beat the model it was given.** Reordering ALS's top 20, it found fewer hidden games than ALS's
  own order (0.149 against 0.160), though 100 users cannot tell that gap from zero (interval -0.038 to +0.015):
  at best a tie, at 10 seconds per user instead of 2 milliseconds. On validation I
  tried telling it the list was already ranked by a model trained on 1.5M players and to keep that order unless
  it had a clear reason; the score did not move. It scored higher on the alternating ALS + content list (0.120 to
  0.137, again inside the noise: -0.011 to +0.047); my guess, not measured, is that it moved the loose tag matches down, partly undoing my own mixing.
  Where it may be worth its cost is the one-line reason per pick, which none of the other systems can write;
  whether people find those reasons useful is the next thing to test, with real users.
- **gemma4 alone recommends from memory, and memory is not a catalogue.** Without a candidate list it named 990
  games for 100 users: 20% of the titles are not in the August 2024 list of 50,872 Steam apps (made up,
  newer than the list, or named differently), 21 were games the player already owned, and it found hidden games
  at a rate below popularity (0.041). The candidate schema is what makes the reranker safe: the same model,
  given a list it must choose from, never invented a game.
- **So I split the jobs: ALS ranks, gemma4 explains** ([recs/explain.py](recs/explain.py)). ALS keeps its order
  (no accuracy lost) and gemma4 writes one sentence per pick, again through a schema that only allows the ten
  games it was given. On 100 test users, **97% of the 1000 sentences name a game the player actually played**
  ("The physics-based interaction and creative sandbox elements will appeal to a fan of Garry's Mod"), none
  came back empty, no call failed. The check is strict (exact titles), so a played game listed under a longer
  edition name counts as a miss. The failure it does have: 14% of sentences mention the recommended game's own
  title. Most of those are harmless ("Portal 2 offers..."), but some cite it as if the player had played it
  ("...similar to State of Decay 2 and Dying Light's atmosphere", for Dying Light). On my own library, 3 of 10
  reasons did this; telling the model not to cut it to 1, so the code now drops any reason that names the
  game it explains.
- **Cold start: tags alone found nothing; fresh tags times popularity found a little.** The idea was that ALS
  cannot know games released after its data ends, but tags can. I simulated it: the 626 games released from
  July 2022 were taken out of training, and the systems had to find them among the hidden games of the test
  users who had one (125). With the 2024 tags, content found none on test. Then I added an October 2026 store
  catalogue (130K apps: tags and review counts) and weighted tag similarity by `log1p(positive reviews)`. On
  validation that went from 0.011 to 0.066; on test it is **0.016**, two users out of 125, with an interval
  that starts at zero (0.000 to 0.040). Barely there, and much smaller than validation promised: the weight was chosen on validation, and with 125 users one hit moves
  recall by 0.008. This is the kind of number a frozen test set exists to catch.
- **Warm start: a game's first week beats every tag.** Stores do not wait for a new game to be old; a patent on
  game recommendations adds a co-play graph built from a new title's first players. So I gave each of the 626
  new games back its early positive reviews, from training users only (an eval user's own review never goes
  in), retrained EASE and scored the same 125 test users ([eval/warm_start.py](eval/warm_start.py)):

  | early data per new game | val recall@10 | test recall@10 |
  |---|---|---|
  | none (cold start, EASE) | 0.000 | 0.000 |
  | tags x popularity, 2026 catalogue | 0.066 | 0.016 |
  | first 7 days of reviews | 0.101 | **0.052** |
  | first 30 days | 0.109 | 0.060 |
  | first 90 days | 0.134 | 0.071 |
  | everything to the end of 2022 | 0.150 | 0.073 |

  One week of reviews already gives 70% of what the full data gives on test, and three times what tags gave.
  My first version of this cut each game at its first k reviews instead of its first k days, and found almost
  nothing (0.005 on test even at 1,000 reviews): capping every game at the same count erases the one thing
  EASE learns from a new game, that the hit gets many more reviews than the rest. A week keeps that
  difference. The data ends in December 2022, so "everything" is at most six months for these games.
- **A learned second stage did not beat EASE.** Large recommenders run in two stages: one model proposes a few
  hundred candidates, a second reorders them with richer features. I built that over EASE
  ([recs/ranker.py](recs/ranker.py)): EASE proposes 50, gradient boosting (scikit-learn) reorders them with
  EASE's score and rank, popularity, release year, the store's review share and count, tag similarity and
  library size. It learns from 60,000 training users with 20% of their games hidden, who are left out of EASE's
  training too, so their scores look like a new user's (`python -m eval.train_ranker`):

  | | recall@10 | ndcg@10 | coverage | mean pop rank |
  |---|---|---|---|---|
  | EASE, val | 0.171 | 0.138 | 1128 | 293 |
  | EASE + ranker, val | 0.173 | 0.141 | 1291 | 366 |
  | EASE, test | 0.171 | 0.140 | 1120 | 294 |
  | EASE + ranker, test | 0.172 | 0.141 | 1253 | 357 |

  No real gain in accuracy (the test gap's 95% interval is -0.003 to +0.006; a second try with three times the users and a relative-score feature gave the
  same), but 12% more of the catalogue reaches someone's top 10. EASE already encodes what these features
  say about popularity and co-play; what makes the second stage pay off in industry (time of day, session,
  price, what was shown and skipped) is not in a review dataset. Kept as an eval, not used by the page.

## Fresher data

The training data ends in 2022, so I looked for something newer:

- **A store catalogue from October 2026** ([130K Steam Games](https://www.kaggle.com/datasets/sibamsamanta07/130k-steam-games-prices-tags-reviews-ccu) on Kaggle, the same columns as the [FronkonGames Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) it appears to be built from). Its header merges "Discount" and "DLC count"
  into one name while every row has both values, so every later column reads one off (Positive shows 0 and
  Tags holds the genres). [recs/data.py](recs/data.py) splits the header and refuses any header or row width
  it does not expect.
- **Public store reviews, anonymised** ([recs/collect.py](recs/collect.py)): newest reviews of 331 popular
  games over the last year. Each author id becomes a keyed hash (HMAC with a key that stays on my machine) the
  moment it arrives; Steam ids are never written, and no profiles or libraries are read. 299K reviews, but 91%
  of the 261K authors appear once, and only 1,634 reviewed five games or more, too few links between games to
  train on (the 2022 data has 1.9M such users). It stays a source of recent popularity, not training data.
  Valve switched off the store's review endpoint on 22 October 2026, mid-project; the collector uses its Web
  API replacement, `IUserReviewsService/GetAppReviews`.

## Your own profile

```bash
python -m recs.profile https://steamcommunity.com/id/<name>
```

[recs/steam.py](recs/steam.py) reads one library from a profile link, only at that person's request (as the
Steam Web API terms require), with your own key in `.env`; nothing is stored. [recs/profile.py](recs/profile.py)
then prints, in about 3 seconds once the EASE weights are cached:

- **your taste**: Steam tags weighted by how rare they are across 130K games and by `sqrt(hours)`. Tags that
  describe how a game is played (Multiplayer, Free to Play, Singleplayer) are left out of the taste and shown as
  a separate share: how many of your hours went to multiplayer or co-op games.
- **play next**: EASE's top 10, with programs filtered out by their Steam genre.
- **new since 2024**: games the 2022 model never saw, ranked by tag similarity times `log1p(positive reviews)`.

Or as a page, standard library only:

```bash
python -m recs.web      # http://127.0.0.1:8000
```

Paste a profile link and it draws the taste as a bubble graph (D3): one bubble per tag, sized by its share,
linked to the tags that describe the same games across the store (Jaccard over well-reviewed games), so related
tastes pull together. A switch shows the last two weeks instead of all time, which is where a change in taste
shows up. Below it, cards for "play next" and "new since 2024", with links to the store pages. Two sliders, popular to niche and
older to newer, borrowed from Steam's own interactive recommender, re-sort EASE's top 60 in the page: each adds
the game's percentile among those 60 (review count, release year) to EASE's order, so at the middle it is plain
EASE. "Niche" is relative: EASE only knows games with 50 or more reviews in 2022, so the far end is less known,
not obscure. A button
asks gemma4 for one line per "play next" card, from the games you played (schema-limited to those ten, in
EASE's order); the 2023 to 2026 games it names come from the store catalogue, so a library full of recent games
still reads correctly. Ollama is started for that request and stopped right after.

Trying it on my own library found two problems the offline eval could not: a wallpaper program in the top 10,
and a genre I had put about 2,000 hours into missing from the taste entirely. Hours were weighted by
`log1p`, which squashed them so hard that a hundred short games outweighed three very long ones; `sqrt` fixed it.
Playtime is still a rough signal: idle games left running inflate it.

I then tried what Steam's own recommender is described as using, playtime: weight each liked game in EASE's
sum by how long it was played, either `sqrt(hours)` or hours against the game's median playtime from the store
catalogue (`python -m eval.run_eval --systems ease,ease-w:sqrt-hours,ease-w:rel-sqrt,ease-w:rel-log`). On val
no variant beat plain EASE (recall@10 0.166 to 0.168 against 0.172; every gap's 95% interval holds 0, the
widest -0.015 to +0.002), and on my
library the median version made things worse in a way I had not expected: idle games have tiny medians
(most owners open them once), so 2,300 hours of one is 209 times its median, the strongest signal in the
library. EASE learned from liked or not; at least for it, "how much" adds noise rather than taste. Kept as an
eval option, not used by the page.

## Running it

```bash
python -m venv .venv && .venv/Scripts/pip install duckdb numpy scipy scikit-learn implicit pytest
python -m recs.data path/to/steam_game_recommendations_archive.zip   # extract, check, build (~15 s)
python -m recs.data --store path/to/steam_game_price.zip             # 2026 store catalogue (~2 min)
python -m eval.run_eval --set val --systems popularity,content,als,hybrid
ollama pull gemma4:12b                                              # only for the LLM rows
python -m eval.run_eval --set val --systems rerank-als20 --users 100
python -m eval.explain_eval --set val --users 100   # how grounded the reasons are
python -m recs.recommend --user <id from eval/split.json>   # ten picks with reasons
```

On Windows with more than 24 logical cores, OpenBLAS (as shipped with numpy) crashes inside implicit; the
package sets `OPENBLAS_NUM_THREADS=1` before numpy loads.

## Tests

```bash
pytest
```

65 tests, no network, no model, no real data: the SQL build and the quality checks on tiny CSVs, the
frozen split, the metrics and their bootstrap intervals, each recommender (owned games never recommended, unknown games ignored, empty
libraries fall back to popularity), the reranker's schema and its fallback when the model fails or returns
duplicates, the store catalogue's broken header, the collector (paging, hashing, no raw ids), the popularity weight, reading a profile link (vanity and numeric, private libraries), the taste profile (rare tags over common ones, long-played over many short games), the non-game filter, the tag links for the graph, the page's data (owned games and programs never recommended, reasons in the page's order), the explainer keeping ALS's order and checking which sentences are grounded, title matching with ™ and odd spacing, and stopping Ollama with its child process.

## What's next

- A Telegram bot in front of the same engine
- For 2023 to 2026 games, the warm start result says what to collect: a week of reviews per new game beats its
  tags, so the anonymised review collector pointed at new releases is the next data step

Not affiliated with Valve. No user data is stored; the dataset's user ids are anonymised.
