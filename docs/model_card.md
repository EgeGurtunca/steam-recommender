# Model card: the page's recommender

The format follows Mitchell et al., "Model Cards for Model Reporting" (2019). Numbers come from
[docs/findings.md](findings.md), where each one is explained.

## What it is

Ten Steam games a player does not own, for the library behind a profile link. Two models in sequence:

- **EASE** ([recs/ease.py](../recs/ease.py)): a game-to-game weight matrix with a closed-form solution, trained on
  every positive in the data. It proposes 100 candidates.
- **SASRec** ([eval/sasrec.py](../eval/sasrec.py), run in numpy by [recs/sasrec_np.py](../recs/sasrec_np.py)): a
  two-layer causal transformer over the player's last 50 games, ordered by Steam's last-played times. Each of EASE's
  candidates is moved up by its place in SASRec's own top 100.

The page then reranks the top 60 for tag variety (lambda 0.8), and each card names the played game with the
biggest term in EASE's score. The same ranker, fed only the last two weeks' games, gives a second list; and for
two players it ranks multiplayer and co-op games by least misery (the worse of their two places). Without the SASRec weights (they need PyTorch to train once) the page falls back to
EASE nudged by transition counts.

## Data

- **Training**: Kaggle's "Game Recommendations on Steam" (August 2024 snapshot): 41.2M public reviews from 14.3M
  anonymised users, 2010 to the end of 2022. A positive is a review that recommends a game after at least two
  hours of play. Users with fewer than 5 positives and games with fewer than 50 positive users are dropped:
  1,495,210 users, 10,816 games, 15.8M positives.
- **Catalogue**: a Steam store catalogue from October 2026 (130K apps) for titles, tags, release dates and review
  counts, and for the "new since 2024" section.
- **At use**: one library, read from the Steam Web API at the player's request with their own key, and not stored.

## Intended use

- A player, or someone they asked, looking for what to play next from their own library.
- Two friends, each giving their own link, looking for a game to play together.
- A portfolio piece on recommender evaluation: every number below was measured on users the models never saw.

Not intended for: ranking a store front for many users at once, any decision about a person, or reading
libraries of people who did not ask.

## Performance

Recall@10: the share of a user's hidden liked games found in the top ten (at most ten). Gaps carry 95% intervals
from a paired bootstrap over users.

| setting | EASE | EASE + SASRec (the page) |
|---|---|---|
| next year's likes, 1,000 test users | 0.064 | **0.087** |
| the same, a holdout group of 1,000 kept out of everything | 0.070 | **0.095** (+0.016 to +0.035) |
| the same test users shown only their latest 3 games | 0.050 | **0.063** |
| deployed a year earlier (2021 likes hidden), 1,000 test users | 0.063 | **0.088** (+0.016 to +0.034) |
| a random 20% of each library hidden (EASE alone, 1,000 users) | 0.172 | |

For scale: popularity finds 0.021 to 0.026 on the same next-year tasks. The random split's numbers are more than
twice the next-year ones; the next-year setting is the one a deployed recommender faces.

Speed and size: the page starts in about 1.2 s once the weights exist (EASE trains in about 20 s the first time),
ranks one library in about 0.3 s on a laptop CPU and holds about 770 MB, most of it EASE's matrix. A copy of EASE
with each game's 200 largest weights in float16 is 8 MB and scored the same on test (+0.000, -0.001 to +0.002).

## Limits and biases

- **Popular games are favoured.** Hold-out evaluation rewards them and the models learn them; the page's
  popular-to-niche slider and the variety rerank push back, they do not remove it.
- **The model's world ends in 2022.** Games released later are unknown to EASE and SASRec; they appear only in the
  "new since 2024" section, by tags and review counts. On the next-year task a quarter of what players liked was a
  game the models had never seen.
- **Only games with 50 or more positive users in the data.** Small and niche games are outside the model.
- **Reviews are not play.** A positive is a recommending review; players who never review are not in the data, and
  what reviewers like may differ from what everyone plays.
- **Playtime is a weak signal.** Weighting by hours did not help, and idle games left running inflate it.
- **Short libraries are weaker.** Three games give about three quarters of what a full library gives.
- **gemma4's one-line reasons** (optional, local) name a played game 97% of the time; the "because you played"
  line on each card is exact by construction.

## Privacy

- The training data's user ids were anonymised by the dataset's authors.
- A library read for the page is used for that request and not written anywhere.
- Reviews collected for the newer-games experiment ([recs/collect.py](../recs/collect.py)) are public store reviews;
  each author id is replaced by a keyed hash on arrival and the key never leaves the machine. They are not in this
  repository.

## Maintenance

Retraining is two commands once the data is in place: EASE retrains itself when its weights are missing, and
`python -m eval.sasrec --export --page` then `python -m eval.sasrec --page` (with PyTorch) retrains SASRec. The
evaluation sets are frozen in [eval/](../eval/) so any change can be scored the same way.
