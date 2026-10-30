# What I found

The findings behind the [README](../README.md), with their numbers and how each was measured. Test numbers
come from one run per system, on users frozen before any result existed; every gap has a 95% interval from
a paired bootstrap over users ([eval/intervals.py](../eval/intervals.py)).

## The random split

### EASE beats ALS by a fifth

EASE beats ALS by a fifth (0.172 against 0.143), with no training loop at all; on validation, weighting
by hours and fusing it with ALS changed nothing, which is where I stopped. Realistically this data tops out
around 0.2: some liked games have no trace in the rest of a player's library.

### EASE's lead grows with the library

Split by how many games each test user liked
(`python -m eval.intervals <file> als ease --by-size`):

| liked games | users | ALS | EASE | gap, 95% interval |
|---|---|---|---|---|
| 10 to 14 | 541 | 0.148 | 0.168 | +0.020 (+0.001 to +0.039) |
| 15 to 29 | 348 | 0.146 | 0.185 | +0.040 (+0.020 to +0.059) |
| 30 or more | 111 | 0.106 | 0.152 | +0.046 (+0.023 to +0.071) |

ALS squeezes a big library into 64 numbers and loses detail; EASE adds up one row per game, so every game
keeps its say. With a small library the two are close. Every eval user here liked at least 10 games; a library
of three, which the page does get, is measured on the time split ([below](#with-three-games-the-order-still-holds)).

### ALS is three times better than popularity

ALS is three times better than popularity, and it recommends 648 different games to 1000 people where
popularity recommends the same 18 to everyone. Tuning it was cheap: of four settings (64 or 128 factors,
alpha 1 or 4), 64 factors with alpha 4 found the most hidden games on validation.

### Tags alone lose to popularity on this metric

Content recommends from almost the whole catalogue (mean
popularity rank 3262), but the games people hide are mostly popular ones, so the metric rewards popularity.
It is a real limit of hold-out evaluation, not only of tag matching.

### Weighting by playtime did not help

I then tried what Steam's own recommender is described as using, playtime: weight each liked game in EASE's
sum by how long it was played, either `sqrt(hours)` or hours against the game's median playtime from the store
catalogue (`python -m eval.run_eval --systems ease,ease-w:sqrt-hours,ease-w:rel-sqrt,ease-w:rel-log`). On val
no variant beat plain EASE (recall@10 0.166 to 0.168 against 0.172; every gap's 95% interval holds 0, the
widest -0.015 to +0.002), and on my
library the median version made things worse in a way I had not expected: idle games have tiny medians
(most owners open them once), so 2,300 hours of one is 209 times its median, the strongest signal in the
library. EASE learned from liked or not; at least for it, "how much" adds noise rather than taste. Kept as an
eval option, not used by the page.

### A learned second stage did not beat EASE

Large recommenders run in two stages: one model proposes a few
hundred candidates, a second reorders them with richer features. I built that over EASE
([recs/ranker.py](../recs/ranker.py)): EASE proposes 50, gradient boosting (scikit-learn) reorders them with
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

## The local LLM

### gemma4 did not beat the model it was given

Reordering ALS's top 20, it found fewer hidden games than ALS's
own order (0.149 against 0.160), though 100 users cannot tell that gap from zero (interval -0.038 to +0.015):
at best a tie, at 10 seconds per user instead of 2 milliseconds. On validation I
tried telling it the list was already ranked by a model trained on 1.5M players and to keep that order unless
it had a clear reason; the score did not move. It scored higher on the alternating ALS + content list (0.120 to
0.137, again inside the noise: -0.011 to +0.047); my guess, not measured, is that it moved the loose tag matches down, partly undoing my own mixing.
Where it may be worth its cost is the one-line reason per pick, which none of the other systems can write;
whether people find those reasons useful is the next thing to test, with real users.

### gemma4 alone recommends from memory, and memory is not a catalogue

Without a candidate list it named 990
games for 100 users: 20% of the titles are not in the August 2024 list of 50,872 Steam apps (made up,
newer than the list, or named differently), 21 were games the player already owned, and it found hidden games
at a rate below popularity (0.041). The candidate schema is what makes the reranker safe: the same model,
given a list it must choose from, never invented a game.

### So I split the jobs: ALS ranks, gemma4 explains

So I split the jobs: ALS ranks, gemma4 explains ([recs/explain.py](../recs/explain.py)). ALS keeps its order
(no accuracy lost) and gemma4 writes one sentence per pick, again through a schema that only allows the ten
games it was given. On 100 test users, **97% of the 1000 sentences name a game the player actually played**
("The physics-based interaction and creative sandbox elements will appeal to a fan of Garry's Mod"), none
came back empty, no call failed. The check is strict (exact titles), so a played game listed under a longer
edition name counts as a miss. The failure it does have: 14% of sentences mention the recommended game's own
title. Most of those are harmless ("Portal 2 offers..."), but some cite it as if the player had played it
("...similar to State of Decay 2 and Dying Light's atmosphere", for Dying Light). On my own library, 3 of 10
reasons did this; telling the model not to cut it to 1, so the code now drops any reason that names the
game it explains.

## Predicting the next year (the time split)

### Asked to predict the future, every number falls by more than half

The main split hides a random 20% of
each library, so a game liked in 2022 can help find one liked in 2015, and the other users' later reviews are
in training too. A deployed recommender has neither. So I froze a second split
([eval/time_split.py](../eval/time_split.py), [eval/time_split.json](../eval/time_split.json)): everything trains
on positives from before 1 January 2022, and each of 1000 val and 1000 test users (8 or more liked games before,
2 or more after) is shown their past and asked for what they liked in 2022:

| test, 1000 users | random split | in time | in time, games training knows |
|---|---|---|---|
| popularity | 0.048 | 0.021 | 0.027 |
| content (tags) | 0.027 | 0.015 | 0.019 |
| ALS | 0.143 | 0.054 | 0.072 |
| EASE | 0.172 | 0.064 | 0.085 |

A quarter of what these users liked in 2022 is a game training never saw (released or first liked later),
which is the cold start problem again, and no model here can find it. Even counting only games it knows,
EASE finds half as much as the random split says. Part of that is how people review: a quarter of the hidden
games in the random split were reviewed within a day of a game the model was shown, and EASE finds those
more often (18% of them against 15% of the rest). EASE's lead over ALS shrinks from 0.030 to 0.011 on test
(interval +0.003 to +0.019) and is a tie on val (-0.008 to +0.008). The ranking of systems holds; the size
of every number does not, and the random split is the one most papers report.

### Weighing recent likes more helped on val and not on test

On the time split, recency is fair to use, so I
tried what deployed systems do: EASE trained with every like losing half its weight per half-life
(`ease-decay:<days>`, through a `binary=False` option in [recs/ease.py](../recs/ease.py)), and EASE's top 100
nudged by how many players liked each game in the 30 days before the cutoff (`ease+recent:<beta>`), and the
user's side of the same idea, each game they were shown weighted by how long ago they liked it
(`ease-userdecay:<days>`, what the page's "last two weeks" switch would feed):

| time split | val recall@10 | gap vs EASE, 95% interval | test recall@10 | gap vs EASE, 95% interval |
|---|---|---|---|---|
| EASE | 0.062 | | 0.064 | |
| EASE, half-life 730 days (best of 180 to 2,920 on val) | 0.072 | +0.010 (+0.005 to +0.015) | 0.068 | +0.003 (-0.002 to +0.009) |
| EASE nudged by the last 30 days (best beta on val) | 0.065 | +0.003 (-0.000 to +0.006) | 0.064 | -0.000 (-0.004 to +0.003) |
| the user's own recent likes weigh more, half-life 1,095 days | 0.068 | +0.006 (+0.002 to +0.009) | 0.065 | +0.001 (-0.003 to +0.004) |
| popularity over the last 30 days | 0.031 | | 0.026 | |

The decay was the one change whose interval cleared zero on val, and it was not a side effect of
regularisation (plain EASE with lambda from 150 to 4,000 stays at 0.058 to 0.062). On test the gain shrank to
something the eval cannot tell from nothing, the same pattern as the cold start weight: chosen on 1000
users, it was partly fit to them. Recent popularity beats all-time popularity as a baseline (0.031 against
0.019 on val), but adds nothing on top of EASE, and the user's own recent likes went the same way as the
decay: a gap on val, none on test. Three small val wins that did not survive one test run is the clearest
case in this project for keeping the test set frozen. The page uses none of them.

### Order helps, and it held on test

EASE sees a library as a set. A sequential recommender reads it in
order; without a neural network the classic version is a transition count: over the training users' likes
in date order, how often game B was liked within the next three likes after game A
(`transitions()` in [recs/sequence.py](../recs/sequence.py)). A user's score for B is the sum over their
games, weighted toward the latest (half-life 180 days), of P(B soon after A):

| time split | val recall@10 | test recall@10 | test gap vs EASE, 95% interval |
|---|---|---|---|
| EASE | 0.062 | 0.064 | |
| transitions alone (`markov`) | 0.061 | 0.066 | +0.002 (-0.007 to +0.010) |
| EASE's top 100 nudged by transitions (`ease+markov:3`, best of 0.3 to 30 on val) | 0.072 | **0.073** | +0.009 (+0.001 to +0.016) |

Counting what came next, a few lines of numpy, already matches EASE on its own; the two together find 14%
more on test than EASE, the first change in this project tuned on val that survived the test run. They
are wrong in different places: EASE knows which games go together, the transitions know which one tends to
come next. Every number here is on the time split; on the random split, order means nothing.

### A small transformer reading the library in order (SASRec) is the best system on the time split

[eval/sasrec.py](../eval/sasrec.py) is SASRec (Kang and McAuley, 2018) in PyTorch: two causal self-attention
layers over the last 50 likes in date order, trained to predict each next like over all 9,899 games, on the
same 1.45M training users before the cutoff. Three epochs take about five minutes on a laptop RTX 4090:

| time split | val recall@10 | test recall@10 | test gap vs EASE, 95% interval |
|---|---|---|---|
| EASE | 0.062 | 0.064 | |
| EASE + transitions | 0.072 | 0.073 | +0.009 (+0.001 to +0.016) |
| SASRec, 128 dimensions, 3 epochs (1 to 3 epochs, 64 or 128 tried on val: 0.080 to 0.082) | 0.082 | 0.078 | +0.014 (+0.004 to +0.024) |
| **EASE's top 100 nudged by SASRec's ranking** (`ease+sasrec:3`, best of 0.5 to 3 on val) | 0.089 | **0.088** | +0.024 (+0.015 to +0.033) |

Four seeds on val give 0.082 to 0.086, so training luck moves it by about 0.004, far less than its lead;
a third layer gave 0.078 on val, so it stays at two, and adding how long each game was played (a log2-hours
embedding, `--hours`) gave 0.082, the same as without. A longer history would change little: the median user
shows 13 games, well under the 50 it reads.
It finds 22% more than EASE on test, and its edge over the few lines of transition counting is inside the
noise (+0.005, -0.004 to +0.015). The best is both: EASE's candidates reordered with SASRec's view finds
38% more than EASE, and beats SASRec alone too (+0.010, +0.002 to +0.017; retrained, SASRec gave the same
0.078). Nor does it buy accuracy with popularity, the usual trade (test):

| time split | recall@10 | different games recommended | mean popularity rank |
|---|---|---|---|
| EASE | 0.064 | 990 | 243 |
| EASE + transitions | 0.073 | 701 | 172 |
| SASRec | 0.078 | 1176 | 592 |
| EASE + SASRec | 0.088 | 996 | 298 |
| LightGCN (below) | 0.052 | 753 | 138 |

The transition counts gain by leaning on what is popular; SASRec gains while recommending more of the
catalogue and less popular games than EASE. On the random split EASE was the clear winner; on the question a deployed system faces, what comes
next, the models that read order win, and a set model and an order model together win most. The page uses
that pair without torch: [recs/sasrec_np.py](../recs/sasrec_np.py) is the same forward pass in numpy, trained
once on every positive (`python -m eval.sasrec --page`), and gives torch's top 10 for every checked library
(largest score difference 6e-6) in 0.03 s.

### The headline held on users no system had seen

By then 13 systems had been scored once each on the
time split's test set; picking the best of 13 on one set flatters the winner. So I froze a third group of
1,000 users ([eval/time_holdout.json](../eval/time_holdout.json)), retrained everything without them, checked
the new pipeline on val first, and scored them once:

| holdout, 1000 users | recall@10 | gap vs EASE, 95% interval |
|---|---|---|
| popularity | 0.026 | |
| EASE | 0.070 | |
| EASE + transitions | 0.084 | +0.014 (+0.006 to +0.022) |
| SASRec | 0.084 | +0.014 (+0.004 to +0.024) |
| **EASE + SASRec** | **0.095** | +0.025 (+0.016 to +0.035) |

Same order, same size of gap: EASE + SASRec finds 36% more than EASE, and beats SASRec alone again (+0.011,
+0.004 to +0.019).

### With three games, the order still holds

The page often gets a new account, but every eval user had at
least eight liked games. Shown only their latest three or five (`--last N`), the same test users give:

| test | latest 3 games | latest 5 | whole library |
|---|---|---|---|
| popularity | 0.021 | 0.021 | 0.021 |
| EASE | 0.050 | 0.055 | 0.064 |
| EASE + transitions | 0.060 | 0.066 | 0.073 |
| SASRec | 0.058 | 0.066 | 0.075 |
| EASE + SASRec | **0.063** | **0.072** | **0.087** |

(SASRec here was retrained without the holdout group, hence 0.075 for the whole library against 0.078
above.) Three games already find two to three times what popularity does, and the pair stays on top
(+0.013 over EASE with three games, +0.006 to +0.020). A short library costs less than I expected: the
latest three games give three quarters of what the whole library gives.

### Variety cost nothing measurable

The demo's top ten held three Souls games. Stores rerank for variety;
the usual way is maximal marginal relevance: pick the list one game at a time, weighing its place in the
ranking (lambda) against its tag overlap with the games already picked (`diversify` in
[recs/core.py](../recs/core.py), `--diversify` in the time split eval). For EASE + SASRec:

| lambda | val recall@10 | test recall@10 | tag variety, test (1 - mean overlap) | gap vs none, test |
|---|---|---|---|---|
| 1 (none) | 0.089 | 0.087 | 0.979 | |
| 0.8 | 0.087 | 0.086 | 0.988 | -0.001 (-0.003 to +0.001) |
| 0.6 | 0.084 | | 0.995 on val | -0.005 on val (-0.009 to -0.001) |

At 0.8 the loss is inside the noise on both sets, so the page uses it: on the demo, Dark Souls Remastered
makes way for Satisfactory and Dark Souls II moves down. Most lists are already varied, which is why the
measure starts near 1; the rerank matters for the libraries that are not.

### A graph neural network did not beat EASE either

Stores describe their recommenders as graphs of who
played what, so I trained LightGCN (He et al., 2020) on the same likes before the cutoff
([eval/lightgcn.py](../eval/lightgcn.py)): game and player embeddings smoothed over three hops of the like graph,
trained with BPR on the GPU in about 30 minutes, a new player folded in from their games by the same rule.
One run with its settings fixed beforehand, scored on both sets: 0.053 on val and **0.052** on test, below EASE
(-0.012, -0.019 to -0.006) and level with ALS (0.054), while leaning harder on popular games than any of them
(mean popularity rank 138). It is a known result that a well tuned simple model often beats a newer neural one
on this kind of data (Dacrema et al., 2019); I tuned LightGCN far less than EASE, so read it as "not for free"
rather than "worse".

## New games

### Cold start: tags alone found nothing; fresh tags times popularity found a little

The idea was that ALS
cannot know games released after its data ends, but tags can. I simulated it: the 626 games released from
July 2022 were taken out of training, and the systems had to find them among the hidden games of the test
users who had one (125). With the 2024 tags, content found none on test. Then I added an October 2026 store
catalogue (130K apps: tags and review counts) and weighted tag similarity by `log1p(positive reviews)`. On
validation that went from 0.011 to 0.066; on test it is **0.016**, two users out of 125, with an interval
that starts at zero (0.000 to 0.040). Barely there, and much smaller than validation promised: the weight was chosen on validation, and with 125 users one hit moves
recall by 0.008. This is the kind of number a frozen test set exists to catch.

### Warm start: a game's first week beats every tag

Stores do not wait for a new game to be old; a patent on
game recommendations adds a co-play graph built from a new title's first players. So I gave each of the 626
new games back its early positive reviews, from training users only (an eval user's own review never goes
in), retrained EASE and scored the same 125 test users ([eval/warm_start.py](../eval/warm_start.py)):

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

### A store catalogue from October 2026

The training data ends in 2022, so I looked for something newer: a store catalogue from October 2026 ([130K Steam Games](https://www.kaggle.com/datasets/sibamsamanta07/130k-steam-games-prices-tags-reviews-ccu) on Kaggle, the same columns as the [FronkonGames Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) it appears to be built from). Its header merges "Discount" and "DLC count"
into one name while every row has both values, so every later column reads one off (Positive shows 0 and
Tags holds the genres). [recs/data.py](../recs/data.py) splits the header and refuses any header or row width
it does not expect.

### Public store reviews, anonymised

Public store reviews, anonymised ([recs/collect.py](../recs/collect.py)): newest reviews of 331 popular
games over the last year. Each author id becomes a keyed hash (HMAC with a key that stays on my machine) the
moment it arrives; Steam ids are never written, and no profiles or libraries are read. 299K reviews, but 91%
of the 261K authors appear once, and only 1,634 reviewed five games or more, too few links between games to
train on (the 2022 data has 1.9M such users). It stays a source of recent popularity, not training data.
Then the warm start result made me look again, and I pointed the collector at new releases too: a random
sample of 1,200 games released in 2025 (`--new-since`, 97K reviews), not the top lists. With the collected
reviewers added to EASE's training and 1,537 of them held out ([eval/fresh.py](../eval/fresh.py)), every system
ranks the 1,011 games from 2023 on that the reviews cover, and has to find each held-out reviewer's ones
(test, share of hidden games found):

| | hits from the top lists (304) | random 2025 releases (679) |
|---|---|---|
| popularity | **0.967** | 0.000 |
| tags x popularity (the page now) | 0.605 | **0.085** |
| EASE + collected reviews | 0.306 | 0.077 |

Popularity finds the hits and never a new game that is not one. On those, co-likes from these reviews do no
better than tags (-0.009, interval -0.035 to +0.016). With the first 400 games EASE was ahead (0.148
against 0.102), inside the noise; tripling the sample took it away. The difference from the warm start eval
is the reviewers: there each new game's early players came with a full 2022 library, here a collected
reviewer has a handful of reviews among the games I happened to collect, too little to link a new game to
the rest.

The same eval prices the page's own knob, how much popularity weighs against tag similarity
(`log1p(positive reviews) ** power`, test):

| power | hits found | random 2025 releases found |
|---|---|---|
| 1 (the page) | 0.605 | **0.085** |
| 2 | 0.757 | 0.062 |
| 3 | 0.822 | 0.052 |
| 5 | 0.908 | 0.038 |

Every step buys hits with new games nobody has heard of, and both sides of each trade clear their
intervals. Recall over everyone rises with the power (0.261 to 0.325), because hits are what most people
liked; I keep power 1, since a "new for you" list that mostly shows the year's hits tells a player little
they did not know. That is a product choice the eval can price but not make.
Valve switched off the store's review endpoint on 22 October 2026, mid-project; the collector uses its Web
API replacement, `IUserReviewsService/GetAppReviews`.
