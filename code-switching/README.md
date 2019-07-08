## Data

The data provided is derived from http://www.care4lang.seas.gwu.edu/cs2/call.html. Since the data is small I put it with the code under `code-switching/data` folder.

This data is a collection of tweets, in particular you have three files for the training set and three for the validation set:

- offsets_mod.tsv
- tweets.tsv
- data.tsv

The first file has the id information about the tweets, together with the tokens positions and the gold labels. The second has the ids and the actual tweet text, and the third
has the combination of the previous files, with the tokens of each sentence and the gold labels associated. More specifically, the columns are:

* `offsets_mod.tsv`:

```
tweet_id, user_id, start, end, gold label
```

* `tweets.tsv`

```
tweet_id, user_id, tweet text
```

* `data.tsv`:

```
tweet_id, user_id, start, end, token, gold label
```

The gold labels can be one of three:

* en
* es
* other
