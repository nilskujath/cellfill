# Readme

## Data

French verb paradigms from Beukman (2020), as distributed with Heitmeier, Chuang & 
Baayen (2026); stored at OSF project `vpdt2`, no licence stated, so the data is not in 
this repo, though I have provided a small script to download it called
`data/get_french_data.sh`.
Run it like this:
```bash
bash data/get_french_data.sh
```


`data/reproduce_french_split.jl` reproduces the train/test split of the book's §12.8
French model (notebook `16_Ch12.8_french.ipynb` in
github.com/quantling/JudiLing_tutorial):
2000 held-out forms, seed 314.
Every triphone (column Phon2) and every feature value in test also occurs in train.
Writes `data/french_train.csv` and `data/french_test.csv` next to `data/french.csv`.
Requires Julia 1.10 with the packages JudiLing, CSV, DataFrames.
Run it like this:

```bash
julia data/reproduce_french_split.jl
```


`data/simulate_french_S.jl` creates the simulated semantic matrix S for the split,
following the same notebook (`make_combined_S_matrix`, 1000 dimensions).
The  notebook does not seed this step, so I seed it (314 again) to make the files
reproducible.
Writes `data/french_S_train.bin` and `data/french_S_test.bin`, plus a `.shape` file 
with the dimensions next to each; row i of each matrix belongs to row i of the 
corresponding csv.
Run it like this:

```bash
   julia data/simulate_french_S.jl
```


## LDL

`ldl/ldl.jl` is the reference run of the book's §12.8 French triphone model with
JudiLing itself, on the frozen split and semantic matrix from `data/`.
It follows the same notebook (`16_Ch12.8_french.ipynb`); the `learn_paths` settings are
the ones of the book's Table 12.5 (threshold 0.01, tolerance mode with at most one
weak cue).
It prints comprehension and production accuracy on the 2000 test forms and writes
`results/judiling_french_test.csv` (JudiLing's top 10 candidate forms for every test
form, with supports) and `results/judiling_french_numbers.txt` (the two accuracies).
These two files are committed, since they are the reference the Python reimplementation 
is checked against.
Run it like this:

```bash
julia ldl/ldl.jl
```

Results:

| model | comprehension | production |
|---|---|---|
| JudiLing (`ldl/ldl.jl`) | 0.882 | 0.696 |
| book, Table 12.5 | 0.882 | 0.696 |