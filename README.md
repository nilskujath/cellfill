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