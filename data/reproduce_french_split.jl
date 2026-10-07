# reproduce_french_split.jl implements the careful train/test (see JudiLings function
# `.loading_data_careful_split`) split of the book's §12.8 French model.
# This follows notebooks/16_Ch12.8_french.ipynb in
# github.com/quantling/JudiLing_tutorial:
# 2000 held-out forms, seed 314.
# Every triphone (column Phon2) and every feature value in test also occurs in train.
# Writes french_train.csv and french_test.csv next to french.csv.


using DataFrames, CSV, JudiLing

const DATA = @__DIR__

train, test = JudiLing.loading_data_careful_split(
    joinpath(DATA, "french.csv"), "french", joinpath(DATA, "careful_tmp"),
    ["Lexeme", "Person", "Number", "Gender", "Tense", "Aspect", "Class", "Mood"],
    n_grams_target_col = "Phon2", grams = 3,
    val_sample_size = 2000, random_seed = 314)
rm(joinpath(DATA, "careful_tmp"), recursive = true, force = true)

CSV.write(joinpath(DATA, "french_train.csv"), train)
CSV.write(joinpath(DATA, "french_test.csv"), test)
println("french_train: ", nrow(train), "  french_test: ", nrow(test))