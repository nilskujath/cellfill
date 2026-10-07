# ldl.jl implements a reference run of the book's §12.8 French triphone model with
# JudiLing, on the frozen split (data/french_train.csv, french_test.csv) and semantic
# matrix (data/french_S_*.bin).
# Follows notebooks/16_Ch12.8_french.ipynb in github.com/quantling/JudiLing_tutorial;
# the learn_paths settings are those of the book's Table 12.5's caption (threshold
# 0.01, tolerance mode with at most one weak cue).
# Expected: comprehension 0.882, production 0.696.
# Writes results/judiling_french_test.csv (JudiLing's predicted form for every
# test row) and results/judiling_french_numbers.txt.

using DataFrames, CSV, JudiLing

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA = joinpath(ROOT, "data")
const RESULTS = joinpath(ROOT, "results")
mkpath(RESULTS)

function load_matrix(path)
    r, c = parse.(Int, split(readline(path * ".shape")))
    M = Array{Float64}(undef, c, r)
    read!(path, M)
    permutedims(M)
end

train = CSV.read(joinpath(DATA, "french_train.csv"), DataFrame)
test  = CSV.read(joinpath(DATA, "french_test.csv"), DataFrame)
S_train = load_matrix(joinpath(DATA, "french_S_train.bin"))
S_test  = load_matrix(joinpath(DATA, "french_S_test.bin"))

# Form matrices (ch. 4): triphones on Phon2, one cue index shared by train and test.
cue_train, cue_test = JudiLing.make_combined_cue_matrix(
    train, test, grams = 3, target_col = "Phon2")
println("triphone cues: ", size(cue_train.C, 2))

# Comprehension (ch. 6–7): F maps C to S; test evaluated against train ∪ test.
F = JudiLing.make_transform_matrix(cue_train.C, S_train)
Shat_test = cue_test.C * F
acc_comp = JudiLing.eval_SC(Shat_test, S_test, S_train)
println("comprehension: ", acc_comp)

# Production (ch. 6, 8): G maps S to C; learn_paths orders the triphones.
G = JudiLing.make_transform_matrix(S_train, cue_train.C)
Chat_test = S_test * G
max_t = JudiLing.cal_max_timestep(train, test, "Phon2")

prod = JudiLing.learn_paths(
    train, test, cue_train.C, S_test, F, Chat_test,
    cue_test.A, cue_train.i2f, cue_train.f2i,
    max_t = max_t, threshold = 0.01, grams = 3, target_col = "Phon2",
    is_tolerant = true, tolerance = -0.1, max_tolerance = 1, verbose = false)
acc_prod = JudiLing.eval_acc(prod, cue_test)
println("production: ", acc_prod)

# Output (ch. 9).
JudiLing.write2csv(prod, test, cue_train, cue_test,
    joinpath(RESULTS, "judiling_french_test.csv"), grams = 3, target_col = :Phon2)
open(joinpath(RESULTS, "judiling_french_numbers.txt"), "w") do io
    println(io, "comprehension ", acc_comp)
    println(io, "production ", acc_prod)
end