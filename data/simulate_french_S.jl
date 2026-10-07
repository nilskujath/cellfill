# simulate_french_S.jl implements the creation of the simulated semantic matrix S for
# the French split.
# Follows notebooks/16_Ch12.8_french.ipynb in github.com/quantling/JudiLing_tutorial
# (JudiLing.make_combined_S_matrix, 1000 dimensions): one random vector per lexeme
# and per feature value, a form's vector is the sum of the vectors of its parts.
#
# The notebook does not seed the random generator before this step, so the S  behind
# the book's Table 12.5 was a one-off draw that is not stored anywhere and cannot be
# reproduced.
# I seed it here (any fixed value will do; 314 is reused from the split) so that the
# files written below are the same on every run, and the Python reimplementation can be
# checked against the Julia run form by form.
#
# Writes french_S_train.bin and french_S_test.bin (Float64, row-major) and a .shape
# file with "rows cols" next to each.

using DataFrames, CSV, JudiLing, Random

const DATA = @__DIR__
const FEATURES = ["Person", "Number", "Gender", "Tense", "Aspect", "Class", "Mood"]

train = CSV.read(joinpath(DATA, "french_train.csv"), DataFrame)
test  = CSV.read(joinpath(DATA, "french_test.csv"), DataFrame)

Random.seed!(314)
S_train, S_test = JudiLing.make_combined_S_matrix(
    train, test, ["Lexeme"], FEATURES, ncol = 1000)

function dump_matrix(M, path)
    open(path, "w") do io
        write(io, permutedims(Matrix{Float64}(M)))
    end
    open(path * ".shape", "w") do io
        println(io, size(M, 1), " ", size(M, 2))
    end
end

dump_matrix(S_train, joinpath(DATA, "french_S_train.bin"))
dump_matrix(S_test,  joinpath(DATA, "french_S_test.bin"))
println("S_train: ", size(S_train), "  S_test: ", size(S_test))