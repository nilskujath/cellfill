"""
This script reimplements what ldl/ldl.jl does with JudiLing (book ch. 4, 6, 7, 8),
with the same settings, and writes the same two result files for the Python run.
"""

import enum
import pathlib
import time

import numpy as np
import numpy.typing as npt
import pandas as pd

type FloatMatrix = npt.NDArray[np.float64]  # matrix whose elements are float64
type BoolVector = npt.NDArray[np.bool_]  # vector whose elements are bools
type BoolMatrix = npt.NDArray[np.bool_]  # matrix whose elements are bools

type CueIndex = int  # identifier of a column in C that corresponds to a cue
type CuePath = list[CueIndex]  # a way to represent a form as a sequence of cues

type NgramSize = int
type Form = str  # a form as a string of symbols, e.g. phones or characters
type Ngram = str  # a window of n consecutive symbols


ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
DATA: pathlib.Path = ROOT / "data"
RESULTS: pathlib.Path = ROOT / "results"


class FrozenSplit:
    """
    Class used to load the frozen French split written by the scripts in data/:
    the train and test forms (column Phon2) and their semantic matrices.
    """

    def __init__(self, data_dir: pathlib.Path, language: str) -> None:
        train: pd.DataFrame = pd.read_csv(data_dir / f"{language}_train.csv")
        test: pd.DataFrame = pd.read_csv(data_dir / f"{language}_test.csv")
        self.train_forms: list[Form] = train.Phon2.tolist()
        self.test_forms: list[Form] = test.Phon2.tolist()
        self.S_train: FloatMatrix = self._load_semantic_matrix_S(
            data_dir / f"{language}_S_train.bin"
        )
        self.S_test: FloatMatrix = self._load_semantic_matrix_S(
            data_dir / f"{language}_S_test.bin"
        )

    @staticmethod
    def _load_semantic_matrix_S(path: pathlib.Path) -> FloatMatrix:
        """
        Read a matrix written by data/simulate_french_S.jl: 8-byte floats laid down
        row by row, no header; rows and columns in a companion .shape file.
        """
        shape_file: pathlib.Path = path.with_name(path.name + ".shape")
        rows, cols = map(int, shape_file.read_text().split())
        return np.fromfile(path, dtype=np.float64).reshape(rows, cols)


class NgramInventory:
    """
    Class used to create and hold a shared inventory of n-grams over a collection of
    forms.
    """

    FORM_BOUNDARY_MARKER: str = "#"

    @staticmethod
    def split_into_ngrams(form: Form, n: NgramSize) -> list[Ngram]:
        """
        Split a form into its constituting n-grams.

        Example (n = 3): 'Zab@dOn' -> ['#Za', 'Zab', 'ab@', 'b@d', '@dO', 'dOn', 'On#']
        """
        padded_form: str = (
            NgramInventory.FORM_BOUNDARY_MARKER
            + form
            + NgramInventory.FORM_BOUNDARY_MARKER
        )
        return [padded_form[i : i + n] for i in range(len(padded_form) - n + 1)]

    def __init__(self, forms: list[Form], n: NgramSize) -> None:
        self.n: NgramSize = n

        self.ngram_to_index: dict[Ngram, CueIndex] = {}
        next_index: CueIndex = 0
        for form in forms:
            for ngram in self.split_into_ngrams(form=form, n=self.n):
                if ngram not in self.ngram_to_index:
                    self.ngram_to_index[ngram] = next_index
                    next_index += 1

        self.index_to_ngram: dict[CueIndex, Ngram] = {
            index: ngram for ngram, index in self.ngram_to_index.items()
        }

        self.ngram_adjacency_matrix: BoolMatrix = np.zeros(
            (num_ngrams := len(self.ngram_to_index), num_ngrams), dtype=np.bool_
        )
        for form in forms:
            ngrams: list[Ngram] = self.split_into_ngrams(form=form, n=self.n)
            for left, right in zip(ngrams, ngrams[1:]):
                row_index: CueIndex = self.ngram_to_index[left]
                column_index: CueIndex = self.ngram_to_index[right]
                self.ngram_adjacency_matrix[row_index, column_index] = True

        self.is_form_initial_ngram: BoolVector = np.zeros(num_ngrams, dtype=np.bool_)
        self.is_form_final_ngram: BoolVector = np.zeros(num_ngrams, dtype=np.bool_)
        for ngram, index in self.ngram_to_index.items():
            if ngram[0] == self.FORM_BOUNDARY_MARKER:
                self.is_form_initial_ngram[index] = True
            if ngram[-1] == self.FORM_BOUNDARY_MARKER:
                self.is_form_final_ngram[index] = True

    def build_form_matrix(self, forms: list[Form]) -> FloatMatrix:
        """
        Build the form matrix (referred to as C in Heitmeier et al. 2026) for a list of
        forms.

        One row per form and one column per n-gram in the inventory.
        1.0 where form contains the n-gram, 0.0 otherwise.
        """
        form_matrix: FloatMatrix = np.zeros((len(forms), len(self.ngram_to_index)))
        for row_index, form in enumerate(forms):
            for ngram in self.split_into_ngrams(form=form, n=self.n):
                form_matrix[row_index, self.ngram_to_index[ngram]] = 1.0

        return form_matrix

    def gold_cue_path(self, form: Form) -> CuePath:
        """
        Get the column indices of a form's n-grams, in order (gold_ind in JudiLing).
        """
        return [
            self.ngram_to_index[ngram]
            for ngram in self.split_into_ngrams(form=form, n=self.n)
        ]

    def cue_path_to_form(self, cue_path: CuePath) -> Form:
        """
        Given a list of column indices of ngrams, stich the corresponding form
        together.

        Each n-gram after the first repeats the last n - 1 symbols of the one before it
        and adds one new symbol, so the form is the first n-gram plus that one new
        symbol from each further n-gram.
        """
        form: Form = self.index_to_ngram[cue_path[0]]
        for index in cue_path[1:]:
            form += self.index_to_ngram[index][-1]
        return form.strip(self.FORM_BOUNDARY_MARKER)


class LearningMode(enum.Enum):
    ENDSTATE = enum.auto()
    INCREMENTAL = enum.auto()


class LinearNetwork:
    """
    Class used to learn a two-layer linear network from paired cue and outcome rows
    (see Baayen et al. 2019; Heitmeier et al. 2026, ch. 1 and 6).

    Row i of `cues` is the input of learning event i and row i of `outcomes` its
    target.
    Fitting yields a weight matrix W such that cues @ W approximates outcomes.
    The same class gives the comprehension network (cues = C, outcomes = S, W = F) and
    the production network (cues = S, outcomes = C, W = G); the direction is only a
    matter of which matrix is passed as which.
    """

    def __init__(
        self,
        cues: FloatMatrix,
        outcomes: FloatMatrix,
        mode: LearningMode = LearningMode.ENDSTATE,
    ) -> None:
        self.cues: FloatMatrix = cues
        self.outcomes: FloatMatrix = outcomes
        self.mode: LearningMode = mode

    def fit(self) -> FloatMatrix:
        """
        Learn and return the weight matrix W according to the learning mode.
        """

        match self.mode:
            case LearningMode.ENDSTATE:
                return self._fit_endstate()
            case LearningMode.INCREMENTAL:
                return self._fit_incremental()

    def _fit_endstate(self, ridge: float = 0.02) -> FloatMatrix:
        """
        Compute the endstate of learning directly, with the least-squares formula of
        Heitmeier et al. (2026, ch. 6; see `make_transform_matrix` in JudiLing).

        The ridge term is a small number added to the diagonal so that the system
        always has a unique solution; `0.02` is JudiLing's default.
        """
        X: FloatMatrix = self.cues
        Y: FloatMatrix = self.outcomes
        A: FloatMatrix = X.T @ X + ridge * np.eye(X.shape[1])
        B: FloatMatrix = X.T @ Y
        return np.linalg.solve(A, B)

    def _fit_incremental(self) -> FloatMatrix:
        raise NotImplementedError


class Comprehension:
    """
    Class used to map forms to meanings with a linear network (F in Heitmeier et al.
    2026) and to evaluate the mapping on held-out forms (book ch. 7).
    """

    def __init__(
        self,
        C_train: FloatMatrix,
        S_train: FloatMatrix,
        mode: LearningMode = LearningMode.ENDSTATE,
    ) -> None:
        self.S_train: FloatMatrix = S_train
        self.F: FloatMatrix = LinearNetwork(
            cues=C_train, outcomes=S_train, mode=mode
        ).fit()

    def predict(self, C: FloatMatrix) -> FloatMatrix:
        """
        Predict the meaning vectors of forms (S_hat in Heitmeier et al. 2026).
        """
        return C @ self.F

    def accuracy(self, C_test: FloatMatrix, S_test: FloatMatrix) -> float:
        """
        Share of test forms whose predicted meaning is closest, by correlation, to
        their own meaning among all meanings in train and test (eval_SC in JudiLing).
        """
        S_all: FloatMatrix = np.vstack([S_test, self.S_train])  # test row i at i
        correlations: FloatMatrix = self.correlation_matrix(self.predict(C_test), S_all)
        nearest: npt.NDArray[np.intp] = correlations.argmax(axis=1)
        return float(np.mean(nearest == np.arange(len(S_test))))

    @staticmethod
    def correlation_matrix(A: FloatMatrix, B: FloatMatrix) -> FloatMatrix:
        """
        Pearson correlation of every row of A with every row of B, as a matrix of shape
        (rows of A, rows of B).
        """
        A_standardised: FloatMatrix = (A - A.mean(axis=1, keepdims=True)) / A.std(
            axis=1, keepdims=True
        )
        B_standardised: FloatMatrix = (B - B.mean(axis=1, keepdims=True)) / B.std(
            axis=1, keepdims=True
        )
        return A_standardised @ B_standardised.T / A.shape[1]


def main() -> None:
    split: FrozenSplit = FrozenSplit(DATA, "french")

    inventory: NgramInventory = NgramInventory(
        forms=split.train_forms + split.test_forms, n=3
    )
    print("n-grams in inventory:", len(inventory.ngram_to_index))
    C_train: FloatMatrix = inventory.build_form_matrix(split.train_forms)
    C_test: FloatMatrix = inventory.build_form_matrix(split.test_forms)

    comprehension: Comprehension = Comprehension(C_train, split.S_train)
    comprehension_accuracy: float = comprehension.accuracy(C_test, split.S_test)
    print(f"comprehension: {comprehension_accuracy:.3f}")


if __name__ == "__main__":
    main()
