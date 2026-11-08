# Letter Recognition Classification

## Problem Statement

The goal is to tell which capital letter (A to Z) an image shows, using 16
numbers that were already computed from each image. I trained five different
classifiers on the same data, compared them on a held-out test set, and built a
small Streamlit app where you can upload a test CSV and see how each saved
model does.

## Dataset Description

I used the [UCI Letter Recognition dataset](https://archive.ics.uci.edu/dataset/59/letter+recognition).
It has 20,000 rows and 16 features, all whole numbers from 0 to 15. The target
is the letter, with 26 classes. The classes are close to balanced, with 734 to
813 rows per letter. There are no missing values. About 1,332 rows are exact
duplicates; I left them in as they came.

The features describe images of letters from 20 different fonts that were
distorted a bit. They are x_box, y_box, width, high, onpix, x_bar, y_bar, x2bar,
y2bar, xybar, x2ybr, xy2br, x_ege, xegvy, y_ege and yegvx. I changed the
hyphens in the original names to underscores and renamed the target column
(`lettr`) to `Class`.

I split the data 80/20 with stratification, using random state `20261106`. That
gives 16,000 training rows and 4,000 test rows. `test_data.csv` has all 16
features plus the `Class` column.

The dataset is under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Citation: Slate, D. (1991). Letter Recognition [Dataset]. UCI Machine Learning
Repository. [DOI: 10.24432/C5ZP40](https://doi.org/10.24432/C5ZP40).

## GitHub Repository Link

[GitHub Repository](https://github.com/2026AJ05974-Arun/MLAssignment2)

Live Streamlit app: _add the Streamlit Community Cloud link here after deploying_

The repository contains:

```text
MLAssignment2/
|-- app.py
|-- prepare_data.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model_metrics.csv
|-- .gitignore
|-- .streamlit/
|   |-- config.toml
|-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- gaussian_naive_bayes.joblib
    |-- support_vector_machine.joblib
```

`train_data.csv` is not in the repository because the app never needs it. It
is recreated by `prepare_data.py`. The saved models total about 21 MB (the SVM
file is the largest at about 18 MB), well under GitHub's 100 MB file limit.

## Models Used

All five models were trained on the same training rows and tested on the same
held-out rows. Since there are 26 classes, precision, recall and F1 are macro
averages. AUC is one-vs-rest, macro averaged, and computed from predicted
probabilities. MCC is the multiclass version. The numbers below come from my
local run, rounded to four decimals.

Settings used (everything else is the scikit-learn default):

| Model | Settings |
|---|---|
| Logistic Regression | `StandardScaler`, `max_iter=2000` |
| Decision Tree | no depth limit |
| KNN | `StandardScaler`, `n_neighbors=5` |
| Naive Bayes | `GaussianNB` |
| Support Vector Machine | `StandardScaler`, `SVC` with the default RBF kernel, wrapped in 5-fold sigmoid `CalibratedClassifierCV` |

All models use random state `20261106` where they have one. Predicted classes
are taken as the most probable class from `predict_proba`, so each model is
scored in one pass.

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.7738   | 0.9788 | 0.7744    | 0.7723 | 0.7721 | 0.7648 |
| Decision Tree          | 0.8718   | 0.9331 | 0.8727    | 0.8714 | 0.8717 | 0.8666 |
| KNN                    | 0.9370   | 0.9948 | 0.9381    | 0.9367 | 0.9369 | 0.9345 |
| Naive Bayes (Gaussian) | 0.6322   | 0.9533 | 0.6435    | 0.6312 | 0.6271 | 0.6183 |
| Support Vector Machine | 0.9333   | 0.9957 | 0.9353    | 0.9328 | 0.9332 | 0.9306 |

### Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | 0.7738 accuracy. A straight-line boundary is not enough to separate 26 letters, even though the AUC (0.9788) looks good. The high AUC means it ranks the right letter well, but its top pick is wrong too often. It is a linear model, so it cannot capture the curved boundaries between similar letters. |
| Decision Tree | 0.8718 accuracy, but the lowest AUC (0.9331). It has no depth limit, so it fits the training data closely, and its probabilities come from leaves that are almost pure and so quite coarse. That gives good hard predictions but poor ranking, which probably explains the low AUC. |
| KNN | Best accuracy (0.9370) and also best precision, recall, F1 and MCC. Similar letters sit close together once the features are scaled, which suits a distance-based method. Its remaining mistakes are letters with similar shapes, such as Q predicted as O (11 times), J as I (10) and P as F (8) in the test set. It has no training step, but it has to compare every test row against all 16,000 training rows when predicting, so it is slower to evaluate. |
| Naive Bayes | Worst model (0.6322 accuracy, 0.6183 MCC). The features are small whole numbers and are correlated with each other (for example width, high and the box statistics), so the Gaussian and independence assumptions do not really hold. It is the fastest model and still a usable baseline, but it is well behind the others. |
| Support Vector Machine | Very close to KNN (0.9333 accuracy) and the highest AUC (0.9957). The RBF kernel handles the non-linear boundaries well, and the calibration step gives it good probabilities. Its most common mistakes are different from KNN's (E as G 9 times, K as R and H as D 7 times each). It is the largest saved model (about 18 MB) and slower to train than the others. |
| Overall, Winner for your dataset? | KNN, because it is best on accuracy, F1 and MCC. SVM is only slightly behind (0.0037 lower accuracy, which is about 15 test rows out of 4,000) and has the best AUC, so on one split I would not call the gap meaningful. Ranking by accuracy: KNN, SVM, Decision Tree, Logistic Regression, Naive Bayes. |

Logistic Regression, KNN and SVM have a `StandardScaler` inside their saved
pipelines. For SVM I wrapped the model in 5-fold sigmoid calibration so it can
give probabilities, which lets AUC be computed the same way for all five models.

## Reproduce the Local Run

`prepare_data.py` downloads the UCI archive, checks the rows, features, classes
and missing values, and writes the train and test CSVs. `train_models.py` fits
each classifier on the training CSV only, evaluates on the test CSV, saves the
models in `model/` and writes `model_metrics.csv`.

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe prepare_data.py
.venv\Scripts\python.exe train_models.py
.venv\Scripts\python.exe -m streamlit run app.py
```

On the BITS Virtual Lab the steps are the same: install the dependencies, run
both scripts, then start the app. If the lab cannot reach UCI, copy the train
and test CSVs over and run only `train_models.py`.

## Streamlit App

The app uses the saved models and starts with the bundled `test_data.csv`. You
can upload your own CSV instead, as long as it has the same 16 numeric columns
and a `Class` column. If the file does not have all 26 letters, AUC is averaged
over the letters present. Nothing is retrained on the uploaded data.

The sidebar has the CSV upload first and the model dropdown below it, so you
can upload a file once and switch between models without uploading it again.
There is also a button to download the sample test CSV.

The main page shows, from top to bottom:

1. Labels with the data source, row count, number of letters and the model
   with the best F1.
2. Six metric cards (Accuracy, AUC, Precision, Recall, F1, MCC) for the
   selected model. Each card shows how far the model is behind the best of the
   five, or marks it as the best.
3. Four tabs:
   - **Model comparison**: a table of all five models, with the best score for
     each metric highlighted in green, and a bar chart for a chosen metric.
   - **Confusion matrix**: a heatmap with correct predictions in green and
     mistakes in red, plus the ten most common mistakes.
   - **Per-letter report**: a bar chart of F1 per letter and the full
     classification report, with the weakest letter highlighted.
   - **Data preview**: the first 100 rows of the data being evaluated.

Results are cached, so switching models for the same file is quick.

The UI follows a Google Material style: Google blue as the main color, Google
Sans and Roboto fonts, rounded cards and buttons, and a multicolor "Letters"
heading. The base theme is set in `.streamlit/config.toml`, and the extra
styling is CSS inside `app.py`. The fonts are loaded from Google Fonts, so
without internet access the app falls back to a standard sans-serif font.

To deploy, push the repo to GitHub, create a new app on
[Streamlit Community Cloud](https://streamlit.io/cloud) from the `main` branch
with `app.py` as the entry point, and check that the live link opens without
errors.

Deployment notes:

- `requirements.txt` pins the exact versions the models were saved with and
  tested against (Python 3.11). In the app's advanced settings pick Python
  3.11 or newer. The `.joblib` files only load reliably with the same
  scikit-learn version, so keep `scikit-learn==1.9.1`. If a pin cannot be
  installed, retrain with `train_models.py` on the version that is available.
- The `matplotlib` and `seaborn` packages from the assignment example are not
  needed, because the charts use Streamlit's built-in charts.
- Live app link: add the Streamlit Community Cloud URL at the top of this
  README after deploying.
