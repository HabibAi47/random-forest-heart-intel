# Random Forest Experiments (Question 2, Part B)

Experiments exploring the behaviour of Random Forests (Breiman, 2001) with scikit-learn on two datasets:

- **Tabular:** Heart Disease (Cleveland, UCI) - 297 samples, 13 features, binary label (`condition`)
- **Image:** Intel Image Classification - natural scenes in 6 classes (buildings, forest, glacier, mountain, sea, street)

## Experiments

For each dataset: standard train/test split, preprocessing, then

1. **Trees vs accuracy:** vary `n_estimators` over 1, 10, 50, 100, 300 and plot test accuracy.
2. **Decision Tree vs Random Forest:** default parameters, comparing test accuracy and training time.
3. **Comparative analysis with SVM:** Decision Tree vs Random Forest vs SVM (RBF and linear), with classification reports, confusion matrices and 5-fold cross-validation (Heart).

## Framework (end to end)

1. **Load data** - CSV with pandas (Heart); images with PIL resized to 32x32 (Intel).
2. **Split** - stratified 80/20 split (Heart); provided `seg_train` / `seg_test` folders (Intel).
3. **Preprocess** - `StandardScaler` for both datasets; PCA (95% variance) for Intel only. All preprocessing is fitted on training data only to avoid leakage.
4. **Train** - Decision Tree, Random Forest, SVM.
5. **Evaluate** - test accuracy, training time, classification report, confusion matrix, cross-validation.

## Results

**Accuracy vs number of trees**

| n_estimators | Heart (mean +/- std, 5 seeds) | Intel (single run) |
|---|---|---|
| 1 | 0.7433 +/- 0.0429 | 0.2322 |
| 10 | 0.8367 +/- 0.0531 | 0.3867 |
| 50 | 0.8467 +/- 0.0194 | 0.4633 |
| 100 | 0.8500 +/- 0.0105 | 0.4722 |
| 300 | 0.8467 +/- 0.0125 | 0.5233 |

**Model comparison - Intel Images**

| Model | Test accuracy | Training time (s) |
|---|---|---|
| Decision Tree | 0.3222 | 0.16 |
| Random Forest (100 trees) | 0.4722 | 0.34 |
| SVM (RBF) | 0.5644 | 0.13 |

**Model comparison - Heart Disease (5-fold CV accuracy)**

| Model | CV accuracy |
|---|---|
| Decision Tree | 0.7411 +/- 0.0594 |
| Random Forest | 0.8185 +/- 0.0501 |
| SVM (RBF) | 0.8254 +/- 0.0553 |

### Key findings

- Accuracy rises quickly with the first few trees and then plateaus on Heart (about 50-100 trees); the standard deviation across seeds shrinks from 0.043 to about 0.01, i.e. more trees mainly give stability.
- On Intel, accuracy was still improving at 300 trees: the harder, higher-dimensional problem needs a larger ensemble.
- Random Forest clearly beats a single Decision Tree on both datasets, at a modest extra training cost.
- SVM is best on Intel; on Heart, SVM and Random Forest are statistically tied.
- Intel accuracy is limited (47-56%) because flattened 32x32 pixels discard spatial structure; a CNN would be expected to do much better.

## Repository structure

```
.
├── notebook.ipynb      # step-by-step experiments (rename to your notebook file)
├── app11.py            # Streamlit web app (Intel + Heart)
├── requirements.txt
└── README.md
```

## Datasets (not included - too large)

- Heart Disease (Cleveland): https://archive.ics.uci.edu/dataset/45/heart+disease (or the Kaggle `heart_cleveland_upload.csv` version)
- Intel Image Classification: https://www.kaggle.com/datasets/puneet6060/intel-image-classification

Expected layout after downloading:

```
seg_train/seg_train/<class>/*.jpg
seg_test/seg_test/<class>/*.jpg
seg_pred/seg_pred/*.jpg
heart_cleveland_upload.csv
```

## How to reproduce

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Download the datasets and update the paths (at the top of the notebook, or in the app sidebar).
3. Run the notebook cells in order, **or** launch the web app:
   ```
   python -m streamlit run app11.py
   ```
4. In the app: choose a dataset in the sidebar, click **Load & Preprocess**, then run Experiments 1-3 from the tabs.


## Reference

Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
