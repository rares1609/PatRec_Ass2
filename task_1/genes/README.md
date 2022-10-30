# PatRec_Ass2
Pattern recognition Assignment2


Feature extraction/ dimensionality reduction:

- Gene dataset -> PCA (due to high dimensionality)

- Cat dataset -> Fourier Transform


Classification:

- Gene dataset -> Logistic Regression

- Cat dataset -> Decision Tree/Support Vector Machine


Clustering:

- Gene dataset -> K-means/DBSCAN

- Cat dataset -> K-means 


Evaluation metrices:

- Gene dataset -> 

			1. Classification: ROC-AUC
			2. Clustering: silhouette coefficient

- Cat dataset ->
			1. Classification: Confusion Matrix
			2. Clustering: mutual information values


Validation methods:

- Gene dataset -> Train:Test <=> 80%:20%

- Cat dataset ->  i-folds cross validation, i ∈ [3, 5, 7]


Ensemble:

TBD
