import pandas as pd
import matplotlib.pyplot as plt

pca_svm = pd.read_csv('src/results/experiment_two/cross_validation_pca_svm_15000.csv')

pca_svm.plot.box(x = 'param_pca__n_components', y = 'mean_test_score')
plt.title('PCA SVM accuracy as a function of number of \nPrincipal Components training size: 15000')

plt.savefig('src/results/experiment_two/experiment_two.png')
