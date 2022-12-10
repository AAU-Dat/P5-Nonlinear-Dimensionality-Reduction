import matplotlib.pyplot as plt
import pandas as pd


pca_svm = pd.read_csv('src/results/experiment_two/cross_validation_pca_svm_15000.csv')

x_axis = 'param_pca__n_components'
y_axis = 'mean_test_score'

diff  = []
for i in range(0, 49):
    diff.append(pca_svm[y_axis][com+1]-pca_svm[y_axis][com])

new_diff = diff[:48]
# new_diff.reverse()
# print(new_diff)



plt.bar(range(0,48), new_diff)
plt.title('PCA SVM accuracy as a function of number of \nPrincipal Components training size: 15000')

plt.savefig('src/results/experiment_two/bar_pca_svm_15000.png')

# print(pca_svm[y_axis][0])