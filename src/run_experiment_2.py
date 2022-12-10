import pandas as pd
import matplotlib.pyplot as plt

# df = pd.DataFrame({
#     'LDA':['100%','66%','33%'],
#     'PCA':['62%','32%','20%']
# })

df = pd.DataFrame({
    'Thresholds':['1%','5%', '10%'],
    'LDA Accuracy':[100,66,33],
    'PCA Accuracy':[62,32,20]
})

ax = plt.gca()

df.plot(kind='line', x='Thresholds', y='LDA Accuracy', ax=ax)
df.plot(kind='line', x='Thresholds', y='PCA Accuracy',color='red', ax=ax)
plt.savefig('src/results/experiment_two/lda_accuracy.png')
print(df)



# pca_svm = pd.read_csv('src/results/cross_validation_kernel_pca_svm_lars15000.csv')

# pca_svm.plot(kind = 'scatter', x = 'param_kernel_pca__n_components', y = 'mean_test_score')
# plt.title('KPCA SVM accuracy as a function of number of \nPrincipal Components training size: 15000')

# plt.savefig('src/results/experiment_two/kpca_svm_15000.png')
