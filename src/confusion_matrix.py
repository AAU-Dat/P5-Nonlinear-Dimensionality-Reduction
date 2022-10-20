from sklearn.metrics import confusion_matrix
#from sklearn import svm
#import load_mnist as load
from sklearn.model_selection import train_test_split

# def test_create_confusion_matrix():
#     (training_data, test_data) = load.load_mnist()
#     model = svm.SVC(gamma=0.001)
#     model.fit(training_data[0], training_data[1])
#     cm = create_confusion_matrix(model)

#creates a confusion matrix for for the model and returns it, needs the trained model and the test data
def create_confusion_matrix(model, test):
    predictions = model.predict(test[0])
    return confusion_matrix(test[1], predictions)
