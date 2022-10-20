from sklearn.metrics import confusion_matrix
from sklearn import svm
import load_mnist as load
from sklearn.model_selection import train_test_split

def test_create_confusion_matrix():
    (training_data, test_data) = load.load_mnist()
    model = svm.SVC(gamma=0.001)
    model.fit(training_data[0], training_data[1])
    cm = create_confusion_matrix(model)


def create_confusion_matrix(model, test_images, test_labels):
    predictions = model.predict(test_images)
    return confusion_matrix(test_labels, predictions)
