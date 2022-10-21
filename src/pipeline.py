import load_mnist
import svm_model
import confusion_matrix
from sklearn import svm
def main():
    #load the data
    (training_data, test_data) = load_mnist.load_mnist()
    #create the model
    model = svm_model.train_model(training_data, svm.SVC())
    #create the confusion matrix
    print(confusion_matrix.create_confusion_matrix(model, test_data))

main()