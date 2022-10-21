from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#creates a confusion matrix for for the model and returns it, needs the trained model and the test data
def create_confusion_matrix(model, test_data):
    predictions = model.predict(test_data[0])
    return confusion_matrix(test_data[1], predictions)
