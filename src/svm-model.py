from sklearn import svm
import load_mnist
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


def train_model(training_data, model):
    train_image = training_data[0]
    train_label = training_data[1]
    model.fit(train_image, train_label)
    return model


def hyperparam_tuning_svm(model):

    # creating a KFold object with 5 splits
    folds = KFold(n_splits=5, shuffle=True, random_state=10)

    # specify range of hyperparameters
    # Set the parameters by cross-validation
    hyper_params = [{'gamma': [1e-2, 1e-3, 1e-4],
                    'C': [5, 10]}]

    # set up GridSearchCV()
    model_cv = GridSearchCV(estimator=model,
                            param_grid=hyper_params,
                            scoring='accuracy',
                            cv=folds,
                            verbose=1,
                            return_train_score=True)

    # fit the model
    return model_cv

