import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
%matplotlib inline

# Different Candidate Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# Classification Performance
from sklearn import model_selection
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Define a function to Describe Features extracted
def describeData(X,y):
    print('Total number of images: {}'.format(len(X)))
    print('Number of Benign Images: {}'.format(np.sum(y==0)))
    print('Number of Malignant Images: {}'.format(np.sum(y==1)))
    print('Percentage of positive images: {:.2f}%'.format(100*np.mean(y)))
    print('Image shape (Samples, Rows, Columns, Features): {}'.format(X[0].shape))
    print()


# Spliting the features into training and testing set at 80%/20% ratio
def train_test_group_split(X, y, p):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2)
    for train_index, test_index in gss.split(X, y, p):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        p_train, p_test = p[train_index], p[test_index]
    return X_train, y_train, X_test, y_test, p_train, p_test


def feature_dimension(model, factor):
    X = np.load('../data/features_' + model + '/' + str(factor) + '/X.npy')
    y = np.load('../data/features_' + model + '/' + str(factor) + '/y.npy')
    p = np.load('../data/features_' + model + '/' + str(factor) + '/p.npy')
    print('Discription of features extracted from ' + str(factor) + 'x images by model ' + model + ":")
    X_train, y_train, X_test, y_test, p_train, p_test = train_test_group_split(X, y, p)
    print('Training Set')
    describeData(X_train, y_train)
    print('Testing Set')
    describeData(X_test, y_test)
    print("-----------------------------------------------------------------------")


# Showing the features dimension after train_test_split
magnification_factors = ['40', '100', '200', '400']
models_list = ['xception', 'vgg16', 'vgg19']
for factor in magnification_factors:
    for model in models_list:
        feature_dimension(model, factor)

# Describing abbreviation and its corresponding algorithm
def defineClassifiers():
    """
    This function just defines each abbreviation used in the previous function (e.g. LR = Logistic Regression)
    """
    print('')
    print('LR = LogisticRegression')
    print('LDA = LinearDiscriminantAnalysis')
    print('DTC = DecisionTreeClassifier')
    print('RF = RandomForestClassifier')
    print('GBC = GradientBoostingClassifier')
    print('KNN = KNeighborsClassifier')
    print('SVM = Support Vector Machine SVC')
    print('LSVM = LinearSVC')
    print('GNB = GaussianNB')
    print('')
    return
defineClassifiers()


classifiers = {'LR': (LogisticRegression(), LR_parameters),
               'LDA': LinearDiscriminantAnalysis(),
               'DTC': DecisionTreeClassifier(),
               'RF': RandomForestClassifier(),
               'GBC': GradientBoostingClassifier(),
               'KNN': KNeighborsClassifier(),
               'SVM': (SVC(), SVM_parameters),
               'LSVM': LinearSVC(),
               'GNB': GaussianNB()}

SVM_parameters = {'kernel':('linear', 'rbf'),
                  'C':(1, 10)}

#LR_parameters = {'penalty': ('l1', 'l2'),
                 #'C' : (0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000),
                # 'class_weight' : ({1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}),
                # 'solver' : ('liblinear', 'saga')}

LR_parameters = {'solver' : ('liblinear', 'saga')}


# Tuning hyperparameter to get optimized classifier using GridSearchCV
def optimized_classifier(X_train, y_train, p_train, classifier_name):
    """Function that use logistic regression as classifier and return Cross-validated F1 Score"""
    # Hyperparameter Tuning using group k folds cross validation
    group_kfold = GroupKFold(n_splits=3)
    parameters = classifiers[classifier_name][1]
    model = classifiers[classifier_name][0]
    grid = GridSearchCV(estimator=model,
                       param_grid=parameters,
                       cv=group_kfold,
                       scoring=['roc_auc','f1'],
                       verbose=1,
                       n_jobs=-1,
                       refit='f1')
    grid_result = grid.fit(X_train, y_train, p_train)
    # summarize results
    print("Best F1 score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_f1']
    stds = grid_result.cv_results_['std_test_f1']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_estimator_


# Test the Function
X = np.load('../data/features_xception/40/X.npy')
X_Shape = X.shape[1]*X.shape[2]*X.shape[3]*X.shape[4]
X_Flat = X.reshape(X.shape[0], X_Shape)
y = np.load('../data/features_xception/40/y.npy')
p = np.load('../data/features_xception/40/p.npy')
optimized_classifier(X_Flat, y, p, 'LR' )


# Test Model Performance on Testing Set
def test_performance(X_test, y_test, p_test, classifier):
    """This function will give performance measurement for a given classifier"""
    # Classification report
    labels = ["Benign", "Malignant"]
    y_predict = classifier.predict(X_test)
    print(classification_report(y_test, y_predict, target_names=labels))
    # Image Level F1 Score
    print("Image Level F1 Score: {:.4f}".format(max(f1_score(y_test, y_predict, pos_label=1, average=None))))
    # Patient level F1 score
    result = {}
    for true, pred, people in zip(y_test, y_predict, p_test):
        if people in result:
            result[people][0].append(true)
            result[people][1].append(pred)
        else:
            if pred == true:
                result[people] = [[true], [pred]]
    f1_scores = []
    for people, true_pred in result.items():
        true_class = np.asarray(true_pred[0])
        predicted_class = np.asarray(true_pred[1])
        f1_scores.append(max(f1_score(true_class, predicted_class, pos_label=1, average=None)))
    mean_f1_patient_level = sum(f1_scores) / len(f1_scores)
    print("Patient Level Mean F1 Score: {:.4f}".format(mean_f1_patient_level))
    # ROC curve
    y_predict_proba = classifier.predict_proba(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_predict_proba[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# Function returning pre_trained model performance
def Model_Performance(model, magnification_factor):
    """This function will compare performance of 3 pre-trained models"""
    # Make Features from Training Set 1D for compatability with standard classifiers
    X = np.load('../data/features_' + model + '/' + magnification_factor + '/X.npy')
    X_Shape = X.shape[1] * X.shape[2] * X.shape[3] * X.shape[4]
    X_Flat = X.reshape(X.shape[0], X_Shape)
    y = np.load('../data/features_' + model + '/' + magnification_factor + '/y.npy')
    p = np.load('../data/features_' + model + '/' + magnification_factor + '/p.npy')
    # Split into training and testing set with defined
    X_trainFlat, y_train, X_testFlat, y_test, p_train, p_test = train_test_group_split(X_Flat, y, p)
    # Hyperparameter tuning
    # optimized_classifier(X_trainFlat, y_train, p_train, 'LR' )
    # classifier = optimized_classifier(X_trainFlat, y_train, p_train, 'SVM')
    # Classification Performance on testing data
    print('Performance of pre-trained CNN model ' + model + ' for images at ' + magnification_factor + 'x:')
    classifier = LogisticRegression(solver='liblinear').fit(X_trainFlat, y_train)
    test_performance(X_testFlat, y_test, p_test, classifier)


# Classification performance of pre-trained models for images at each magnification level
for factor in magnification_factors:
    Model_Performance('xception', factor)
    Model_Performance('vgg16', factor)
    Model_Performance('vgg19', factor)
    print("--------------------------------------------------------------------------------------------")

