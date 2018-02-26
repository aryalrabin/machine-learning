# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals_ide as vs

def donor() :

    # Load the Census dataset
    data = pd.read_csv("census.csv")

    # Success - Display the first record
    # display(data.head(n=1))

    # TODO: Total number of records
    n_records = np.size(data['income'])

    # TODO: Number of records where individual's income is more than $50,000
    n_greater_50k = np.size(np.where(data['income'] == '>50K'))

    # TODO: Number of records where individual's income is at most $50,000
    n_at_most_50k = np.size(np.where(data['income'] == '<=50K'))

    # TODO: Percentage of individuals whose income is more than $50,000
    greater_percent = float (n_greater_50k/n_records) * 100

    # Print the results
    # print("Total number of records: {}".format(n_records))
    # print("Individuals making more than $50,000: {}".format(n_greater_50k))
    # print("Individuals making at most $50,000: {}".format(n_at_most_50k))
    # print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))

    # Split the data into features and target label
    income_raw = data['income']
    features_raw = data.drop('income', axis = 1)

    # Visualize skewed continuous features of original data
    # vs.distribution(data)

    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data = features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

    # display(data.head(n=1))
    # display(features_log_transformed.head(n=1))
    # Visualize the new log distributions
    # vs.distribution(features_log_transformed, transformed = True)

        # Import sklearn.preprocessing.StandardScaler
    from sklearn.preprocessing import MinMaxScaler

    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler() # default=(0, 1)
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

    # Show an example of a record with scaling applied
    # display(features_log_minmax_transform.head(n = 5))

    features_final = pd.get_dummies(features_log_minmax_transform)
    # display(features_final.head())

        # TODO: Encode the 'income_raw' data to numerical values
    income = income_raw.map(lambda x: 0 if x == "<=50K" else 1)
    display(income.head())

    # Print the number of features after one-hot encoding
    encoded = list(features_final.columns)
    # print("{} total features after one-hot encoding.".format(len(encoded)))

    # print (np.logspace(-1, 1, 3));
    # Import train_test_split
    from sklearn.cross_validation import train_test_split

    # Split the 'features' and 'income' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                        income,
                                                        test_size = 0.2,
                                                        random_state = 0)
    from sklearn.metrics import make_scorer, fbeta_score
    from sklearn.grid_search import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(random_state=21)

    # TODO: Create the parameters list you wish to tune
    parameters = {'C': np.logspace(-1, 1, 11), 'penalty':['l1', 'l2']}

    # TODO: Make an fbeta_score scoring object
    scorer = make_scorer(fbeta_score, beta=0.5)

    # TODO: Perform grid search on the classifier using 'scorer' as the scoring method
    grid_obj = GridSearchCV(clf, parameters, scoring = scorer, cv=10, n_jobs =10)

    # TODO: Fit the grid search object to the training data and find the optimal parameters
    grid_fit = grid_obj.fit(X_train, y_train)

    # Get the estimator
    best_clf = grid_fit.best_estimator_

    # Make predictions using the unoptimized and model
    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    from sklearn.metrics import fbeta_score
    from sklearn.metrics import accuracy_score

    # Report the before-and-afterscores
    print("Unoptimized model\n------")
    print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
    print("\nOptimized Model\n------")
    print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))

    # Uncomment the following line to see the encoded feature names
    # print (encoded)
if __name__ == "__main__":
    donor()