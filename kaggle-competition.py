import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import Perceptron
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier, BernoulliRBM
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree


class Kaggle:

    x_train = []
    y_train = []
    x_test = []

    def __init__(self):

        training_data = pd.read_csv('self_train.csv')
        training_data.apply(lambda x: sum(x.isnull()), axis=0)

        training_data['Employment_Info_1'].fillna(training_data['Employment_Info_1'].mean(), inplace=True)
        training_data['Employment_Info_4'].fillna(training_data['Employment_Info_4'].mean(), inplace=True)
        training_data['Employment_Info_6'].fillna(training_data['Employment_Info_6'].mean(), inplace=True)
        training_data['Insurance_History_5'].fillna(training_data['Insurance_History_5'].mean(), inplace=True)
        training_data['Family_Hist_2'].fillna(training_data['Family_Hist_2'].mean(), inplace=True)
        training_data['Family_Hist_3'].fillna(training_data['Family_Hist_3'].mean(), inplace=True)
        training_data['Family_Hist_4'].fillna(training_data['Family_Hist_4'].mean(), inplace=True)
        training_data['Family_Hist_5'].fillna(training_data['Family_Hist_5'].mean(), inplace=True)
        training_data['Medical_History_1'].fillna(training_data['Medical_History_1'].mean(), inplace=True)
        training_data['Medical_History_10'].fillna(training_data['Medical_History_10'].mean(), inplace=True)
        training_data['Medical_History_32'].fillna(training_data['Medical_History_32'].mean(), inplace=True)
        training_data['Medical_History_15'].fillna(training_data['Medical_History_15'].mean(), inplace=True)
        training_data['Medical_History_24'].fillna(training_data['Medical_History_24'].mean(), inplace=True)

        training_data = training_data.drop('Ins_Age', 1)
        training_data = training_data.drop('Ht', 1)
        training_data = training_data.drop('Wt', 1)
        training_data = training_data.drop('Medical_Keyword_48', 1)
        # training_data = training_data.drop('Medical_Keyword_15', 1)
        training_data.dropna(axis=1)

        x_train_data = training_data.values[:, 1:123]
        y_train_data = training_data.values[:, 123]

        label_encoder = LabelEncoder()
        x_train_data[:, 1] = label_encoder.fit_transform(x_train_data[:, 1])

        self.x_train = x_train_data[:, :]
        self.y_train = y_train_data[:]
        self.y_train = self.y_train.astype('float')
        self.x_train = pd.DataFrame(data=self.x_train)

        # Replacing Infinite values or nan values
        self.x_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.x_train.fillna(self.x_train.mean(), inplace=True)

        vt = VarianceThreshold(threshold=0.3)
        vt.fit(self.x_train)
        feature_indices = vt.get_support(indices=True)
        feature_names = [self.x_train.columns[idx]
                         for idx, _ in enumerate(self.x_train.columns) if idx in feature_indices]
        self.x_train = pd.DataFrame(data=self.x_train, columns=feature_names)

        # Testing
        testing_data = pd.read_csv('self_Test_wo_response.csv')

        testing_data['Employment_Info_1'].fillna(testing_data['Employment_Info_1'].mean(), inplace=True)
        testing_data['Employment_Info_4'].fillna(testing_data['Employment_Info_4'].mean(), inplace=True)
        testing_data['Employment_Info_6'].fillna(testing_data['Employment_Info_6'].mean(), inplace=True)
        testing_data['Insurance_History_5'].fillna(testing_data['Insurance_History_5'].mean(), inplace=True)
        testing_data['Family_Hist_2'].fillna(testing_data['Family_Hist_2'].mean(), inplace=True)
        testing_data['Family_Hist_3'].fillna(testing_data['Family_Hist_3'].mean(), inplace=True)
        testing_data['Family_Hist_4'].fillna(testing_data['Family_Hist_4'].mean(), inplace=True)
        testing_data['Family_Hist_5'].fillna(testing_data['Family_Hist_5'].mean(), inplace=True)
        testing_data['Medical_History_1'].fillna(testing_data['Medical_History_1'].mean(), inplace=True)
        testing_data['Medical_History_10'].fillna(testing_data['Medical_History_10'].mean(), inplace=True)
        testing_data['Medical_History_32'].fillna(testing_data['Medical_History_32'].mean(), inplace=True)
        testing_data['Medical_History_15'].fillna(testing_data['Medical_History_15'].mean(), inplace=True)
        testing_data['Medical_History_24'].fillna(testing_data['Medical_History_24'].mean(), inplace=True)

        testing_data = testing_data.dropna(axis=1)
        testing_data = testing_data.drop('Ins_Age', 1)
        testing_data = testing_data.drop('Ht', 1)
        testing_data = testing_data.drop('Wt', 1)
        testing_data = testing_data.drop('Medical_Keyword_48', 1)
        # testing_data = testing_data.drop('Medical_Keyword_15', 1)

        x_test_data = testing_data.values[:, 1:123]

        label_encoder = LabelEncoder()
        x_test_data[:, 1] = label_encoder.fit_transform(x_test_data[:, 1])

        self.x_test = x_test_data[:, :]
        self.x_test = pd.DataFrame(data=self.x_test)

        # Replacing Infinite values or nan values
        self.x_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.x_test.fillna(self.x_test.mean(), inplace=True)

        vt = VarianceThreshold(threshold=0.3)
        vt.fit(self.x_test)
        feature_indices = vt.get_support(indices=True)
        feature_names = [self.x_test.columns[idx]
                         for idx, _ in enumerate(self.x_test.columns) if idx in feature_indices]
        self.x_test = pd.DataFrame(data=self.x_test, columns=feature_names)

    def training(self):
        pass

    def testing(self):
        pass

    def model(self):
        # clf1 = LogisticRegression(random_state=1)
        # clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
        # clf3 = GaussianNB()

        # Just change the model here.
        model = MLPClassifier()
        model.fit(self.x_train, self.y_train)
        y_predict = model.predict(self.x_test)
        pd.DataFrame(y_predict, columns=['Response']).to_csv('result.csv')


obj = Kaggle()
# obj.testing()
# obj.training()
obj.model()
