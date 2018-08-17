from dnn import DeepNeuralNetwork
import numpy as np
import pandas as pd
from sklearn import svm
np.set_printoptions(threshold=np.nan)
np.random.seed(16)


def main():
	tr_X, tr_Y, t_X = load_data()
	train_X, train_Y, test_X, test_Y = split_set(tr_X, tr_Y)
	train_X, test_X = scale(train_X, test_X)
			
	return (train_X, train_Y, test_X, test_Y)
	#execute_with_DNN(train_X, train_Y, test_X, test_Y)

def scale(tr_X, t_X):
	mew = np.mean(tr_X, axis=0)
	s = np.std(tr_X, axis=0)
	tr_X = (tr_X - mew) / s 
	t_X = (t_X - mew) / s 
	return tr_X, t_X 


def get_sets():
	tr_X, tr_Y, t_X = load_data()
	train_X, train_Y, test_X, test_Y = split_set(tr_X, tr_Y)

def execute_with_svm(X, Y, test_X, test_Y):
	clf = svm.SVC()
	clf.fit(X, Y)
	clf.predict(test_X)
	

def execute_with_DNN(X, Y, test_X, test_Y):
	dnn = DeepNeuralNetwork()
	X = X.T
	Y = Y.T
	dnn.train_model(X, Y)
	dnn.predict(X, Y, test_X.T, test_Y.T)


def split_set(tr_X, tr_Y):
	X = np.array(tr_X)
	Y = np.array(tr_Y)
	Y = Y.reshape(Y.shape[0], -1)
	idx = np.random.permutation(X.shape[0])
	split = int(Y.shape[0] * 0.8)
	train_idx, test_idx = idx[:split], idx[split:]
	train_X, test_X = X[train_idx, :], X[test_idx, :]
	train_Y, test_Y = Y[train_idx, :], Y[test_idx, :]
	return train_X, train_Y, test_X, test_Y
	

def load_data():
	df_train = pd.read_csv("~/.kaggle/competitions/titanic/train.csv")
	df_test = pd.read_csv("~/.kaggle/competitions/titanic/test.csv")
	df_train, df_test = strip_columns(["Cabin", "Ticket", "PassengerId", "Fare", "Name"], [df_train, df_test])
	df_train, df_test = fill_na_mean(["Age"], [df_train, df_test])
	df_train, df_test = fill_na_mode(["Embarked"], [df_train, df_test])
	df_train, df_test = encode(["Embarked", "Sex", "Pclass", "SibSp", "Parch"], [df_train, df_test])

	train_X, train_Y = extract_label("Survived", [df_train])
	df_test = strip_columns("Parch_9", [df_test])[0]
	return train_X, train_Y, df_test

def extract_label(label, dataframes):
	Xs = []
	Ys = []
	for df in dataframes:
		Ys.append(df[label])
		Xs.append(df.drop(labels=[label], axis=1))
	return Xs + Ys 

def strip_columns(columns, dataframes):
	r = []
	for df in dataframes:
		r.append(df.drop(labels=columns, axis=1))
	return r

def fill_na_mean(columns, dataframes):
	r = []
	for df in dataframes:
		for col in columns:
			df[col].fillna(df[col].mean(), inplace=True)
		r.append(df)
	return r

def fill_na_mode(columns, dataframes):
	r = []
	for df in dataframes:
		for col in columns:
			df[col].fillna(df[col].mode(), inplace=True)
		r.append(df)
	return r

def encode(cols, dataframes):
	r = []
	for df in dataframes:
		r.append(pd.get_dummies(df, columns=cols))
	return r

train_X, train_Y, test_X, test_Y = main()
if __name__=="__main__":
	main()
