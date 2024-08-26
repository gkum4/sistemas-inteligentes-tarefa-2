import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Dataset inicial
dataset = pd.read_csv('treino_sinais_vitais_sem_label.csv', sep=',', header=None, index_col=0)

#Separar features do target e dividir o dataset entre dados de treino e teste
features = dataset.drop(labels=6, axis=1)
target = dataset[6]
f_train, f_test, t_train, t_test = train_test_split(features, target, test_size=0.1)

#Criar uma instância do modelo e realizar o fit para treinar o modelo
rfr = RandomForestRegressor(n_estimators = 300, max_features = 'sqrt', max_depth = 5, random_state = 18)
rfr.fit(f_train, t_train)

#Testar o modelo e ver acurácia
prediction = rfr.predict(f_test)
mse = mean_squared_error(t_test, prediction)
rmse = mse**.5
print(f'MSE: {mse}\nRMSE: {rmse}')