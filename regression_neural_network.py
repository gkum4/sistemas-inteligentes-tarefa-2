import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.src.models import Sequential
from keras.src.layers import Dense

#Dataset inicial
dataset = pd.read_csv('treino_sinais_vitais_sem_label.csv', sep=',', header=None, index_col=0)

#Separar features do target
features = dataset.drop(labels=6, axis=1)
target = dataset[6]
target = target.values.reshape(-1,1)

#Normalizar os dados com o StandartScaler
features_scaler_fit = StandardScaler().fit(features)
features = features_scaler_fit.transform(features)
target_scaler_fit = StandardScaler().fit(target)
target = target_scaler_fit.transform(target)

#Dividir entre dados de treinamento e teste
f_train, f_test, t_train, t_test = train_test_split(features, target, test_size=0.1, random_state=42)

#Crir um modelo ANN 
model = Sequential()
#Primeira camada, input/oculta
model.add(Dense(units=5, input_dim=features.shape[1], kernel_initializer='normal', activation='relu'))
#Segunda camada
model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))
#Output, um único neurônio
model.add(Dense(units=1, kernel_initializer='normal'))

#Compilar o modelo e treiná-lo
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(f_train, t_train, batch_size=1, epochs=50, verbose=1)

#Usar os dados de teste para gerar as predições
prediction = model.predict(f_test)

#Organizando resultados para serem exibidos, revertendo a escala da predição e dos dados de teste para o que era antes
prediction = target_scaler_fit.inverse_transform(prediction)
target_test_org = target_scaler_fit.inverse_transform(t_test)
features_test_org = features_scaler_fit.inverse_transform(f_test)
results_df = pd.DataFrame(data=features_test_org, columns=['pSist', 'pDiast', 'qPA', 'pulso', 'respiracao'])
results_df['gravidade'] = target_test_org
results_df['gradidadePred'] = prediction
print(results_df)

#Computando o Root Mean Squared Error a partir do Mean Squared Error/loss
mse = mean_squared_error(target_test_org, prediction)
rmse = mse**.5
print(f'MSE: {mse}\nRMSE: {rmse}')
