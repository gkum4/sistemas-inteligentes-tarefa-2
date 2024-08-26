import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


#Dataset inicial
dataset = pd.read_csv('treino_sinais_vitais_com_label.csv', sep=',', header=None, index_col=0)

#Separar features do target e dividir o dataset entre dados de treino e teste
features = dataset.drop(labels=7, axis=1)
target = dataset[7]
f_train, f_test, t_train, t_test = train_test_split(features, target, test_size=0.1)

#Criar uma instância do modelo e realizar o fit para treinar o modelo
rfc = RandomForestClassifier()
rfc.fit(f_train, t_train)

#Testar o modelo e ver acurácia
prediction = rfc.predict(f_test)
accuracy = accuracy_score(t_test, prediction)
print(f'Accuracy: {accuracy}')

#Matriz de confusão
cm = confusion_matrix(t_test, prediction)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()