import numpy as np
import pandas as pd
import pathlib
from models_pya.ml import Regresion_Softmax, MLP

actual_path = pathlib.Path(__file__).resolve().parent

data_path = actual_path.parent / 'dataset' / 'data' / 'mnist_train.csv'
weights_path = actual_path / 'params_digits.npz'

data = pd.read_csv(data_path)

datos = data.iloc[:]

columna_clave = 'label'
nuevo_orden = [col for col in datos.columns if col != columna_clave] + [columna_clave]

datos = datos[nuevo_orden].to_numpy(dtype=float)

datos[:, :-1] /= 255


###         CLAVES: 
# soft_w (pesos de softmax)
# soft_b (sesgos de softmax) 
# mlp_wi (pesos de mlp)
# mlp_bi (sesgos de mlp)
# mlp_arch (arquitectura del mlp)
# mlp_act (activacion de mlp)

try:
    save_data = dict(np.load(weights_path))
except:
    save_data = dict()

def train_softmax(epochs: int, learning_rate: float):

    softmax = Regresion_Softmax(epochs=epochs, learning_rate=learning_rate, architecture=(784,10), mini_batch=(True, 128))

    softmax.entrenar(datos=datos, estandarizar=False)

    save_data['soft_w'] = softmax._pesos
    save_data['soft_b'] = softmax._sesgos

def train_mlp(epochs: int, learning_rate: float):

    archic_red = (784, 512, 256, 10)
    activation_red = 'relu'

    mlp = MLP(epochs=epochs, learning_rate=learning_rate, architecture=archic_red, loss='log', activation=activation_red, optimizer=('minibatch', 128))

    mlp.entrenar(datos=datos, estandarizar=False)

    save_data['mlp_arch'] = archic_red
    save_data['mlp_act'] = activation_red

    for idx, peso in enumerate(mlp._pesos):

        save_data[f'mlp_w{idx + 1}'] = peso
        save_data[f'mlp_b{idx + 1}'] = mlp._sesgos[idx]


np.savez(weights_path, **save_data)