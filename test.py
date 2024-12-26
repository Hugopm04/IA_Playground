from IA import BPNetwork, sigmoid
import numpy as np
import time


def main():
    test_network()

def test_network():
    np.random.seed(10)

    training_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 0]])

    network = BPNetwork()
    network.add_layer(4, 3, sigmoid)
    network.add_layer(3, 4, sigmoid)
    network.add_layer(2, 3, sigmoid)
    network.add_layer(3, 2, sigmoid)

    print("Al comienzo: " + str(network))

    start = time.perf_counter()
    network.database_train(training_set_inputs, training_set_outputs, 10000)
    print("Ha tardado en entrenar:", time.perf_counter() - start)

    print("Al final: " + str(network))

    #Correct output: [0, 1, 0]
    newCase = np.array([1, 1, 0])
    output = network.think(newCase)
    print("Resultado del nuevo caso: " + str(output))

    print("Guardando red neuronal en la carpeta test_guardar.")
    network.save("test_guardar")
    print("Cargándola de nuevo.")
    new_network = BPNetwork.load("test_guardar")
    print("La red cargada es: " + str(new_network))
    output = new_network.think(newCase)
    print("Resultado del nuevo caso: " + str(output))

if __name__ == '__main__':
    main()
