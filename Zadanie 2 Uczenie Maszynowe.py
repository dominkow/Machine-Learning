#Zadanie 2 Uczenie Maszynowe
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import mnist_loader
from mnist_loader import load_data_wrapper
from tqdm import tqdm


# Funkcje aktywacji
def sigmoid(z: np.ndarray, pochodna: bool=False) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-z))
    if pochodna:
        return s * (1 - s)
    return s

def softmax(z: np.ndarray, pochodna: bool=False) -> np.ndarray:
    # Zapewnia stabilność numeryczną
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e_z / np.sum(e_z, axis=0, keepdims=True)

# Funkcja straty
def entropia_krzyzowa(y_pred: np.ndarray, test: np.ndarray, pochodna: bool=False) -> float | np.ndarray:
    epsilon = 1e-9
    m = test.shape[1]
    if pochodna:
        pass   
    # Obliczanie straty: -sum(y * log(y_pred)) / m
    # interesują nas tylko prawdopodobieństwa dla poprawnej klasy.
    log_prawd = -np.log(y_pred[test.argmax(axis=0), range(m)] + epsilon)
    strata = np.sum(log_prawd) / m
    return strata


# Klasa główna - pierwotna
class warstwa():
    def __init__(self, n: int, f_aktywacji: Callable|None):
        self.n = n  # Liczba neuronów w tej warstwie
        self.f_aktywacji = f_aktywacji

    def inicjuj(self, rozmiar_poprzedniej: int):
        m = rozmiar_poprzedniej
        #wagi W rozmiar (n, m), Biacy b rozmiar (n, 1)
        #zgodnie z konwencją W @ a + b
        self.wagi = np.random.normal(scale=np.sqrt(2/m), size=(self.n, m)) # He lub Xavier, dla Sigmoid Xavier (sqrt(6/m))
        self.b = np.zeros((self.n, 1))

# Warstwa wejściowa - bez zmian
class warstwa_wejsciowa(warstwa):
    def __init__(self, n: int):
        super().__init__(n, None)
       
    def inicjuj(self, rozmiar_poprzedniej: int=0):
        pass
       
    def w_przod(self, wejscie: np.ndarray):
        if wejscie.shape[0] != self.n:
            raise ValueError(f"Wielkość wektora wejściowego ({wejscie.shape[0]}) jest inna niż rozmiar warstwy ({self.n})")
        if len(wejscie.shape) == 1:
            wejscie = wejscie.reshape(-1, 1)
        self.wyjscie = wejscie
        return self.wyjscie
       
    def w_tyl(self, pochodna_nastepnej: np.ndarray, krok: float):
        return None

# warstwa końcowa 
class warstwa_wyjsciowa(warstwa):
    def __init__(self, n: int, f_aktywacji: Callable):
        super().__init__(n, f_aktywacji)
       
    def w_przod(self, a_poprzedniej):
        self.wejscie = a_poprzedniej
        # Z = W @ a + b
        self.z = self.wagi @ self.wejscie + self.b
        self.wyjscie = softmax(self.z)   
        return self.wyjscie

    def w_tyl(self, test: np.ndarray, krok: float):
        # m = liczba próbek w batchu
        m = test.shape[1] 
        
        #dL/dz = a - y (dostajemy gradient dla Softmax + Entropia Krzyżowa)
        #przez m dzielimy potem 
        self.pochodna_z = (self.wyjscie - test)
        
        #dW = dL/dz * x.T / m (Musimy uśrednić wagi)
        dW = (self.pochodna_z @ self.wejscie.T) / m
        
        #db = sum(dL/dz) / m
        db = np.sum(self.pochodna_z, axis=1, keepdims=True) / m
        
        #dL/da_prev = W.T * dL/dz
        pochodna_dla_poprzedniej = self.wagi.T @ self.pochodna_z
        
        #aktualizacja parametrów
        self.wagi -= krok * dW
        self.b -= krok * db
        
        return pochodna_dla_poprzedniej

#Warstwa ukryta 
class warstwa_ukryta(warstwa):
    def __init__(self, n: int, f_aktywacji: Callable = sigmoid):
        super().__init__(n, f_aktywacji)
       
    def w_przod(self, a_poprzedniej):
        self.wejscie = a_poprzedniej
        # Z = W @ a + b
        self.z = self.wagi @ self.wejscie + self.b
        self.wyjscie = self.f_aktywacji(self.z, pochodna=False)
        return self.wyjscie
       
    def w_tyl(self, pochodna_nastepnej: np.ndarray, krok: float):
        # pochodna_nastepnej - dL/da (gradient z warstwy kolejnej)
        m = pochodna_nastepnej.shape[1]
       
        #dL/dz = dL/da * da/dz = dL/da * f'(z)
        pochodna_z = pochodna_nastepnej * self.f_aktywacji(self.z, pochodna=True)
       
        #gradienty wag i biasu
        #dL/dW = dL/dz * x.T / m
        dW = (pochodna_z @ self.wejscie.T) / m
        #dL/db = sum(dL/dz) / m
        db = np.sum(pochodna_z, axis=1, keepdims=True) / m
       
        #pochodna straty względem aktywacji poprzedniej warstwy
        #dL/da_prev = W.T * dL/dz
        pochodna_dla_poprzedniej = self.wagi.T @ pochodna_z
       
        #aktualizujemy wagi i biasy
        self.wagi -= krok * dW
        self.b -= krok * db
       
        return pochodna_dla_poprzedniej

#klasa sieci neuronowej
class siec_neuronowa():
    def __init__(self, warstwy: list):
        self.warstwy = warstwy
        #inicjalizacja wymiarów
        rozmiar_wejscia = warstwy[0].n
        for i in range(len(self.warstwy)):
            if i > 0: #pomijamy warstwę wejściową przy inicjalizacji wag
                self.warstwy[i].inicjuj(rozmiar_wejscia)
                rozmiar_wejscia = self.warstwy[i].n
           
    def predict(self, x):
        #przekazanie danych przez wszystkie warstwy
        a = x
        for warstwa in self.warstwy:
            a = warstwa.w_przod(a)
        return a

    def evaluate(self, data):
        data = list(data)
        if len(data) == 0:
            return 0.0
       
        good = 0
        for x, y in data:
            #upewnienie się, że x jest (784, 1) jeśli jest zdefiniowane jako (784,)
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            out = self.predict(x)
            pred_digit = np.argmax(out)
            if pred_digit == y:
                good += 1
        return good / len(data)

    def fit(self, X_train_data: list,
            epochs : int = 20, batch_size: int = 32, krok: float = 0.1,
            validation_data: list | None = None, plot : bool = False):
        training_data = list(X_train_data)
        n = len(training_data)
       
        if validation_data:
            validation_data = list(validation_data)
       
        testy_straty = []
       
        for k in range(1, epochs + 1):
            np.random.shuffle(training_data)
            batches = [training_data[i:i + batch_size]
                       for i in range(0, n, batch_size)]
           
            total_loss = 0.0
           
            for batch in tqdm(batches, desc=f'Epoka {k}/{epochs}'):
                #mini batch
                X = np.hstack([x.reshape(-1, 1) for (x, y) in batch]) # Upewnienie się o formacie
                Y = np.hstack([y for (x, y) in batch])
               
                #krok 'w przód' (forward pass)
                Y_pred = self.predict(X)
               
                #krok 'w tył' (backpropagation)
                grad = self.warstwy[-1].w_tyl(Y, krok)
               
                for l in range(len(self.warstwy) - 2, 0, -1):
                    warstwa = self.warstwy[l]
                    grad = warstwa.w_tyl(grad, krok)
           
            #obliczanie straty i dokładności po epoce
            X_all = np.hstack([x.reshape(-1, 1) for (x,y) in training_data]) # Upewnienie się o formacie
            Y_pred_all = self.predict(X_all)
            Y_true_all = np.hstack([y for (x,y) in training_data])
            loss_function_val = entropia_krzyzowa(Y_pred_all, Y_true_all)
           
            msg = f"Epoka: {k}, Strata (trening): {loss_function_val:.4f}"
           
            if plot:
                testy_straty.append(loss_function_val)

            if validation_data:
                acc = self.evaluate(validation_data)
                msg += f", Dokładność (walidacja): {acc:.2%}"
               
            print(msg)

        if plot:
            plt.plot(np.arange(1, epochs + 1), testy_straty, ls='-.')
            plt.title('Wartość funkcji straty na zbiorze treningowym po każdej epoce')
            plt.xlabel('Epoka')
            plt.ylabel('Strata (Entropia Krzyżowa)')
            plt.show()



def main():
    #wczytanie danych
    training_data, validation_data, test_data = load_data_wrapper()

    training_data = list(training_data)
    validation_data = list(validation_data)
    test_data = list(test_data)

    #zbudowanie sieci 
    print("Budowanie sieci neuronowej [784, 500, 500, 500, 10]...")
    ann = siec_neuronowa([
        warstwa_wejsciowa(n=784),
        warstwa_ukryta(n=500),
        warstwa_ukryta(n=500),
        warstwa_ukryta(n=500),
        warstwa_wyjsciowa(n=10, f_aktywacji=softmax)
    ])
   
    #Trenowanie
    #używamy parametrów ze schematu (krok=0.1)
    ann.fit(training_data,
            epochs=20,
            batch_size=32,
            krok=0.1,
            validation_data=validation_data,
            plot=True)
   
    #test na zbiorze testowym
    print("\nTestowanie sieci na zbiorze testowym...")
    test_acc = ann.evaluate(test_data)
    print(f"Ostateczna dokładność na zbiorze testowym: {test_acc:.2%}")

if __name__=="__main__":
    main()