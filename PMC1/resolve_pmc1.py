import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    # Cliping to prevent overflow
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1.0 - sig)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, precision=1e-6, seed=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.precision = precision
        
        if seed is not None:
            np.random.seed(seed)
            
        # Initializing weights randomly between 0 and 1
        self.W1 = np.random.uniform(0, 1, (self.input_size, self.hidden_size))
        self.b1 = np.random.uniform(0, 1, (1, self.hidden_size))
        
        self.W2 = np.random.uniform(0, 1, (self.hidden_size, self.output_size))
        self.b2 = np.random.uniform(0, 1, (1, self.output_size))
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
        
    def backward(self, X, y):
        m = X.shape[0]
        
        # Erro na camada de saída
        # dE/da2 = -(y - a2) ou (a2 - y) (depende de como eqm é definido)
        # Eqm = 1/N * sum((d - y)^2), backprop delta2 = -(d - y)*sig'(z2)
        error = y - self.a2
        delta2 = error * sigmoid_derivative(self.z2)
        
        # Erro na camada oculta
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.z1)
        
        # Atualização dos pesos (gradient ascent because delta has the - sign implicitly)
        self.W2 += self.lr * np.dot(self.a1.T, delta2) / m
        self.b2 += self.lr * np.sum(delta2, axis=0, keepdims=True) / m
        
        self.W1 += self.lr * np.dot(X.T, delta1) / m
        self.b1 += self.lr * np.sum(delta1, axis=0, keepdims=True) / m

    def train(self, X, y, max_epochs=100000):
        mse_history = []
        prev_mse = float('inf')
        
        for epoch in range(max_epochs):
            predictions = self.forward(X)
            mse = np.mean(np.square(y - predictions))
            mse_history.append(mse)
            
            if abs(prev_mse - mse) <= self.precision:
                break
                
            self.backward(X, y)
            prev_mse = mse
            
        return epoch + 1, mse, mse_history

    def predict(self, X):
        return self.forward(X)

def load_data():
    # Load training data
    df_train = pd.read_csv('PMC/PMC1_table_3.csv', header=None)
    # df_train has 15 columns. They are triplets of Amostra, x1, x2, x3, d
    X_train = []
    y_train = []
    for row in range(1, len(df_train)):
        for col_offset in [0, 5, 10]:
            if col_offset + 4 < df_train.shape[1]:
                amostra = str(df_train.iloc[row, col_offset]).strip()
                if amostra and amostra.isdigit():
                    x1 = float(df_train.iloc[row, col_offset + 1])
                    x2 = float(df_train.iloc[row, col_offset + 2])
                    x3 = float(df_train.iloc[row, col_offset + 3])
                    d = float(df_train.iloc[row, col_offset + 4])
                    X_train.append([x1, x2, x3])
                    y_train.append([d])
                    
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Load test data
    df_test = pd.read_csv('PMC/PMC1_table_2.csv', header=None)
    X_test = []
    y_test = []
    for row in range(1, 21):
        amostra = str(df_test.iloc[row, 0]).strip()
        if amostra and amostra.isdigit():
            x1 = float(df_test.iloc[row, 1])
            x2 = float(df_test.iloc[row, 2])
            x3 = float(df_test.iloc[row, 4])
            d = float(df_test.iloc[row, 5])
            X_test.append([x1, x2, x3])
            y_test.append([d])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    return X_train, y_train, X_test, y_test

def main():
    X_train, y_train, X_test, y_test = load_data()
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    results = []
    models = []
    histories = []
    
    for i in range(5):
        # Using fixed seeds for reproducibility while maintaining variety
        seed = 42 + i
        mlp = MLP(input_size=3, hidden_size=10, output_size=1, lr=0.1, precision=1e-6, seed=seed)
        epochs, final_mse, history = mlp.train(X_train, y_train)
        results.append((i+1, final_mse, epochs))
        models.append(mlp)
        histories.append((i+1, history))
        print(f"Treinamento {i+1} concluído: {epochs} épocas, EQM final = {final_mse:.6f}")
        
    # Plot learning curves for the 2 models with the longest epochs
    histories_sorted = sorted(histories, key=lambda x: len(x[1]), reverse=True)
    top2 = histories_sorted[:2]
    
    plt.figure(figsize=(10, 6))
    for idx, history in top2:
        plt.plot(history, label=f'Treinamento {idx} ({len(history)} épocas)')
        
    plt.title('Curvas de Aprendizado (EQM x Épocas)')
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Médio (EQM)')
    plt.legend()
    plt.grid(True)
    plt.savefig('PMC/curvas_aprendizado.png')
    print("Gráfico salvo em PMC/curvas_aprendizado.png")

    # Avaliação no Test Set
    test_results = []
    erros_relativos = []
    
    for idx, mlp in enumerate(models):
        y_pred = mlp.predict(X_test)
        # Erro relativo médio para cada treinamento: 1/N * sum(|d - y|/d) * 100
        # However, looking at standard definitions, often relative error is |d - y| / |d|
        
        # Prevent division by zero if d == 0
        d_safe = np.where(y_test == 0, 1e-10, y_test)
        erro_rel_percent = np.abs((y_test - y_pred) / d_safe) * 100
        
        erm = np.mean(erro_rel_percent)
        var = np.var(erro_rel_percent)
        
        test_results.append({
            'T': idx + 1,
            'predictions': y_pred.flatten(),
            'erm': erm,
            'var': var
        })

    # Imprimir os resultados para construir o documento depois
    print("\nResultados Treinamento (Tabela 1):")
    for r in results:
        print(f"T{r[0]} - EQM: {r[1]:.6f}, Épocas: {r[2]}")
        
    print("\nResultados Teste (Tabela 2):")
    for tr in test_results:
        print(f"T{tr['T']} - ERM: {tr['erm']:.4f}%, Var: {tr['var']:.4f}%")
        
    # Let's save a summary to a text file to read easily later
    with open('PMC/pmc1_output.txt', 'w') as f:
        f.write("Treinamento,EQM,Epocas\n")
        for r in results:
            f.write(f"T{r[0]},{r[1]:.6f},{r[2]}\n")
            
        f.write("\nValidacao\n")
        f.write("Treinamento,ERM(%),Variancia(%)\n")
        for tr in test_results:
            f.write(f"T{tr['T']},{tr['erm']:.4f},{tr['var']:.4f}\n")
            
        # Writing predictions column by column for easy extraction
        f.write("\nPredicoes(T1_a_T5)\n")
        for j in range(20):
            preds = [str(test_results[i]['predictions'][j]) for i in range(5)]
            f.write(",".join(preds) + "\n")

if __name__ == "__main__":
    main()
