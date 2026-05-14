import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1.0 - sig)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, momentum=0.0, precision=1e-6, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.momentum = momentum
        self.precision = precision
        
        np.random.seed(seed)
        
        self.W1 = np.random.uniform(0, 1, (self.input_size, self.hidden_size))
        self.b1 = np.random.uniform(0, 1, (1, self.hidden_size))
        
        self.W2 = np.random.uniform(0, 1, (self.hidden_size, self.output_size))
        self.b2 = np.random.uniform(0, 1, (1, self.output_size))
        
        # Momentum variables
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        
    def get_weights(self):
        return {
            'W1': np.copy(self.W1), 'b1': np.copy(self.b1),
            'W2': np.copy(self.W2), 'b2': np.copy(self.b2)
        }
        
    def set_weights(self, weights):
        self.W1 = np.copy(weights['W1'])
        self.b1 = np.copy(weights['b1'])
        self.W2 = np.copy(weights['W2'])
        self.b2 = np.copy(weights['b2'])
        
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
        
    def backward(self, X, y):
        m = X.shape[0]
        
        error = y - self.a2
        delta2 = error * sigmoid_derivative(self.z2)
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.z1)
        
        grad_W2 = np.dot(self.a1.T, delta2) / m
        grad_b2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        grad_W1 = np.dot(X.T, delta1) / m
        grad_b1 = np.sum(delta1, axis=0, keepdims=True) / m
        
        # Apply momentum
        self.v_W2 = (self.momentum * self.v_W2) + (self.lr * grad_W2)
        self.v_b2 = (self.momentum * self.v_b2) + (self.lr * grad_b2)
        self.v_W1 = (self.momentum * self.v_W1) + (self.lr * grad_W1)
        self.v_b1 = (self.momentum * self.v_b1) + (self.lr * grad_b1)
        
        # Update weights
        self.W2 += self.v_W2
        self.b2 += self.v_b2
        self.W1 += self.v_W1
        self.b1 += self.v_b1

    def train(self, X, y, max_epochs=200000):
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
    df_train = pd.read_csv('PMC2/PMC2_table_3.csv', header=None)
    X_train = []
    y_train = []
    for row in range(1, len(df_train)):
        for col_offset in [0, 8]:
            if col_offset + 7 < df_train.shape[1]:
                amostra = str(df_train.iloc[row, col_offset]).strip()
                if amostra and amostra.replace('.0', '').isdigit():
                    x1 = float(df_train.iloc[row, col_offset + 1])
                    x2 = float(df_train.iloc[row, col_offset + 2])
                    x3 = float(df_train.iloc[row, col_offset + 3])
                    x4 = float(df_train.iloc[row, col_offset + 4])
                    d1 = float(df_train.iloc[row, col_offset + 5])
                    d2 = float(df_train.iloc[row, col_offset + 6])
                    d3 = float(df_train.iloc[row, col_offset + 7])
                    X_train.append([x1, x2, x3, x4])
                    y_train.append([d1, d2, d3])
                    
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    df_test = pd.read_csv('PMC2/PMC2_table_2.csv', header=None)
    X_test = []
    y_test = []
    for row in range(1, len(df_test)):
        amostra = str(df_test.iloc[row, 0]).strip()
        if amostra and amostra.replace('.0', '').isdigit():
            x1 = float(df_test.iloc[row, 1])
            x2 = float(df_test.iloc[row, 2])
            x3 = float(df_test.iloc[row, 3])
            x4 = float(df_test.iloc[row, 4])
            d1 = float(df_test.iloc[row, 5])
            d2 = float(df_test.iloc[row, 6])
            d3 = float(df_test.iloc[row, 7])
            X_test.append([x1, x2, x3, x4])
            y_test.append([d1, d2, d3])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    return X_train, y_train, X_test, y_test

def main():
    X_train, y_train, X_test, y_test = load_data()
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Init MLP
    mlp = MLP(input_size=4, hidden_size=15, output_size=3, lr=0.1, momentum=0.0, precision=1e-6, seed=123)
    initial_weights = mlp.get_weights()
    
    print("--- Treinamento Padrão ---")
    start_time = time.time()
    epochs_std, mse_std, hist_std = mlp.train(X_train, y_train)
    time_std = time.time() - start_time
    print(f"Tempo: {time_std:.2f}s | Épocas: {epochs_std} | EQM: {mse_std:.6f}")
    
    # Test Padrão
    pred_std_raw = mlp.predict(X_test)
    pred_std_round = np.where(pred_std_raw >= 0.5, 1, 0)
    acertos_std = np.sum(np.all(pred_std_round == y_test, axis=1))
    taxa_std = (acertos_std / len(y_test)) * 100
    print(f"Taxa de Acerto (Padrão): {taxa_std:.2f}%\n")
    
    print("--- Treinamento Momentum ---")
    # Restore weights and set momentum
    mlp.set_weights(initial_weights)
    mlp.momentum = 0.9
    
    start_time = time.time()
    epochs_mom, mse_mom, hist_mom = mlp.train(X_train, y_train)
    time_mom = time.time() - start_time
    print(f"Tempo: {time_mom:.2f}s | Épocas: {epochs_mom} | EQM: {mse_mom:.6f}")
    
    # Test Momentum
    pred_mom_raw = mlp.predict(X_test)
    pred_mom_round = np.where(pred_mom_raw >= 0.5, 1, 0)
    acertos_mom = np.sum(np.all(pred_mom_round == y_test, axis=1))
    taxa_mom = (acertos_mom / len(y_test)) * 100
    print(f"Taxa de Acerto (Momentum): {taxa_mom:.2f}%\n")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(hist_std, label=f'Padrão ({epochs_std} épocas)')
    plt.plot(hist_mom, label=f'Momentum ({epochs_mom} épocas)')
    plt.title('Comparativo de Convergência (Padrão vs Momentum)')
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Médio (EQM)')
    plt.legend()
    plt.grid(True)
    plt.savefig('PMC2/comparativo_momentum.png')
    
    # Saving to text for easy extraction into Markdown
    with open('PMC2/pmc2_output.txt', 'w') as f:
        f.write(f"Tempo_Std:{time_std:.4f}\n")
        f.write(f"Epocas_Std:{epochs_std}\n")
        f.write(f"EQM_Std:{mse_std:.6f}\n")
        f.write(f"Taxa_Std:{taxa_std:.2f}\n")
        
        f.write(f"Tempo_Mom:{time_mom:.4f}\n")
        f.write(f"Epocas_Mom:{epochs_mom}\n")
        f.write(f"EQM_Mom:{mse_mom:.6f}\n")
        f.write(f"Taxa_Mom:{taxa_mom:.2f}\n")
        
        f.write("\nValidacao_Std\n")
        for p in pred_std_round:
            f.write(",".join(map(str, p)) + "\n")
            
        f.write("\nValidacao_Mom\n")
        for p in pred_mom_round:
            f.write(",".join(map(str, p)) + "\n")

if __name__ == "__main__":
    main()
