import os
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
    def __init__(self, input_size, hidden_size, output_size=1, lr=0.1, momentum=0.8, precision=0.5e-6, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.momentum = momentum
        self.precision = precision
        
        np.random.seed(seed)
        
        self.W1 = np.random.uniform(0, 0.1, (self.input_size, self.hidden_size))
        self.b1 = np.random.uniform(0, 0.1, (1, self.hidden_size))
        
        self.W2 = np.random.uniform(0, 0.1, (self.hidden_size, self.output_size))
        self.b2 = np.random.uniform(0, 0.1, (1, self.output_size))
        
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
        
        self.v_W2 = (self.momentum * self.v_W2) + (self.lr * grad_W2)
        self.v_b2 = (self.momentum * self.v_b2) + (self.lr * grad_b2)
        self.v_W1 = (self.momentum * self.v_W1) + (self.lr * grad_W1)
        self.v_b1 = (self.momentum * self.v_b1) + (self.lr * grad_b1)
        
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


def load_time_series():
    df_train = pd.read_csv('PMC3/PMC3_table_3.csv', header=None)
    # The table has 4 blocks.
    # Col 1: t=1..25, Col 3: t=26..50, Col 5: t=51..75, Col 7: t=76..100
    series_train = np.zeros(100)
    for row in range(1, len(df_train)):
        for col_offset in [0, 2, 4, 6]:
            if col_offset + 1 < df_train.shape[1]:
                amostra = str(df_train.iloc[row, col_offset]).strip()
                if amostra.startswith('t ='):
                    t_idx = int(amostra.split('=')[1].strip())
                    val = float(df_train.iloc[row, col_offset + 1])
                    series_train[t_idx - 1] = val

    df_val = pd.read_csv('PMC3/PMC3_table_2.csv', header=None)
    series_val = np.zeros(20)
    for row in range(1, len(df_val)):
        amostra = str(df_val.iloc[row, 0]).strip()
        if amostra.startswith('t ='):
            t_idx = int(amostra.split('=')[1].strip())
            val = float(df_val.iloc[row, 1])
            series_val[t_idx - 101] = val

    full_series = np.concatenate([series_train, series_val])
    return full_series

def create_datasets(full_series, p):
    # Train: target t=p+1 to 100 (indices p to 99)
    X_train, y_train = [], []
    for t in range(p, 100):
        # inputs: x(t-1) to x(t-p). 
        # The prompt figure shows: x(t-1), x(t-2) ... x(t-p).
        # We slice from t-p to t (exclusive) which gives p elements.
        window = full_series[t-p : t]
        # Reverse it to match [x(t-1), x(t-2), ... x(t-p)] order, 
        # though standard order [x(t-p), ..., x(t-1)] works identically for the MLP.
        X_train.append(window[::-1])
        y_train.append([full_series[t]])
        
    # Val: target t=101 to 120 (indices 100 to 119)
    X_val, y_val = [], []
    for t in range(100, 120):
        window = full_series[t-p : t]
        X_val.append(window[::-1])
        y_val.append([full_series[t]])
        
    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

def main():
    full_series = load_time_series()
    
    redes_config = [
        {"name": "Rede 1", "p": 5, "N1": 10},
        {"name": "Rede 2", "p": 10, "N1": 15},
        {"name": "Rede 3", "p": 15, "N1": 25}
    ]
    
    results = {}
    best_models = {}
    
    for r_idx, config in enumerate(redes_config):
        p = config["p"]
        N1 = config["N1"]
        name = config["name"]
        
        X_train, y_train, X_val, y_val = create_datasets(full_series, p)
        
        results[name] = []
        best_erm = float('inf')
        
        for t_idx in range(3):
            # 3 treinamentos distintos
            seed = 42 + t_idx + r_idx * 10
            mlp = MLP(input_size=p, hidden_size=N1, output_size=1, lr=0.1, momentum=0.8, precision=0.5e-6, seed=seed)
            epochs, eqm, history = mlp.train(X_train, y_train)
            
            # Predict validation
            preds = mlp.predict(X_val)
            d_safe = np.where(y_val == 0, 1e-10, y_val)
            er_percent = np.abs((y_val - preds) / d_safe)
            erm = np.mean(er_percent)
            var = np.var(er_percent)
            
            results[name].append({
                "T": t_idx + 1,
                "epochs": epochs,
                "eqm": eqm,
                "erm": erm,
                "var": var,
                "preds": preds.flatten(),
                "history": history
            })
            
            if erm < best_erm:
                best_erm = erm
                best_models[name] = results[name][-1]
                
    # --- Gerando arquivo de log ---
    with open('PMC3/pmc3_output.txt', 'w') as f:
        for name in redes_config:
            r_name = name['name']
            f.write(f"--- {r_name} ---\n")
            for t_data in results[r_name]:
                f.write(f"T{t_data['T']} - EQM: {t_data['eqm']:.6f}, Epocas: {t_data['epochs']}, ERM: {t_data['erm']:.6f}, Var: {t_data['var']:.6f}\n")
            f.write("\n")
            
        f.write("Predicoes Validaçao (t=101..120)\n")
        # Escrever colunas de T1,T2,T3 para cada rede para facilitar cópia
        for config in redes_config:
            r_name = config["name"]
            f.write(f"--- {r_name} ---\n")
            for j in range(20):
                pr = [str(results[r_name][i]["preds"][j]) for i in range(3)]
                f.write(",".join(pr) + "\n")
            f.write("\n")

    # --- Gráfico 1: EQM x Épocas ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Curvas de Aprendizado - Melhor Treinamento por Rede", fontsize=16)
    
    for i, config in enumerate(redes_config):
        name = config["name"]
        best = best_models[name]
        axes[i].plot(best["history"], label=f"{name} (T{best['T']}) - {best['epochs']} épocas")
        axes[i].set_title(f"{name} - Melhor Treino: T{best['T']}")
        axes[i].set_xlabel("Épocas")
        axes[i].set_ylabel("EQM")
        axes[i].legend()
        axes[i].grid(True)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("PMC3/eqm_comparativo.png")
    
    # --- Gráfico 2: Desejado vs Estimado ---
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12))
    fig2.suptitle("Desejado vs Estimado (Validação t=101..120) - Melhor Treinamento", fontsize=16)
    t_axis = range(101, 121)
    
    # y_val is the same for all (target t=101 to 120)
    _, _, _, y_val_ref = create_datasets(full_series, 5) # any p gets same y_val length
    
    for i, config in enumerate(redes_config):
        name = config["name"]
        best = best_models[name]
        axes2[i].plot(t_axis, y_val_ref, 'ko-', label='Desejado f(t)', linewidth=2)
        axes2[i].plot(t_axis, best["preds"], 'r*--', label=f'Estimado (T{best["T"]})', markersize=8)
        axes2[i].set_title(f"{name} - Previsão na Validação")
        axes2[i].set_xlabel("Tempo (t)")
        axes2[i].set_ylabel("Valor")
        axes2[i].set_xticks(t_axis)
        axes2[i].legend()
        axes2[i].grid(True)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("PMC3/predicao_comparativa.png")

if __name__ == "__main__":
    main()
