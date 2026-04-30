import numpy as np

# Dataset (Padrão, x1, x2, x3, d)
data = np.array([
    [-0.6508, 0.1097, 4.0009, -1.0000],
    [-1.4492, 0.8896, 4.4005, -1.0000],
    [2.0850, 0.6876, 12.0710, -1.0000],
    [0.2626, 1.1476, 7.7985, 1.0000],
    [0.6418, 1.0234, 7.0427, 1.0000],
    [0.2569, 0.6730, 8.3265, -1.0000],
    [1.1155, 0.6043, 7.4446, 1.0000],
    [0.0914, 0.3399, 7.0677, -1.0000],
    [0.0121, 0.5256, 4.6316, 1.0000],
    [-0.0429, 0.4660, 5.4323, 1.0000],
    [0.4340, 0.6870, 8.2287, -1.0000],
    [0.2735, 1.0287, 7.1934, 1.0000],
    [0.4839, 0.4851, 7.4850, -1.0000],
    [0.4089, -0.1267, 5.5019, -1.0000],
    [1.4391, 0.1614, 8.5843, -1.0000],
    [-0.9115, -0.1973, 2.1962, -1.0000],
    [0.3654, 1.0475, 7.4858, 1.0000],
    [0.2144, 0.7515, 7.1699, 1.0000],
    [0.2013, 1.0014, 6.5489, 1.0000],
    [0.6483, 0.2183, 5.8991, 1.0000],
    [-0.1147, 0.2242, 7.2435, -1.0000],
    [-0.7970, 0.8795, 3.8762, 1.0000],
    [-1.0625, 0.6366, 2.4707, 1.0000],
    [0.5307, 0.1285, 5.6883, 1.0000],
    [-1.2200, 0.7777, 1.7252, 1.0000],
    [0.3957, 0.1076, 5.6623, -1.0000],
    [-0.1013, 0.5989, 7.1812, -1.0000],
    [2.4482, 0.9455, 11.2095, 1.0000],
    [2.0149, 0.6192, 10.9263, -1.0000],
    [0.2012, 0.2611, 5.4631, 1.0000]
])

X = data[:, :3]
d = data[:, 3]

# Inserindo o termo de bias x0 = -1 na primeira coluna
X_bias = np.insert(X, 0, -1, axis=1)

learning_rate = 0.01

def train_perceptron():
    # Inicializando pesos aleatórios entre 0 e 1 (4 pesos: w0 para o bias e w1, w2, w3)
    # A semente global é reiniciada antes de chamar a função
    w = np.random.uniform(0, 1, 4)
    initial_w = w.copy()
    
    epochs = 0
    while True:
        error_in_epoch = False
        
        for i in range(len(X_bias)):
            x_i = X_bias[i]
            d_i = d[i]
            
            # Cálculo da saída v = w^T * x
            v = np.dot(w, x_i)
            
            # Função sinal de ativação
            y = 1 if v >= 0 else -1
            
            # Se houver erro, atualizar com a Regra de Hebb (supervisionada e corretiva)
            if y != d_i:
                w = w + learning_rate * d_i * x_i
                error_in_epoch = True
                
        epochs += 1
        
        # Critério de parada: sem erros na época
        if not error_in_epoch:
            break
            
        # Prevenção contra loop infinito
        if epochs > 10000:
            print("Aviso: Limite de 10.000 épocas alcançado. Os dados podem não ser linearmente separáveis.")
            break
            
    return initial_w, w, epochs

if __name__ == "__main__":
    print("=" * 60)
    print(" Treinamento de Perceptron (Destilação de Óleo)")
    print(" Regra de Hebb supervisionada | ETA = 0.01 | Bias = -1")
    print("=" * 60)
    
    weights_list = []
    for execution in range(1, 6):
        # A entropia do SO alimentará a nova semente (garante diferentes inícios)
        np.random.seed()
        
        initial_w, final_w, epochs = train_perceptron()
        weights_list.append(final_w)
        
        print(f"\n--- Treinamento {execution} ---")
        print(f"Pesos Iniciais : {np.round(initial_w, 4)}")
        print(f"Pesos Finais   : {np.round(final_w, 4)}")
        print(f"Épocas até convergência: {epochs}")
        
    # Dados de Teste
    test_data = np.array([
        [-0.3565, 0.0620, 5.9891],
        [-0.7842, 1.1267, 5.5912],
        [0.3012, 0.5611, 5.8234],
        [0.7757, 1.0648, 8.0677],
        [0.1570, 0.8028, 6.3040],
        [-0.7014, 1.0316, 3.6005],
        [0.3748, 0.1536, 6.1537],
        [-0.6920, 0.9404, 4.4058],
        [-1.3970, 0.7141, 4.9263],
        [-1.8842, -0.2805, 1.2548]
    ])
    
    # Inserindo o termo de bias x0 = -1
    X_test_bias = np.insert(test_data, 0, -1, axis=1)

    print("\n" + "=" * 60)
    print(" Classificação de Novas Amostras (Tabela de Resultados)")
    print("=" * 60)
    print(f"{'Amostra':^9} | {'y(T1)':^6} | {'y(T2)':^6} | {'y(T3)':^6} | {'y(T4)':^6} | {'y(T5)':^6}")
    print("-" * 55)
    
    for i, x_t in enumerate(X_test_bias):
        preds = []
        for w in weights_list:
            v = np.dot(w, x_t)
            y = 1 if v >= 0 else -1
            preds.append(y)
        print(f"{i+1:^9} | {preds[0]:^6} | {preds[1]:^6} | {preds[2]:^6} | {preds[3]:^6} | {preds[4]:^6}")
        
    print("\nConcluído!")
