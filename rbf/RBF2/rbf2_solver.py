import math
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# Read extracted_tables.txt to get data
train_data = []
test_data = []

with open('c:/Users/Gi/Lab_IA_I/IA_2026.1/rbf/RBF2/extracted_tables.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

table2_idx = lines.index("Table 2:\n")
table3_idx = lines.index("Table 3:\n")

# Parse Table 2 (Test data)
# It has 15 samples
for line in lines[table2_idx+3:table2_idx+18]:
    line = line.strip()
    if line.startswith("['"):
        parts = eval(line) # it's a string representation of list
        amostra = int(parts[0])
        x1 = float(parts[1])
        x2 = float(parts[2])
        x3 = float(parts[3])
        d = float(parts[4])
        test_data.append((amostra, x1, x2, x3, d))

# Parse Table 3 (Train data)
for line in lines[table3_idx+2:]:
    line = line.strip()
    if not line: continue
    if line.startswith("['"):
        parts = eval(line)
        if len(parts) >= 5 and parts[0].isdigit():
            train_data.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        if len(parts) >= 10 and parts[5].isdigit():
            train_data.append((float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])))
        if len(parts) >= 15 and parts[10].isdigit():
            train_data.append((float(parts[11]), float(parts[12]), float(parts[13]), float(parts[14])))

print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# RBF function
def rbf_func(x, c, var):
    dist_sq = np.sum((np.array(x) - np.array(c))**2)
    # add small epsilon to variance to avoid division by zero
    return np.exp(-dist_sq / (2 * (var + 1e-8)))

X_train = np.array([ [t[0], t[1], t[2]] for t in train_data ])
D_train = np.array([ t[3] for t in train_data ])

X_test = np.array([ [t[1], t[2], t[3]] for t in test_data ])
D_test = np.array([ t[4] for t in test_data ])

eta = 0.01
epsilon = 1e-7

results_md = "# Respostas - Trabalho RBF2\n\n"

# We need to run for N1 in [5, 10, 15]
topologies = [5, 10, 15]

# To store all eqm curves to plot
eqm_curves = {}
best_models = {}

md_table1 = "| Treinamento | Rede 1 (N1=5) EQM | Rede 1 Épocas | Rede 2 (N1=10) EQM | Rede 2 Épocas | Rede 3 (N1=15) EQM | Rede 3 Épocas |\n"
md_table1 += "|---|---|---|---|---|---|---|\n"

results_by_topology_run = {}

for top_idx, n_clusters in enumerate(topologies):
    print(f"Training Topology N1={n_clusters}...")
    
    # Train hidden layer (K-means)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_train)
    centers = kmeans.cluster_centers_
    
    variances = []
    for i in range(n_clusters):
        cluster_points = X_train[kmeans.labels_ == i]
        if len(cluster_points) > 0:
            var = np.mean(np.sum((cluster_points - centers[i])**2, axis=1))
        else:
            var = 1.0
        # Sometimes variance can be 0 if only 1 point, so handle it
        if var == 0: var = 1.0
        variances.append(var)
        
    # Prepare H matrix (precompute h values for train set since centers are fixed)
    H_train = np.zeros((len(train_data), n_clusters + 1))
    for i in range(len(train_data)):
        H_train[i, 0] = 1.0 # bias
        for j in range(n_clusters):
            H_train[i, j+1] = rbf_func(X_train[i], centers[j], variances[j])
            
    H_test = np.zeros((len(test_data), n_clusters + 1))
    for i in range(len(test_data)):
        H_test[i, 0] = 1.0
        for j in range(n_clusters):
            H_test[i, j+1] = rbf_func(X_test[i], centers[j], variances[j])
            
    for run in range(3):
        print(f"  Run {run+1}/3...")
        np.random.seed(run * 100 + top_idx * 10)
        # "inicializando a matriz de pesos da camada de saída com valores aleatórios entre 0 e 1"
        W = np.random.uniform(0, 1, n_clusters + 1)
        
        mse_prev = float('inf')
        epochs = 0
        eqm_history = []
        
        while True:
            # Batch or Online? The instruction says "regra delta generalizada", usually SGD online.
            # RBF output is linear, we can just do online SGD.
            for i in range(len(train_data)):
                y_in = np.dot(W, H_train[i])
                e = D_train[i] - y_in
                W += eta * e * H_train[i]
                
            # Calc MSE
            Y_pred = np.dot(H_train, W)
            mse = np.mean((D_train - Y_pred)**2)
            eqm_history.append(mse)
            
            if abs(mse_prev - mse) < epsilon:
                break
            mse_prev = mse
            epochs += 1
            if epochs > 10000: # safeguard
                break
                
        # Save results
        Y_test_pred = np.dot(H_test, W)
        
        # Erro relativo médio (%)
        rel_errors = np.abs((Y_test_pred - D_test) / D_test) * 100
        mean_rel_err = np.mean(rel_errors)
        var_rel_err = np.var(rel_errors)
        
        results_by_topology_run[(top_idx, run)] = {
            'eqm': mse,
            'epochs': epochs,
            'eqm_history': eqm_history,
            'Y_test_pred': Y_test_pred,
            'mean_rel_err': mean_rel_err,
            'var_rel_err': var_rel_err,
            'W': W.copy()
        }

# Generate Markdown Tables

# Table 1: Training Results
for run in range(3):
    r1 = results_by_topology_run[(0, run)]
    r2 = results_by_topology_run[(1, run)]
    r3 = results_by_topology_run[(2, run)]
    md_table1 += f"| {run+1}º (T{run+1}) | {r1['eqm']:.6f} | {r1['epochs']} | {r2['eqm']:.6f} | {r2['epochs']} | {r3['eqm']:.6f} | {r3['epochs']} |\n"

results_md += "## 1. Resultados dos Treinamentos\n\n"
results_md += md_table1 + "\n\n"

# Table 2: Test Results
md_table2 = "## 2. Validação\n\n"
md_table2 += "| Amostra | $x_1$ | $x_2$ | $x_3$ | $d$ | Rede 1 (T1) | Rede 1 (T2) | Rede 1 (T3) | Rede 2 (T1) | Rede 2 (T2) | Rede 2 (T3) | Rede 3 (T1) | Rede 3 (T2) | Rede 3 (T3) |\n"
md_table2 += "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"

for i in range(len(test_data)):
    amostra, x1, x2, x3, d = test_data[i]
    row = f"| {amostra:02d} | {x1:.4f} | {x2:.4f} | {x3:.4f} | {d:.4f} "
    for top_idx in range(3):
        for run in range(3):
            y_pred = results_by_topology_run[(top_idx, run)]['Y_test_pred'][i]
            row += f"| {y_pred:.4f} "
    row += "|\n"
    md_table2 += row

# Append Error/Var rows
row_err = "| **Erro Rel. Médio (%)** | | | | | "
row_var = "| **Variância (%)** | | | | | "
for top_idx in range(3):
    for run in range(3):
        mre = results_by_topology_run[(top_idx, run)]['mean_rel_err']
        vre = results_by_topology_run[(top_idx, run)]['var_rel_err']
        row_err += f"{mre:.2f} | "
        row_var += f"{vre:.2f} | "

md_table2 += row_err + "\n" + row_var + "\n\n"

results_md += md_table2

# Identify best training per topology (minimum MSE on test, or minimum EQM on train?)
# The prompt says: "Para cada uma das topologias apresentadas... considerando o melhor treinamento {T1, T2 ou T3}... trace o gráfico..."
# Usually "melhor treinamento" means lowest EQM on training or lowest Error on test. Let's use lowest EQM on train.
best_runs = []
for top_idx in range(3):
    best_run = 0
    best_eqm = float('inf')
    for run in range(3):
        if results_by_topology_run[(top_idx, run)]['eqm'] < best_eqm:
            best_eqm = results_by_topology_run[(top_idx, run)]['eqm']
            best_run = run
    best_runs.append(best_run)

# Plotting graphs
plt.figure(figsize=(15, 5))
for top_idx in range(3):
    run = best_runs[top_idx]
    hist = results_by_topology_run[(top_idx, run)]['eqm_history']
    plt.subplot(1, 3, top_idx+1)
    plt.plot(hist)
    plt.title(f"Rede {top_idx+1} (N1={topologies[top_idx]}) - T{run+1}")
    plt.xlabel("Épocas")
    plt.ylabel("EQM")
    plt.grid(True)

plt.tight_layout()
plt.savefig('c:/Users/Gi/Lab_IA_I/IA_2026.1/rbf/RBF2/graficos_eqm.png')

results_md += "## 3. Gráficos de EQM\n\n"
results_md += "![Gráficos EQM](c:/Users/Gi/Lab_IA_I/IA_2026.1/rbf/RBF2/graficos_eqm.png)\n\n"

# Conclusion
# Best overall model based on test Error Relativo Médio
best_overall_top = 0
best_overall_run = 0
min_mre = float('inf')
for top_idx in range(3):
    for run in range(3):
        if results_by_topology_run[(top_idx, run)]['mean_rel_err'] < min_mre:
            min_mre = results_by_topology_run[(top_idx, run)]['mean_rel_err']
            best_overall_top = top_idx
            best_overall_run = run

results_md += "## 4. Conclusão\n\n"
results_md += f"A topologia mais adequada é a **Rede {best_overall_top+1}** com a configuração de treinamento **T{best_overall_run+1}**, pois apresentou o menor erro relativo médio ({min_mre:.2f}%) nos dados de validação, demonstrando melhor capacidade de generalização.\n"

with open('c:/Users/Gi/Lab_IA_I/IA_2026.1/rbf/RBF2/respostas.md', 'w', encoding='utf-8') as f:
    f.write(results_md)

print("Done")
