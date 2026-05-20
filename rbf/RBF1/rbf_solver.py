import math
import numpy as np
from sklearn.cluster import KMeans

train_data = [
    (0.2563, 0.9503, -1), (0.2405, 0.9018, -1), (0.1157, 0.3676, 1), (0.5147, 0.0167, 1),
    (0.4127, 0.3275, 1), (0.2809, 0.583, 1), (0.8263, 0.9301, -1), (0.9359, 0.8724, -1),
    (0.1096, 0.9165, -1), (0.5158, 0.8545, -1), (0.1334, 0.1362, 1), (0.6371, 0.1439, 1),
    (0.7052, 0.6277, -1), (0.8703, 0.8666, -1), (0.2612, 0.6109, 1), (0.0244, 0.5279, 1),
    (0.9588, 0.3672, -1), (0.9332, 0.5499, -1), (0.9623, 0.2961, -1), (0.7297, 0.5776, -1),
    (0.456, 0.1871, 1), (0.1715, 0.7713, 1), (0.5571, 0.5485, -1), (0.3344, 0.0259, 1),
    (0.4803, 0.7635, -1), (0.9721, 0.485, -1), (0.8318, 0.7844, -1), (0.1373, 0.0292, 1),
    (0.366, 0.8581, -1), (0.3626, 0.7302, -1), (0.6474, 0.3324, 1), (0.3461, 0.2398, 1),
    (0.1353, 0.812, 1), (0.3463, 0.1017, 1), (0.9086, 0.1947, -1), (0.5227, 0.2321, 1),
    (0.5153, 0.2041, 1), (0.1832, 0.0661, 1), (0.5015, 0.9812, -1), (0.5024, 0.5274, -1)
]

test_data = [
    (1, 0.8705, 0.9329, -1), (2, 0.0388, 0.2703, 1), (3, 0.8236, 0.4458, -1),
    (4, 0.7075, 0.1502, 1), (5, 0.9587, 0.8663, -1), (6, 0.6115, 0.9365, -1),
    (7, 0.3534, 0.3646, 1), (8, 0.3268, 0.2766, 1), (9, 0.6129, 0.4518, -1),
    (10, 0.9948, 0.4962, -1)
]

X_rad = np.array([[x1, x2] for x1, x2, d in train_data if d == 1])

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_rad)
centers = kmeans.cluster_centers_

variances = []
for i in range(2):
    cluster_points = X_rad[kmeans.labels_ == i]
    if len(cluster_points) > 0:
        var = np.mean(np.sum((cluster_points - centers[i])**2, axis=1))
    else:
        var = 1.0
    variances.append(var)

def rbf_func(x, c, var):
    dist_sq = np.sum((np.array(x) - np.array(c))**2)
    return np.exp(-dist_sq / (2 * var))

np.random.seed(42)
w = np.random.uniform(-0.5, 0.5, 3) # w0, w1, w2
eta = 0.01
epsilon = 1e-7

mse_prev = float('inf')
epochs = 0
while True:
    mse = 0
    for x1, x2, d in train_data:
        h1 = rbf_func([x1, x2], centers[0], variances[0])
        h2 = rbf_func([x1, x2], centers[1], variances[1])
        y_in = w[0]*1 + w[1]*h1 + w[2]*h2
        
        e = d - y_in
        w[0] += eta * e * 1
        w[1] += eta * e * h1
        w[2] += eta * e * h2
        
    for x1, x2, d in train_data:
        h1 = rbf_func([x1, x2], centers[0], variances[0])
        h2 = rbf_func([x1, x2], centers[1], variances[1])
        y_in = w[0]*1 + w[1]*h1 + w[2]*h2
        mse += (d - y_in)**2
    mse /= len(train_data)
    
    if abs(mse_prev - mse) < epsilon:
        break
    mse_prev = mse
    epochs += 1
    if epochs > 100000:
        break

results = []
correct = 0
for idx, x1, x2, d in test_data:
    h1 = rbf_func([x1, x2], centers[0], variances[0])
    h2 = rbf_func([x1, x2], centers[1], variances[1])
    y = w[0]*1 + w[1]*h1 + w[2]*h2
    y_pos = 1 if y >= 0 else -1
    if y_pos == d:
        correct += 1
    results.append((idx, x1, x2, d, y, y_pos))

acc = correct / len(test_data) * 100

md = f"""# Respostas - Trabalho RBF1

## Treinamento da Camada Escondida (K-Means)
Apenas padrões com presença de radiação ($d = 1$) foram considerados.

| Cluster | Centro ($x_1$, $x_2$) | Variância ($\sigma^2$) |
|---------|--------|-----------|
| 1 | ({centers[0][0]:.4f}, {centers[0][1]:.4f}) | {variances[0]:.4f} |
| 2 | ({centers[1][0]:.4f}, {centers[1][1]:.4f}) | {variances[1]:.4f} |

## Treinamento da Camada de Saída (Regra Delta Generalizada)
- Taxa de aprendizado $\eta = 0.01$
- Precisão $\epsilon = 10^{{-7}}$
- Épocas de convergência: {epochs}

| Peso | Valor Final |
|------|-------|
| $W_{{21,0}}$ (Bias) | {w[0]:.6f} |
| $W_{{21,1}}$ | {w[1]:.6f} |
| $W_{{21,2}}$ | {w[2]:.6f} |

## Validação e Pós-processamento

| Amostra | $x_1$ | $x_2$ | $d$ | $y$ (Real) | $y_{{pós}}$ (Sinal) |
|---------|-------|-------|-----|-----|---------|
"""
for r in results:
    md += f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]} | {r[4]:.4f} | {r[5]} |\n"

md += f"\n**Taxa de Acerto (%):** {acc:.2f}%\n"
md += "\n## Estratégias para Aumentar a Taxa de Acerto\n"
md += "Caso a taxa de acerto não seja ideal, as seguintes estratégias podem ser adotadas para aprimorar a RBF:\n\n"
md += "1. **Aumentar o número de neurônios na camada oculta:** Utilizar um valor maior para $k$ no K-means (por exemplo, 3 ou mais) pode capturar melhor as regiões de distribuição dos dados da classe de radiação. O problema de classificação pode não ser separável com apenas duas gaussianas centradas na classe de radiação.\n"
md += "2. **Incluir os padrões sem radiação no K-means:** Realizar o agrupamento considerando ambas as classes e gerar gaussianas para todos os sub-padrões, o que ajudaria a mapear a fronteira de decisão de forma muito mais precisa ao invés de apenas focar em padrões com radiação.\n"
md += "3. **Ajuste heurístico ou supervisionado da Variância ($\sigma^2$):** Em vez da variância amostral, testar heurísticas como a distância máxima entre os centros dos clusters ($d_{max} / \sqrt{2k}$), ou até usar o gradiente (regra delta generalizada) para otimizar os próprios centros e variâncias (além dos pesos) reduzindo o erro médio quadrático.\n"
md += "4. **Aumentar a base de dados de treinamento:** Um número maior de amostras representativas evitaria o overfitting aos dados de treinamento e melhoraria a generalização do modelo no conjunto de teste.\n"

with open('c:/Users/Gi/Lab_IA_I/IA_2026.1/rbf/RBF1/respostas.md', 'w', encoding='utf-8') as f:
    f.write(md)

print("Done")
