"""
=============================================================================
ADALINE (Adaptive Linear Neuron) — Regra Delta
=============================================================================
Atividade acadêmica: classificação de sinais em Válvula A (-1) ou Válvula B (+1).
Rede com 4 entradas (x1..x4) + bias (x0 = -1).

Parâmetros:
  - Taxa de aprendizado (η)  = 0.0025
  - Precisão (ε)             = 1e-6
  - Ativação: degrau bipolar  y = 1 se u ≥ 0, senão y = -1

Autor: Gerado automaticamente
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# 1. PARÂMETROS GLOBAIS
# =====================================================================
ETA     = 0.0025   # taxa de aprendizado
EPSILON = 1e-6     # precisão (critério de parada)
NUM_TREINAMENTOS = 5

# =====================================================================
# 2. DADOS
# =====================================================================

# Conjunto de Treinamento (x1, x2, x3, x4, d)
dados_treinamento = np.array([
    [ 0.4329, -1.3719,  0.7022, -0.8535,  1.0],
    [ 0.3024,  0.2286,  0.8630,  2.7909, -1.0],
    [ 0.1349, -0.6445,  1.0530,  0.5687, -1.0],
    [ 0.3374, -1.7163,  0.3670, -0.6283, -1.0],
    [ 1.1434, -0.0485,  0.6637,  1.2606,  1.0],
    [ 1.3749, -0.5071,  0.4464,  1.3009,  1.0],
    [ 0.7221, -0.7587,  0.7681, -0.5592,  1.0],
    [ 0.4403, -0.8072,  0.5154, -0.3129,  1.0],
    [-0.5231,  0.3548,  0.2538,  1.5776, -1.0],
    [ 0.3255, -2.0000,  0.7112, -1.1209,  1.0],
    [ 0.5824,  1.3915, -0.2291,  4.1735, -1.0],
    [ 0.1340,  0.6081,  0.4450,  3.2230, -1.0],
    [ 0.1480, -0.2988,  0.4778,  0.8649,  1.0],
    [ 0.7359,  0.1869, -0.0872,  2.3584,  1.0],
    [ 0.7115, -1.1469,  0.3394,  0.9573, -1.0],
    [ 0.8251, -1.2840,  0.8452,  1.2382, -1.0],
    [ 0.1569,  0.3712,  0.8825,  1.7633,  1.0],
    [ 0.0033,  0.6835,  0.5389,  2.8249, -1.0],
    [ 0.4243,  0.8313,  0.2634,  3.5855, -1.0],
    [ 1.0490,  0.1326,  0.9138,  1.9792,  1.0],
    [ 1.4276,  0.5331, -0.0145,  3.7286,  1.0],
    [ 0.5971,  1.4865,  0.2904,  4.6069, -1.0],
    [ 0.8475,  2.1479,  0.3179,  5.8235, -1.0],
    [ 1.3967, -0.4171,  0.6443,  1.3927,  1.0],
    [ 0.0044,  1.5378,  0.6099,  4.7755, -1.0],
    [ 0.2201, -0.5668,  0.0515,  0.7829,  1.0],
    [ 0.6300, -1.2480,  0.8591,  0.8093, -1.0],
    [-0.2479,  0.8960,  0.0547,  1.7381,  1.0],
    [-0.3088, -0.0929,  0.8659,  1.5483, -1.0],
    [-0.5180,  1.4974,  0.5453,  2.3993,  1.0],
    [ 0.6833,  0.8266,  0.0829,  2.8864,  1.0],
    [ 0.4353, -1.4066,  0.4207, -0.4879,  1.0],
    [-0.1069, -3.2329,  0.1856, -2.4572, -1.0],
    [ 0.4662,  0.6261,  0.7304,  3.4370, -1.0],
    [ 0.8298, -1.4089,  0.3119,  1.3235, -1.0],
])

# Conjunto de Teste (x1, x2, x3, x4)
dados_teste = np.array([
    [ 0.9694,  0.6909,  0.4334,  3.4965],
    [ 0.5427,  1.3832,  0.6390,  4.0352],
    [ 0.6081, -0.9196,  0.5925,  0.1016],
    [-0.1618,  0.4694,  0.2030,  3.0117],
    [ 0.1870, -0.2578,  0.6124,  1.7749],
    [ 0.4891, -0.5276,  0.4378,  0.6439],
    [ 0.3777,  2.0149,  0.7423,  3.3932],
    [ 1.1498, -0.4067,  0.2469,  1.5866],
    [ 0.9325,  1.0950,  1.0359,  3.3591],
    [ 0.5060,  1.3317,  0.9222,  3.7174],
    [ 0.0497, -2.0656,  0.6124, -0.6585],
    [ 0.4004,  3.5369,  0.9766,  5.3532],
    [-0.1874,  1.3343,  0.5374,  3.2189],
    [ 0.5060,  1.3317,  0.9222,  3.7174],
    [ 1.6375, -0.7911,  0.7537,  0.5515],
])


# =====================================================================
# 3. FUNÇÕES AUXILIARES
# =====================================================================

def degrau_bipolar(u):
    """Função de ativação degrau bipolar.
    Retorna +1 se u >= 0, senão -1.
    """
    return np.where(u >= 0, 1.0, -1.0)


def adicionar_bias(X):
    """Adiciona a coluna de bias x0 = -1 como primeira coluna.
    Entrada X tem shape (N, 4). Saída tem shape (N, 5).
    """
    N = X.shape[0]
    bias = -1.0 * np.ones((N, 1))
    return np.hstack((bias, X))


def calcular_eqm(X, d, w):
    """Calcula o Erro Quadrático Médio (EQM).
    
    EQM = (1/N) * Σ (d_i - u_i)²
    
    onde u_i = w · x_i  (saída linear, antes da ativação).
    
    Parâmetros:
        X : array (N, 5) — entradas com bias
        d : array (N,)   — saídas desejadas
        w : array (5,)   — vetor de pesos
    
    Retorna:
        eqm : float
    """
    N = X.shape[0]
    u = X @ w           # saída linear (N,)
    erros = d - u
    eqm = np.sum(erros ** 2) / N
    return eqm


def treinar_adaline(X_bias, d, eta, epsilon, seed):
    """Treina a rede ADALINE usando a Regra Delta.
    
    Algoritmo:
        1. Inicializa pesos aleatórios em [0, 1).
        2. Para cada época:
           a. Para cada amostra i:
              - Calcula u_i = w · x_i
              - Atualiza: w = w + η * (d_i - u_i) * x_i
           b. Calcula EQM da época.
           c. Se |EQM_atual - EQM_anterior| < ε → para.
    
    Parâmetros:
        X_bias  : array (N, 5) — entradas com bias
        d       : array (N,)   — saídas desejadas
        eta     : float        — taxa de aprendizado
        epsilon : float        — precisão para parada
        seed    : int          — semente do gerador aleatório
    
    Retorna:
        w_inicial   : array (5,) — pesos iniciais
        w_final     : array (5,) — pesos finais
        num_epocas  : int        — número de épocas até convergência
        historico_eqm : list     — EQM por época
    """
    rng = np.random.default_rng(seed)
    w = rng.random(X_bias.shape[1])    # pesos iniciais em [0, 1)
    w_inicial = w.copy()

    N = X_bias.shape[0]
    historico_eqm = []
    eqm_anterior = float('inf')
    epoca = 0

    while True:
        # ---- Fase de atualização (amostra a amostra) ----
        for i in range(N):
            x_i = X_bias[i]
            u_i = np.dot(w, x_i)          # saída linear
            erro_i = d[i] - u_i
            w = w + eta * erro_i * x_i     # Regra Delta

        # ---- Calcula EQM após a época ----
        eqm_atual = calcular_eqm(X_bias, d, w)
        historico_eqm.append(eqm_atual)
        epoca += 1

        # ---- Critério de parada ----
        if abs(eqm_atual - eqm_anterior) < epsilon:
            break

        eqm_anterior = eqm_atual

    return w_inicial, w.copy(), epoca, historico_eqm


def classificar(X_bias, w):
    """Classifica amostras usando os pesos treinados.
    
    Retorna:
        y : array (N,) — predições (+1 ou -1)
    """
    u = X_bias @ w
    return degrau_bipolar(u)


# =====================================================================
# 4. EXECUÇÃO PRINCIPAL
# =====================================================================

def main():
    # ----- Preparar dados -----
    X_treino = dados_treinamento[:, :4]
    d_treino = dados_treinamento[:, 4]
    X_treino_bias = adicionar_bias(X_treino)

    X_teste_bias = adicionar_bias(dados_teste)

    # Seeds diferentes para cada treinamento
    seeds = [42, 123, 7, 2024, 99]

    # ----- Executar 5 treinamentos -----
    resultados = []
    historicos = []

    for t in range(NUM_TREINAMENTOS):
        w_ini, w_fim, epocas, hist = treinar_adaline(
            X_treino_bias, d_treino, ETA, EPSILON, seeds[t]
        )
        resultados.append({
            'Treinamento': f'T{t+1}',
            'w0_inicial': w_ini[0],
            'w1_inicial': w_ini[1],
            'w2_inicial': w_ini[2],
            'w3_inicial': w_ini[3],
            'w4_inicial': w_ini[4],
            'w0_final':   w_fim[0],
            'w1_final':   w_fim[1],
            'w2_final':   w_fim[2],
            'w3_final':   w_fim[3],
            'w4_final':   w_fim[4],
            'Épocas':     epocas,
        })
        historicos.append(hist)

    # =================================================================
    # SAÍDA 1 — Tabela de Pesos
    # =================================================================
    df_pesos = pd.DataFrame(resultados)
    df_pesos = df_pesos.set_index('Treinamento')

    print("=" * 100)
    print("SAÍDA 1 — TABELA DE PESOS (5 Treinamentos)")
    print("=" * 100)
    # Formatar para melhor exibição
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.6f}'.format)
    print(df_pesos.to_string())
    print()

    # =================================================================
    # SAÍDA 2 — Gráfico EQM (T1 e T2)
    # =================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(1, len(historicos[0]) + 1), historicos[0],
            label='T1', color='#2196F3', linewidth=2)
    ax.plot(range(1, len(historicos[1]) + 1), historicos[1],
            label='T2', color='#F44336', linewidth=2, linestyle='--')

    ax.set_xlabel('Época', fontsize=13)
    ax.set_ylabel('Erro Quadrático Médio (EQM)', fontsize=13)
    ax.set_title('Convergência do EQM — ADALINE (Regra Delta)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    caminho_grafico = 'grafico_eqm.png'
    fig.savefig(caminho_grafico, dpi=150, bbox_inches='tight')
    print(f"Gráfico salvo em: {caminho_grafico}")
    print()

    # =================================================================
    # SAÍDA 3 — Tabela de Classificação (Teste)
    # =================================================================
    classificacoes = {}
    for t in range(NUM_TREINAMENTOS):
        w_fim = np.array([
            resultados[t]['w0_final'],
            resultados[t]['w1_final'],
            resultados[t]['w2_final'],
            resultados[t]['w3_final'],
            resultados[t]['w4_final'],
        ])
        y_pred = classificar(X_teste_bias, w_fim)
        classificacoes[f'y(T{t+1})'] = y_pred.astype(int)

    df_teste = pd.DataFrame({
        'Amostra': range(1, len(dados_teste) + 1),
        'x1': dados_teste[:, 0],
        'x2': dados_teste[:, 1],
        'x3': dados_teste[:, 2],
        'x4': dados_teste[:, 3],
    })

    for col, vals in classificacoes.items():
        df_teste[col] = vals

    df_teste = df_teste.set_index('Amostra')

    print("=" * 100)
    print("SAÍDA 3 — CLASSIFICAÇÃO DO CONJUNTO DE TESTE")
    print("=" * 100)
    print(df_teste.to_string())
    print()

    # ----- Resumo rápido -----
    print("=" * 100)
    print("RESUMO")
    print("=" * 100)
    for t in range(NUM_TREINAMENTOS):
        print(f"  T{t+1}: {resultados[t]['Épocas']:>6} épocas | "
              f"EQM final = {historicos[t][-1]:.8f}")
    print()


if __name__ == '__main__':
    main()
