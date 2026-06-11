"""
LVQ-1 (Learning Vector Quantization 1) - Classificação de Perfis de Potência Elétrica
=======================================================================================
Centro Federal de Educação Tecnológica de Minas Gerais
Disciplina: Lab. Inteligência Artificial
Professor: Lázaro Eduardo da Silva

Descrição:
    Implementação e treinamento de uma rede LVQ-1 para classificação de perfis
    de demanda de potência elétrica, com base em medições das 7h às 12h.
    
    - 16 amostras de treinamento divididas em 4 classes (perfis de demanda)
    - 6 atributos de entrada (potência medida de 7h a 12h)
    - Taxa de aprendizagem α = 0.05
"""

import numpy as np
import os

# =============================================================================
# 1. DADOS DE TREINAMENTO (16 amostras, 4 classes)
# =============================================================================
# Cada amostra: [7h, 8h, 9h, 10h, 11h, 12h]

dados_treino = np.array([
    # Classe 1 (amostras 1–4)
    [2.3976, 1.5328, 1.9044, 1.1937, 2.4184, 1.8649],   # Amostra 1
    [2.3936, 1.4804, 1.9907, 1.2732, 2.2719, 1.8110],   # Amostra 2
    [2.2880, 1.4585, 1.9867, 1.2451, 2.3389, 1.8099],   # Amostra 3
    [2.2904, 1.4766, 1.8876, 1.2706, 2.2966, 1.7744],   # Amostra 4
    # Classe 2 (amostras 5–8)
    [1.1201, 0.0587, 1.3154, 5.3783, 3.1849, 2.4276],   # Amostra 5
    [0.9913, 0.1524, 1.2700, 5.3808, 3.0714, 2.3331],   # Amostra 6
    [1.0915, 0.1881, 1.1387, 5.3701, 3.2561, 2.3383],   # Amostra 7
    [1.0535, 0.1229, 1.2743, 5.3226, 3.0950, 2.3193],   # Amostra 8
    # Classe 3 (amostras 9–12)
    [1.4871, 2.3448, 0.9918, 2.3160, 1.6783, 5.0850],   # Amostra 9
    [1.3312, 2.2553, 0.9618, 2.4702, 1.7272, 5.0645],   # Amostra 10
    [1.3646, 2.2945, 1.0562, 2.4763, 1.8051, 5.1470],   # Amostra 11
    [1.4392, 2.2296, 1.1278, 2.4230, 1.7259, 5.0876],   # Amostra 12
    # Classe 4 (amostras 13–16)
    [2.9364, 1.5233, 4.6109, 1.3160, 4.2700, 6.8749],   # Amostra 13
    [2.9034, 1.4640, 4.6061, 1.4598, 4.2912, 6.9142],   # Amostra 14
    [3.0181, 1.4918, 4.7051, 1.3521, 4.2623, 6.7966],   # Amostra 15
    [2.9374, 1.4896, 4.7219, 1.3977, 4.1863, 6.8336],   # Amostra 16
])

rotulos_treino = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

# =============================================================================
# 2. DADOS DE TESTE (8 amostras, classes desconhecidas)
# =============================================================================

dados_teste = np.array([
    [2.9817, 1.5656, 4.8391, 1.4311, 4.1916, 6.9718],   # Dia 1
    [1.5537, 2.2615, 1.3169, 2.5873, 1.7570, 5.0958],   # Dia 2
    [1.2240, 0.2445, 1.3595, 5.4192, 3.2027, 2.5675],   # Dia 3
    [2.5828, 1.5146, 2.1119, 1.2859, 2.3414, 1.8695],   # Dia 4
    [2.4168, 1.4857, 1.8959, 1.3013, 2.4500, 1.7868],   # Dia 5
    [1.0604, 0.2276, 1.2806, 5.4732, 3.2133, 2.4839],   # Dia 6
    [1.5246, 2.4254, 1.1353, 2.5325, 1.7569, 5.2640],   # Dia 7
    [3.0565, 1.6259, 4.7743, 1.3654, 4.2904, 6.9808],   # Dia 8
])

# =============================================================================
# 3. IMPLEMENTAÇÃO DA LVQ-1
# =============================================================================

class LVQ1:
    """
    Rede LVQ-1 (Learning Vector Quantization 1).
    
    Algoritmo:
        1. Inicializa protótipos (um por classe) usando a média das amostras de cada classe
        2. Para cada época:
           a. Embaralha as amostras de treinamento
           b. Para cada amostra x com rótulo c:
              - Encontra o protótipo mais próximo (vencedor) w_j usando distância Euclidiana
              - Se o rótulo do protótipo vencedor == c (acerto):
                  w_j = w_j + α * (x - w_j)   → aproxima o protótipo da amostra
              - Se o rótulo do protótipo vencedor != c (erro):
                  w_j = w_j - α * (x - w_j)   → afasta o protótipo da amostra
    """
    
    def __init__(self, taxa_aprendizagem=0.05, n_epocas=1000, seed=42):
        self.alpha = taxa_aprendizagem
        self.n_epocas = n_epocas
        self.seed = seed
        self.prototipos = None
        self.rotulos_prototipos = None
        self.historico_erros = []
        
    def _distancia_euclidiana(self, x, w):
        """Calcula a distância Euclidiana entre x e w."""
        return np.sqrt(np.sum((x - w) ** 2))
    
    def _encontrar_vencedor(self, x):
        """
        Encontra o índice do protótipo mais próximo de x (BMU - Best Matching Unit).
        Retorna o índice do protótipo vencedor.
        """
        distancias = np.array([self._distancia_euclidiana(x, w) for w in self.prototipos])
        return np.argmin(distancias)
    
    def _inicializar_prototipos(self, X, y):
        """
        Inicializa os protótipos como a média das amostras de cada classe.
        Essa estratégia garante que os protótipos já começam em posições representativas.
        """
        classes = np.unique(y)
        self.rotulos_prototipos = classes.copy()
        self.prototipos = np.zeros((len(classes), X.shape[1]))
        
        for i, c in enumerate(classes):
            amostras_classe = X[y == c]
            self.prototipos[i] = np.mean(amostras_classe, axis=0)
    
    def treinar(self, X, y, verbose=True):
        """
        Treina a rede LVQ-1.
        
        Parâmetros:
            X: array (n_amostras, n_atributos) - dados de treinamento
            y: array (n_amostras,) - rótulos das amostras
            verbose: se True, imprime informações durante o treinamento
        """
        np.random.seed(self.seed)
        
        # Inicializar protótipos
        self._inicializar_prototipos(X, y)
        
        if verbose:
            print("=" * 70)
            print("TREINAMENTO LVQ-1")
            print("=" * 70)
            print(f"Taxa de aprendizagem (α): {self.alpha}")
            print(f"Número de épocas: {self.n_epocas}")
            print(f"Número de amostras: {X.shape[0]}")
            print(f"Número de atributos: {X.shape[1]}")
            print(f"Número de classes: {len(np.unique(y))}")
            print(f"Número de protótipos: {len(self.prototipos)}")
            print()
            print("Protótipos Iniciais (média das amostras de cada classe):")
            for i, (p, r) in enumerate(zip(self.prototipos, self.rotulos_prototipos)):
                print(f"  Protótipo {i+1} (Classe {r}): {np.round(p, 4)}")
            print()
        
        # Treinamento
        for epoca in range(self.n_epocas):
            # Embaralhar índices
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            
            erros_epoca = 0
            
            for idx in indices:
                x = X[idx]
                rotulo_real = y[idx]
                
                # Encontrar protótipo vencedor (mais próximo)
                idx_vencedor = self._encontrar_vencedor(x)
                rotulo_vencedor = self.rotulos_prototipos[idx_vencedor]
                
                # Atualizar protótipo vencedor
                if rotulo_vencedor == rotulo_real:
                    # Acerto: aproximar o protótipo da amostra
                    self.prototipos[idx_vencedor] += self.alpha * (x - self.prototipos[idx_vencedor])
                else:
                    # Erro: afastar o protótipo da amostra
                    self.prototipos[idx_vencedor] -= self.alpha * (x - self.prototipos[idx_vencedor])
                    erros_epoca += 1
            
            self.historico_erros.append(erros_epoca)
            
            # Log a cada 100 épocas ou na última
            if verbose and (epoca + 1) % 200 == 0 or epoca == 0:
                taxa_acerto = ((X.shape[0] - erros_epoca) / X.shape[0]) * 100
                print(f"  Época {epoca+1:4d}: Erros = {erros_epoca:2d}/{X.shape[0]}, "
                      f"Acurácia = {taxa_acerto:.1f}%")
        
        if verbose:
            print()
            print("Protótipos Finais (após treinamento):")
            for i, (p, r) in enumerate(zip(self.prototipos, self.rotulos_prototipos)):
                print(f"  Protótipo {i+1} (Classe {r}): {np.round(p, 4)}")
            print()
            
            # Verificar classificação do conjunto de treinamento
            print("Verificação no conjunto de treinamento:")
            predicoes_treino = self.classificar(X)
            acertos = np.sum(predicoes_treino == y)
            print(f"  Acurácia no treinamento: {acertos}/{X.shape[0]} "
                  f"({(acertos/X.shape[0])*100:.1f}%)")
            print()
    
    def classificar(self, X):
        """
        Classifica um conjunto de amostras.
        
        Parâmetros:
            X: array (n_amostras, n_atributos) - dados a classificar
            
        Retorna:
            Array com os rótulos preditos para cada amostra.
        """
        predicoes = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            idx_vencedor = self._encontrar_vencedor(X[i])
            predicoes[i] = self.rotulos_prototipos[idx_vencedor]
        return predicoes
    
    def classificar_detalhado(self, X, nomes_amostras=None):
        """
        Classifica amostras e retorna informações detalhadas incluindo distâncias.
        """
        resultados = []
        for i in range(X.shape[0]):
            distancias = np.array([self._distancia_euclidiana(X[i], w) for w in self.prototipos])
            idx_vencedor = np.argmin(distancias)
            classe_predita = self.rotulos_prototipos[idx_vencedor]
            
            nome = nomes_amostras[i] if nomes_amostras else f"Amostra {i+1}"
            
            resultados.append({
                'nome': nome,
                'entrada': X[i],
                'classe_predita': classe_predita,
                'distancias': distancias,
                'distancia_vencedor': distancias[idx_vencedor]
            })
        
        return resultados


# =============================================================================
# 4. EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + " LVQ-1: Classificação de Perfis de Potência Elétrica ".center(68) + "║")
    print("║" + " CEFET-MG / Lab. Inteligência Artificial ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # =========================================================================
    # 4.1 Criar e treinar a rede LVQ-1
    # =========================================================================
    rede = LVQ1(taxa_aprendizagem=0.05, n_epocas=1000, seed=42)
    rede.treinar(dados_treino, rotulos_treino, verbose=True)
    
    # =========================================================================
    # 4.2 Classificação dos dados de treinamento (verificação)
    # =========================================================================
    print("=" * 70)
    print("CLASSIFICAÇÃO DO CONJUNTO DE TREINAMENTO (Verificação)")
    print("=" * 70)
    
    nomes_treino = [f"Amostra {i+1}" for i in range(len(dados_treino))]
    resultados_treino = rede.classificar_detalhado(dados_treino, nomes_treino)
    
    print(f"\n{'Amostra':<12} {'Classe Real':<12} {'Classe Predita':<15} {'Distância BMU':<15} {'Status'}")
    print("-" * 70)
    
    for i, res in enumerate(resultados_treino):
        classe_real = rotulos_treino[i]
        status = "✓ Correto" if res['classe_predita'] == classe_real else "✗ ERRO"
        print(f"{res['nome']:<12} {classe_real:<12} {res['classe_predita']:<15} "
              f"{res['distancia_vencedor']:<15.4f} {status}")
    
    # =========================================================================
    # 4.3 Classificação dos dados de teste
    # =========================================================================
    print()
    print("=" * 70)
    print("CLASSIFICAÇÃO DOS DADOS DE TESTE")
    print("=" * 70)
    
    nomes_teste = [f"Dia {i+1}" for i in range(len(dados_teste))]
    resultados_teste = rede.classificar_detalhado(dados_teste, nomes_teste)
    
    print(f"\n{'Dia':<8} {'Entrada (7h-12h)':<45} {'Classe':<8} {'Dist. Vencedor'}")
    print("-" * 70)
    
    for res in resultados_teste:
        entrada_str = "[" + ", ".join(f"{v:.4f}" for v in res['entrada']) + "]"
        print(f"{res['nome']:<8} {entrada_str:<45} {res['classe_predita']:<8} "
              f"{res['distancia_vencedor']:.4f}")
    
    # Distâncias detalhadas por protótipo
    print("\nDistâncias detalhadas para cada protótipo:")
    print(f"{'Dia':<8} {'d(P1/Classe1)':<15} {'d(P2/Classe2)':<15} {'d(P3/Classe3)':<15} {'d(P4/Classe4)':<15} {'Classe'}")
    print("-" * 80)
    
    for res in resultados_teste:
        dists = " ".join(f"{d:<15.4f}" for d in res['distancias'])
        print(f"{res['nome']:<8} {dists} {res['classe_predita']}")
    
    # =========================================================================
    # 4.4 Resumo
    # =========================================================================
    print()
    print("=" * 70)
    print("RESUMO DA CLASSIFICAÇÃO DOS DADOS DE TESTE")
    print("=" * 70)
    
    for res in resultados_teste:
        print(f"  {res['nome']}: Classe {res['classe_predita']} "
              f"(distância ao protótipo vencedor: {res['distancia_vencedor']:.4f})")
    
    print()
    print("Protótipos finais da rede:")
    for i, (p, r) in enumerate(zip(rede.prototipos, rede.rotulos_prototipos)):
        print(f"  Classe {r}: {np.round(p, 4)}")
    
    # =========================================================================
    # 4.5 Histórico de convergência
    # =========================================================================
    print()
    print("=" * 70)
    print("CONVERGÊNCIA DO TREINAMENTO")
    print("=" * 70)
    
    # Mostra erros em pontos-chave
    pontos = [0, 9, 49, 99, 199, 499, 999]
    pontos = [p for p in pontos if p < len(rede.historico_erros)]
    
    for p in pontos:
        erros = rede.historico_erros[p]
        acuracia = ((len(dados_treino) - erros) / len(dados_treino)) * 100
        print(f"  Época {p+1:4d}: {erros} erros ({acuracia:.1f}% acurácia)")
    
    # Época em que zerou erros (se alcançou)
    for i, erros in enumerate(rede.historico_erros):
        if erros == 0:
            print(f"\n  → Convergência completa (0 erros) alcançada na época {i+1}")
            break
    else:
        print(f"\n  → Erros finais na última época: {rede.historico_erros[-1]}")
    
    # Retornar resultados para uso no markdown
    return rede, resultados_treino, resultados_teste


if __name__ == "__main__":
    rede, resultados_treino, resultados_teste = main()
