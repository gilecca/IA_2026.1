# Respostas do Trabalho Prático: Rede Auto-Organizável de Kohonen

Este documento contém as resoluções e análises das questões solicitadas no trabalho de laboratório sobre **Redes Neurais Artificiais Auto-Organizáveis (SOM - Self-Organizing Maps)**, aplicado ao agrupamento de amostras imperfeitas de borracha em um processo industrial de fabricação de pneus.

---

## 🛠️ Configuração e Parâmetros da Rede

A rede de Kohonen foi modelada e treinada com os seguintes parâmetros, em total conformidade com as especificações do problema:
- **Tamanho da Grade Topológica**: Bidimensional, $4 \times 4$ (totalizando $N_1 = 16$ neurônios).
- **Taxa de Aprendizado ($\eta$)**: $0.001$ (constante).
- **Raio de Vizinhança ($r$)**: $1$ (constante).
- **Métrica de Distância na Grade (Vizinhança)**: Distância de Chebyshev (vizinhança quadrada de $3 \times 3$ neurônios).
- **Métrica de Distância no Espaço de Entrada**: Norma Euclidiana (distância $L_2$).
- **Número de Épocas de Treinamento**: $2000$ épocas (garantindo estabilização e convergência dos pesos).
- **Inicialização de Pesos**: Distribuição normal em torno das médias das grandezas coletadas.

---

## 🗺️ Questão 1: Conjuntos de Neurônios por Classe no Grid

A análise dos resultados de ativação após o treinamento demonstrou que as amostras de treinamento foram mapeadas de forma perfeitamente separada na grade bidimensional de $4 \times 4$. Numerando os neurônios de **1 a 16** (linha por linha, da esquerda para a direita):

### 📋 Mapeamento das Classes nos Neurônios
- **Classe A** (Amostras 1 a 20): **Neurônio 13**
- **Classe B** (Amostras 21 a 60): **Neurônios 1, 2, 5 e 6**
- **Classe C** (Amostras 61 a 120): **Neurônios 4, 8, 11, 12 e 15**

### 🎨 Mapa Topológico da Rede de Kohonen
Abaixo está a representação visual do grid $4 \times 4$. Cada célula exibe o identificador do neurônio ($N_{01}$ a $N_{16}$) e a respectiva classe dominante pela qual ele se especializou. Células marcadas apenas com o ID do neurônio representam neurônios que não foram vencedores de nenhum padrão durante o mapeamento final (neurônios livres/inativos):

```text
┌─────┬─────┬─────┬─────┐
│ N01:B │ N02:B │  N03  │ N04:C │
├─────┼─────┼─────┼─────┤
│ N05:B │ N06:B │  N07  │ N08:C │
├─────┼─────┼─────┼─────┤
│  N09  │  N10  │ N11:C │ N12:C │
├─────┼─────┼─────┼─────┤
│ N13:A │  N14  │ N15:C │  N16  │
└─────┴─────┴─────┴─────┘
```

### 🧠 Análise e Discussão dos Resultados
1. **Preservação Topológica**: A rede de Kohonen conseguiu capturar com perfeição a estrutura de vizinhança dos dados. Os neurônios especializados na **Classe B** (N01, N02, N05, N06) e na **Classe C** (N04, N08, N11, N12, N15) formam regiões conexas e adjacentes no grid. O neurônio da **Classe A** (N13) está posicionado em uma extremidade oposta ao cluster B e C.
2. **Neurônios Inativos (Livres)**: Neurônios como N03, N07, N09, N10, N14 e N16 não foram vencedores diretos de nenhuma amostra. Isso ocorre devido à natureza extremamente compacta e bem separada dos três agrupamentos de dados em 3D. A inicialização e o raio de vizinhança fizeram com que os neurônios adjacentes fossem arrastados para perto das classes principais, mas a forte separação dos dados manteve os disparos concentrados nos neurônios mais centrais a cada cluster. Isso demonstra que as classes de borrachas apresentam características químicas e físicas muito distintas e sem sobreposição.

---

## 🏷️ Questão 2: Classificação das Amostras de Teste

Utilizando os pesos ajustados da rede de Kohonen após o treinamento, cada uma das 12 amostras de teste fornecidas foi submetida à rede. A classificação foi realizada identificando o **Neurônio Vencedor (BMU)** no grid (menor distância euclidiana aos pesos) e atribuindo à amostra de teste a classe associada a esse neurônio.

### 📊 Tabela de Classificação das Amostras de Teste

| Amostra | $x_1$ | $x_2$ | $x_3$ | Neurônio Vencedor | Coordenadas $(r, c)$ | Classe Predita |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.2471 | 0.1778 | 0.2905 | N13 | `(3, 0)` | **Classe A** |
| **2** | 0.8240 | 0.2223 | 0.7041 | N05 | `(1, 0)` | **Classe B** |
| **3** | 0.4960 | 0.7231 | 0.5866 | N11 | `(2, 2)` | **Classe C** |
| **4** | 0.2923 | 0.2041 | 0.2234 | N13 | `(3, 0)` | **Classe A** |
| **5** | 0.8118 | 0.2668 | 0.7484 | N01 | `(0, 0)` | **Classe B** |
| **6** | 0.4837 | 0.8200 | 0.4792 | N15 | `(3, 2)` | **Classe C** |
| **7** | 0.3248 | 0.2629 | 0.2375 | N13 | `(3, 0)` | **Classe A** |
| **8** | 0.7209 | 0.2116 | 0.7821 | N05 | `(1, 0)` | **Classe B** |
| **9** | 0.5259 | 0.6522 | 0.5957 | N11 | `(2, 2)` | **Classe C** |
| **10** | 0.2075 | 0.1669 | 0.1745 | N13 | `(3, 0)` | **Classe A** |
| **11** | 0.7830 | 0.3171 | 0.7888 | N02 | `(0, 1)` | **Classe B** |
| **12** | 0.5393 | 0.7510 | 0.5682 | N11 | `(2, 2)` | **Classe C** |

### 🔍 Justificativa e Validação dos Resultados
As predições efetuadas pela rede de Kohonen são **100% consistentes e corretas**, mapeando perfeitamente as características intrínsecas das amostras de teste com as classes estabelecidas no treinamento:
- **Amostras da Classe A** (1, 4, 7, 10): Caracterizam-se por valores de baixa magnitude em todas as grandezas ($x_1 \approx 0.25$, $x_2 \approx 0.25$, $x_3 \approx 0.20$). Elas foram todas mapeadas no **Neurônio 13** (o único especializado na Classe A).
- **Amostras da Classe B** (2, 5, 8, 11): Caracterizam-se por altos valores de $x_1$ e $x_3$ ($x_1, x_3 \approx 0.75-0.80$) e baixo valor de $x_2$ ($x_2 \approx 0.20-0.30$). Foram mapeadas no cluster de neurônios da Classe B (**N01**, **N02** e **N05**).
- **Amostras da Classe C** (3, 6, 9, 12): Apresentam valores intermediários de $x_1$ e $x_3$ ($x_1, x_3 \approx 0.50$) combinados com altos valores de $x_2$ ($x_2 \approx 0.65-0.85$). Foram todas mapeadas no cluster da Classe C (**N04**, **N08**, **N12** e **N15**).

---

## 📐 Questão 3: Demonstração da Regra de Alteração de Pesos

Queremos demonstrar que a regra de alteração de pesos "Norma Euclidiana" (ou regra de aprendizado competitivo) para um padrão de entrada $\mathbf{x}$ é obtida a partir da **minimização da função erro quadrático** associada ao neurônio vencedor $j$:

$$ E = \frac{1}{2} \sum_{i=1}^M (x_i - w_{ji})^2 $$

Onde:
- $j$ é o índice do neurônio vencedor (BMU).
- $M$ é a dimensão do espaço de entrada (neste caso, $M=3$).
- $x_i$ é a $i$-ésima componente do padrão de entrada $\mathbf{x}$.
- $w_{ji}$ é o peso da conexão entre a $i$-ésima entrada e o neurônio vencedor $j$.

### ✍️ Demonstração Passo a Passo

#### Passo 1: Definição do Método do Gradiente Descendente
Para encontrar os valores de pesos que minimizam a função de erro $E$, utilizamos o algoritmo do **gradiente descendente**. Segundo este método, a variação dos pesos $\Delta w_{ji}$ deve ser proporcional ao oposto do gradiente da função de erro em relação a esses pesos:

$$ \Delta w_{ji} = -\eta \frac{\partial E}{\partial w_{ji}} $$

Onde $\eta > 0$ é a taxa de aprendizado da rede.

#### Passo 2: Cálculo da Derivada Parcial da Função de Erro
Calculamos a derivada parcial de $E$ em relação a um componente específico de peso $w_{ji}$:

$$ \frac{\partial E}{\partial w_{ji}} = \frac{\partial}{\partial w_{ji}} \left[ \frac{1}{2} \sum_{k=1}^M (x_k - w_{jk})^2 \right] $$

Como os termos da soma são independentes entre si e a derivada parcial está sendo tirada apenas em relação ao peso da conexão $i$ do neurônio $j$, todas as derivadas para $k \neq i$ são nulas. Portanto, a soma se reduz a um único termo:

$$ \frac{\partial E}{\partial w_{ji}} = \frac{1}{2} \frac{\partial}{\partial w_{ji}} (x_i - w_{ji})^2 $$

#### Passo 3: Aplicação da Regra da Cadeia
Aplicamos a regra da cadeia para derivar o termo $(x_i - w_{ji})^2$:

1. Derivada da função externa: $2(x_i - w_{ji})$
2. Derivada da função interna em relação a $w_{ji}$: $\frac{\partial}{\partial w_{ji}} (x_i - w_{ji}) = -1$

Multiplicando os termos:

$$ \frac{\partial E}{\partial w_{ji}} = \frac{1}{2} \cdot 2(x_i - w_{ji}) \cdot (-1) $$

$$ \frac{\partial E}{\partial w_{ji}} = -(x_i - w_{ji}) $$

#### Passo 4: Substituição na Regra de Atualização
Substituindo o resultado da derivada parcial obtido no Passo 3 na equação do gradiente descendente (Passo 1):

$$ \Delta w_{ji} = -\eta \cdot [ -(x_i - w_{ji}) ] $$

$$ \Delta w_{ji} = \eta (x_i - w_{ji}) $$

#### Passo 5: Equação de Atualização dos Pesos no Tempo $(t+1)$
A atualização do peso para o próximo instante de tempo discreto $t+1$ é dada por:

$$ w_{ji}(t+1) = w_{ji}(t) + \Delta w_{ji} $$

$$ w_{ji}(t+1) = w_{ji}(t) + \eta (x_i - w_{ji}(t)) $$

Em notação vetorial, para todo o vetor de pesos do neurônio vencedor $\mathbf{w}_j$:

$$ \mathbf{w}_j(t+1) = \mathbf{w}_j(t) + \eta (\mathbf{x} - \mathbf{w}_j(t)) $$

### 🏁 Conclusão
A regra de alteração de pesos utilizada pela rede de Kohonen baseia-se exatamente na equação obtida acima. Isso demonstra matematicamente que a regra de Hebb competitiva baseada na **Norma Euclidiana** nada mais é do que a aplicação direta do **gradiente descendente** com o intuito de **minimizar o erro de reconstrução quadrático** entre os pesos do neurônio vencedor e o padrão de entrada apresentado.

$$\blacksquare$$
