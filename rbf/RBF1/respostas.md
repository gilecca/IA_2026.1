# Respostas - Trabalho RBF1

## Treinamento da Camada Escondida (K-Means)
Apenas padrões com presença de radiação ($d = 1$) foram considerados.

| Cluster | Centro ($x_1$, $x_2$) | Variância ($\sigma^2$) |
|---------|--------|-----------|
| 1 | (0.1648, 0.6121) | 0.0298 |
| 2 | (0.3990, 0.1571) | 0.0385 |

## Treinamento da Camada de Saída (Regra Delta Generalizada)
- Taxa de aprendizado $\eta = 0.01$
- Precisão $\epsilon = 10^{-7}$
- Épocas de convergência: 340

| Peso | Valor Final |
|------|-------|
| $W_{21,0}$ (Bias) | -1.002659 |
| $W_{21,1}$ | 2.378065 |
| $W_{21,2}$ | 2.697717 |

## Validação e Pós-processamento

| Amostra | $x_1$ | $x_2$ | $d$ | $y$ (Real) | $y_{pós}$ (Sinal) |
|---------|-------|-------|-----|-----|---------|
| 1 | 0.8705 | 0.9329 | -1 | -1.0025 | -1 |
| 2 | 0.0388 | 0.2703 | 1 | -0.3231 | -1 |
| 3 | 0.8236 | 0.4458 | -1 | -0.9140 | -1 |
| 4 | 0.7075 | 0.1502 | 1 | -0.2201 | -1 |
| 5 | 0.9587 | 0.8663 | -1 | -1.0026 | -1 |
| 6 | 0.6115 | 0.9365 | -1 | -0.9878 | -1 |
| 7 | 0.3534 | 0.3646 | 1 | 0.9665 | 1 |
| 8 | 0.3268 | 0.2766 | 1 | 1.3232 | 1 |
| 9 | 0.6129 | 0.4518 | -1 | -0.4682 | -1 |
| 10 | 0.9948 | 0.4962 | -1 | -0.9967 | -1 |

**Taxa de Acerto (%):** 80.00%

## Estratégias para Aumentar a Taxa de Acerto
Caso a taxa de acerto não seja ideal, as seguintes estratégias podem ser adotadas para aprimorar a RBF:

1. **Aumentar o número de neurônios na camada oculta:** Utilizar um valor maior para $k$ no K-means (por exemplo, 3 ou mais) pode capturar melhor as regiões de distribuição dos dados da classe de radiação. O problema de classificação pode não ser separável com apenas duas gaussianas centradas na classe de radiação.
2. **Incluir os padrões sem radiação no K-means:** Realizar o agrupamento considerando ambas as classes e gerar gaussianas para todos os sub-padrões, o que ajudaria a mapear a fronteira de decisão de forma muito mais precisa ao invés de apenas focar em padrões com radiação.
3. **Ajuste heurístico ou supervisionado da Variância ($\sigma^2$):** Em vez da variância amostral, testar heurísticas como a distância máxima entre os centros dos clusters ($d_{max} / \sqrt{2k}$), ou até usar o gradiente (regra delta generalizada) para otimizar os próprios centros e variâncias (além dos pesos) reduzindo o erro médio quadrático.
4. **Aumentar a base de dados de treinamento:** Um número maior de amostras representativas evitaria o overfitting aos dados de treinamento e melhoraria a generalização do modelo no conjunto de teste.
