# Requisitos do Projeto

## REQ-001: Arquitetura Perceptron (Destilação de Óleo)
- **Topologia:** Perceptron de camada única.
- **Função de Aprendizagem:** Regra de Hebb supervisionada ($w = w + \eta \cdot d \cdot x$).
- **Taxa de Aprendizagem ($\eta$):** 0.01.
- **Função de Ativação:** Função Sinal ($y = 1$ se $v \ge 0$, senão $y = -1$).
- **Inicialização de Pesos:** Valores aleatórios contínuos entre 0 e 1 ($0 \le w \le 1$).

## REQ-002: Requisitos de Execução
- **Treinamentos Executados:** 5 (cinco).
- **Independência:** Cada treinamento deve usar pesos iniciais únicos e não viciados (diferentes sementes).
- **Convergência:** O laço de treinamento (épocas) deve continuar até que o erro de classificação no conjunto seja nulo.
