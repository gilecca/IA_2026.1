---
title: Implementar Perceptron para classificação de óleo
date: 2026-04-30
priority: medium
---

# Tarefa

Implementar um script em Python que aplique um Perceptron com a Regra de Hebb (aprendizagem supervisionada) para classificar 30 amostras de óleo.

## Passos

1. Configurar o vetor de dados de entrada com 3 características ($x_1, x_2, x_3$).
2. Adicionar o "bias" (limiar de ativação).
3. Definir a função de ativação do tipo sinal (para as classes -1 e +1).
4. Implementar o laço de treinamento executando **5 treinamentos independentes**.
5. Para cada treinamento, inicializar os pesos com números pseudoaleatórios **entre 0 e 1**, garantindo que a semente de aleatoriedade mude a cada iteração (ou simplesmente não fixar semente).
6. Calcular as épocas até a convergência (erro zero) e exibir os pesos finais encontrados em cada uma das 5 execuções.
