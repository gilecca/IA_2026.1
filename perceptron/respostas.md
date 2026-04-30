# Resultados e Análise - Classificador Perceptron

## 2. Resultados dos 5 Treinamentos

Foram executados 5 treinamentos independentes para o Perceptron, inicializando os pesos com valores aleatórios entre 0 e 1, taxa de aprendizado $\eta = 0.01$ e considerando um bias ($x_0 = -1$). Como os dados são linearmente separáveis, todos atingiram convergência (erro zero) no conjunto de treinamento.

```text
--- Treinamento 1 ---
Pesos Iniciais : [0.9314, 0.0472, 0.3772, 0.2252]
Pesos Finais   : [-1.5186, 0.7743, 1.2312, -0.3631]
Épocas até convergência: 449

--- Treinamento 2 ---
Pesos Iniciais : [0.8008, 0.8573, 0.3074, 0.3108]
Pesos Finais   : [-1.5192, 0.7719, 1.2304, -0.3634]
Épocas até convergência: 438

--- Treinamento 3 ---
Pesos Iniciais : [0.5244, 0.0400, 0.6200, 0.7008]
Pesos Finais   : [-1.5456, 0.7802, 1.2387, -0.3684]
Épocas até convergência: 465

--- Treinamento 4 ---
Pesos Iniciais : [0.1164, 0.0991, 0.2539, 0.4910]
Pesos Finais   : [-1.5436, 0.7846, 1.2444, -0.3680]
Épocas até convergência: 407

--- Treinamento 5 ---
Pesos Iniciais : [0.0254, 0.1607, 0.1935, 0.0673]
Pesos Finais   : [-1.5146, 0.7721, 1.2221, -0.3616]
Épocas até convergência: 385
```

---

## 3. Tabela de Classificação de Novas Amostras

Após o treinamento do modelo, aplicamos as novas amostras na rede neural. A tabela abaixo exibe os resultados da classificação, onde é possível notar que, independentemente da variação dos pesos iniciais em cada um dos 5 treinamentos (T1 a T5), todos os modelos convergiram para resultados idênticos, provando a estabilidade da fronteira de decisão alcançada.

| Amostra | $x_1$ | $x_2$ | $x_3$ | $y(T1)$ | $y(T2)$ | $y(T3)$ | $y(T4)$ | $y(T5)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | -0.3565 | 0.0620 | 5.9891 | -1 | -1 | -1 | -1 | -1 |
| **2** | -0.7842 | 1.1267 | 5.5912 | 1 | 1 | 1 | 1 | 1 |
| **3** | 0.3012 | 0.5611 | 5.8234 | 1 | 1 | 1 | 1 | 1 |
| **4** | 0.7757 | 1.0648 | 8.0677 | 1 | 1 | 1 | 1 | 1 |
| **5** | 0.1570 | 0.8028 | 6.3040 | 1 | 1 | 1 | 1 | 1 |
| **6** | -0.7014 | 1.0316 | 3.6005 | 1 | 1 | 1 | 1 | 1 |
| **7** | 0.3748 | 0.1536 | 6.1537 | -1 | -1 | -1 | -1 | -1 |
| **8** | -0.6920 | 0.9404 | 4.4058 | 1 | 1 | 1 | 1 | 1 |
| **9** | -1.3970 | 0.7141 | 4.9263 | -1 | -1 | -1 | -1 | -1 |
| **10** | -1.8842 | -0.2805 | 1.2548 | -1 | -1 | -1 | -1 | -1 |

---

## 4. Explicação sobre a variação do número de épocas

**Por que o número de épocas varia a cada execução?**

O número de épocas varia exclusivamente porque **o vetor de pesos iniciais ($w$) é inicializado de forma aleatória** a cada novo treinamento. 

O espaço de soluções de um problema linearmente separável não é um ponto único, mas sim uma "região" de hiperplanos válidos que separam as classes corretamente. A Regra de Hebb com correção de erro dá passos em direção a essa região de solução. 

Se os pesos aleatórios iniciais caírem "perto" de um hiperplano válido no espaço multidimensional, a rede precisará de poucas correções e convergirá em poucas épocas. Por outro lado, se os pesos iniciais começarem "longe" ou na direção diametralmente oposta ao plano de separação, o algoritmo terá que realizar muitos ajustes e percorrerá mais iterações sobre o conjunto de dados (mais épocas) até chegar na zona de erro zero.

---

## 5. Principal limitação do Perceptron

A principal limitação de um Perceptron simples (de camada única) é que ele **só consegue resolver problemas que sejam perfeitamente linearmente separáveis**. 

Isso significa que o Perceptron só converge e funciona se for possível traçar uma reta (em 2D), um plano (em 3D) ou um hiperplano que separe perfeitamente todas as amostras de uma classe das amostras da outra classe, sem nenhuma sobreposição. Se os dados não puderem ser separados por uma forma linear simples, a rede ficará corrigindo seus pesos em um loop infinito sem nunca atingir o erro zero.

O exemplo clássico dessa falha é o **problema do XOR (Ou Exclusivo)**. Um único Perceptron é matematicamente incapaz de aprender a função XOR. Para solucionar problemas não-lineares, tornou-se necessário o uso de Redes Neurais com múltiplas camadas (Multilayer Perceptrons) associadas a funções de ativação não-lineares.
