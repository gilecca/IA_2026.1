# Trabalho: Rede LVQ-1 — Classificação de Perfis de Potência Elétrica

**Centro Federal de Educação Tecnológica de Minas Gerais — Campus VIII – Varginha**  
**Bacharelado em Sistemas de Informação**  
**Disciplina:** Lab. Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 11/06/2026

---

## 1. Descrição do Problema

A previsão de potência elétrica é essencial para o planejamento operacional de sistemas de energia. O escalonamento da manutenção, o planejamento de expansão e a análise de despacho dependem da potência prevista.

A previsão para um dia específico pode ser realizada a partir da potência medida nas primeiras horas do dia. Através da **classificação de curvas**, é possível traçar o perfil da potência necessária para as demais horas.

**Objetivo:** Implementar e treinar uma rede **LVQ-1** que detecte similaridades e regularidades entre vetores de cada classe, classificando perfis de potência futuros.

---

## 2. Dados de Treinamento

16 amostras agrupadas em 4 classes (perfis de demanda), com 6 atributos cada (potência medida de 7h a 12h):

| Amostra | 7 horas | 8 horas | 9 horas | 10 horas | 11 horas | 12 horas | Classe |
|---------|---------|---------|---------|----------|----------|----------|--------|
| 1       | 2.3976  | 1.5328  | 1.9044  | 1.1937   | 2.4184   | 1.8649   | 1      |
| 2       | 2.3936  | 1.4804  | 1.9907  | 1.2732   | 2.2719   | 1.8110   | 1      |
| 3       | 2.2880  | 1.4585  | 1.9867  | 1.2451   | 2.3389   | 1.8099   | 1      |
| 4       | 2.2904  | 1.4766  | 1.8876  | 1.2706   | 2.2966   | 1.7744   | 1      |
| 5       | 1.1201  | 0.0587  | 1.3154  | 5.3783   | 3.1849   | 2.4276   | 2      |
| 6       | 0.9913  | 0.1524  | 1.2700  | 5.3808   | 3.0714   | 2.3331   | 2      |
| 7       | 1.0915  | 0.1881  | 1.1387  | 5.3701   | 3.2561   | 2.3383   | 2      |
| 8       | 1.0535  | 0.1229  | 1.2743  | 5.3226   | 3.0950   | 2.3193   | 2      |
| 9       | 1.4871  | 2.3448  | 0.9918  | 2.3160   | 1.6783   | 5.0850   | 3      |
| 10      | 1.3312  | 2.2553  | 0.9618  | 2.4702   | 1.7272   | 5.0645   | 3      |
| 11      | 1.3646  | 2.2945  | 1.0562  | 2.4763   | 1.8051   | 5.1470   | 3      |
| 12      | 1.4392  | 2.2296  | 1.1278  | 2.4230   | 1.7259   | 5.0876   | 3      |
| 13      | 2.9364  | 1.5233  | 4.6109  | 1.3160   | 4.2700   | 6.8749   | 4      |
| 14      | 2.9034  | 1.4640  | 4.6061  | 1.4598   | 4.2912   | 6.9142   | 4      |
| 15      | 3.0181  | 1.4918  | 4.7051  | 1.3521   | 4.2623   | 6.7966   | 4      |
| 16      | 2.9374  | 1.4896  | 4.7219  | 1.3977   | 4.1863   | 6.8336   | 4      |

---

## 3. Arquitetura da Rede LVQ-1

### 3.1 Parâmetros

| Parâmetro               | Valor  |
|--------------------------|--------|
| Taxa de aprendizagem (α) | 0.05   |
| Número de épocas          | 1000   |
| Número de atributos       | 6      |
| Número de classes         | 4      |
| Número de protótipos      | 4 (1 por classe) |
| Inicialização dos protótipos | Média das amostras de cada classe |

### 3.2 Algoritmo LVQ-1

1. **Inicialização:** Os protótipos são inicializados como a **média** das amostras pertencentes a cada classe.
2. **Para cada época:**
   - Embaralham-se as amostras de treinamento.
   - Para cada amostra **x** com rótulo **c**:
     - Calcula-se a **distância Euclidiana** de **x** a todos os protótipos.
     - O protótipo mais próximo (**vencedor**) **w_j** é identificado.
     - **Se** o rótulo do protótipo vencedor == **c** (classificação correta):  
       `w_j = w_j + α × (x − w_j)` → **aproxima** o protótipo da amostra.
     - **Se** o rótulo do protótipo vencedor ≠ **c** (classificação incorreta):  
       `w_j = w_j − α × (x − w_j)` → **afasta** o protótipo da amostra.

---

## 4. Treinamento

### 4.1 Protótipos Iniciais (Média por Classe)

| Protótipo | Classe | 7 horas | 8 horas | 9 horas | 10 horas | 11 horas | 12 horas |
|-----------|--------|---------|---------|---------|----------|----------|----------|
| 1         | 1      | 2.3424  | 1.4871  | 1.9424  | 1.2456   | 2.3315   | 1.8150   |
| 2         | 2      | 1.0641  | 0.1305  | 1.2496  | 5.3630   | 3.1518   | 2.3546   |
| 3         | 3      | 1.4055  | 2.2810  | 1.0344  | 2.4214   | 1.7341   | 5.0960   |
| 4         | 4      | 2.9488  | 1.4922  | 4.6610  | 1.3814   | 4.2524   | 6.8548   |

### 4.2 Protótipos Finais (Após 1000 Épocas)

| Protótipo | Classe | 7 horas | 8 horas | 9 horas | 10 horas | 11 horas | 12 horas |
|-----------|--------|---------|---------|---------|----------|----------|----------|
| 1         | 1      | 2.3436  | 1.4877  | 1.9422  | 1.2450   | 2.3324   | 1.8158   |
| 2         | 2      | 1.0653  | 0.1296  | 1.2501  | 5.3635   | 3.1533   | 2.3560   |
| 3         | 3      | 1.4051  | 2.2805  | 1.0346  | 2.4219   | 1.7342   | 5.0959   |
| 4         | 4      | 2.9476  | 1.4919  | 4.6600  | 1.3825   | 4.2527   | 6.8560   |

### 4.3 Convergência

A rede convergiu **imediatamente na época 1** com **0 erros** (100% de acurácia), mantendo esse desempenho durante todas as 1000 épocas. Isso se deve ao fato de que as 4 classes possuem perfis bastante distintos entre si, tornando a separação natural com protótipos inicializados pelas médias.

| Época | Erros | Acurácia |
|-------|-------|----------|
| 1     | 0     | 100.0%   |
| 10    | 0     | 100.0%   |
| 50    | 0     | 100.0%   |
| 100   | 0     | 100.0%   |
| 200   | 0     | 100.0%   |
| 500   | 0     | 100.0%   |
| 1000  | 0     | 100.0%   |

---

## 5. Verificação no Conjunto de Treinamento

Todas as 16 amostras de treinamento foram classificadas corretamente:

| Amostra | Classe Real | Classe Predita | Distância ao Protótipo Vencedor | Status    |
|---------|-------------|----------------|---------------------------------|-----------|
| 1       | 1           | 1              | 0.1372                          | ✓ Correto |
| 2       | 1           | 1              | 0.0969                          | ✓ Correto |
| 3       | 1           | 1              | 0.0774                          | ✓ Correto |
| 4       | 1           | 1              | 0.0979                          | ✓ Correto |
| 5       | 2           | 2              | 0.1365                          | ✓ Correto |
| 6       | 2           | 2              | 0.1180                          | ✓ Correto |
| 7       | 2           | 2              | 0.1657                          | ✓ Correto |
| 8       | 2           | 2              | 0.0848                          | ✓ Correto |
| 9       | 3           | 3              | 0.1647                          | ✓ Correto |
| 10      | 3           | 3              | 0.1215                          | ✓ Correto |
| 11      | 3           | 3              | 0.1136                          | ✓ Correto |
| 12      | 3           | 3              | 0.1122                          | ✓ Correto |
| 13      | 4           | 4              | 0.0928                          | ✓ Correto |
| 14      | 4           | 4              | 0.1284                          | ✓ Correto |
| 15      | 4           | 4              | 0.1074                          | ✓ Correto |
| 16      | 4           | 4              | 0.0953                          | ✓ Correto |

**Acurácia no treinamento: 16/16 (100%)**

---

## 6. Classificação dos Dados de Teste

### 6.1 Dados de Teste

| Dia | 7 horas | 8 horas | 9 horas | 10 horas | 11 horas | 12 horas |
|-----|---------|---------|---------|----------|----------|----------|
| 1   | 2.9817  | 1.5656  | 4.8391  | 1.4311   | 4.1916   | 6.9718   |
| 2   | 1.5537  | 2.2615  | 1.3169  | 2.5873   | 1.7570   | 5.0958   |
| 3   | 1.2240  | 0.2445  | 1.3595  | 5.4192   | 3.2027   | 2.5675   |
| 4   | 2.5828  | 1.5146  | 2.1119  | 1.2859   | 2.3414   | 1.8695   |
| 5   | 2.4168  | 1.4857  | 1.8959  | 1.3013   | 2.4500   | 1.7868   |
| 6   | 1.0604  | 0.2276  | 1.2806  | 5.4732   | 3.2133   | 2.4839   |
| 7   | 1.5246  | 2.4254  | 1.1353  | 2.5325   | 1.7569   | 5.2640   |
| 8   | 3.0565  | 1.6259  | 4.7743  | 1.3654   | 4.2904   | 6.9808   |

### 6.2 Resultados da Classificação

| Dia | Classe Predita | Distância ao Protótipo Vencedor |
|-----|----------------|--------------------------------|
| 1   | **4**          | 0.2412                         |
| 2   | **3**          | 0.3606                         |
| 3   | **2**          | 0.3172                         |
| 4   | **1**          | 0.3023                         |
| 5   | **1**          | 0.1592                         |
| 6   | **2**          | 0.2063                         |
| 7   | **3**          | 0.2940                         |
| 8   | **4**          | 0.2453                         |

### 6.3 Distâncias Detalhadas para Cada Protótipo

A tabela abaixo mostra a distância Euclidiana de cada amostra de teste a todos os 4 protótipos. O menor valor (protótipo vencedor) determina a classe.

| Dia | d(Protótipo 1 / Classe 1) | d(Protótipo 2 / Classe 2) | d(Protótipo 3 / Classe 3) | d(Protótipo 4 / Classe 4) | Classe Predita |
|-----|---------------------------|---------------------------|---------------------------|---------------------------|----------------|
| 1   | 6.2354                    | 7.5142                    | 5.2925                    | **0.2412**                | **4**          |
| 2   | 3.8085                    | 4.6852                    | **0.3606**                | 4.9488                    | **3**          |
| 3   | 4.6781                    | **0.3172**                | 4.6709                    | 7.1561                    | **2**          |
| 4   | **0.3023**                | 4.7420                    | 3.8991                    | 5.9291                    | **1**          |
| 5   | **0.1592**                | 4.6263                    | 3.8878                    | 6.0725                    | **1**          |
| 6   | 4.7720                    | **0.2063**                | 4.7660                    | 7.3151                    | **2**          |
| 7   | 4.0100                    | 4.8904                    | **0.2940**                | 5.0405                    | **3**          |
| 8   | 6.2508                    | 7.5690                    | 5.3240                    | **0.2453**                | **4**          |

---

## 7. Análise dos Resultados

### 7.1 Perfis de Demanda Identificados

- **Classe 1** — Perfil com potência moderada e estável ao longo do dia (valores entre ~1.2 e ~2.5).
- **Classe 2** — Perfil com potência muito baixa pela manhã (7h–9h) e pico às 10h (~5.3).
- **Classe 3** — Perfil com pico na 8h (~2.3) e forte elevação ao meio-dia (~5.1).
- **Classe 4** — Perfil de alta demanda com valores elevados às 9h (~4.7), 11h (~4.3) e 12h (~6.9).

### 7.2 Classificação dos Dias de Teste

| Dia | Classe | Interpretação |
|-----|--------|---------------|
| 1   | 4      | Dia de alta demanda com picos às 9h, 11h e 12h |
| 2   | 3      | Dia com pico na 8h e forte elevação ao meio-dia |
| 3   | 2      | Dia com baixa demanda matinal e pico às 10h |
| 4   | 1      | Dia de demanda moderada e estável |
| 5   | 1      | Dia de demanda moderada e estável |
| 6   | 2      | Dia com baixa demanda matinal e pico às 10h |
| 7   | 3      | Dia com pico na 8h e forte elevação ao meio-dia |
| 8   | 4      | Dia de alta demanda com picos às 9h, 11h e 12h |

### 7.3 Desempenho da Rede

- **Acurácia no treinamento:** 100% (16/16 amostras classificadas corretamente)
- **Convergência:** Imediata (época 1), indicando que as classes são bem separadas no espaço de atributos
- **Confiança na classificação:** As distâncias dos dados de teste aos protótipos vencedores são todas pequenas (0.15 a 0.36), enquanto as distâncias aos protótipos de classes erradas são muito maiores (3.8 a 7.6), demonstrando alta confiança na separação

---

## 8. Código-Fonte

O código-fonte completo da implementação está disponível no arquivo `lvq1.py` na pasta `RNA_LVQ`.

### Estrutura do código:
- **Classe `LVQ1`** — Implementação completa da rede LVQ-1 com métodos:
  - `treinar()` — Treina a rede com os dados fornecidos
  - `classificar()` — Classifica novas amostras
  - `classificar_detalhado()` — Classifica e retorna distâncias a todos os protótipos
- **Dados de treinamento** — 16 amostras × 6 atributos, organizados em 4 classes
- **Dados de teste** — 8 amostras para classificação

### Execução:
```bash
python -X utf8 lvq1.py
```

---

## 9. Conclusão

A rede LVQ-1 implementada com taxa de aprendizagem α = 0.05 e 1000 épocas de treinamento demonstrou excelente capacidade de classificação dos perfis de potência elétrica. A convergência imediata e a acurácia de 100% no treinamento refletem a natureza bem separada das 4 classes de demanda.

Os 8 dias de teste foram classificados com alta confiança, sendo distribuídos como:
- **2 dias** classificados como **Classe 1** (demanda moderada/estável)
- **2 dias** classificados como **Classe 2** (baixa demanda matinal, pico às 10h)
- **2 dias** classificados como **Classe 3** (pico às 8h, elevação ao meio-dia)
- **2 dias** classificados como **Classe 4** (alta demanda geral)

A LVQ-1 provou ser uma ferramenta eficiente para este problema de classificação de padrões de consumo de potência elétrica, permitindo o planejamento operacional adequado do sistema de energia.
