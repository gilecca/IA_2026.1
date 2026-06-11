# Trabalho — Rede ART-1: Classificação de Situações de Processo Industrial

**Disciplina:** Lab. Inteligência Artificial

**Professor:** Lázaro Eduardo da Silva

**Data:** 11/06/2026

---

## Descrição do Problema

O comportamento de um processo industrial é analisado considerando 16 variáveis de status binárias ao longo de 10 situações. A rede ART-1 é utilizada para classificar e agrupar as situações "parecidas", de modo a obter um provável diagnóstico para eventual manutenção.

## Dados de Entrada

| Situação | x1 | x2 | x3 | x4 | x5 | x6 | x7 | x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15 | x16 |
|----------|----|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|
| Situação 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| Situação 2 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| Situação 3 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| Situação 4 | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| Situação 5 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| Situação 6 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| Situação 7 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| Situação 8 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| Situação 9 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| Situação 10 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |

## Parâmetros da Rede ART-1

- **Dimensão de entrada (n):** 16
- **Parâmetro L:** 2.0
- **Inicialização dos pesos top-down (T):** todos 1s (ou igual à primeira entrada do cluster)
- **Inicialização dos pesos bottom-up (B):** L / (L - 1 + n)
- **Critério de vigilância:** |T_j ∩ I| / |I| ≥ ρ

---

## Parâmetro de Vigilância ρ = 0.5

**Número de classes ativas:** 4

### Agrupamentos

| Classe | Situações Agrupadas | Padrão Representativo (pesos top-down) |
|--------|---------------------|----------------------------------------|
| Classe 1 | Situação 1, Situação 2 | (ver detalhes abaixo) |
| Classe 2 | Situação 3, Situação 4, Situação 8, Situação 9 | (ver detalhes abaixo) |
| Classe 3 | Situação 5, Situação 7, Situação 10 | (ver detalhes abaixo) |
| Classe 4 | Situação 6 | (ver detalhes abaixo) |

### Detalhamento do Treinamento (passo a passo)

**Situação 1** — Entrada: `[0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]`

- **NOVO Cluster 1 criado**

**Situação 2** — Entrada: `[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]`

- Testou Classe 1: |T∩I|=5, |I|=10, match_ratio=0.5000, ρ=0.5 → **RESSONÂNCIA → atribuído ao Cluster 1**

**Situação 3** — Entrada: `[1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]`

- Testou Classe 1: |T∩I|=5, |I|=12, match_ratio=0.4167, ρ=0.5 → **RESET (match_ratio=0.4167 < rho=0.5)**
- **NOVO Cluster 2 criado**

**Situação 4** — Entrada: `[1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]`

- Testou Classe 1: |T∩I|=4, |I|=10, match_ratio=0.4000, ρ=0.5 → **RESET (match_ratio=0.4000 < rho=0.5)**
- Testou Classe 2: |T∩I|=7, |I|=10, match_ratio=0.7000, ρ=0.5 → **RESSONÂNCIA → atribuído ao Cluster 2**

**Situação 5** — Entrada: `[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]`

- Testou Classe 1: |T∩I|=3, |I|=9, match_ratio=0.3333, ρ=0.5 → **RESET (match_ratio=0.3333 < rho=0.5)**
- Testou Classe 2: |T∩I|=4, |I|=9, match_ratio=0.4444, ρ=0.5 → **RESET (match_ratio=0.4444 < rho=0.5)**
- **NOVO Cluster 3 criado**

**Situação 6** — Entrada: `[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]`

- Testou Classe 1: |T∩I|=4, |I|=11, match_ratio=0.3636, ρ=0.5 → **RESET (match_ratio=0.3636 < rho=0.5)**
- Testou Classe 2: |T∩I|=5, |I|=11, match_ratio=0.4545, ρ=0.5 → **RESET (match_ratio=0.4545 < rho=0.5)**
- Testou Classe 3: |T∩I|=4, |I|=11, match_ratio=0.3636, ρ=0.5 → **RESET (match_ratio=0.3636 < rho=0.5)**
- **NOVO Cluster 4 criado**

**Situação 7** — Entrada: `[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]`

- Testou Classe 1: |T∩I|=4, |I|=11, match_ratio=0.3636, ρ=0.5 → **RESET (match_ratio=0.3636 < rho=0.5)**
- Testou Classe 2: |T∩I|=5, |I|=11, match_ratio=0.4545, ρ=0.5 → **RESET (match_ratio=0.4545 < rho=0.5)**
- Testou Classe 3: |T∩I|=6, |I|=11, match_ratio=0.5455, ρ=0.5 → **RESSONÂNCIA → atribuído ao Cluster 3**

**Situação 8** — Entrada: `[1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]`

- Testou Classe 2: |T∩I|=7, |I|=12, match_ratio=0.5833, ρ=0.5 → **RESSONÂNCIA → atribuído ao Cluster 2**

**Situação 9** — Entrada: `[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]`

- Testou Classe 2: |T∩I|=6, |I|=9, match_ratio=0.6667, ρ=0.5 → **RESSONÂNCIA → atribuído ao Cluster 2**

**Situação 10** — Entrada: `[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]`

- Testou Classe 3: |T∩I|=6, |I|=9, match_ratio=0.6667, ρ=0.5 → **RESSONÂNCIA → atribuído ao Cluster 3**

---

## Parâmetro de Vigilância ρ = 0.8

**Número de classes ativas:** 5

### Agrupamentos

| Classe | Situações Agrupadas | Padrão Representativo (pesos top-down) |
|--------|---------------------|----------------------------------------|
| Classe 1 | Situação 1, Situação 6 | (ver detalhes abaixo) |
| Classe 2 | Situação 2, Situação 7 | (ver detalhes abaixo) |
| Classe 3 | Situação 3, Situação 8 | (ver detalhes abaixo) |
| Classe 4 | Situação 4, Situação 9 | (ver detalhes abaixo) |
| Classe 5 | Situação 5, Situação 10 | (ver detalhes abaixo) |

### Detalhamento do Treinamento (passo a passo)

**Situação 1** — Entrada: `[0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]`

- **NOVO Cluster 1 criado**

**Situação 2** — Entrada: `[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]`

- Testou Classe 1: |T∩I|=5, |I|=10, match_ratio=0.5000, ρ=0.8 → **RESET (match_ratio=0.5000 < rho=0.8)**
- **NOVO Cluster 2 criado**

**Situação 3** — Entrada: `[1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]`

- Testou Classe 1: |T∩I|=9, |I|=12, match_ratio=0.7500, ρ=0.8 → **RESET (match_ratio=0.7500 < rho=0.8)**
- Testou Classe 2: |T∩I|=8, |I|=12, match_ratio=0.6667, ρ=0.8 → **RESET (match_ratio=0.6667 < rho=0.8)**
- **NOVO Cluster 3 criado**

**Situação 4** — Entrada: `[1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]`

- Testou Classe 2: |T∩I|=7, |I|=10, match_ratio=0.7000, ρ=0.8 → **RESET (match_ratio=0.7000 < rho=0.8)**
- Testou Classe 1: |T∩I|=7, |I|=10, match_ratio=0.7000, ρ=0.8 → **RESET (match_ratio=0.7000 < rho=0.8)**
- Testou Classe 3: |T∩I|=7, |I|=10, match_ratio=0.7000, ρ=0.8 → **RESET (match_ratio=0.7000 < rho=0.8)**
- **NOVO Cluster 4 criado**

**Situação 5** — Entrada: `[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]`

- Testou Classe 2: |T∩I|=7, |I|=9, match_ratio=0.7778, ρ=0.8 → **RESET (match_ratio=0.7778 < rho=0.8)**
- Testou Classe 3: |T∩I|=7, |I|=9, match_ratio=0.7778, ρ=0.8 → **RESET (match_ratio=0.7778 < rho=0.8)**
- Testou Classe 4: |T∩I|=5, |I|=9, match_ratio=0.5556, ρ=0.8 → **RESET (match_ratio=0.5556 < rho=0.8)**
- Testou Classe 1: |T∩I|=5, |I|=9, match_ratio=0.5556, ρ=0.8 → **RESET (match_ratio=0.5556 < rho=0.8)**
- **NOVO Cluster 5 criado**

**Situação 6** — Entrada: `[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]`

- Testou Classe 1: |T∩I|=10, |I|=11, match_ratio=0.9091, ρ=0.8 → **RESSONÂNCIA → atribuído ao Cluster 1**

**Situação 7** — Entrada: `[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]`

- Testou Classe 2: |T∩I|=9, |I|=11, match_ratio=0.8182, ρ=0.8 → **RESSONÂNCIA → atribuído ao Cluster 2**

**Situação 8** — Entrada: `[1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]`

- Testou Classe 3: |T∩I|=12, |I|=12, match_ratio=1.0000, ρ=0.8 → **RESSONÂNCIA → atribuído ao Cluster 3**

**Situação 9** — Entrada: `[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]`

- Testou Classe 4: |T∩I|=8, |I|=9, match_ratio=0.8889, ρ=0.8 → **RESSONÂNCIA → atribuído ao Cluster 4**

**Situação 10** — Entrada: `[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]`

- Testou Classe 5: |T∩I|=9, |I|=9, match_ratio=1.0000, ρ=0.8 → **RESSONÂNCIA → atribuído ao Cluster 5**

---

## Parâmetro de Vigilância ρ = 0.9

**Número de classes ativas:** 7

### Agrupamentos

| Classe | Situações Agrupadas | Padrão Representativo (pesos top-down) |
|--------|---------------------|----------------------------------------|
| Classe 1 | Situação 1, Situação 6 | (ver detalhes abaixo) |
| Classe 2 | Situação 2 | (ver detalhes abaixo) |
| Classe 3 | Situação 3, Situação 8 | (ver detalhes abaixo) |
| Classe 4 | Situação 4 | (ver detalhes abaixo) |
| Classe 5 | Situação 5, Situação 10 | (ver detalhes abaixo) |
| Classe 6 | Situação 7 | (ver detalhes abaixo) |
| Classe 7 | Situação 9 | (ver detalhes abaixo) |

### Detalhamento do Treinamento (passo a passo)

**Situação 1** — Entrada: `[0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]`

- **NOVO Cluster 1 criado**

**Situação 2** — Entrada: `[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]`

- Testou Classe 1: |T∩I|=5, |I|=10, match_ratio=0.5000, ρ=0.9 → **RESET (match_ratio=0.5000 < rho=0.9)**
- **NOVO Cluster 2 criado**

**Situação 3** — Entrada: `[1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]`

- Testou Classe 1: |T∩I|=9, |I|=12, match_ratio=0.7500, ρ=0.9 → **RESET (match_ratio=0.7500 < rho=0.9)**
- Testou Classe 2: |T∩I|=8, |I|=12, match_ratio=0.6667, ρ=0.9 → **RESET (match_ratio=0.6667 < rho=0.9)**
- **NOVO Cluster 3 criado**

**Situação 4** — Entrada: `[1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]`

- Testou Classe 2: |T∩I|=7, |I|=10, match_ratio=0.7000, ρ=0.9 → **RESET (match_ratio=0.7000 < rho=0.9)**
- Testou Classe 1: |T∩I|=7, |I|=10, match_ratio=0.7000, ρ=0.9 → **RESET (match_ratio=0.7000 < rho=0.9)**
- Testou Classe 3: |T∩I|=7, |I|=10, match_ratio=0.7000, ρ=0.9 → **RESET (match_ratio=0.7000 < rho=0.9)**
- **NOVO Cluster 4 criado**

**Situação 5** — Entrada: `[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]`

- Testou Classe 2: |T∩I|=7, |I|=9, match_ratio=0.7778, ρ=0.9 → **RESET (match_ratio=0.7778 < rho=0.9)**
- Testou Classe 3: |T∩I|=7, |I|=9, match_ratio=0.7778, ρ=0.9 → **RESET (match_ratio=0.7778 < rho=0.9)**
- Testou Classe 4: |T∩I|=5, |I|=9, match_ratio=0.5556, ρ=0.9 → **RESET (match_ratio=0.5556 < rho=0.9)**
- Testou Classe 1: |T∩I|=5, |I|=9, match_ratio=0.5556, ρ=0.9 → **RESET (match_ratio=0.5556 < rho=0.9)**
- **NOVO Cluster 5 criado**

**Situação 6** — Entrada: `[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]`

- Testou Classe 1: |T∩I|=10, |I|=11, match_ratio=0.9091, ρ=0.9 → **RESSONÂNCIA → atribuído ao Cluster 1**

**Situação 7** — Entrada: `[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]`

- Testou Classe 2: |T∩I|=9, |I|=11, match_ratio=0.8182, ρ=0.9 → **RESET (match_ratio=0.8182 < rho=0.9)**
- Testou Classe 4: |T∩I|=7, |I|=11, match_ratio=0.6364, ρ=0.9 → **RESET (match_ratio=0.6364 < rho=0.9)**
- Testou Classe 3: |T∩I|=8, |I|=11, match_ratio=0.7273, ρ=0.9 → **RESET (match_ratio=0.7273 < rho=0.9)**
- Testou Classe 5: |T∩I|=6, |I|=11, match_ratio=0.5455, ρ=0.9 → **RESET (match_ratio=0.5455 < rho=0.9)**
- Testou Classe 1: |T∩I|=5, |I|=11, match_ratio=0.4545, ρ=0.9 → **RESET (match_ratio=0.4545 < rho=0.9)**
- **NOVO Cluster 6 criado**

**Situação 8** — Entrada: `[1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]`

- Testou Classe 3: |T∩I|=12, |I|=12, match_ratio=1.0000, ρ=0.9 → **RESSONÂNCIA → atribuído ao Cluster 3**

**Situação 9** — Entrada: `[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]`

- Testou Classe 4: |T∩I|=8, |I|=9, match_ratio=0.8889, ρ=0.9 → **RESET (match_ratio=0.8889 < rho=0.9)**
- Testou Classe 1: |T∩I|=7, |I|=9, match_ratio=0.7778, ρ=0.9 → **RESET (match_ratio=0.7778 < rho=0.9)**
- Testou Classe 3: |T∩I|=7, |I|=9, match_ratio=0.7778, ρ=0.9 → **RESET (match_ratio=0.7778 < rho=0.9)**
- Testou Classe 5: |T∩I|=5, |I|=9, match_ratio=0.5556, ρ=0.9 → **RESET (match_ratio=0.5556 < rho=0.9)**
- Testou Classe 2: |T∩I|=5, |I|=9, match_ratio=0.5556, ρ=0.9 → **RESET (match_ratio=0.5556 < rho=0.9)**
- Testou Classe 6: |T∩I|=5, |I|=9, match_ratio=0.5556, ρ=0.9 → **RESET (match_ratio=0.5556 < rho=0.9)**
- **NOVO Cluster 7 criado**

**Situação 10** — Entrada: `[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]`

- Testou Classe 5: |T∩I|=9, |I|=9, match_ratio=1.0000, ρ=0.9 → **RESSONÂNCIA → atribuído ao Cluster 5**

---

## Parâmetro de Vigilância ρ = 0.99

**Número de classes ativas:** 8

### Agrupamentos

| Classe | Situações Agrupadas | Padrão Representativo (pesos top-down) |
|--------|---------------------|----------------------------------------|
| Classe 1 | Situação 1 | (ver detalhes abaixo) |
| Classe 2 | Situação 2 | (ver detalhes abaixo) |
| Classe 3 | Situação 3, Situação 8 | (ver detalhes abaixo) |
| Classe 4 | Situação 4 | (ver detalhes abaixo) |
| Classe 5 | Situação 5, Situação 10 | (ver detalhes abaixo) |
| Classe 6 | Situação 6 | (ver detalhes abaixo) |
| Classe 7 | Situação 7 | (ver detalhes abaixo) |
| Classe 8 | Situação 9 | (ver detalhes abaixo) |

### Detalhamento do Treinamento (passo a passo)

**Situação 1** — Entrada: `[0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]`

- **NOVO Cluster 1 criado**

**Situação 2** — Entrada: `[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]`

- Testou Classe 1: |T∩I|=5, |I|=10, match_ratio=0.5000, ρ=0.99 → **RESET (match_ratio=0.5000 < rho=0.99)**
- **NOVO Cluster 2 criado**

**Situação 3** — Entrada: `[1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]`

- Testou Classe 1: |T∩I|=9, |I|=12, match_ratio=0.7500, ρ=0.99 → **RESET (match_ratio=0.7500 < rho=0.99)**
- Testou Classe 2: |T∩I|=8, |I|=12, match_ratio=0.6667, ρ=0.99 → **RESET (match_ratio=0.6667 < rho=0.99)**
- **NOVO Cluster 3 criado**

**Situação 4** — Entrada: `[1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]`

- Testou Classe 2: |T∩I|=7, |I|=10, match_ratio=0.7000, ρ=0.99 → **RESET (match_ratio=0.7000 < rho=0.99)**
- Testou Classe 1: |T∩I|=7, |I|=10, match_ratio=0.7000, ρ=0.99 → **RESET (match_ratio=0.7000 < rho=0.99)**
- Testou Classe 3: |T∩I|=7, |I|=10, match_ratio=0.7000, ρ=0.99 → **RESET (match_ratio=0.7000 < rho=0.99)**
- **NOVO Cluster 4 criado**

**Situação 5** — Entrada: `[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]`

- Testou Classe 2: |T∩I|=7, |I|=9, match_ratio=0.7778, ρ=0.99 → **RESET (match_ratio=0.7778 < rho=0.99)**
- Testou Classe 3: |T∩I|=7, |I|=9, match_ratio=0.7778, ρ=0.99 → **RESET (match_ratio=0.7778 < rho=0.99)**
- Testou Classe 4: |T∩I|=5, |I|=9, match_ratio=0.5556, ρ=0.99 → **RESET (match_ratio=0.5556 < rho=0.99)**
- Testou Classe 1: |T∩I|=5, |I|=9, match_ratio=0.5556, ρ=0.99 → **RESET (match_ratio=0.5556 < rho=0.99)**
- **NOVO Cluster 5 criado**

**Situação 6** — Entrada: `[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]`

- Testou Classe 1: |T∩I|=10, |I|=11, match_ratio=0.9091, ρ=0.99 → **RESET (match_ratio=0.9091 < rho=0.99)**
- Testou Classe 3: |T∩I|=9, |I|=11, match_ratio=0.8182, ρ=0.99 → **RESET (match_ratio=0.8182 < rho=0.99)**
- Testou Classe 4: |T∩I|=7, |I|=11, match_ratio=0.6364, ρ=0.99 → **RESET (match_ratio=0.6364 < rho=0.99)**
- Testou Classe 2: |T∩I|=5, |I|=11, match_ratio=0.4545, ρ=0.99 → **RESET (match_ratio=0.4545 < rho=0.99)**
- Testou Classe 5: |T∩I|=4, |I|=11, match_ratio=0.3636, ρ=0.99 → **RESET (match_ratio=0.3636 < rho=0.99)**
- **NOVO Cluster 6 criado**

**Situação 7** — Entrada: `[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]`

- Testou Classe 2: |T∩I|=9, |I|=11, match_ratio=0.8182, ρ=0.99 → **RESET (match_ratio=0.8182 < rho=0.99)**
- Testou Classe 4: |T∩I|=7, |I|=11, match_ratio=0.6364, ρ=0.99 → **RESET (match_ratio=0.6364 < rho=0.99)**
- Testou Classe 3: |T∩I|=8, |I|=11, match_ratio=0.7273, ρ=0.99 → **RESET (match_ratio=0.7273 < rho=0.99)**
- Testou Classe 5: |T∩I|=6, |I|=11, match_ratio=0.5455, ρ=0.99 → **RESET (match_ratio=0.5455 < rho=0.99)**
- Testou Classe 1: |T∩I|=6, |I|=11, match_ratio=0.5455, ρ=0.99 → **RESET (match_ratio=0.5455 < rho=0.99)**
- Testou Classe 6: |T∩I|=6, |I|=11, match_ratio=0.5455, ρ=0.99 → **RESET (match_ratio=0.5455 < rho=0.99)**
- **NOVO Cluster 7 criado**

**Situação 8** — Entrada: `[1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]`

- Testou Classe 3: |T∩I|=12, |I|=12, match_ratio=1.0000, ρ=0.99 → **RESSONÂNCIA → atribuído ao Cluster 3**

**Situação 9** — Entrada: `[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]`

- Testou Classe 4: |T∩I|=8, |I|=9, match_ratio=0.8889, ρ=0.99 → **RESET (match_ratio=0.8889 < rho=0.99)**
- Testou Classe 1: |T∩I|=8, |I|=9, match_ratio=0.8889, ρ=0.99 → **RESET (match_ratio=0.8889 < rho=0.99)**
- Testou Classe 6: |T∩I|=7, |I|=9, match_ratio=0.7778, ρ=0.99 → **RESET (match_ratio=0.7778 < rho=0.99)**
- Testou Classe 3: |T∩I|=7, |I|=9, match_ratio=0.7778, ρ=0.99 → **RESET (match_ratio=0.7778 < rho=0.99)**
- Testou Classe 5: |T∩I|=5, |I|=9, match_ratio=0.5556, ρ=0.99 → **RESET (match_ratio=0.5556 < rho=0.99)**
- Testou Classe 2: |T∩I|=5, |I|=9, match_ratio=0.5556, ρ=0.99 → **RESET (match_ratio=0.5556 < rho=0.99)**
- Testou Classe 7: |T∩I|=5, |I|=9, match_ratio=0.5556, ρ=0.99 → **RESET (match_ratio=0.5556 < rho=0.99)**
- **NOVO Cluster 8 criado**

**Situação 10** — Entrada: `[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]`

- Testou Classe 5: |T∩I|=9, |I|=9, match_ratio=1.0000, ρ=0.99 → **RESSONÂNCIA → atribuído ao Cluster 5**

---

## Análise Comparativa

| ρ (Vigilância) | Nº de Classes Ativas | Observação |
|----------------|----------------------|------------|
| 0.5 | 4 | Baixa vigilância — agrupamentos mais amplos (generalização) |
| 0.8 | 5 | Vigilância moderada — equilíbrio entre generalização e especificidade |
| 0.9 | 7 | Alta vigilância — classes mais específicas |
| 0.99 | 8 | Vigilância muito alta — quase cada padrão é uma classe separada |

### Conclusão

À medida que o parâmetro de vigilância **ρ** aumenta, a rede ART-1 se torna mais exigente quanto à similaridade necessária para que um padrão seja incorporado a um cluster existente. Isso resulta em:

- **ρ baixo (0.5):** Poucos clusters, agrupamentos amplos. A rede generaliza bastante, agrupando situações que têm pelo menos 50% de semelhança.
- **ρ moderado (0.8):** Mais clusters surgem, com agrupamentos mais coerentes.
- **ρ alto (0.9):** Clusters muito específicos, apenas situações muito similares ficam juntas.
- **ρ muito alto (0.99):** Quase cada situação forma seu próprio cluster, exceto pares praticamente idênticos (como Situações 3 e 8, e Situações 5 e 10).

Para fins de **diagnóstico de manutenção**, um valor intermediário de ρ (como 0.8) pode oferecer o melhor equilíbrio entre agrupar situações suficientemente parecidas sem perder informação relevante de diferenças.
