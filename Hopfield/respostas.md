# Resultados da Rede de Hopfield

A rede de Hopfield foi implementada com 45 neurônios. A matriz de pesos foi obtida pela regra do produto externo, com a diagonal zerada. As 4 imagens utilizadas (C, E, F, T) foram extraídas e convertidas em vetores de 45 posições (9 linhas x 5 colunas).

## Simulações de Transmissão com 20% de ruído

O ruído de 20% corrompe aproximadamente 9 pixels de cada imagem (45 * 0.2 = 9).

### Padrão C

**Simulação 1:**
```text
Imagem Transmitida    Imagem Distorcida     Imagem Limpa
(livre de ruído)      (com ruído)           (sem ruído)
░░███                 ░████                 ░████
░████                 ░░░░█                 ░████
██░░░                 ██░░█                 ░█░░░
██░░░                 ░██░░                 ░██░░
██░░░                 ██░░░                 ░███░
██░░░                 ██░░░                 ██░░░
██░░░                 ██░░█                 ██░░░
█████                 █████                 █████
░███░                 ████░                 ████░
```

**Simulação 2:**
```text
Imagem Transmitida    Imagem Distorcida     Imagem Limpa
(livre de ruído)      (com ruído)           (sem ruído)
░░███                 ░█░██                 ░████
░████                 █████                 ░████
██░░░                 ██░██                 ░█░░░
██░░░                 ██░░░                 ░██░░
██░░░                 ██░░░                 ░███░
██░░░                 ███░░                 ██░░░
██░░░                 ███░░                 ██░░░
█████                 ██░██                 █████
░███░                 ░████                 ████░
```

**Simulação 3:**
```text
Imagem Transmitida    Imagem Distorcida     Imagem Limpa
(livre de ruído)      (com ruído)           (sem ruído)
░░███                 ░████                 ░████
░████                 ░████                 ░████
██░░░                 ███░█                 ░█░░░
██░░░                 ███░░                 ░██░░
██░░░                 ██░░░                 ░███░
██░░░                 █░░█░                 ██░░░
██░░░                 ██░░█                 ██░░░
█████                 ░██░█                 █████
░███░                 ░███░                 ████░
```

### Padrão E

**Simulação 1:**
```text
Imagem Transmitida    Imagem Distorcida     Imagem Limpa
(livre de ruído)      (com ruído)           (sem ruído)
░████                 █████                 ░████
░████                 ░░██░                 ░████
░█░░░                 ██░█░                 ░█░░░
░███░                 ░█░█░                 ░██░░
░███░                 ░░██░                 ░███░
██░░░                 ███░░                 ██░░░
██░░░                 ██░░░                 ██░░░
█████                 █████                 █████
█████                 █░███                 ████░
```

**Simulação 2:**
```text
Imagem Transmitida    Imagem Distorcida     Imagem Limpa
(livre de ruído)      (com ruído)           (sem ruído)
░████                 ░█░██                 ░████
░████                 █░███                 ░████
░█░░░                 ░█░░░                 ░█░░░
░███░                 ░░██░                 ░██░░
░███░                 ░█░██                 ░███░
██░░░                 ██░░█                 ██░░░
██░░░                 ██░░░                 ██░░░
█████                 ███░█                 █████
█████                 █░███                 ████░
```

**Simulação 3:**
```text
Imagem Transmitida    Imagem Distorcida     Imagem Limpa
(livre de ruído)      (com ruído)           (sem ruído)
░████                 ░████                 ░████
░████                 ██░██                 ░████
░█░░░                 ░██░░                 ░█░░░
░███░                 ░██░░                 ░██░░
░███░                 ░██░█                 ░███░
██░░░                 ██░░░                 ██░░░
██░░░                 ██░░█                 ██░░░
█████                 ░░███                 █████
█████                 █████                 ████░
```

### Padrão F

**Simulação 1:**
```text
Imagem Transmitida    Imagem Distorcida     Imagem Limpa
(livre de ruído)      (com ruído)           (sem ruído)
░████                 ░█░█░                 ░████
░████                 █░███                 ░████
░█░░░                 ░█░░█                 ░█░░░
░██░░                 ░███░                 ░██░░
░███░                 ░████                 ░███░
██░░░                 ██░░░                 ██░░░
██░░░                 ██░░░                 ██░░░
██░░░                 ██░░░                 █████
██░░░                 █░█░░                 ████░
```

**Simulação 2:**
```text
Imagem Transmitida    Imagem Distorcida     Imagem Limpa
(livre de ruído)      (com ruído)           (sem ruído)
░████                 ░████                 ░████
░████                 ░████                 ░████
░█░░░                 ██░█░                 ░█░░░
░██░░                 ░█░░░                 ░██░░
░███░                 ░█░░░                 ░███░
██░░░                 ██░░█                 ██░░░
██░░░                 ██░░░                 ██░░░
██░░░                 ██░░░                 ███░░
██░░░                 ░░░░█                 ██░░░
```

**Simulação 3:**
```text
Imagem Transmitida    Imagem Distorcida     Imagem Limpa
(livre de ruído)      (com ruído)           (sem ruído)
░████                 ░███░                 ░████
░████                 ░████                 ░████
░█░░░                 ██░░░                 ░█░░░
░██░░                 ███░░                 ░██░░
░███░                 ░███░                 ░███░
██░░░                 ██░░░                 ██░░░
██░░░                 ██░░█                 ██░░░
██░░░                 ██░██                 █████
██░░░                 █░░██                 ████░
```

### Padrão T

**Simulação 1:**
```text
Imagem Transmitida    Imagem Distorcida     Imagem Limpa
(livre de ruído)      (com ruído)           (sem ruído)
█████                 ███░█                 ░████
████░                 ████░                 ░████
░██░░                 ░██░█                 ░█░░░
░██░░                 ░█░░░                 ░██░░
░██░█                 ███░░                 ░███░
░██░█                 ░██░░                 ██░░░
░██░░                 ░██░░                 ██░░░
░██░░                 ███░█                 █████
░█░░░                 ██░░░                 ████░
```

**Simulação 2:**
```text
Imagem Transmitida    Imagem Distorcida     Imagem Limpa
(livre de ruído)      (com ruído)           (sem ruído)
█████                 ██░█░                 █████
████░                 ████░                 ████░
░██░░                 ░███░                 ░██░░
░██░░                 ░█░░░                 ░██░░
░██░█                 ███░█                 ░██░█
░██░█                 ███░█                 ░██░█
░██░░                 ░██░░                 ░██░░
░██░░                 ░░█░█                 ░██░░
░█░░░                 ░█░█░                 ░█░░░
```

**Simulação 3:**
```text
Imagem Transmitida    Imagem Distorcida     Imagem Limpa
(livre de ruído)      (com ruído)           (sem ruído)
█████                 █████                 █████
████░                 ███░█                 ████░
░██░░                 ░██░░                 ░██░░
░██░░                 █░█░░                 ░██░░
░██░█                 ░██░█                 ░██░█
░██░█                 ███░░                 ░██░█
░██░░                 ░░█░░                 ░██░░
░██░░                 ░██░░                 ░██░░
░█░░░                 ░░░░█                 ░█░░░
```

## O que acontece quando aumentamos excessivamente o nível de ruído?

Quando o nível de ruído é aumentado excessivamente (por exemplo, acima de 40-50%), a rede de Hopfield perde a sua capacidade de recuperar os padrões originais com precisão. Isso ocorre por alguns motivos principais:
1. **Afastamento da Bacia de Atração:** Cada padrão armazenado possui uma 'bacia de atração' no espaço de estados da rede. Se o ruído for muito alto, a imagem distorcida pode 'cair' na bacia de atração de outro padrão armazenado (ou de um estado espúrio), fazendo com que a rede convirja para uma imagem completamente diferente da que foi transmitida originalmente.
2. **Estados Espúrios:** A regra de aprendizado de Hebb (produto externo) cria não apenas os atratores desejados (os padrões), mas também misturas deles e estados invertidos (estados espúrios). Com muito ruído, a rede tem uma alta probabilidade de estabilizar em um desses estados indesejados, resultando em uma 'mistura' dos padrões ou em uma imagem que não faz sentido.
3. **Limite de Capacidade:** A capacidade teórica de uma rede de Hopfield com N neurônios é de aproximadamente 0.138N padrões. Com 45 neurônios, o limite é de cerca de 6 padrões. Embora estejamos usando apenas 4 padrões (dentro do limite), o aumento de ruído exige bacias de atração maiores e mais robustas, o que se torna difícil de garantir quando o número de padrões se aproxima da capacidade da rede.
