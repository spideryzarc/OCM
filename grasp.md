---
marp: true
theme: default
paginate: true
---

# Greedy Randomized Adaptive Search Procedure (GRASP)

---

## Conceito

- O Greedy Randomized Adaptive Search Procedure (GRASP) é uma metaheurística que combina a construção gulosa com a aleatoriedade para explorar o espaço de busca.

---

## Funcionamento

- O GRASP constrói soluções iterativamente, selecionando a cada iteração a melhor opção localmente com uma probabilidade controlada.
- A probabilidade de escolha é ajustada dinamicamente com base no histórico de escolhas anteriores.
- O processo é repetido várias vezes, resultando em múltiplas soluções que são refinadas por uma busca local.

---

## Estratégia Gulosa

- A estratégia gulosa do GRASP seleciona, em cada iteração, a melhor opção localmente dentre um conjunto de candidatos.
- Essa estratégia permite uma construção rápida de soluções viáveis, mas também introduz aleatoriedade para explorar diferentes soluções.

