---
marp: true
theme: default
paginate: true
---

# Simulated Annealing

---

## Conceito

- Simulated Annealing é uma metaheurística inspirada no processo físico de recozimento simulado de materiais.

---

## Funcionamento

- O algoritmo inicia com uma solução inicial e uma temperatura inicial alta.
- Durante a busca, movimentos são feitos aleatoriamente e aceitos com base em uma função de aceitação, que considera a diferença de custo e a temperatura.
- À medida que a busca avança, a temperatura é reduzida gradualmente, o que permite que o algoritmo escape de mínimos locais.

---

## Estratégia de Resfriamento

- A temperatura é reduzida de acordo com uma estratégia de resfriamento, como:
  - Decaimento exponencial.
  - Decaimento linear.
  - Decaimento por recálculo adaptativo.

