---
marp: true
theme: default
paginate: true
---

# Variable Neighborhood Descent (VND)

---

## Conceito

- O Variable Neighborhood Descent (VND) é uma metaheurística que combina a busca local com diferentes estruturas de vizinhança.

---

## Funcionamento

- O VND realiza iterações alternadas entre diferentes estruturas de vizinhança.
- Em cada iteração, ele aplica uma busca local em uma estrutura de vizinhança específica.
- Se uma melhoria é encontrada, a busca continua na mesma estrutura de vizinhança. Caso contrário, ele muda para uma estrutura de vizinhança diferente.

---

## Estratégias de Vizinhança

- As estruturas de vizinhança podem variar em complexidade, cobrindo diferentes regiões do espaço de busca.
- Estratégias comuns incluem:
  - Vizinhança de troca: troca de posições entre elementos da solução.
  - Vizinhança de inserção: insere ou remove elementos da solução.
  - Vizinhança de reversão: inverte a ordem de elementos em uma parte da solução.

