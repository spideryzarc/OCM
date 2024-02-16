---
marp: true
theme: default
paginate: true
---

# Variable Neighborhood Search (VNS)

---

## Conceito

- O Variable Neighborhood Search (VNS) é uma metaheurística que combina a busca local com a mudança de estruturas de vizinhança durante a busca.

---

## Funcionamento

- O VNS começa com uma solução inicial e uma estrutura de vizinhança.
- Ele aplica a busca local na estrutura de vizinhança atual até que não haja mais melhorias.
- Em seguida, muda para uma estrutura de vizinhança diferente e continua a busca.
- O processo de mudança de estrutura de vizinhança é repetido até que um critério de parada seja atingido.

---

## Estratégia de Vizinhança

- As estruturas de vizinhança no VNS podem variar em complexidade e cobertura do espaço de busca.
- Estratégias comuns incluem:
  - Vizinhança de troca: troca de posições entre elementos da solução.
  - Vizinhança de inserção: insere ou remove elementos da solução.
  - Vizinhança de reversão: inverte a ordem de elementos em uma parte da solução.

---

# Variable Neighborhood Descent (VND) vs. Variable Neighborhood Search (VNS)

---

## Diferenciação

- **Variable Neighborhood Descent (VND):**
  - Combina busca local com diferentes estruturas de vizinhança em iterações alternadas.
  - Explora diferentes vizinhanças para buscar melhorias na solução atual.

- **Variable Neighborhood Search (VNS):**
  - Começa com uma solução inicial e uma estrutura de vizinhança.
  - Aplica busca local até que não haja mais melhorias e muda para uma estrutura de vizinhança diferente.
  - Repete o processo de mudança de estrutura de vizinhança até atingir um critério de parada.

---

## Estratégias de Vizinhança

- Ambos VND e VNS podem usar uma variedade de estruturas de vizinhança, como:
  - Vizinhança de troca.
  - Vizinhança de inserção.
  - Vizinhança de reversão.

