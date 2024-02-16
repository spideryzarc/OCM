---
marp: true
theme: default
paginate: true
---

# Tabu Search

---

## Conceito

- A Tabu Search é uma metaheurística que utiliza uma estratégia de lista tabu para evitar movimentos repetitivos durante a busca.

---

## Funcionamento

- Durante a busca, certos movimentos são marcados como tabu e não podem ser executados temporariamente.
- Essa restrição tabu é usada para evitar ciclos e para encorajar a exploração de diferentes regiões do espaço de busca.
- A lista tabu é atualizada dinamicamente, removendo movimentos mais antigos e adicionando novos movimentos à medida que a busca avança.

---

## Estratégia Tabu

- A estratégia tabu pode incluir diferentes critérios para determinar quais movimentos devem ser proibidos, como:
  - Proibir movimentos que levam de volta a soluções visitadas anteriormente.
  - Proibir movimentos que causem violações de restrições ou piora da solução.

