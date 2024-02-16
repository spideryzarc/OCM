---
marp: true
theme: default
paginate: true
---

# Método Random Multistart

---

## Conceito

- Consiste em iniciar múltiplas execuções de uma metaheurística a partir de soluções iniciais aleatórias.
- Cada execução é independente uma da outra.

---

## Funcionamento

- Gera várias soluções iniciais aleatórias.
- Aplica uma metaheurística, como a busca local, a partir de cada solução inicial.
- Retorna a melhor solução encontrada entre todas as execuções.

---

## Vantagens

- Aumenta a chance de encontrar uma solução de boa qualidade.
- Ajuda a evitar mínimos locais ao explorar diferentes regiões do espaço de busca.

---

## Desvantagens

- Pode ser computacionalmente custoso devido ao grande número de execuções necessárias.
- Não garante a obtenção da melhor solução global.
