---
marp: true
theme: default
paginate: true
---

# Método Iterated Local Search

---

## Conceito

- Combinação de busca local com perturbações aleatórias e uma estratégia de intensificação.
- Realiza iterações alternadas entre a melhoria da solução local e a perturbação da solução.

---

## Funcionamento

- Começa com uma solução inicial obtida aleatoriamente ou por outra heurística.
- Realiza uma busca local para melhorar a solução.
- Aplica uma perturbação para escapar de ótimos locais.
- Retorna à solução melhorada e repete o processo.

---

## Vantagens

- Combina eficiência da busca local com a diversidade da perturbação aleatória.
- Pode escapar de mínimos locais, explorando diferentes regiões do espaço de busca.

---

## Desvantagens

- Pode ser sensível aos parâmetros de controle, exigindo ajustes cuidadosos.
- Nem sempre garante a obtenção da melhor solução global.

---

## Diversificação vs Intensificação

- **Diversificação:** Exploração do espaço de busca em busca de diferentes regiões.
- **Intensificação:** Concentração nos arredores de uma solução promissora.

