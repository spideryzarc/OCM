---
marp: true
theme: default
paginate: true
---

# Conceitos Fundamentais

---

## Definição de Metaheurísticas

- **O que são metaheurísticas?**
- **Como as metaheurísticas se diferenciam de outros métodos de otimização?**
- **Qual é o papel das metaheurísticas na resolução de problemas de otimização combinatória?**

---

## Elementos para Implementação

- **Espaço de Busca:**
  - O que é espaço de busca e qual é a sua relação com o problema de otimização?
  - Como a estrutura do espaço de busca influencia o desempenho de uma metaheurística?
- **Representação do Problema:**
  - Qual é a importância da representação adequada de um problema em uma metaheurística?
  - Quais são algumas formas comuns de representação de problemas em otimização combinatória?
- **Vizinhança:**
  - O que é vizinhança em relação a uma solução em um problema de otimização?
  - Como a definição de vizinhança impacta o processo de busca em uma metaheurística?

---

## Algoritmos Construtivos

- **Definição e Funcionamento:**
  - Constroem soluções passo a passo, adicionando elementos gradualmente.
  - Estratégias comuns: algoritmos gulosos, heurísticas de inserção.

---

## Busca Local

- **Conceito e Componentes:**
  - Procura por soluções melhores em uma vizinhança local.
  - Componentes principais: solução inicial, função de avaliação, operador de movimento.

---

## Métodos de Descida

- **Características e Aplicações:**
  - Utilizados para encontrar mínimos locais em uma função de custo.
  - Exemplos: Gradiente Descendente, Método de Descida de Vizinho Mais Próximo.

---

# Ótimo Local

---

## Definição

- Um ótimo local é uma solução em um espaço de busca que é melhor do que todas as soluções em sua vizinhança imediata, mas não necessariamente melhor do que todas as outras soluções no espaço de busca.

---

## Desafios

- O principal desafio ao lidar com problemas de otimização é evitar ficar preso em ótimos locais.
- Ficar preso em um ótimo local pode impedir a descoberta da melhor solução global.

---

## Estratégias de Escape

- Para escapar de ótimos locais, muitas metaheurísticas usam estratégias como:
  - Perturbações aleatórias para explorar novas regiões do espaço de busca.
  - Intensificação e diversificação para equilibrar a exploração e a explotação.
  - Reinicialização para reiniciar a busca a partir de soluções diferentes.


