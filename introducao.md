---
marp: true
theme: default
paginate: true
---

# Metaheurísticas em Otimização Combinatória

---

## Introdução e Contextualização

- **Propósito:** Explorar técnicas avançadas para resolver problemas de otimização complexos.
- **Relevância:** Problemas de otimização combinatória são comuns em diversos campos, como logística, engenharia e computação.
- **Abordagem:** Utilização de métodos aproximados e heurísticos para encontrar soluções próximas do ótimo em tempo razoável.
- **Desafios:** A natureza combinatória desses problemas os torna difíceis de resolver de forma exata em tempo viável.

---

### Bibliografia

- Glover, Fred W., and Gary A. Kochenberger, eds. **Handbook of metaheuristics**. Vol. 57. Springer Science & Business Media, 2006.
- Talbi, El-Ghazali. **Metaheuristics: from design to implementation**. Vol. 74. John Wiley & Sons, 2009.
- Michalewicz, Zbigniew, and David B. Fogel. **How to solve it: modern heuristics**. Springer Science & Business Media, 2013.

---

### Avaliação

- **Provas Escritas:** Duas avaliações escritas ao longo do semestre. 
- **Projeto Final:** Resolução de um problema de otimização combinatória utilizando metaheurísticas. 

$$ Nota = \frac{p_1+p_2+pf}{3}$$

---

### Linguagem de Programação

- Os exemplos e projetos serão desenvolvidos em **Python**. Embora não seja a linguagem adequada, é de fácil entendimento e rápida prototipação.
- O projeto final pode ser desenvolvido em qualquer linguagem de programação "não obscura".
- Tentaremos atingir o máximo de performance possível, mas sem otimizações extremas.

---

## Por que Problemas de Otimização Combinatória são Relevantes?

- **Aplicações Práticas:** Exemplos incluem roteamento de veículos, escalonamento de produção e design de redes.
- **Complexidade:** O grande número de possíveis soluções torna a busca pelo ótimo uma tarefa árdua.
- **Impacto Econômico:** Soluções eficientes podem levar a economias significativas de recursos e tempo.

---

## Exemplos de Problemas de Otimização Combinatória

- **Caixeiro Viajante (TSP):** Encontrar o caminho mais curto que visita todas as cidades exatamente uma vez e retorna à cidade de origem.
- **Problema da Mochila (KP):** Determinar quais itens selecionar para maximizar o valor total, considerando uma capacidade máxima de carga.
- **Escalonamento de Produção:** Decidir a sequência de produção de diferentes itens para minimizar os custos ou maximizar a eficiência.
- **Roteamento de Veículos (VRP):** Designar rotas para veículos de forma a atender a demanda de vários clientes, minimizando os custos totais.

---

## Modelagem

- O primeiro passo para resolver computacionalmente um problema real é sua modelagem.
- Um modelo nunca será igual ao problema original, sempre haverão simplificações.
- Um modelo útil deve ser simples o bastante para ser resolvido, e preciso o suficiente para que seu resultado possa ser aplicado.

---

<!-- _backgroundColor: black -->
<!-- _color: white -->

"Todos os modelos são errados, mas alguns são úteis!"
-

---

## Otimização combinatória

- Um problema combinatório é aquele cuja solução pode ser representada na forma de uma sequência de decisões discretas. 
- Exemplos:
    - Caminho mínimo.
    - Árvore geradora mínima.
    - Localização de facilidades.
    - Roteirização de veículos.
    - Escala de horários. 

---

## Abordagens para Resolver Problemas de Otimização Combinatória

- **Métodos Exatos:** Algoritmos que garantem a solução ótima, mas podem ser inviáveis para problemas grandes.
- **Métodos Aproximados:** Estratégias que buscam soluções de boa qualidade, mas não necessariamente ótimas.
- **Heurísticas:** Técnicas que exploram informações específicas do problema para guiar a busca por soluções.
- **Metaheurísticas:** Estratégias gerais e flexíveis que podem ser aplicadas a uma ampla variedade de problemas.
  
---

## Métodos Exatos

- **Enumeração Exaustiva:** Avalia todas as possíveis soluções para encontrar a melhor.
- **Programação Dinâmica:** Explora a estrutura recursiva do problema para evitar recálculos redundantes.
- **Programação Linear Inteira:** Formula o problema como um modelo de programação linear com variáveis inteiras, para posterior resolução usando algoritmos específicos, e.g Branch and Bound, Branch and Cut, etc.
- **Vantagens:** Garantem a solução ótima, quando encontrada.
- **Desvantagens:** Podem ser inviáveis para problemas grandes devido à complexidade computacional.
---

## Métodos Aproximados e Heurísticos

- **Métodos Aproximados:** Buscam soluções de boa qualidade, mas não necessariamente ótimas, em tempo razoável.
- **Heurísticas:** Estratégias que exploram informações específicas do problema para guiar a busca por soluções.
- **Vantagens:** Permitem lidar com problemas complexos de forma eficiente, mesmo sem garantias de otimalidade.
- **Desvantagens:** Podem não encontrar a solução ótima, mas são úteis para problemas práticos.

---

## Por que a Resolução de Problemas deste Tipo é Considerada Difícil?

- **Combinatorialidade:** O grande espaço de busca resulta em um número exponencial de possíveis soluções.
- **Complexidade Computacional:** Muitos problemas são NP-difíceis, o que significa que não existe algoritmo polinomial conhecido para resolvê-los de forma exata em tempo viável.
- **Necessidade de Metaheurísticas:** Estratégias inteligentes são necessárias para encontrar soluções de boa qualidade em um tempo aceitável.

---

## Metaheurísticas

- **Definição:** Métodos gerais e flexíveis para resolver problemas de otimização, que podem ser aplicados a uma ampla variedade de problemas.
- **Características:** Não garantem a solução ótima, mas são capazes de encontrar soluções de boa qualidade em tempo razoável.


---

## Conceitos Básicos

---
### Espaço de Busca

O que faz das metaheurísticas uma abordagem genérica é o fato de serem projetadas para resolver o **problema de busca**

- **Problema de Busca:** Dado um conjunto de soluções, encontrar a melhor solução de acordo com um critério de otimização.
- **Espaço de Busca:** Conjunto de todas as possíveis soluções para um problema.

> As metaheurísticas não são descritas em termos de um problema específico, mas sim em termos de um **espaço de busca**.

---
- Em geral, usamos espaços de busca para descrever de forma metafórica o funcionamento de uma metaheurística.
  - Gráfico 2D: Espaço de busca x Qualidade da solução.
  - Gráfico mapa de calor: Espaço de busca (2D) x Função objetivo (gradiente de cor). 

---
### Função Objetivo

- **Função Objetivo:** Função que atribui um valor numérico a cada solução candidata, representando sua qualidade de acordo com o critério de otimização.
- **Função de Avaliação:** Certas abordagens de otimização, adaptam a função objetivo para uma função de avaliação, que é uma função que não necessariamente representa a qualidade da solução, mas sim a adequação da solução para a metaheurística.

---
### Vizinhança

A vizinhança é uma noção fundamental em muitas metaheurísticas, como busca local e algoritmos genéticos.

- **Vizinhança:** Conjunto de soluções que são "próximas" de uma solução dada.
- Geralmente, um vizinho é obtido a partir de uma solução atual por meio de uma pequena modificação.


---
### Princípio da Localidade

- **Princípio da Localidade:** A ideia de que soluções de alta qualidade tendem a estar próximas umas das outras no espaço de busca.
- Pequenas modificações em uma solução de alta qualidade devem resultar em soluções de qualidade semelhante.
- Se a função objetivo for muito sensível ou nada sensível a pequenas modificações, as metaheurísticas podem não ser eficazes.
- Nestes casos podemos usar um função de avaliação ou trocar a natureza das modificações.
  
---
### Ótimo Global e Ótimo Local

- **Ótimo Global:** A melhor solução possível para um problema de otimização.
- **Ótimo Local:** Uma solução que é a melhor em sua vizinhança, mas não necessariamente a melhor globalmente.

> O desafio das metaheurísticas é encontrar o ótimo global, evitando ficar preso em ótimos locais.

---
### Método Construtivo

As metaheurísticas precisam de um ponto de partida para iniciar a busca. 

- **Definição:** Abordagem que constrói uma solução passo a passo, adicionando elementos de acordo com um critério específico.
- Na maioria das vezes, criar uma solução viável não é complicado,
- Em certos casos, criar umas solução viável pode ser um problema em si. Neste casos podemos admitir soluções inviáveis e corrigi-las posteriormente.
- Comumente temos métdos construtivos:
  - Totalmente aleatórios.
  - Gulosos,
  - Guloso Randomizado.  
---
### Busca Local e Métodos de Descida

A maior parte das metaheurísticas usam a busca local como um componente fundamental.

- **Busca Local:** Estratégia que explora a vizinhança de uma solução atual para encontrar uma solução melhor.
- **Métodos de Descida:** Algoritmos que exploram a vizinhança de uma solução atual e se movem para um vizinho melhor, até que não haja mais melhorias possíveis (**ótimo local**). Podem ser:
  - **Primeiro Melhor:** Aceita o primeiro vizinho melhor encontrado para se mover.
  - **Melhor Melhor:** Explora todos os vizinhos e escolhe o melhor deles para se mover.

---
### Intensificação vs Diversificação
Um conceito que lidamos no projeto de metaheurísticas é a **intensificação** vs **diversificação**.

- **Intensificação:** Concentrar a busca em torno de regiões promissoras do espaço de busca.
- **Diversificação:** Explorar diferentes regiões do espaço de busca para evitar ficar preso em ótimos locais.

- Se um método demora muito para apresentar um bom resultado, pode ser que ele esteja muito diversificado
- Se um método não apresenta melhores soluções mesmo aumentando o número de iterações, pode ser que ele esteja muito intensificado.
> O desafio é encontrar um equilíbrio entre intensificação e diversificação para encontrar o ótimo global.
---
### Metaheurísticas Populares
Há muitas metaheurísticas diferentes, cada uma com suas próprias características e aplicações.
- **s-Metaheuriticas**: Baseados em uma solução corrente:
  - RMS (Random Multi Start)
  - ILS (Iterated Local Search)
  - VNS (Variable Neighborhood Search)
  - GRASP (Greedy Randomized Adaptive Search Procedure)
  - TS (Tabu Search)
  - SA (Simulated Annealing)
  - GLS (Guided Local Search)
---
- **p-Metaheurísticas**: Baseados em uma população de soluções:
  - GA (Genetic Algorithm)
  - PSO (Particle Swarm Optimization)
  - ACO (Ant Colony Optimization)
  - EDA (Estimation of Distribution Algorithm)

> Nada impede de combinar diferentes metaheurísticas para resolver um problema, essa é uma abordagem comum e chamada de híbrida.
---
- Algumas metaheurísticas pitorescas:
  - **Harmony Search:** Inspirada no processo de improvisação musical.
  - **Firefly Algorithm:** Baseada no comportamento de vaga-lumes.
  - **Cuckoo Search:** Inspirada no parasitismo de aves.
  - **Bat Algorithm:** Baseada no comportamento de morcegos.
  - **Flower Pollination Algorithm:** Inspirada na polinização de flores.
---

### Matheuristicas

- **Matheurísticas:** Combinação de métodos exatos e heurísticos para resolver problemas de otimização.
- **Vantagens:** Aproveitam a eficiência dos métodos exatos e a flexibilidade das heurísticas.
- **Desvantagens:** Podem ser mais complexas de implementar e ajustar do que métodos puramente heurísticos.
- **Exemplos:** Algoritmos híbridos que combinam programação linear inteira com busca local ou algoritmos genéticos.
- **Aplicações:** Problemas de otimização complexos que requerem uma abordagem mista para serem resolvidos de forma eficiente.

---

## Conclusão

- As metaheurísticas oferecem abordagens poderosas para enfrentar problemas desafiadores de otimização combinatória.
- Embora não garantam soluções ótimas, podem encontrar resultados satisfatórios em um tempo razoável.
- A compreensão dessas técnicas é essencial para resolver problemas do mundo real de forma eficiente.

