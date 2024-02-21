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

## "Todos os modelos são errados, mas alguns são úteis!"

---

## Otimização combinatória

- Um problema combinatório é aquele cuja solução pode ser representada na forma de uma sequência de decisões discretas. 
- Exemplos:
    - Caminho mínimo.
    - Árvore geradora mínima.
    - Localização de facilidades.
    - Roteirização de veículos. 

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

## Conclusão

- As metaheurísticas oferecem abordagens poderosas para enfrentar problemas desafiadores de otimização combinatória.
- Embora não garantam soluções ótimas, podem encontrar resultados satisfatórios em um tempo razoável.
- A compreensão dessas técnicas é essencial para resolver problemas do mundo real de forma eficiente.

