---
marp: true
theme: default
paginate: true
---

# Conceitos Fundamentais

---

## Definição de Metaheurísticas

- São métodos de solução que orquestram uma interação entre procedimentos de melhoria local e estratégias de nível mais alto.
 
- Ao longo do tempo, passaram a incluir quaisquer procedimentos que empreguem estratégias para superar a armadilha da optimalidade local.


- Ao contrário das heuristicas clássicas, as metaheurísticas são mais genéricas e podem ser aplicadas a uma ampla variedade de problemas de otimização.
  
- Metaheurísticas são frequentemente usadas em problemas de otimização combinatória, onde os métodos exatos são impraticáveis.

---
<!-- _backgroundColor: black -->
<!-- _color: white -->

## "Melhor aproximadamente agora do que exatamente nunca!"
---

## Elementos para Modelagem de Problemas

  - Representação do problema
  - Função objetivo
  - Função de avaliação


---

### Representação do Problema

  - Qual é a importância da representação adequada de um problema em uma metaheurística?
  - Quais são algumas formas comuns de representação de problemas em otimização combinatória?

---
### Função Objetivo
- Maximizar lucro!
- Minimizar custos!
- Multi critério.
- Exemplo:
  - Problema do Cacheiro Viajante: Minimizar o tempo total para se visitar todas as cidades.
  - Problema da Mínima Latência: Minimizar a soma dos tempos que cada cidade aguardou pela visita.

---
### Função de Avaliação 
- Mapeia soluções em um conjunto numérico.
- Pode englobar outros fatores como diversidade ou viabilidade.
- Geralmente se usa a Função Objetivo.
- Pode ser alterada durante a busca.
- Podem impactar a eficiência da busca.
---

## Principios para o entendimento de Metaheurísticas

- Problema de busca
- Espaço de busca
- Vizinhança
- Ótimo local
- Princípio da localidade

---

### Problema de Busca
  - Dado um conjunto de soluções, encontrar a melhor solução de acordo com um critério de otimização.
  - Seja $S$ um conjunto de soluções, $f: S \rightarrow R$ uma função objetivo, e $x^*$ uma solução ótima, o problema de busca é encontrar $x^*$ tal que $f(x^*) \le f(x)$ para todo $x \in S$.
  - Chamamos essa solução de **ótima global**.
---
### Espaço de Busca
- Conjunto de todas as soluções possíveis.
- Análise combinatória pode calcular seu tamanho ou ordem de grandeza.
- Serve como base intuitiva para a compreensão e desenvolvimento de processos de busca.
- Em geral, é limitado por restrições.
- Em geral, quanto maior for espaço de soluções, mais difícil será o problema

---

### Vizinhança
-Uma região do Espaço de Solução próxima a uma dada solução.
-A proximidade é determinada por uma métrica apropriada.

---

### Ótimo Local

- Um ótimo local é uma solução em um espaço de busca que é melhor do que todas as soluções em sua vizinhança imediata, mas não necessariamente melhor do que todas as outras soluções no espaço de busca.


### Princípio da Localidade

- O princípio da localidade é a ideia de que soluções boas tendem a estar próximas umas das outras no espaço de busca.
- Em outras palavras, uma boa função de avaliação tende a ser contínua e suave.

---

# Primeiros passos para implementação de Metaheurísticas

- Algoritmos Construtivos
- Busca Local
- Métodos de Descida

---

## Algoritmos Construtivos

- Constroem soluções **viáveis**, do zero, passo a passo, adicionando elementos gradualmente.
- Estratégias comuns: algoritmos gulosos, heurísticas de inserção.
- Por si só, podem ser desafiadores para problemas complexos, e.g. k-VRP.

---

## Busca Local

- Consiste em **varrer** o espaço de busca na vizinhança de uma dada solução a procura de uma solução melhor.
- **primeiro melhoramento**: a busca acaba quando qualquer melhoramento é obtido,
- **melhor vizinho**: enumerar todos os vizinhos e escolher o melhor.

- Quando a busca local não encontra uma solução melhor, dizer que a busca local está em um ótimo local.

---

## Métodos de Descida

- Utilizados para encontrar mínimos locais.
- Algorítmo:
  1. Dada uma solução, busque uma solução melhor em sua vizinhança.
  2. Mova-se para esta solução.
  3. Repita os passos 1-2 até que não haja qualquer solução melhor na vizinhança.
  4. Fim.
- Hill-Climbing

---

- Depende do ponto inicial.
- Não é global.
- Não informa GAP de otimalidade.
- É um procedimento genérico.
- Depende da “aspereza” do espaço de solução.
- Os vizinhos devem ser avaliados rapidamente.

---

# Desafios

- O principal desafio ao lidar com problemas de otimização é evitar ficar preso em ótimos locais.
- Ficar preso em um ótimo local pode impedir a descoberta da melhor solução global.

---

## Estratégias de Escape

- Para escapar de ótimos locais, muitas metaheurísticas usam estratégias como:
  - Perturbações aleatórias para explorar novas regiões do espaço de busca.
  - Intensificação e diversificação para equilibrar a exploração e a explotação.
  - Reinicialização para reiniciar a busca a partir de soluções diferentes.


