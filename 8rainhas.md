---
marp: true
theme: default
paginate: true
title: 8 Rainhas
footer: '8 Rainhas - Metaheurísticas em Otimização Combinatória'
---

# 8 Rainhas

---

## O Problema das 8 Rainhas

- **Propósito:** Encontrar uma disposição das 8 rainhas em um tabuleiro de xadrez de forma que nenhuma delas se ataque.
- **Relevância:** Problema clássico de otimização combinatória, com aplicações em jogos e algoritmos genéticos.
- **Abordagem:** Utilização de métodos exatos e heurísticos para encontrar soluções viáveis.
  
--- 

## Descrição do Problema

- **Tabuleiro de Xadrez:** 8x8
- **Rainhas:** 8
- **Objetivo:** Dispor as 8 rainhas de forma que nenhuma delas se ataque.
- **Restrições:** Nenhuma rainha pode atacar outra na mesma linha, coluna ou diagonal.

---

## Implementação em Python

- Funções preliminares
  - **Estrutura de dados**: O que vamos usar para representar o tabuleiro?
  - **Função de vizualização**: Como vamos mostrar o tabuleiro?
  - **Função de avaliação**: Como vamos determinar se uma solução é válida?

---

## Estrutura de Dados

### Estrutura elementar

A forma mais direta de se representar um tabuleiro de xadrez é através de uma matriz 8x8. Cada posição da matriz pode ser ocupada por uma rainha ou não.

```python
import numpy as np
tabuleiro = np.zeros((8, 8), dtype=bool)
```

---

### Um pouco de combinatória

- Chamaremos de **solução** qualquer disposição das 8 rainhas no tabuleiro.
- Chamaremos de **solução válida** aquela em que nenhuma rainha se ataca.

Quantas **soluções** existem para o problema das 8 rainhas?

Em um tabuleiro de xadrez 8x8, existem $64 \choose 8$ possíveis disposições de 8 rainhas.

$$4.426.165.368$$

---

### Usando um pouco a cabeça

- Muitas destas soluções são obviamente inválidas.
  - Aquelas que tem duas rainhas na mesma linha ou coluna.

- Sabemos que exite exatamente uma rainha em cada linha e coluna. Podemos usar o princípio da multiplicação para calcular o número de soluções sem ataque em linhas ou colunas.
  - $d_1$: Qual linha ocupar na coluna 1? *--8 possibiliades--*
  - $d_2$: Qual linha ocupar na coluna 2? *--7 possibiliades--*
  - ...
  - $d_8$: Qual linha ocupar na coluna 8? *--1 possibiliade--*
- Pelo princípio da multiplicação, o número total de soluções é $8! = 40.320$.

---

### Repensando a estrutura de dados

- Parece desperdício usar uma matriz 8x8 para este problema.
- Uma estrutura capaz de armazenar 4.426.165.368 quando só precisamos de 40.320, ainda menos.
- Nós só precisamos indicar a linha em que cada rainha está para cada coluna.
- Uma permutação de 8 elementos basta, mas não há esse tipo de estrutura em Python, então usaremos um array de inteiros.

```python
import numpy as np
solucao = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
```
- Faremos que se comportem como permutações via lógica de programação.

---

## Função de Vizualização
### Básico
```python
def printar(solucao):
    n = len(solucao)
    for i in range(n):
        for j in range(n):
            if solucao[j] == i:
                print('\u2655', end='')
            else:
                print('\u2B1C', end='')
        print()
    print()
```
- Os caracteres `'\u2655'` e `'\u2B1C'` são respectivamente o símbolo de uma rainha e um quadrado vazio.
---

### Avançado
- Vamos usar a biblioteca `matplotlib` 
```python
import matplotlib.pyplot as plt
solucao = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
def plotar(solucao):
    n = len(solucao)
    dx, dy = 0.015, 0.05 
    x = np.arange(-4.0, 4.0, dx) 
    y = np.arange(-4.0, 4.0, dy) 
    X, Y = np.meshgrid(x,y) 
    min_max = np.min(x), np.max(x), np.min(y), np.max(y) 
    res = np.add.outer(range(n), range(n))%2 
    plt.imshow(res, cmap="binary_r")
    plt.xticks([])
    plt.yticks([])
    for i in range(n):
        plt.text(i, solucao[i], '\u2655', color="orange", fontsize=20, ha="center", va="center")
    plt.show()
```

---

## Função de Avaliação

- Há três tipos de problemas na computação:
  - **Satisfação:** Existe uma solução?
  - **Localização:** Onde está uma solução?
  - **Otimização:** Qual é a melhor solução?
- O problema das 8 rainhas é um problema de localização.
- Mas podemos transformá-lo em um problema de otimização.

---

### Função de Avaliação

- A função de avaliação é uma função que atribui um valor numérico a uma solução.
- Vamos contar o número de ataques entre as rainhas. Considerando que não há ataques entre rainhas na mesma linha ou coluna, só precisamos verificar as diagonais.

```python
def avaliar(solucao):    
    ataques = 0
    n = len(solucao)
    for i in range(n):
        for j in range(i+1, n):
            if abs(solucao[i] - solucao[j]) == j - i:
                ataques += 1
    return ataques  
```
---
### Ovo de Colombo

`abs(solucao[i] - solucao[j]) == j - i` 
- é uma condição que verifica se duas rainhas estão na mesma diagonal.
- Parece fácil, será que é mesmo?
- Após encontrar soluções para problemas difíceis, eles parecem fáceis.

---

## Solução por Força Bruta

- Vamos gerar todas as soluções possíveis e avaliá-las.
- Se encontrarmos uma solução sem ataques, podemos parar a busca.
- Vamos usar a biblioteca `itertools` para gerar todas as permutações possíveis.

```python
import itertools
def forca_bruta(n=8):
    melhor_solucao = np.array(range(n), dtype=int)
    menor_ataques = avaliar(melhor_solucao)
    for solucao in itertools.permutations(range(n)):
        ataques = avaliar(solucao)
        if ataques < menor_ataques:
            menor_ataques = ataques
            melhor_solucao = solucao
            if ataques == 0:
                break
    return melhor_solucao
``` 

---
- A função `itertools.permutations` gera todas as permutações possíveis de um conjunto.
- Acabamos por gerar e avaliar muitas soluções inválidas que poderiam ser descartadas.
- Se implementarmos nosso próprio gerador de permutações, podemos evitar isso.	
- Vamos implementar um gerador de permutações que gere apenas soluções válidas.

---
## Gerador Apenas Permutações Válidas

- Vamos usar um algoritmo recursivo para gerar todas as permutações possíveis de um conjunto.

```python
def forca_bruta_otimizada(n=8):    
    solucao = np.full(n, -1, dtype=int)
    usados = np.full(n, False, dtype=bool)    
    for i in range(n):
        if coloca(i,0, solucao, usados):
            break
    return solucao
```

---
- A função `coloca` é uma função recursiva que tenta colocar uma rainha em uma coluna.
- Se a rainha puder ser colocada, a função chama a si mesma para a próxima coluna.
- Se a função chegar na última coluna, encontramos uma solução válida.
- Se a função não conseguir colocar uma rainha em uma coluna, ela retorna `False`.
```python	
def coloca(linha , coluna, solucao, usados):
    solucao[coluna] = linha
    usados[linha] = True
    if ha_ataque(coluna, solucao):
        usados[linha] = False
        return False
    if coluna == len(solucao) - 1:
        return True
    for i in range(len(solucao)):
        if not usados[i]:
            if coloca(i, coluna+1, solucao, usados):
                return True
    usados[linha] = False
    return False
```	
---
- A função `ha_ataque` verifica se uma rainha na coluna `coluna` ataca alguma outra rainha.
- É uma versão modificada da função `avaliar` que verifica se uma solução é válida.
- Esta função leva em conta apenas as rainhas que já foram colocadas em relação com a última.

```python
def ha_ataque(coluna, solucao):
    for i in range(coluna):
        if abs(solucao[i] - solucao[coluna]) in (0, coluna - i):
            return True
    return False
```

---
### Comparação de Desempenho

- Vamos comparar o desempenho das duas funções.
- Vamos usar a função mágica `%timeit` do Jupyter para medir o tempo de execução.
- Os parâmetros `-n1 -r3` indicam que a função será executada uma vez em cada repetição, e que serão feitas 3 repetições.

```python
for n in range(8, 12):
    print(n)
    %timeit -n1 -r3 forca_bruta(n)
    %timeit -n1 -r3 forca_bruta_otimizada(n)
    print()
```





