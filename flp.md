---
marp: true
theme: default
paginate: true
---

# Capacitated Facility Location Problems with Single Sourcing

---

## Descrição do problema

- Dado um conjunto de clientes e um conjunto de possíveis localizações para a instalação de uma fábrica, o problema consiste em determinar a localização da fábrica de forma a minimizar a soma das distâncias entre a fábrica e os clientes.
- O problema é conhecido como problema de localização de instalações de fonte única (single source facilities location problem).
- O problema é NP-difícil.

---

## Formulação do problema

- Conjuntos:
  - Um conjunto de clientes $I = \{1,2, \ldots, n\}$.
  - Um conjunto de possíveis localizações para a fábrica $J = \{1, 2, \ldots, m\}$.
- Parâmetros:
  - Uma matriz de custo $c_{ij}$, que representa o custo de atender o cliente $i$ a partir da localização $j$.
  - Um custo fixo $f_j$ associado à instalação da fábrica na localização $j$.
  - Uma demanda $d_i$ associada a cada cliente $i$.
  - Uma capacidade $M_j$ associada à fábrica.
  
---

- Variáveis de decisão:
  - $y_j = 1$, se a fábrica é instalada na localização $j$; $y_j = 0$, caso contrário.
  - $x_{ij} = 1$, se o cliente $i$ é atendido a partir da localização $j$; $x_{ij} = 0$, caso contrário.

---

## Modelo de Programação Linear Inteira

- Função objetivo:

$$ min: \sum_{j \in J} f_jy_j  + \sum_{i \in I}\sum_{j \in J} x_{ij}c_{ij}$$

- Restrições:
  
$$\sum_{j \in J} x_{ij} = 1, \forall i \in I$$
> Cada cliente deve ser atendido a partir de uma única localização:

  
$$\sum_{i \in I} d_i x_{ij} \leq M_jy_j, \forall j \in J$$
>A capacidade da fábrica não pode ser excedida:
---

$$ x_{ij} \leq y_j, \forall i \in I, j \in J$$
> Se a fábrica não for instalada na localização $j$, então o cliente $i$ não pode ser atendido a partir dessa localização. Restrição para fortalecer a formulação.

$$x_{ij} \in \{0,1\}, \forall i \in I, j \in J$$
$$y_j \in \{0,1\}, \forall j \in J$$

> As variáveis $x_{ij}$ e $y_j$ são binárias:

---
