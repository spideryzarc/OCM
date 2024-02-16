---
marp: true
theme: default
paginate: true
---

# Path Relinking

---

## Conceito

- Path Relinking é uma metaheurística que combina duas soluções para explorar regiões não visitadas do espaço de busca.

---

## Funcionamento

- No Path Relinking, é selecionado um par de soluções candidatas, uma chamada solução inicial e outra chamada solução final.
- Um caminho é construído entre as duas soluções, iterativamente, movendo-se de uma solução para a outra em direção a uma solução de referência, utilizando operadores de movimento.
- O processo é repetido até que uma solução de qualidade desejada seja encontrada ou um critério de parada seja atingido.

---

## Operadores de Movimento

- Os operadores de movimento utilizados no Path Relinking podem variar dependendo do problema, mas geralmente incluem:
  - Perturbações para introduzir variações nas soluções.
  - Busca local para melhorar a qualidade das soluções durante o caminho.

---

## Vantagens

- O Path Relinking é eficaz para explorar regiões não visitadas do espaço de busca.
- Pode ser combinado com outras metaheurísticas para melhorar o desempenho em problemas complexos.

