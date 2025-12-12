# Compilado das atividades da Semana 9 - Criptografia e Computação Quântica

Este repositório contém as atividades práticas realizadas durante a semana 9 da optativa de Criptografia e Computação Quântica.

**Grupo:**

- Ana Goes
- Gabriel Coletto
- Mauro das Chagas
- Vitto Mazeto

## Estrutura

### 1. [Algoritmo de Grover](./article)
Implementação do algoritmo de Grover para busca quântica em bancos de dados, aplicado ao problema de transplante renal.

**Arquivos:**
- [`grover_algorithm.ipynb`](./article/grover_algorithm.ipynb) - Notebook com implementação completa
- [`grover_algorithm.py`](./article/grover_algorithm.py) - Script Python

**Implementações:**
- 2 qubits (4 candidatos) - Manual e usando Qiskit
- 4 qubits (16 candidatos) - Busca do estado `1010`
- 6 qubits (64 candidatos) - Busca do estado `101010`
- Aplicação ao Kidney Exchange Problem (KEP)

### 2. [Algoritmo de Shor](./shors)
Implementação do algoritmo de Shor para fatoração de números inteiros, demonstrando a quebra de criptografia RSA.

**Arquivos:**
- [`shors_algorithm.ipynb`](./shors/shors_algorithm.ipynb) - Notebook completo com teoria e prática
- [`shors-algorithm.py`](./shors/shors-algorithm.py) - Script Python

**Implementações:**
- Fatoração de N=15 com a=2
- Estimativa de fase quântica
- Operadores de exponenciação modular
- Execução em hardware IBM Quantum (`ibm_marrakesh`)

### 3. [Atividades do Curso](./course)
Diretórios individuais para atividades práticas de cada membro do grupo:
- [`/aninha`](./course/aninha)
- [`/colecria`](./course/colecria)
- [`/mauro`](./course/mauro)
- [`/vitto`](./course/vitto)
