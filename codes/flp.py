import numpy as np


class FLP:
    '''Facility Location Problem    
    '''

    def __init__(self, **args):
        if 'n' in args:
            # n: number of customers
            self.n = args['n']
            # m: number of facilities
            self.m = args.get('m', 0)
            self.demand = np.zeros(self.n, dtype=int)
            self.supply = np.zeros(self.m, dtype=int)
            self.opening_cost = np.zeros(self.m, dtype=int)
            self.capacity = np.zeros(self.m, dtype=int)
            self.transport_cost = np.zeros((self.n, self.m), dtype=int)
        elif 'filename' in args:
            self.read_file(args['filename'])
        else:
            raise ValueError('Invalid arguments')

    def read_file(self, filename):
        with open(filename, 'r') as f:
            self.n, self.m = map(int, f.readline().split())
            self.demand = np.array(list(map(int, f.readline().split())))
            self.supply = np.array(list(map(int, f.readline().split())))
            self.opening_cost = np.array(list(map(int, f.readline().split())))
            self.capacity = np.array(list(map(int, f.readline().split())))
            self.transport_cost = np.array(
                [list(map(int, f.readline().split())) for _ in range(self.n)])

    def write_file(self, filename):
        with open(filename, 'w') as f:
            f.write(f'{self.n} {self.m}\n')
            f.write(' '.join(map(str, self.demand)) + '\n')
            f.write(' '.join(map(str, self.supply)) + '\n')
            f.write(' '.join(map(str, self.opening_cost)) + '\n')
            f.write(' '.join(map(str, self.capacity)) + '\n')
            for i in range(self.n):
                f.write(' '.join(map(str, self.transport_cost[i])) + '\n')


class FLP_Solution:
    '''Facility Location Problem Solution
    '''

    def __init__(self, flp: FLP, filename=None):
        self.flp = flp
        self.opened = np.zeros(flp.m, dtype=int)
        self.transported = np.zeros((flp.n, flp.m), dtype=float)
        self.objective = 0
        if filename:
            self.read_file(filename)

    def evaluate(self):
        self.objective = self.opened @ self.flp.opening_cost
        self.objective += np.sum(self.transported * self.flp.transport_cost)
        return self.objective
    
    def is_feasible(self):
        for j in range(self.flp.m):
            if self.opened[j] == 1 and np.sum(self.transported[:, j]) > self.flp.capacity[j]:
                return False
        if np.any(np.sum(self.transported, axis=1) < self.flp.demand):
            return False
        return True

    def __str__(self):
        s = f'Objective: {self.objective}\n'
        s += 'Opened: ' + ' '.join(map(str, self.opened)) + '\n'
        s += 'Transported:\n'
        for i in range(self.flp.n):
            s += ' '.join(map(str, self.transported[i])) + '\n'
        return s

    def write_file(self, filename):
        with open(filename, 'w') as f:
            f.write(f'{self.objective}\n')
            f.write(' '.join(map(str, self.opened)) + '\n')
            for i in range(self.flp.n):
                f.write(' '.join(map(str, self.transported[i])) + '\n')

    def read_file(self, filename):
        with open(filename, 'r') as f:
            obj = int(f.readline())
            self.opened = np.array(
                list(map(int, f.readline().split())), dtype=int)
            self.transported = np.array(
                [list(map(int, f.readline().split())) for _ in range(self.flp.n)])
        if self.transported.shape != (self.flp.n, self.flp.m):
            raise ValueError('Invalid solution shape')
        if not self.is_feasible():
            raise ValueError('Infesible solution')
        self.evaluate()
        if self.objective != obj:
            raise ValueError('Invalid objective value')
