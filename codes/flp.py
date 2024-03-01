import numpy as np


class FLP:
    '''Single Source Capacitated Facility Location Problem (SSCFLP)
    or Warehouse Location Problem (WLP)
    '''

    def __init__(self, **args):
        if 'filename' in args:
            self.read_file(args['filename'])
        elif 'n' in args:
            # n: number of customers
            self.n = args['n']
            # m: number of facilities
            self.m = args.get('m', 0)
            # demand: array of size n with the demand of each customer
            self.demand = np.zeros(self.n, dtype=int)
            # supply: array of size m with the supply/capacity of each facility
            self.supply = np.zeros(self.m, dtype=int)
            # opening_cost: array of size m with the cost of opening each facility
            self.opening_cost = np.zeros(self.m, dtype=int)
            # assignment_cost: array of size (n, m) with the cost of assigning each customer to each facility
            self.assignment_cost = np.zeros((self.n, self.m), dtype=float) 
        else:
            raise ValueError('Invalid arguments')

    def read_file(self, filename):
        with open(filename, 'r') as f:
            self.m, self.n = map(int, f.readline().split())
            self.supply = np.zeros(self.m, dtype=int)
            self.opening_cost = np.zeros(self.m, dtype=int)
            for i in range(self.m):
                self.supply[i], self.opening_cost[i] = map(int, f.readline().split())
            v = []
            while len(v) < self.n:
                v.extend(map(int,map(float, f.readline().split())))
            self.demand = np.array(v[:self.n], dtype=int)
           
            self.assignment_cost = np.zeros((self.n, self.m), dtype=float)
            for i in range(self.n):
                v = []
                while len(v) < self.m:
                    v.extend(map(float, f.readline().split()))
                self.assignment_cost[i, :] = v[:self.m]


    def write_file(self, filename):
        with open(filename, 'w') as f:
            f.write(f'{self.m} {self.n}\n')
            for i in range(self.m):
                f.write(f'{self.supply[i]} {self.opening_cost[i]}\n')
            f.write(' '.join(map(str, self.demand)) + '\n')
            for i in range(self.n):
                f.write(' '.join(map(str, self.assignment_cost[i])) + '\n')

    def __str__(self):
        s = f'Number of facilities: {self.m}\n'
        s += f'Number of customers: {self.n}\n'
        s += 'Demand: ' + ' '.join(map(str, self.demand)) + '\n'
        s += 'Supply: ' + ' '.join(map(str, self.supply)) + '\n'
        s += 'Opening cost: ' + ' '.join(map(str, self.opening_cost)) + '\n'
        s += 'Assignment cost:\n'
        for i in range(self.n):
            s += ' '.join(map(str, self.assignment_cost[i])) + '\n'
        return s
    



###### Solution ######

class FLP_Solution:
    '''Facility Location Problem Solution
    '''

    def __init__(self, flp: FLP, filename=None):
        self.flp = flp
        # facility opened or not 0/1
        self.opened = np.zeros(flp.m, dtype=int)
        # facility assigned to each customer
        self.assigned = np.zeros(flp.n, dtype=int)
        # objective function value
        self.objective = 0
        if filename:
            self.read_file(filename)

    def read_file(self, filename):
        with open(filename, 'r') as f:
            self.opened = np.array(list(map(int, f.readline().split())))
            self.assigned = np.array(list(map(int, f.readline().split())))
            self.objective = float(f.readline())

    def write_file(self, filename):
        with open(filename, 'w') as f:
            f.write(' '.join(map(str, self.opened)) + '\n')
            f.write(' '.join(map(str, self.assigned)) + '\n')
            f.write(f'{self.objective}\n')

    def __str__(self):
        s = 'Opened: ' + ' '.join(map(str, self.opened)) + '\n'
        s += 'Assigned: ' + ' '.join(map(str, self.assigned)) + '\n'
        s += f'Objective: {self.objective}\n'
        return s
    
    def evaluate(self):
        self.objective = self.flp.opening_cost @ self.opened
        for i in range(self.flp.n):
            self.objective += self.flp.assignment_cost[i, self.assigned[i]]
        return self.objective

    def is_valid(self):
        '''Check if the solution is valid'''
        # check if each customer was assigned to a facility
        if np.any(self.assigned < 0) or np.any(self.assigned >= self.flp.m):
            return False
        # check if the demand of each customer is not greater than the supply of the facility
        demand = np.zeros(self.flp.m, dtype=int)
        for i in range(self.flp.n):
            demand[self.assigned[i]] += self.flp.demand[i]
        if np.any(demand > self.flp.supply):
            return False        
        return True
    
#### main ####
        
if __name__ == '__main__':
    flp = FLP(filename='codes\instances\p1')
    print(flp)

