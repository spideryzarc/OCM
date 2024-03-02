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
            self.num_customers = args['n']
            # m: number of facilities
            self.num_facilities = args.get('m', 0)
            # demand: array of size n with the demand of each customer
            self.demand = np.zeros(self.num_customers, dtype=int)
            # supply: array of size m with the supply/capacity of each facility
            self.supply = np.zeros(self.num_facilities, dtype=int)
            # opening_cost: array of size m with the cost of opening each facility
            self.opening_cost = np.zeros(self.num_facilities, dtype=int)
            # assignment_cost: array of size (n, m) with the cost of assigning each customer to each facility
            self.assignment_cost = np.zeros(
                (self.num_customers, self.num_facilities), dtype=float)
        else:
            raise ValueError('Invalid arguments')

    def read_file(self, filename):
        with open(filename, 'r') as f:
            self.num_facilities, self.num_customers = map(
                int, f.readline().split())
            self.supply = np.zeros(self.num_facilities, dtype=int)
            self.opening_cost = np.zeros(self.num_facilities, dtype=int)
            for i in range(self.num_facilities):
                self.supply[i], self.opening_cost[i] = map(
                    int, map(float, f.readline().split()))
            v = []
            while len(v) < self.num_customers:
                v.extend(map(int, map(float, f.readline().split())))
            self.demand = np.array(v[:self.num_customers], dtype=int)

            self.assignment_cost = np.zeros(
                (self.num_facilities, self.num_customers), dtype=float)
            for i in range(self.num_facilities):
                v = []
                while len(v) < self.num_customers:
                    v.extend(map(float, f.readline().split()))
                self.assignment_cost[i, :] = v[:self.num_customers]

    def write_file(self, filename):
        with open(filename, 'w') as f:
            f.write(f'{self.num_facilities} {self.num_customers}\n')
            for i in range(self.num_facilities):
                f.write(f'{self.supply[i]} {self.opening_cost[i]}\n')
            f.write(' '.join(map(str, self.demand)) + '\n')
            for i in range(self.num_customers):
                f.write(' '.join(map(str, self.assignment_cost[i])) + '\n')

    def __str__(self):
        s = f'Number of facilities: {self.num_facilities}\n'
        s += f'Number of customers: {self.num_customers}\n'
        s += 'Demand: ' + ' '.join(map(str, self.demand)) + '\n'
        s += 'Supply: ' + ' '.join(map(str, self.supply)) + '\n'
        s += 'Opening cost: ' + ' '.join(map(str, self.opening_cost)) + '\n'
        s += 'Assignment cost:\n'
        for i in range(self.num_facilities):
            s += ' '.join(map(str, self.assignment_cost[i])) + '\n'
        return s


###### Solution ######

class FLP_Solution:
    '''Facility Location Problem Solution
    '''

    def __init__(self, flp: FLP, filename=None):
        self.flp = flp
        # facility opened or not 0/1
        self.opened = np.zeros(flp.num_facilities, dtype=bool)
        # facility assigned to each customer
        self.assigned = np.full(flp.num_customers, -1, dtype=int)
        # objective function value
        self.objective = np.inf
        if filename:
            self.read_file(filename)

    def read_file(self, filename):
        with open(filename, 'r') as f:
            self.opened[:] = list(map(bool, f.readline().split()))[:]
            self.assigned[:] = list(map(int, f.readline().split()))[:]
            self.objective = float(f.readline())

    def write_file(self, filename):
        with open(filename, 'w') as f:
            f.write(' '.join(map(str, self.opened)) + '\n')
            f.write(' '.join(map(str, self.assigned)) + '\n')
            f.write(f'{self.objective}\n')

    def __str__(self):
        s = 'Opened: ' + \
            ' '.join([str(i) for i in range(self.flp.num_facilities)
                     if self.opened[i]]) + '\n'
        s += 'Assigned: ' + ' '.join(map(str, self.assigned)) + '\n'
        s += f'Objective: {self.objective}\n'
        return s

    def evaluate(self):
        self.objective = self.flp.opening_cost @ self.opened
        for i in range(self.flp.num_customers):
            self.objective += self.flp.assignment_cost[self.assigned[i], i]
        return self.objective

    def is_valid(self):
        '''Check if the solution is valid'''
        # check if each customer was assigned to a facility
        if np.any(self.assigned < 0) or np.any(self.assigned >= self.flp.num_facilities):
            print('Invalid assignment')
            return False
        # check if the demand of each customer is not greater than the supply of the facility
        demand = np.zeros(self.flp.num_facilities, dtype=int)
        for i in range(self.flp.num_customers):
            demand[self.assigned[i]] += self.flp.demand[i]
        if np.any(demand > self.flp.supply):
            print('Invalid supply')
            return False
        return True


class ConstructionHeuristics:
    '''Construction Heuristics for Facility Location Problem
    '''

    def __init__(self, flp: FLP):
        self.flp = flp
        # indices of customers and facilities for shoffle operations
        self.costumers = np.arange(flp.num_customers)
        self.facilities = np.arange(flp.num_facilities)
        self.total_demand = np.sum(flp.demand)

    def random_assignment_solution(self, sol=None, max_tries=1000):
        '''Create a solution by randomly assigning customers to facilities, 
        the feasibility is checked before returning the solution       
        Parameters:
            max_tries: int (default 1000) maximum number of tries to create a feasible solution
            sol: FLP_Solution or None (default None) a solution to be used, if None a new solution is created
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
            '''
        if sol is None:
            sol = FLP_Solution(self.flp)
        else:
            sol.opened[:] = False
            sol.assigned[:] = -1
        for _ in range(max_tries):
            for i in range(self.flp.num_customers):
                sol.assigned[i] = np.random.randint(self.flp.num_facilities)
            for j in sol.assigned:
                sol.opened[j] = 1
            if sol.is_valid():
                sol.evaluate()
                break
        else:
            return None
        return sol

    def greedy(self, sol=None, rd_opening = False):
        '''Create a solution by a greedy heuristic, 
        each customer is assigned to the facility with the lowest cost, 
        considering the opening cost and the assignment cost, 
        the feasibility is guaranteed by the heuristic    
        Parameters:
            sol: FLP_Solution or None (default None) a solution to be used, if None a new solution is created
            rd_opening: bool (default False) if True the random facilities will be opened before customers assignment
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
            '''
        if sol is None:
            sol = FLP_Solution(self.flp)
        else:
            sol.opened[:] = False
            sol.assigned[:] = -1

        if rd_opening:
            np.random.shuffle(self.facilities)
            supply = 0
            for j in self.facilities:
                sol.opened[j] = True
                supply += self.flp.supply[j]
                if supply >= self.total_demand:
                    break
        capacity = self.flp.supply.copy()
        for _ in range(self.flp.num_customers):
            best = np.inf
            arg_cos = -1
            arg_fac = -1
            for i in self.costumers:
                if sol.assigned[i] >= 0:
                    continue
                for j in self.facilities:
                    if capacity[j] >= self.flp.demand[i]:
                        cost = self.flp.assignment_cost[j, i]
                        if not sol.opened[j]:
                            cost += self.flp.opening_cost[j]
                        if cost < best:
                            best = cost
                            arg_cos = i
                            arg_fac = j
            sol.assigned[arg_cos] = arg_fac
            sol.opened[arg_fac] = True
            capacity[arg_fac] -= self.flp.demand[arg_cos]
        assert sol.is_valid(), 'Invalid solution' # should not happen
        sol.evaluate()
        return sol


#### main ####
if __name__ == '__main__':
    flp = FLP(filename='codes\instances\cap61')
    print(flp)
    ch = ConstructionHeuristics(flp)
    sol = ch.random_assignment_solution()
    print(sol)
    sol = ch.greedy()    
    print(sol)
    sol = ch.greedy(rd_opening=True)
    print(sol)
    best = sol.objective
    for i in range(1000):
        sol = ch.greedy(rd_opening=True,sol=sol)
        if best > sol.objective:
            best = sol.objective
            print(i, sol)
        
