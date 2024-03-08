import numpy as np
import time


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
        # how many costumers are assigned to each facility. facility closed if 0 costumers are assigned
        self.facility_customers_count = np.zeros(flp.num_facilities, dtype=int)
        # facility assigned to each customer
        self.assigned = np.full(flp.num_customers, -1, dtype=int)
        # objective function value
        self.objective = np.inf
        # facility remaining capacity
        self.remaining = flp.supply.copy()
        if filename:
            self.read_file(filename)

    def read_file(self, filename):
        with open(filename, 'r') as f:
            self.facility_customers_count[:] = list(
                map(int, f.readline().split()))[:]
            self.assigned[:] = list(map(int, f.readline().split()))[:]
            self.objective = float(f.readline())

    def write_file(self, filename):
        with open(filename, 'w') as f:
            f.write(' '.join(map(str, self.facility_customers_count)) + '\n')
            f.write(' '.join(map(str, self.assigned)) + '\n')
            f.write(f'{self.objective}\n')

    def __str__(self):
        s = 'Opened: ' + \
            ' '.join(map(str,  np.flatnonzero(
                self.facility_customers_count > 0))) + '\n'
        s += 'Assigned: ' + ' '.join(map(str, self.assigned)) + '\n'
        s += f'Objective: {self.objective}\n'
        return s

    def evaluate(self):
        self.objective = \
            np.sum(np.where(self.facility_customers_count > 0, self.flp.opening_cost, 0)) \
            + np.sum(self.flp.assignment_cost[self.assigned, np.arange(self.flp.num_customers)])
        return self.objective

    def is_valid(self):
        '''Check if the solution is feasible and consistent'''
        # consistency
        if np.any(np.bincount(self.assigned, minlength=self.flp.num_facilities) != self.facility_customers_count):
            print('facility_customers_count inconsistency')
            return False
        remaining = self.flp.supply - np.bincount(
            self.assigned, weights=self.flp.demand, minlength=self.flp.num_facilities)
        if np.any(remaining != self.remaining):
            print('remaining inconsistency')
            return False
        objective = np.sum(np.where(self.facility_customers_count > 0, self.flp.opening_cost, 0)) \
            + np.sum(self.flp.assignment_cost[self.assigned, np.arange(self.flp.num_customers)])
        if not np.isclose(objective, self.objective):
            print('objective inconsistency')
            return False
        # feasibility
        if np.any(remaining < 0):
            print('capacity constraint violated')
            return False
        if np.any(self.assigned < 0) or np.any(self.assigned >= self.flp.num_facilities):
            print('invalid facility assignment')
            return False
        return True

    def reset(self):
        '''Reset the solution to the initial state'''
        self.facility_customers_count[:] = 0
        self.assigned[:] = -1
        self.objective = 0
        self.remaining[:] = self.flp.supply

    def copy_from(self, other):
        '''Copy the solution from other'''
        self.facility_customers_count[:] = other.facility_customers_count
        self.assigned[:] = other.assigned
        self.objective = other.objective
        self.remaining[:] = other.remaining

    def assign(self, i, j):
        '''Assign customer i to facility j, i must not be assigned to any facility yet'''
        if self.assigned[i] >= 0:
            raise ValueError('Customer already assigned')
        if self.remaining[j] < self.flp.demand[i]:
            raise ValueError('Facility capacity would be exceeded')
        self.assigned[i] = j
        self.remaining[j] -= self.flp.demand[i]
        self.facility_customers_count[j] += 1
        if self.facility_customers_count[j] == 1:
            self.objective += self.flp.opening_cost[j]
        self.objective += self.flp.assignment_cost[j, i]

    def unassign(self, i):
        '''Unassign customer i from its facility, i must be assigned to some facility
        Returns the facility index where i was assigned'''
        j = self.assigned[i]
        if j < 0:
            raise ValueError('Customer not assigned')
        self.remaining[j] += self.flp.demand[i]
        self.assigned[i] = -1
        self.facility_customers_count[j] -= 1
        if self.facility_customers_count[j] == 0:
            self.objective -= self.flp.opening_cost[j]
        self.objective -= self.flp.assignment_cost[j, i]
        return j


class BruteForce:

    def __init__(self, flp: FLP):
        self.flp = flp
        self.costumers = np.arange(flp.num_customers)
        self.facilities = np.arange(flp.num_facilities)
        self.stop_time = np.inf

    def tries(self, i, j, best: FLP_Solution, working: FLP_Solution):
        '''Try to assign customer i to facility j and then try to assign the next customer recursively'''
        # stop if time is over
        if time.time() > self.stop_time:
            return

        # if facility j has enough capacity to attend customer i
        if working.remaining[j] >= self.flp.demand[i]:
            # assign customer i to facility j
            working.assign(i, j)
            # if current lower bound is better than best found so far
            if working.objective < best.objective:
                if i < self.flp.num_customers - 1:
                    # sort facilities by assignment cost to the next customer
                    facilities = sorted(
                        self.facilities, key=lambda j: self.flp.assignment_cost[j, i+1])
                    # try next customer
                    for k in facilities:
                        self.tries(i+1, k, best, working)
                elif working.objective < best.objective:
                    # update best solution
                    best.copy_from(working)
                    print('bf', best.objective)
                    assert best.is_valid(), 'Invalid solution'  # should not happen
            # undo assignment
            working.unassign(i)
        return

    def solve(self, timeout=None):
        '''Solve the problem by brute force, trying all possible assignments
        Parameters:
            timeout: int or None (default None) maximum time in seconds to run the algorithm
        Returns:
            FLP_Solution - the best solution found
        '''

        if timeout:
            self.stop_time = time.time() + timeout

        best = FLP_Solution(self.flp)
        best.objective = np.inf
        working = FLP_Solution(self.flp)
        working.reset()

        # sort facilities by assignment cost to the customer 0
        facilities = sorted(
            self.facilities, key=lambda j: self.flp.assignment_cost[j, 0])

        for j in facilities:
            self.tries(0, j, best, working)

        return best


class ConstructionHeuristics:
    '''Construction Heuristics for Facility Location Problem
    '''

    def __init__(self, flp: FLP):
        self.flp = flp
        # indices of customers and facilities for shuffle operations
        self.costumers = np.arange(flp.num_customers)
        self.facilities = np.arange(flp.num_facilities)
        # total demand
        self.total_demand = np.sum(flp.demand)

    def random_assignment_solution(self, max_tries=1000):
        '''Create a solution by randomly assigning customers to facilities, 
        the feasibility is checked before returning the solution       
        Parameters:
            max_tries: int (default 1000) maximum number of tries to create a feasible solution
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
            '''

        best = FLP_Solution(self.flp)
        best.objective = np.inf
        working = FLP_Solution(self.flp)
        for _ in range(max_tries):
            # reset remaining capacity
            working.remaining[:] = self.flp.supply
            for i in range(self.flp.num_customers):
                working.assigned[i] = np.random.randint(
                    self.flp.num_facilities)
                working.remaining[working.assigned[i]] -= self.flp.demand[i]
                # check if the assignment violates capacity constraints
                if working.remaining[working.assigned[i]] < 0:
                    break
            else:
                # all customers were assigned without violating capacity constraints
                working.facility_customers_count[:] = np.bincount(
                    working.assigned, minlength=self.flp.num_facilities)
                working.evaluate()
                assert working.is_valid(), 'Invalid solution'  # should not happen
                if working.objective < best.objective:
                    print('ch', working.objective)
                    best.copy_from(working)
        if best.objective < np.inf:
            return best
        return None

    def greedy(self, rd_opening=False):
        '''Create a solution by a greedy heuristic, 
        each customer is assigned to the facility with the lowest cost, 
        considering the opening cost and the assignment cost, 
        the feasibility is guaranteed by the heuristic    
        Parameters:
            rd_opening: bool (default False) if True the random facilities will be opened before customers assignment
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
            '''
        
        sol = FLP_Solution(self.flp)       

        if rd_opening:
            # open random facilities until the total demand is satisfied
            np.random.shuffle(self.facilities)
            supply = 0
            for j in self.facilities:
                sol.facility_customers_count[j] = 1
                supply += self.flp.supply[j]
                if supply >= self.total_demand:
                    break
        
        for _ in range(self.flp.num_customers):
            best = np.inf
            arg_cos = -1
            arg_fac = -1
            # select the customer with the lowest cost to be assigned
            for i in self.costumers:
                if sol.assigned[i] >= 0:
                    continue
                for j in self.facilities:
                    if sol.remaining[j] >= self.flp.demand[i]:
                        cost = self.flp.assignment_cost[j, i]
                        if sol.facility_customers_count[j] == 0:
                            cost += self.flp.opening_cost[j]
                        if cost < best:
                            best = cost
                            arg_cos = i
                            arg_fac = j
            if arg_cos < 0:
                # no customer can be assigned
                return None
            sol.assigned[arg_cos] = arg_fac
            # mark the facility as opened (not counting the customers yet)
            sol.facility_customers_count[arg_fac] = 1
            sol.remaining[arg_fac] -= self.flp.demand[arg_cos]
        # ajust facility_customers_count
        sol.facility_customers_count[:] = np.bincount(
            sol.assigned, minlength=self.flp.num_facilities)
        # evaluate the solution
        sol.evaluate()
        # check if the solution is valid
        assert sol.is_valid(), 'Invalid solution'  # should not happen
        return sol


class LocalSearch:
    '''Local Search for Facility Location Problem
    '''

    def __init__(self, flp: FLP):
        self.flp = flp
        # indices of customers and facilities for shoffle operations
        self.costumers = np.arange(flp.num_customers)
        # indices of customers and facilities for shoffle operations
        self.facilities = np.arange(flp.num_facilities)

    def two_opt(self, sol: FLP_Solution, shuffle=True, first_improvement=True):
        ''' Try to improve the solution by swapping two customers between two facilities,
        if find a better solution, then sol is updated.
        Parameters:
            sol: FLP_Solution - the solution to be improved
            shuffle: bool (default True) if True the customers and facilities are shuffled before the search
            first_improvement: bool (default True) if True the search stops at the first improvement
        Returns:
            bool - True if the solution was improved, False otherwise            
            '''
        if shuffle:
            np.random.shuffle(self.costumers)
            np.random.shuffle(self.facilities)

        best = 0
        arg_i = -1
        arg_j = -1
        for i in self.costumers:
            for j in self.costumers:
                if i >= j:
                    # avoid duplicate pairs
                    continue
                if sol.assigned[i] == sol.assigned[j]:
                    # same facility
                    continue
                if sol.remaining[sol.assigned[i]] - self.flp.demand[i] < self.flp.demand[j] or sol.remaining[sol.assigned[j]] - self.flp.demand[j] < self.flp.demand[i]:
                    # capacity constraint
                    continue
                delta = self.flp.assignment_cost[sol.assigned[j], i] + self.flp.assignment_cost[sol.assigned[i], j] \
                    - self.flp.assignment_cost[sol.assigned[i], i] - \
                    self.flp.assignment_cost[sol.assigned[j], j]
                if delta < 0:
                    if first_improvement:
                        demand_delta = self.flp.demand[i] - self.flp.demand[j]
                        sol.remaining[sol.assigned[i]] += demand_delta
                        sol.remaining[sol.assigned[j]] -= demand_delta
                        sol.assigned[i], sol.assigned[j] = sol.assigned[j], sol.assigned[i]
                        assert sol.is_valid(), 'Invalid solution'  # should not happen
                        new_obj = sol.objective + delta
                        # should not happen
                        assert np.isclose(
                            new_obj, sol.evaluate()), 'Invalid objective'
                        sol.objective = new_obj
                        return True
                    if delta < best:
                        best = delta
                        arg_i = i
                        arg_j = j
        if arg_i >= 0:
            demand_delta = self.flp.demand[arg_i] - self.flp.demand[arg_j]
            sol.remaining[sol.assigned[arg_i]] += demand_delta
            sol.remaining[sol.assigned[arg_j]] -= demand_delta
            sol.assigned[arg_i], sol.assigned[arg_j] = sol.assigned[arg_j], sol.assigned[arg_i]
            sol.objective += best
            assert sol.is_valid(), 'Invalid solution'
            return True
        return False

    def replace(self, sol: FLP_Solution, shuffle=True, first_improvement=True):
        ''' Try to improve the solution by replacing a customer from one facility to another,
        if find a better solution, then sol is updated.
        Parameters:
            sol: FLP_Solution - the solution to be improved
            shuffle: bool (default True) if True the customers and facilities are shuffled before the search
            first_improvement: bool (default True) if True the search stops at the first improvement
        Returns:
            bool - True if the solution was improved, False otherwise            
            '''
        if shuffle:
            np.random.shuffle(self.costumers)
            np.random.shuffle(self.facilities)

        best = 0
        arg_i = -1
        arg_j = -1
        for i in self.costumers:
            for j in self.facilities:
                if j == sol.assigned[i]:
                    # same facility
                    continue
                if sol.remaining[j] < self.flp.demand[i]:
                    # capacity constraint
                    continue
                delta = self.flp.assignment_cost[j, i] - \
                    self.flp.assignment_cost[sol.assigned[i], i]
                if sol.facility_customers_count[sol.assigned[i]] == 1:
                    # facility will be closed
                    delta -= self.flp.opening_cost[sol.assigned[i]]
                if sol.facility_customers_count[j] == 0:
                    # facility will be opened
                    delta += self.flp.opening_cost[j]
                if delta < 0:
                    if first_improvement:
                        sol.facility_customers_count[j] += 1
                        sol.facility_customers_count[sol.assigned[i]] -= 1
                        sol.remaining[sol.assigned[i]] += self.flp.demand[i]
                        sol.remaining[j] -= self.flp.demand[i]
                        sol.assigned[i] = j
                        assert sol.is_valid(), 'Invalid solution'
                        new_obj = sol.objective + delta
                        assert np.isclose(
                            new_obj, sol.evaluate()), 'Invalid objective'
                        sol.objective = new_obj
                        return True
                    if delta < best:
                        best = delta
                        arg_i = i
                        arg_j = j
        if arg_i >= 0:
            sol.facility_customers_count[arg_j] += 1
            sol.facility_customers_count[sol.assigned[arg_i]] -= 1
            sol.remaining[sol.assigned[arg_i]] += self.flp.demand[arg_i]
            sol.remaining[arg_j] -= self.flp.demand[arg_i]
            sol.assigned[arg_i] = arg_j
            assert sol.is_valid(), 'Invalid solution'
            new_obj = sol.objective + best
            assert np.isclose(new_obj, sol.evaluate()), 'Invalid objective'
            sol.objective = new_obj
            return True
        return False


#### main ####
if __name__ == '__main__':
    flp = FLP(filename='codes/instances/p1')
    print(flp)
    # bf = BruteForce(flp)
    # sol = bf.solve(10)
    # print(sol)

    ch = ConstructionHeuristics(flp)
    # sol = ch.random_assignment_solution(max_tries=1000)
    # print(sol)
    sol = ch.greedy()
    print(sol)
    # sol = ch.greedy(rd_opening=True)
    # print(sol)
    ls = LocalSearch(flp)
    while ls.two_opt(sol, first_improvement=False):
        print('** ', sol)
