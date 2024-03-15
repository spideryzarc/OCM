import numpy as np
import time
from itertools import combinations, product
import cProfile

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
        # s += 'Assignment cost:\n'
        # for i in range(self.num_facilities):
        #     s += ' '.join(map(str, self.assignment_cost[i])) + '\n'
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
                    # print('ch', working.objective)
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
                            best, arg_cos, arg_fac = cost, i, j
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
        # indices of customers for generator operations
        self.costumers = np.arange(flp.num_customers)
        # estimation of how many customers are assigned to each open facility
        c = np.ceil(np.mean(flp.supply) / np.mean(flp.demand))
        print('Estimated number of costumer per facility',c)
        # percentile of the assignment cost to be considered in the greedy score
        q = c/flp.num_customers
        print('Percentile of the assignment cost',q)
        # facilities greedy scores
        score = np.quantile(flp.assignment_cost,q=q,axis=1 ) + flp.opening_cost/ flp.supply
        # print(score)
        self.facilities = np.argsort(score)
        # print(score[self.facilities])

    def two_exchange(self, sol: FLP_Solution, first_improvement=True):
        ''' Try to improve the solution by exchanging two customers between their facilities,
        sol is updated if a better solution is found.
        Parameters:
            sol: FLP_Solution - the solution to be improved            
            first_improvement: bool (default True) if True the search stops at the first improvement
                               if False the search returns the best improvement
        Returns:
            bool - True if the solution was improved, False otherwise            
            '''
        # take flp attributes as local variables to avoid multiple lookups, improving performance
        assignment_cost, demand = self.flp.assignment_cost, self.flp.demand
        # take sol attributes as local variables to avoid multiple lookups, improving performance
        remaining, assigned = sol.remaining, sol.assigned

        # nested function to calculate the cost variation of swapping customers i and j between facilities
        def delta(i:int, j:int):
            fi = assigned[i]
            fj = assigned[j]
            return assignment_cost[fj, i] + assignment_cost[fi, j] \
                - assignment_cost[fi, i] - assignment_cost[fj, j]
                    
        # all pairs of costumers (i,j) that are assigned to different facilities,
        # and the facilities have enough capacity to swap the customers
        pairs = ((i, j) for i, j in combinations(self.costumers, 2)
                 if assigned[i] != assigned[j]
                 and remaining[assigned[i]] + demand[i]  >= demand[j]
                 and remaining[assigned[j]] + demand[j]  >= demand[i])
        if first_improvement:
            # generate only pairs that improve the solution
            imp_pairs = (p for p in pairs if delta(*p) < -1e-6)
            # find first improvement
            try:
                i, j = next(imp_pairs)
            except StopIteration:
                return False
        else:
            # find the best improvement
            i,j = min(pairs, key=lambda p: delta(*p))
        d = delta(i,j)
        if d < -1e-6:
            remaining[assigned[i]] += demand[i] - demand[j]
            remaining[assigned[j]] += demand[j] - demand[i]
            assigned[i], assigned[j] = assigned[j], assigned[i]
            sol.objective += d
            assert sol.is_valid(), 'Invalid solution'
            # print('two_opt', sol.objective)
            return True
        return False

    def replace(self, sol: FLP_Solution, first_improvement=True):
        ''' Try to improve the solution by replacing a customer from one facility to another,
        if find a better solution, then sol is updated.
        Parameters:
            sol: FLP_Solution - the solution to be improved
            first_improvement: bool (default True) if True the search stops at the first improvement
        Returns:
            bool - True if the solution was improved, False otherwise            
            '''
        # take flp attributes as local variables to avoid multiple lookups, improving performance
        assignment_cost, opening_cost, demand = self.flp.assignment_cost, self.flp.opening_cost, self.flp.demand
        # take sol attributes as local variables to avoid multiple lookups, improving performance
        facility_customers_count, remaining, assigned = sol.facility_customers_count, sol.remaining, sol.assigned
        
        # nested function to calculate the cost variation of replacing customer i from facility fi to facility j
        def delta(i:int, j:int)->float:
            fi = assigned[i]
            d = assignment_cost[j,i] - assignment_cost[fi, i]
            if facility_customers_count[fi] == 1:
                # facility will be closed
                d -= opening_cost[fi]
            if facility_customers_count[j] == 0:
                # facility will be opened
                d += opening_cost[j]
            return d
        
        # all pairs (i,b) of costumers x facilities,
        # where i is not currently assigned to b
        # and b has enough capacity to attend i demand 
        pairs = ((i, b) for i, b in product(self.costumers, self.facilities)
                 if remaining[b] >= demand[i] and assigned[i] != b)
        if first_improvement:
            # generate only pairs that improve the solution
            imp_pairs = (p for p in pairs if delta(*p) < -1e-6)
            # find the first improvement
            try:
                i, j = next(imp_pairs)
            except StopIteration:
                return False         
        else:
            # find the best improvement
            i,j = min(pairs, key=lambda p: delta(*p))
        d = delta(i,j)
        if d < -1e-6:
            facility_customers_count[j] += 1
            facility_customers_count[assigned[i]] -= 1
            remaining[assigned[i]] += demand[i]
            remaining[j] -= demand[i]
            assigned[i] = j
            sol.objective += d
            assert sol.is_valid(), 'Invalid solution'
            # print('replace', sol.objective)
            return True            
        return False
    
    def exchange_facilities(self, sol: FLP_Solution, first_improvement=True):
        ''' Try to improve the solution by closing a facility and assigning all its customers to a closed facility,
        if find a better solution, then sol is updated.
        Parameters:
            sol: FLP_Solution - the solution to be improved
            first_improvement: bool (default True) if True the search stops at the first improvement
        Returns:
            bool - True if the solution was improved, False otherwise            
            '''
        # take flp attributes as local variables to avoid multiple lookups, improving performance
        assignment_cost, opening_cost, demand, supply = self.flp.assignment_cost, self.flp.opening_cost, self.flp.demand, self.flp.supply
        # take sol attributes as local variables to avoid multiple lookups, improving performance
        facility_customers_count, remaining, assigned = sol.facility_customers_count, sol.remaining, sol.assigned

        # pre separete costumers by facility
        facility_customers = [np.flatnonzero(assigned == i) for i in range(self.flp.num_facilities)]
        # pre calculate the assignment cost plus opening cost of each facility
        assignment_cost_facility = opening_cost + np.bincount(assigned, weights=assignment_cost[assigned, np.arange(self.flp.num_customers)], minlength=self.flp.num_facilities)

        # nested function to calculate the cost variation of closing facility a and opening facility b, moving all customers from a to b
        def delta(a:int, b:int):
            return  opening_cost[b] + np.sum(assignment_cost[b,facility_customers[a]])-assignment_cost_facility[a]

        # closed_facilities = np.flatnonzero(facility_customers_count==0)
        # opened_facilities = np.flatnonzero(facility_customers_count>0)

        # all pairs of facilities (a,b),
        # where a is opened and b closed and b supports all a costumers
        pairs = ((a,b) for a in self.facilities if facility_customers_count[a] > 0
                for b in self.facilities if facility_customers_count[b] == 0
                and supply[b] >= supply[a] - remaining[a])

        if first_improvement:
            # generate only pairs that improve the solution
            imp_pairs = (p for p in pairs if delta(*p) < -1e-6)
            # find the first improvement
            try:
                a, b = next(imp_pairs)
            except StopIteration:
                return False         
        else:
            # find the best improvement
            a,b = min(pairs, key=lambda p: delta(*p))
        d = delta(a,b)
        if d < -1e-6:
            facility_customers_count[b] = facility_customers_count[a]
            facility_customers_count[a] = 0
            remaining[b] = supply[b] - supply[a] + remaining[a] 
            remaining[a] = supply[a]
            for i in np.flatnonzero(assigned == a):
                assigned[i] = b
            sol.objective += d
            assert sol.is_valid(), 'Invalid solution'
            # print('exchange_facilities', sol.objective)
            return True


        return False




    def VND(self, sol: FLP_Solution, use_first_improvement=True):
        '''Variable Neighborhood Descent, a metaheuristic that combines local search methods,
        the search stops when no improvement is found in any neighborhood
        Parameters:
            sol: FLP_Solution - the initial solution
        Returns:
            bool - True if the solution was improved, False otherwise
            '''
        # if use_first_improvement is True, costumers indices are shuffled to avoid bias
        if use_first_improvement:                
            np.random.shuffle(self.costumers)
        
        any_imp = False
        while False\
                or self.replace(sol,first_improvement=use_first_improvement)\
                or self.two_exchange(sol,first_improvement=use_first_improvement)\
                or self.exchange_facilities(sol,first_improvement=use_first_improvement)\
                :
            any_imp = True
           
        return any_imp


class Metaheuristics:
    '''Metaheuristics for Facility Location Problem
    '''

    def __init__(self, flp: FLP):
        self.flp = flp

    def RMS(self, max_tries=1000):
        '''Randomized Multi-Start, a metaheuristic that combines construction heuristics and local search methods
        Parameters:
            max_tries: int (default 1000) maximum number of tries without improvement
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
            '''
        ch = ConstructionHeuristics(self.flp)
        ls = LocalSearch(self.flp)
        best = None
        ite = 0
        while ite < max_tries:
            ite += 1
            sol = ch.random_assignment_solution(10)
            if sol:
                ls.VND(sol)
                if not best or sol.objective  + 1e-6 < best.objective:
                    best = sol
                    ite = 0
                    print ('rms', best.objective)
        return best
    
    def close_facility(self, sol: FLP_Solution):
        '''Close a facility randomly and try to assign its customers to another facility
        Parameters:
            sol: FLP_Solution - the initial solution           '''     
        f = np.random.choice(np.flatnonzero(sol.facility_customers_count > 0))
        unassigned = np.flatnonzero(sol.assigned == f)
        for i in unassigned:
            sol.unassign(i) 
        #greedy assignment avoiding the closed facility
        np.random.shuffle(unassigned)
        for i in unassigned:
            best = np.inf
            arg_j = -1
            for j in range(self.flp.num_facilities):
                if j!=f and sol.remaining[j] >= self.flp.demand[i]:
                    cost = self.flp.assignment_cost[j, i]
                    if sol.facility_customers_count[j] == 0:
                        cost += self.flp.opening_cost[j]
                    if cost < best:
                        best, arg_j = cost, j
            if arg_j < 0:
                sol.assign(i,f)
            else:
                sol.assign(i, arg_j)
        assert sol.is_valid(), 'Invalid solution'

    def ILS(self, max_tries=1000):
        '''Iterated Local Search, a metaheuristic that combines construction heuristics and local search methods
        Parameters:
            max_tries: int (default 1000) maximum number of tries without improvement
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
            '''
        ch = ConstructionHeuristics(self.flp)
        ls = LocalSearch(self.flp)
        best = ch.random_assignment_solution(100)
        sol = FLP_Solution(self.flp)
        sol.copy_from(best)
        ite = 0
        while ite < max_tries:
            ite += 1
            # sol.copy_from(best)
            self.close_facility(sol)
            if sol:
                ls.VND(sol)
                if  sol.objective  + 1e-6 < best.objective:
                    best.copy_from(sol)
                    ite = 0
                    print ('ils', best.objective)
        return best

class MIP:
    ''' Mixed Integer Programming for Facility Location Problem,
    using the ortools library
    '''

    def __init__(self, flp: FLP):
        self.flp = flp
        self.model = None

    def solve(self, timeout=None):
        '''Solve the problem by MIP, using the ortools library
        Parameters:
            timeout: int or None (default None) maximum time in seconds to run the algorithm
        Returns:
            FLP_Solution - the best solution found
        '''
        from ortools.linear_solver import pywraplp

        self.model = pywraplp.Solver.CreateSolver('SCIP')
        if not self.model:
            raise ValueError('Solver not found')

        x = {}
        for i in range(self.flp.num_customers):
            for j in range(self.flp.num_facilities):
                x[i, j] = self.model.IntVar(0, 1, f'x[{i},{j}]')
        y = {}
        for j in range(self.flp.num_facilities):
            y[j] = self.model.IntVar(0, 1, f'y[{j}]')

        # each customer must be assigned to exactly one facility
        for i in range(self.flp.num_customers):
            self.model.Add(
                sum(x[i, j] for j in range(self.flp.num_facilities)) == 1)

        # capacity constraint
        for j in range(self.flp.num_facilities):
            self.model.Add(
                sum(x[i, j] * self.flp.demand[i] for i in range(self.flp.num_customers)) <= self.flp.supply[j]*y[j])

        # streangthening the capacity constraint
        for i in range(self.flp.num_customers):
            for j in range(self.flp.num_facilities):
                self.model.Add(x[i, j] <= y[j])

        # objective function
        self.model.Minimize(
            sum(self.flp.assignment_cost[j, i] * x[i, j] for i in range(
                self.flp.num_customers) for j in range(self.flp.num_facilities))
            + sum(self.flp.opening_cost[j] * y[j]
                  for j in range(self.flp.num_facilities))
        )

        if timeout:
            self.model.SetTimeLimit(timeout * 1000)

        status = self.model.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            sol = FLP_Solution(self.flp)
            for i in range(self.flp.num_customers):
                for j in range(self.flp.num_facilities):
                    if x[i, j].solution_value() > 0:
                        sol.assigned[i] = j
                        sol.facility_customers_count[j] += 1
                        sol.remaining[j] -= self.flp.demand[i]
            sol.evaluate()
            assert sol.is_valid(), 'Invalid solution'
            return sol
        return None


#### main ####
if __name__ == '__main__':
    flp = FLP(filename='codes/instances/cap61')
    print(flp)
    # bf = BruteForce(flp)
    # sol = bf.solve(10)
    # print(sol)

    mip = MIP(flp)
    sol = mip.solve(60)
    print(sol)

    # ch = ConstructionHeuristics(flp)
    # sol = ch.random_assignment_solution(max_tries=10)
    # print(sol)
    # sol = ch.greedy()
    # print(sol)
    # sol = ch.greedy(rd_opening=True)
    # print(sol)
    # ls = LocalSearch(flp)
    # ls.VND(sol)
    # print(sol)
    #fix random seed
    np.random.seed(0)
    meta = Metaheuristics(flp)
    # sol = meta.RMS(100)
    # cProfile.run('sol = meta.RMS(500)', sort='tottime')
    # sol = meta.ILS(100)
    cProfile.run('sol = meta.ILS(500)', sort='tottime')