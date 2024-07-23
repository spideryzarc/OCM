import numpy as np
import time
from itertools import combinations, product
import cProfile
import os
import heapq
from collections import deque
# from numba import njit

class FLP:
    '''Single Source Capacitated Facility Location Problem (SSCFLP)
    or Warehouse Location Problem (WLP)
    '''

    def __init__(self, **args):
        if 'filename' in args:
            self.filename = args['filename']
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

            self.total_demand = np.sum(self.demand)

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

    def random_assignment_solution(self, max_tries=1):
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
        # take flp attributes as local variables to avoid multiple lookups, improving performance
        assignment_cost, opening_cost, demand, supply = self.flp.assignment_cost, self.flp.opening_cost, self.flp.demand, self.flp.supply
        # create an empty solution
        sol = FLP_Solution(self.flp)
        

        if rd_opening:
            # open random facilities until the total demand is satisfied
            sup = 0
            for j in self.facilities:
                sol.facility_customers_count[j] = 1
                sup += supply[j]
                if sup >= self.total_demand:
                    break

        #nested function to calculate the cost of assigning customer i to facility j
        def cost(i: int, j: int) -> float:
            c = assignment_cost[j, i]
            if sol.facility_customers_count[j] == 0:
                c += opening_cost[j]
            return c
        costumers = list(self.costumers)
        for _ in range(self.flp.num_customers):
            try:
                i,j = min(((i,j) for i in costumers 
                            for j in self.facilities if sol.remaining[j] >= demand[i]),
                            key=lambda p: cost(*p))
            except ValueError:
                return None
            sol.assigned[i] = j
            # mark the facility as opened (not counting the customers yet)
            sol.facility_customers_count[j] = 1
            sol.remaining[j] -= demand[i]
            costumers.remove(i)
        
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
        c = np.floor(np.mean(flp.supply) / np.mean(flp.demand))
        if __debug__: print('Estimated number of costumer per facility', c)
        # percentile of the assignment cost to be considered in the greedy score
        q = c/flp.num_customers
        if __debug__: print('Percentile of the assignment cost', q)
        # facilities greedy scores
        score = np.quantile(flp.assignment_cost, q=q, axis=1) + \
            flp.opening_cost / flp.supply
        # print(score)
        self.facilities = np.argsort(score)
        # self.facilities = np.arange(flp.num_facilities)
        # print(score[self.facilities])
        # the closest facility to each customer
        self.closest_facility = np.argmin(flp.assignment_cost, axis=0)

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
        
        def delta(i: int, j: int):
            fi = assigned[i]
            fj = assigned[j]
            return assignment_cost[fj, i] + assignment_cost[fi, j] \
                - assignment_cost[fi, i] - assignment_cost[fj, j]
        
        # nested function to commit the swap
        def commit(i: int, j: int, d: float):
            fi = assigned[i]
            fj = assigned[j]
            remaining[fi] += demand[i] - demand[j]
            remaining[fj] += demand[j] - demand[i]
            assigned[i], assigned[j] = fj, fi
            sol.objective += d
            # print('two_exchange', sol.objective)
            assert sol.is_valid(), 'Invalid solution'

        # all pairs of costumers (i,j) that are assigned to different facilities,
        # and the facilities have enough capacity to swap the customers
        # and at least one of the costumers is assigned to a facility that is not the its closest
        pairs = ((i, j) for i, j in combinations(self.costumers, 2)
                 if assigned[i] != assigned[j]
                 and (self.closest_facility[i] != assigned[i] or self.closest_facility[j] != assigned[j]) 
                 and remaining[assigned[i]] + demand[i] >= demand[j]
                 and remaining[assigned[j]] + demand[j] >= demand[i])
        if first_improvement:
            # commit swap if it improves the solution while searching
            imp = False
            for i, j in pairs:
                d = delta(i, j)
                if d < -1e-6:
                    commit(i, j, d)
                    imp = True
            return imp
        else:
            # find the best improvement
            i, j = min(pairs, key=lambda p: delta(*p))
            d = delta(i, j)
            if d < -1e-6:
                commit(i, j, d)
                return True
        return False

    
    def replace(self, sol: FLP_Solution, first_improvement=True)->bool:
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
        def delta(i: int, j: int) -> float:
            fi = assigned[i]
            d = assignment_cost[j, i] - assignment_cost[fi, i]
            if facility_customers_count[fi] == 1:
                # facility will be closed
                d -= opening_cost[fi]
            if facility_customers_count[j] == 0:
                # facility will be opened
                d += opening_cost[j]
            return d
        
        #nested function to commit the replacement
        def commit(i: int, j: int, delta: float):
            fi = assigned[i]
            facility_customers_count[fi] -= 1
            facility_customers_count[j] += 1
            remaining[fi] += demand[i]
            remaining[j] -= demand[i]
            assigned[i] = j
            sol.objective += delta
            assert sol.is_valid(), 'Invalid solution'

        # all pairs (i,b) of costumers x facilities,
        # where i is not currently assigned to b
        # and b has enough capacity to attend i demand
        # pairs = ((i, b) for i, b in product(self.costumers, self.facilities)
        #          if remaining[b] >= demand[i] and assigned[i] != b)
        # optimize the search by considering only the costumers that are assigned to a facility that is not the closest
        pairs = ((i, b) for i in self.costumers 
                    if self.closest_facility[i] != assigned[i] or sol.facility_customers_count[assigned[i]] == 1
                    for b in self.facilities if remaining[b] >= demand[i] and assigned[i] != b)
        if first_improvement:
            imp = False
            # commit replacement if it improves the solution while searching
            for i, j in pairs:
                d = delta(i, j)
                if d < -1e-6:
                    commit(i, j, d)
                    imp = True
            return imp
        else:
            # find the best improvement
            i, j = min(pairs, key=lambda p: delta(*p))
            d = delta(i, j)
            if d < -1e-6:
                commit(i, j, d)
                return True
        return False

    def exchange_facilities(self, sol: FLP_Solution, first_improvement=True)->bool:
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
        facility_customers = [np.flatnonzero(
            assigned == i) for i in range(self.flp.num_facilities)]
        # pre calculate the assignment cost plus opening cost of each facility
        assignment_cost_facility = opening_cost + \
            np.bincount(assigned, weights=assignment_cost[assigned, np.arange(
                self.flp.num_customers)], minlength=self.flp.num_facilities)

        # nested function to calculate the cost variation of closing facility a and opening facility b, moving all customers from a to b
        def delta(a: int, b: int):
            return opening_cost[b] + np.sum(assignment_cost[b, facility_customers[a]])-assignment_cost_facility[a]

        # all pairs of facilities (a,b),
        # where a is opened and b closed and b supports all a costumers
        pairs = ((a, b) for b in self.facilities if facility_customers_count[b] == 0
                 for a in self.facilities if facility_customers_count[a] > 0
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
            try:
                a, b = min(pairs, key=lambda p: delta(*p))
            except ValueError:
                return False
        d = delta(a, b)
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

    def VND(self, sol: FLP_Solution, first_imp=True):
        '''Variable Neighborhood Descent, a metaheuristic that combines local search methods,
        the search stops when no improvement is found in any neighborhood
        Parameters:
            sol: FLP_Solution - the initial solution
        Returns:
            bool - True if the solution was improved, False otherwise
            '''
        # if use_first_improvement is True, costumers indices are shuffled to avoid bias
        if first_imp:
            np.random.shuffle(self.costumers)

        any_imp = False
        while False\
                or self.replace(sol, first_imp)\
                or self.exchange_facilities(sol, first_imp)\
                or self.two_exchange(sol, first_imp):
            any_imp = True

        return any_imp


class Metaheuristics:
    '''Metaheuristics for Facility Location Problem
    '''

    def __init__(self):
        pass
    @staticmethod
    def RMS(flp, max_tries=1000, first_imp=True):
        '''Randomized Multi-Start, a metaheuristic that combines construction heuristics and local search methods
        Parameters:
            flp: FLP - the problem to be solved
            max_tries: int (default 1000) maximum number of tries without improvement
            first_imp: bool (default True) if True the local search will use first improvement
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
            '''
        ch = ConstructionHeuristics(flp)
        ls = LocalSearch(flp)
        best = None
        ite = 0
        while ite < max_tries:
            ite += 1
            sol = ch.random_assignment_solution()
            if sol:
                ls.VND(sol, first_imp)
                if not best or sol.objective + 1e-6 < best.objective:
                    best = sol
                    ite = 0
                    if __debug__: print('rms', best.objective)
        return best

    @staticmethod
    def close_facility(sol: FLP_Solution, k: int = 1):
        '''Close a facility randomly and try to assign its customers to others ones
        Parameters:
            sol: FLP_Solution - the initial solution   
            k: int (default 1) number of facilities to be closed        
        '''
        # take flp attributes as local variables to avoid multiple lookups, improving performance
        assignment_cost, opening_cost, demand = sol.flp.assignment_cost, sol.flp.opening_cost, sol.flp.demand
        # take sol attributes as local variables to avoid multiple lookups, improving performance
        facility_customers_count, remaining, assigned = sol.facility_customers_count, sol.remaining, sol.assigned
        # choose k facilities to be closed
        facilities = np.random.choice(np.flatnonzero(
            sol.facility_customers_count > 0), k, replace=False)
        # list of costumers assigned to the closed facilities
        costumers = np.flatnonzero(np.isin(assigned, facilities))
        # unassign all costumers from the closed facilities
        facility_customers_count[facilities] = 0
        remaining[facilities] = sol.flp.supply[facilities]
        sol.objective -= np.sum(opening_cost[facilities])\
                        + np.sum(assignment_cost[assigned[costumers], costumers])
        assigned[costumers] = -1
        # nested function to calculate the cost of assigning customer i to facility j

        def cost(i: int, j: int) -> float:
            c = assignment_cost[j, i]
            if facility_customers_count[j] == 0:
                c += opening_cost[j]
            return c

        # shuffle costumers to avoid bias
        np.random.shuffle(costumers)
        # greedy assignment avoiding the closed facility
        for i in costumers:
            try:
                j = min((j for j in range(sol.flp.num_facilities)
                         if j not in facilities and remaining[j] >= demand[i]),
                        key=lambda j: cost(i, j))
            except ValueError:
                # if is not possible to assign i to any other facility
                # use the closed one
                j = min((j for j in facilities if remaining[j] >= demand[i]),
                        key=lambda j: cost(i, j))
            sol.assign(i, j)
        assert sol.is_valid(), 'Invalid solution'

    @staticmethod
    def ILS(flp, max_tries=1000, first_imp=True):
        '''Iterated Local Search, a metaheuristic that perturbs the local minimum solution and then applies local search methods
        Parameters:
            flp: FLP - the problem to be solved
            max_tries: int (default 1000) maximum number of tries without improvement
            first_imp: bool (default True) if True the local search will use first improvement
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
            '''
        ch = ConstructionHeuristics(flp)
        ls = LocalSearch(flp)
        best = ch.greedy(True)
        ls.VND(best, first_imp)
        if __debug__: print('ils', best.objective)
        sol = FLP_Solution(flp)
        sol.copy_from(best)
        ite = 0
        while ite < max_tries:
            ite += 1
            # perturb the local minimum solution
            Metaheuristics.close_facility(sol)
            ls.VND(sol, first_imp)
            # print('vnd', sol.objective)
            if sol.objective + 1e-6 < best.objective:
                best.copy_from(sol)
                ite = 0
                if __debug__: print('ils', best.objective)
        return best

    @staticmethod
    def VNS(flp, max_tries=1000, first_imp=True):
        '''Variable Neighborhood Search, similar to ILS, but the perturbation is not fixed and changes during the search
        to explore different neighborhoods
        Parameters:
            flp: FLP - the problem to be solved
            max_tries: int (default 1000) maximum number of tries without improvement
            first_imp: bool (default True) if True the local search will use first improvement
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
            '''
        ch = ConstructionHeuristics(flp)
        ls = LocalSearch(flp)
        best = None
        while not best:
            best = ch.greedy(True)
        ls.VND(best)
        sol = FLP_Solution(flp)
        sol.copy_from(best)
        if __debug__: print('vns', best.objective)
        ite = 0
        k = 1
        k_max = np.count_nonzero(best.facility_customers_count)//2
        while ite < max_tries:
            ite += 1
            last_objective = sol.objective
            # perturb the local minimum solution
            Metaheuristics.close_facility(sol, k)
            ls.VND(sol, first_imp)
            # change the perturbation if it does not alter the current solution
            if np.isclose(sol.objective, last_objective):
                k = k+1 if k < k_max else 1
                # print('change k', k)
            else:
                k = k-1 if k > 1 else 1
            if sol.objective + 1e-6 < best.objective:
                best.copy_from(sol)
                ite = 0
                if __debug__: print('vns', best.objective)
        return best
    
    @staticmethod
    def GRASP(flp: FLP, max_tries=1000, first_imp=True, K=10):
        '''Greedy Randomized Adaptive Search Procedure, a metaheuristic that combines construction heuristics and local search methods
        Parameters:
            flp: FLP - the problem to be solved
            max_tries: int (default 1000) maximum number of tries without improvement
            first_imp: bool (default True) if True the local search will use first improvement
            K: int (default 10) number of candidates to be considered in the greedy construction
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
            '''
        sol = FLP_Solution(flp)
        best = FLP_Solution(flp)
        ls = LocalSearch(flp)
        costumers = np.arange(flp.num_customers)
        #take flp attributes as local variables to avoid multiple lookups, improving performance
        assignment_cost, opening_cost, demand, supply, total_demand, num_facilities  = \
            flp.assignment_cost, flp.opening_cost, flp.demand, flp.supply, flp.total_demand, flp.num_facilities
        def greedy_rd():
        #generate a greedy solution with randomized decisions
            sol.reset()
            #shuffle costumers to avoid bias
            np.random.shuffle(costumers)
            # current opening cost payed
            opening = 0
            # Open facilities and assign customers to its closest facility when facility is already opened
            for c in costumers:
                #closest facility to customer c
                f = ls.closest_facility[c]
                if sol.remaining[f] < demand[c]:
                    continue
                if sol.facility_customers_count[f]:
                    # if the facility is already opened, assign the customer
                        sol.assigned[c] = f
                        sol.remaining[f] -= demand[c]
                elif opening+opening_cost[f] <= best_opening_cost:
                    # if the facility is closed and the opening cost is affordable, open the facility
                    sol.assigned[c] = f
                    sol.facility_customers_count[f] = 1
                    sol.remaining[f] -= demand[c]
                    opening += opening_cost[f]
            #end_for
                    
            #nested function to calculate the cost of assigning customer i to facility j
            def delta(i: int, j: int) -> float:
                c = assignment_cost[j, i]
                if sol.facility_customers_count[j] == 0:
                    c += opening_cost[j]
                return c
            #end_delta
            #costumers to be assigned
            costumers_to_assign = np.flatnonzero(sol.assigned == -1)
            #assign the remaining costumers using a greedy randomized construction
            for _ in range(len(costumers_to_assign)):
                pairs = ((i, j) for i in costumers_to_assign if sol.assigned[i]==-1 
                            for j in range(num_facilities)
                            if sol.remaining[j] >= demand[i])
                candidates = []
                for i,j in pairs:
                    #retain the K best candidates
                    heapq.heappush(candidates, (-delta(i,j), i, j))
                    if len(candidates) > K:
                        heapq.heappop(candidates)
                        #print(candidates)
                #pick a random candidate
                _ , i, j = candidates[np.random.randint(len(candidates))]
                #assign the customer
                sol.assigned[i] = j
                sol.facility_customers_count[j] = 1
                sol.remaining[j] -= demand[i]
            # ajust sol attributes
            sol.facility_customers_count[:] = np.bincount(
                sol.assigned, minlength=num_facilities)
            sol.evaluate()
            assert sol.is_valid(), 'Invalid solution'
        #end_greedy_rd
        
        # GRASP main loop
        ch = ConstructionHeuristics(flp)
        best = ch.greedy(True)
        ls.VND(best, first_imp)
        # amount of open cost payed in the best solution available
        best_opening_cost = np.sum(opening_cost[np.flatnonzero(best.facility_customers_count)])
            
        ite = 0
        while ite < max_tries:
            ite += 1
            greedy_rd()
            ls.VND(sol, first_imp)
            #print(sol.objective)
            if sol.objective + 1e-6 < best.objective:
                best.copy_from(sol)
                # amount of open cost payed in the best solution available
                best_opening_cost = np.sum(opening_cost[np.flatnonzero(best.facility_customers_count)])
                ite = 0
                if __debug__: print('grasp', best.objective)
        return best
    #end_GRASP
    
    @staticmethod
    def GLS(flp:FLP, max_tries=1000, first_imp=True, alpha=0.8, beta=1.1):
        '''Guided Local Search, a metaheuristic that changes the objective function to guide the search
        Parameters:
            flp: FLP - the problem to be solved
            max_tries: int (default 1000) maximum number of tries without improvement
            first_imp: bool (default True) if True the local search will use first improvement
            alpha: float (default 0.8) weight of the original costs in the perturbed costs
            beta: float (default 1.1) factor to perturb the assignment costs
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
            '''
        
        original_assignment_cost = flp.assignment_cost.copy()
        #nested function to calculate the original cost of a solution
        def cost_original(sol: FLP_Solution)->float:
            return np.sum(original_assignment_cost[sol.assigned, np.arange(flp.num_customers)]) + \
                np.sum(flp.opening_cost[np.flatnonzero(sol.facility_customers_count)])
        #take flp attributes as local variables to avoid multiple lookups, improving performance
        assignment_cost = flp.assignment_cost
        #Apply the perturbation to the costs    
        def perturb_costs(sol: FLP_Solution)->None:
            #restore parcially the original costs
            assignment_cost[:] = original_assignment_cost*(1-alpha) + assignment_cost*alpha
            #perturb the assignment cost for each customer assigned to a facility 
            for i in range(flp.num_customers):
                assignment_cost[sol.assigned[i], i] *= np.random.uniform(1, beta)
            #recalculate the objective function
            sol.evaluate()
        #end_perturb_costs
        ch = ConstructionHeuristics(flp)
        ls = LocalSearch(flp)
        best = ch.greedy(True)
        ls.VND(best, first_imp)
        if __debug__: print('gls', best.objective)
        sol = FLP_Solution(flp)
        sol.copy_from(best)
        tmp = FLP_Solution(flp)
        ite = 0
        while ite < max_tries:
            ite += 1
            # perturb costs
            perturb_costs(sol)
            ls.VND(sol, first_imp)
            #swap assignment costs, return to the original costs
            flp.assignment_cost, original_assignment_cost = original_assignment_cost, flp.assignment_cost
            #apply VND from the perturbed solution under the original costs in a copy of the solution
            tmp.copy_from(sol)
            tmp.evaluate()
            ls.VND(tmp, first_imp)
            if tmp.objective + 1e-6 < best.objective:
                #restore the original costs
                best.copy_from(tmp)
                sol.copy_from(tmp)
                ite = 0
                if __debug__: print('gls', best.objective)            
            #swap assignment costs, return to the perturbed costs
            flp.assignment_cost, original_assignment_cost = original_assignment_cost, flp.assignment_cost
        #restore the original costs before return
        flp.assignment_cost[:] = original_assignment_cost
        return best
    @staticmethod
    def Tabu(flp: FLP, max_tries=1000, first_imp=True, tenure=10):
        '''Tabu Search, a metaheuristic that avoids revisiting the same solutions
        Parameters:
            flp: FLP - the problem to be solved
            max_tries: int (default 1000) maximum number of tries without improvement
            first_imp: bool (default True) if True the local search will use first improvement
            tenure: int (default 10) number of iterations that a rule is considered tabu
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
            '''
        original_assignment_cost = flp.assignment_cost.copy()
        tabu_list = deque(maxlen=tenure+1)
        def add_tabu(sol: FLP_Solution):
            #choose a pair (i,j) in sol to be tabu
            i = np.random.randint(flp.num_customers)
            j = sol.assigned[i]
            # remove client i from facility j
            sol.objective-= flp.assignment_cost[j, i]
            sol.assigned[i] = -1
            sol.remaining[j] += flp.demand[i]
            sol.facility_customers_count[j] -= 1
            if sol.facility_customers_count[j] == 0:
                sol.objective -= flp.opening_cost[j]
            #update the assignment cost
            flp.assignment_cost[j, i] += 1e6
            #add the pair to the end of the tabu list
            tabu_list.append((i,j))
            #pop the oldest pair if the list is full
            if len(tabu_list) > tenure:
                pi,pj = tabu_list.popleft()
                flp.assignment_cost[pj, pi] -= 1e6
                if sol.assigned[pi] == pj:
                    sol.objective -= 1e6
            #assign client i to a cheaper facility
            facilities = np.flatnonzero(sol.remaining >= flp.demand[i])
            j = min(facilities, key=lambda j: flp.assignment_cost[j, i] + (0 if sol.facility_customers_count[j] > 0 else flp.opening_cost[j]))
            sol.assigned[i] = j
            sol.objective += flp.assignment_cost[j, i]
            sol.remaining[j] -= flp.demand[i]
            sol.facility_customers_count[j] += 1
            if sol.facility_customers_count[j] == 1:
                sol.objective += flp.opening_cost[j]      
        #end_add_tabu
        def reset_tabu():
            #restore the original costs
            flp.assignment_cost[:] = original_assignment_cost
            tabu_list.clear()
        #end_reset_tabu        
                    
        ch = ConstructionHeuristics(flp)
        ls = LocalSearch(flp)
        best = ch.greedy(True)
        ls.VND(best, first_imp)
        if __debug__: print('TABU', best.objective)
        sol = FLP_Solution(flp)
        sol.copy_from(best)
        ite = 0
        tmp = FLP_Solution(flp)
        while ite < max_tries:
            ite += 1
            sol.copy_from(best)
            sol.evaluate()
            add_tabu(sol)
            Metaheuristics.close_facility(sol,3)
            ls.VND(sol, first_imp)
            if sol.objective > best.objective*1.05:
                reset_tabu()
                ls.VND(sol,first_imp)
            if sol.objective + 1e-6 < best.objective:
                reset_tabu()
                ls.VND(sol, first_imp)
                best.copy_from(sol)
                ite = 0
                if __debug__: print('TABU', best.objective)            
        #restore the original costs before return
        flp.assignment_cost[:] = original_assignment_cost
        return best
    
    
    @staticmethod
    def SA(flp: FLP, max_tries=1000, first_imp=True, T0=1e3, alpha=0.99):
        '''Simulated Annealing, a metaheuristic that accepts worse solutions with a probability that decreases with time
        Parameters:
            flp: FLP - the problem to be solved
            max_tries: int (default 1000) maximum number of tries without improvement
            first_imp: bool (default True) if True the local search will use first improvement
            T0: float (default 1e6) initial temperature
            alpha: float (default 0.99) cooling factor
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
            '''
        ch = ConstructionHeuristics(flp)
        best = ch.greedy(True) 
        ls = LocalSearch(flp)
        ls.VND(best, first_imp) 
        current = FLP_Solution(flp)
        current.copy_from(best)
        sol = FLP_Solution(flp)
        ite = 0
        print('SA', best.objective)
        while ite < max_tries:
            ite += 1
            sol.copy_from(current)
            Metaheuristics.close_facility(sol,3)
            ls.VND(sol, first_imp)
            delta = (sol.objective - current.objective)/best.objective
            if delta < -1e-3 or np.random.rand() < np.exp(-delta/T0):
                current.copy_from(sol)
                # print('---', current.objective)
                if current.objective+1e-6 < best.objective:
                    best.copy_from(current)
                    ite = 0
                    if __debug__: print('SA', best.objective)
            T0 *= alpha
    #end SA
    
    @staticmethod
    def DEA(flp: FLP, max_tries=1000, first_imp=True, N=1000, K=100, alpha=0.5,elitist = True):
        ''' Distribution Estimation Algorithm, a metaheuristic that uses a probabilistic model to generate new solutions
        Parameters:
            flp: FLP - the problem to be solved
            max_tries: int (default 1000) maximum number of generations without improvement
            first_imp: bool (default True) if True the local search will use first improvement
            N: int (default 1000) number of solutions in the population
            K: int (default 100) number of solutions used to generate a distribution
            alpha: float (default 0.5) weight of the marginal distribution in the new distribution
            elitist: bool (default True) if True the best solution is always kept
        Returns:
            FLP_Solution or None - a feasible solution or None if it was not possible to create a feasible solution
        '''
        
        ls = LocalSearch(flp)
        # probability distribution of facilities initialization with uniform distribution
        D =  np.ones(flp.num_facilities)/flp.num_facilities
        
        best= None
        
        def generate_solution(D: np.ndarray)->FLP_Solution:
            '''Generate a new solution based on the probability distribution D
            Parameters:
                D: np.ndarray - the probability distribution of facilities
            Returns:
                FLP_Solution - a new solution
            '''
            sol = FLP_Solution(flp)
            sol.objective = 0
            # for _ in range(100):
            #     sol.assigned = np.random.choice(flp.num_facilities, flp.num_customers, p=D)
            #     sol.remaining = flp.supply - np.bincount(sol.assigned, weights=flp.demand, minlength=flp.num_facilities)
            #     if np.all(sol.remaining >= 0):                    
            #         break
            # else:
            #     # print('Invalid solution')
            #     return None
            
            assigned_customers = 0
            # greedy assignment
            while assigned_customers < flp.num_customers:
                # choose a facility to assign the next customer
                j = np.random.choice(flp.num_facilities, p=D)
                while True:
                    # generate a list of customers that can be assigned to facility j
                    customers = (i for i in range(flp.num_customers) if sol.assigned[i] == -1 and sol.remaining[j] >= flp.demand[i])
                    # find the best customer to assign to facility j
                    try:
                        i = min(customers, key=lambda i: flp.assignment_cost[j, i])
                    except ValueError:
                        # if there is no customer to be assigned to facility j go to the next facility
                        break
                    # assign customer i to facility j
                    sol.assigned[i] = j
                    sol.facility_customers_count[j] += 1
                    sol.remaining[j] -= flp.demand[i]
                    sol.objective += flp.assignment_cost[j, i]
                    # if the facility was closed, add the opening cost
                    if sol.facility_customers_count[j] == 1:
                        sol.objective += flp.opening_cost[j]
                    assigned_customers += 1
                # end_while

            # sol.facility_customers_count = np.bincount(sol.assigned, minlength=flp.num_facilities)
            # sol.evaluate()
            assert sol.is_valid(), 'Invalid solution'
            ls.VND(sol, first_imp)
            return sol
        
        def maginal_distribution(pop: list[FLP_Solution])->np.ndarray:
            '''Calculate the marginal distribution of facilities
            Parameters:
                pop: list[FLP_Solution] - the population of solutions
            Returns:
                np.ndarray - the marginal distribution of facilities
            '''
            # calculate the frequency of each facility in the population
            freq = np.zeros(flp.num_facilities)
            for sol in pop:
                freq += sol.facility_customers_count
            # normalize the frequency
            freq /= np.sum(freq)
            return freq
        # main loop
        ite = 0
        while ite < max_tries:
            pop = [generate_solution(D) for _ in range(N)]
            #remover None values
            pop = [sol for sol in pop if sol]
            pop.sort(key=lambda s: s.objective)
            if not best or pop[0].objective + 1e-3 < best.objective:
                best = pop[0]
                ite = 0
                if __debug__: print('DEA', best.objective)
            else:
                ite += 1
                
            # update the probability distribution
            pop = pop[:K]
            if elitist and best.objective < pop[0].objective:
                pop.append(best)
            # print(' '.join(f'{s.objective:.2f}' for s in pop))
            M = maginal_distribution(pop)
            D = alpha*M + (1-alpha)*D  
            # print(D)
            # end main loop
        return best
    #end DEA
    
    @staticmethod
    def GA(flp: FLP, max_tries=10, first_imp=True, N=100, M=10,K=5 , elite=True, mutation_rate=0.1):
        '''Genetic Algorithm, a metaheuristic that uses genetic operators to evolve the population
        Parameters:
            flp: FLP - the problem to be solved
            max_tries: int (default 10) maximum number of generations without improvement
            first_imp: bool (default True) if True the local search will use first improvement
            N: int (default 100) number of solutions in the population
            M: int (default 10) number of solutions to be selected for crossover
            K: int (default 5) number of solutions per tournament
            elite: bool (default True) if True the best solution is always kept
            mutation_rate: float (default 0.1) probability of a mutation'''
            
        ls = LocalSearch(flp)
        #nested function to generate a new solution by crossover
        def crossover(sol1: FLP_Solution, sol2: FLP_Solution)->FLP_Solution:
            '''Generate a new solution by crossover
            Parameters:
                sol1: FLP_Solution - the first parent
                sol2: FLP_Solution - the second parent
            Returns:
                FLP_Solution - the child solution
            '''
            child = FLP_Solution(flp)
            for _ in range(100):
                child.reset()
                child.assigned = np.where(np.random.rand(flp.num_customers) < 0.5, sol1.assigned, sol2.assigned)
                child.remaining = flp.supply - np.bincount(child.assigned, weights=flp.demand, minlength=flp.num_facilities)
                if np.all(child.remaining >= 0):
                    break
            else:
                print('No valid child')
                return None
            child.facility_customers_count = np.bincount(child.assigned, minlength=flp.num_facilities)
            child.evaluate()
            assert child.is_valid(), 'Invalid solution'
            ls.VND(child, first_imp)
            return child
        #end_crossover
        
        #nested function to generate a new solution by mutation
        def mutation(sol: FLP_Solution)->FLP_Solution:
            '''Generate a new solution by mutation
            Parameters:
                sol: FLP_Solution - the parent solution
            Returns:
                FLP_Solution - the child solution
            '''
            child = FLP_Solution(flp)
            for _ in range(100):
                child.reset()
                child.assigned[:] = sol.assigned
                child.remaining[:] = sol.remaining
                child.facility_customers_count[:] = sol.facility_customers_count
                i = np.random.randint(flp.num_customers)
                j = np.random.choice(np.flatnonzero(child.remaining >= flp.demand[i]))
                child.remaining[child.assigned[i]] += flp.demand[i]
                child.facility_customers_count[child.assigned[i]] -= 1
                child.assigned[i] = j
                child.remaining[j] -= flp.demand[i]
                child.facility_customers_count[j] += 1
                child.evaluate()
                if child.is_valid():
                    break
            else:
                print('No valid child')
                return None
            assert child.is_valid(), 'Invalid solution'
            return child
        #end_mutation
        
        # initialize the population
        ch = ConstructionHeuristics(flp)
        pop = [ch.greedy(rd_opening=True) for _ in range(N)]
        #remove None values
        pop = [sol for sol in pop if sol]
        for sol in pop:
            ls.VND(sol, first_imp)
        best = min(pop, key=lambda s: s.objective)
        if __debug__: print('GA', best.objective)
        #main loop
        ite = 0
        while ite < max_tries:
            selected = []
            for _ in range(M):
                tournament = np.random.choice(pop, K, replace=False)
                champion = min(tournament, key=lambda s: s.objective)
                selected.append(champion)
                pop.remove(champion)
            if elite and best not in selected:
                selected.append(best)
            pop = selected
            #crossover
            for _ in range(N-M):
                sol1, sol2 = np.random.choice(pop, 2, replace=False)
                child = crossover(sol1, sol2)
                if child:
                    pop.append(child)
            #mutation
            for _ in range(int(N*mutation_rate)):
                sol = np.random.choice(pop)
                child = mutation(sol)
                if child:
                    pop.append(child)
            nb = min(pop, key=lambda s: s.objective)
            if nb.objective + 1e-6 < best.objective:
                best = nb
                ite = 0
                if __debug__: print('GA', best.objective)
            ite += 1
        return best
    #end_GA
            
                
             
            
    
#end_Metaheuristics

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


class Benchmark:
    """Benchmark for Single Source Capacitated Facility Location Problem
    """

    def __init__(self):
        # directory with the instances
        self.dir = 'codes/instances'
        # time limit in seconds for each algorithm
        self.timeout = 60
        # output file
        self.output = 'benchmark.csv'
        #pair of algorithm and list of parameters 
        #ex.: [(meta.RMS, [{'max_tries':10, 'first_imp':True}, {'max_tries':100, 'first_imp':False}])]
        self.methods = []
        #iterator of random seeds
        self.seeds = [7,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97][:10]
        pass

    def add_method(self, method, param_generator):
        self.methods.append((method, param_generator))

    def run(self, run_by='method'):
        '''Run the benchmark
        '''

        #alert if asserts are enabled
        if __debug__: print('\n\n############# Assertions are enabled #########\n\n') 

        #generate FLP instances from directory
        def instances(dir:str):
            for filename in os.listdir(dir):
                try:
                    yield FLP(filename=f'{dir}/{filename}')
                except:
                    pass

        if run_by == 'method':
            runs= ((m, p, f) for m, p in self.methods for f in instances(self.dir))
        else:
            runs= ((m, p, f) for f in instances(self.dir) for m, p in self.methods)

        param_names = set()
        for m,p in self.methods:
            for args in p:
                param_names.update(args.keys())

        param_names = list(param_names)


        with open(self.output, 'w') as f:
            f.write('method,instance,seed,objective,time,')
            f.write(','.join(param_names))
            f.write('\n')
            for method, param_generator, flp in runs:
                for args in param_generator:
                    for s in self.seeds:
                        np.random.seed(s)
                        start = time.time()
                        sol = method(flp, **args)
                        elapsed = time.time() - start
                        instance_name = os.path.basename(flp.filename)
                        obj = format(sol.objective, '.2f') if sol else ''
                        line = f'{method.__name__},{instance_name},{s},{obj},{elapsed:.2f},'
                        f.write(line)
                        params = ','.join([str(args.get(p, '')) for p in param_names])
                        f.write(params)
                        f.write('\n')
                        print(line,params, sep='')
                    f.flush()
        print('Benchmark finished')
        

        
    

#### main ####
if __name__ == '__main__':
    flp = FLP(filename='codes/instances/cap61')
    # flp = FLP(filename='codes/instances/p1')
    # print(flp)
    # bf = BruteForce(flp)
    # sol = bf.solve(10)
    # print(sol)

    # mip = MIP(flp)
    # sol = mip.solve(60)
    # print(sol)

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
    # fix random seed
    # np.random.seed(0)
    meta = Metaheuristics()
    # sol = meta.RMS(100)
    # cProfile.run('sol = meta.RMS(500)', sort='tottime')
    # sol = meta.ILS(1000)
    # cProfile.run('sol = meta.ILS(flp,500)', sort='tottime')
    # cProfile.run('sol = meta.GRASP(flp,100, False, 5)', sort='tottime')
    # sol = meta.VNS(flp,1000)
    # meta.GRASP(flp,100, False, 2)
    # meta.GLS(flp,100,alpha=0.7, beta=1.2)
    # meta.Tabu(flp,100,tenure=30)
    # meta.DEA(flp,max_tries=10,N=10,K=3,alpha=0.5)
    meta.GA(flp,N=10,M=5,K=2,elite=True,mutation_rate=0.1, max_tries=100)
    # meta.SA(flp,100)
    bm= Benchmark()
    # # param = [{'max_tries':100, 'first_imp':True}, {'max_tries':100, 'first_imp':False}]
    # # bm.add_method(meta.RMS, param)
    # # bm.add_method(meta.ILS, param)
    # # bm.add_method(meta.VNS, param)
    # # bm.add_method(meta.GRASP, param)
    bm.seeds = [7,13,17]
    # bm.add_method(meta.GRASP, [{'max_tries':t, 'first_imp':fi, 'K':k} 
    #                            for t in [10, 50, 100] for k in [2, 5, 10] for fi in [True, False]])
    bm.add_method(meta.DEA, [{'max_tries':10, 
                              'N':n, 
                              'K':k, 
                              'alpha':a,
                              'first_imp':True} 
                            for n in [10, 50] for k in  [3, 5]
                            for a in [0.3, 0.5, 0.7]])
                             
    # bm.run()
