# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Memetic Genetic Algorithm with OX crossover + adaptive mutations + embedded VND local search.
"""

import time
import random
import sys
import os

# Add the openevolve_examples directory to the path to import txn_simulator and workloads
# Find the repository root by looking for openevolve_examples directory
def find_repo_root(start_path):
    """Find the repository root by looking for openevolve_examples directory."""
    current = os.path.abspath(start_path)
    # Search up the directory tree
    while current != os.path.dirname(current):  # Stop at filesystem root
        candidate = os.path.join(current, 'openevolve_examples', 'txn_scheduling')
        if os.path.exists(candidate):
            return current
        current = os.path.dirname(current)

    # If not found by searching up, try common locations relative to known paths
    # This handles when the program is copied to a results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_roots = [
        script_dir,  # Current directory
        os.path.dirname(script_dir),  # Parent
        os.path.dirname(os.path.dirname(script_dir)),  # Grandparent
        '/home/ubuntu/ShinkaEvolve',  # Absolute path fallback for Ubuntu
        '/Users/audreycc/Documents/Work/LLMTxn/ADRS-Exps/ShinkaEvolve',  # Absolute path fallback for macOS
    ]
    for root in possible_roots:
        candidate = os.path.join(root, 'openevolve_examples', 'txn_scheduling')
        if os.path.exists(candidate):
            return root

    raise RuntimeError(f"Could not find openevolve_examples directory. Searched from: {start_path}")

repo_root = find_repo_root(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(repo_root, 'openevolve_examples', 'txn_scheduling'))

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3


def get_best_schedule(workload, num_seqs):
    """
    Memetic GA to minimize makespan:
      - Order crossover (OX), adaptive mutations (swap/insert/2-opt)
      - Embedded VND local search (Or-opt(1..3), adjacent swaps, sampled 2-opt)
      - Elitism + immigrant injection on stagnation
    Returns: (best_cost, best_sequence)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # ---------------- Parameterization (adaptive by n and effort) ----------------
    small = n <= 50
    med = 50 < n <= 100
    large = n > 100

    # Population and evolution
    base_pop = 32 if small else (36 if med else 40)
    pop_size = max(24, min(56, base_pop + (num_seqs // 2)))
    generations = 60 if small else (45 if med else 35)
    crossover_rate = 0.9
    mutation_rate = 0.35 if med or large else 0.30
    elite_keep = max(3, min(6, pop_size // 8))
    tournament_k = 3 if small else 4
    stagnation_limit = 10
    immigrant_frac = 0.15

    # Local search intensity
    ls_prob = 0.6  # probability to apply VND to offspring early
    ls_prob_late = 0.9  # increased in late generations
    vnd_rounds = 2 if large else 3

    # Random seed for reproducibility per call
    random.seed((n * 1009 + num_seqs * 9176 + 31) % (2**32 - 1))

    # ---------------- Evaluation cache ----------------
    cost_cache = {}

    def eval_cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(list(seq))
            cost_cache[key] = c
        return c

    # ---------------- Initialization ----------------
    def greedy_append_constructor():
        """Simple append-greedy constructor: append the txn that minimizes incremental cost."""
        rem = set(all_txns)
        seq = []
        while rem:
            best_t = None
            best_c = float('inf')
            # Sample candidates more aggressively when many remain
            k = min(len(rem), 12 if small else (10 if med else 8))
            cand = random.sample(list(rem), k) if len(rem) > k else list(rem)
            for t in cand:
                c = eval_cost(seq + [t])
                if c < best_c:
                    best_c = c
                    best_t = t
            # Fallback if sampling missed improvements
            if best_t is None:
                best_t = rem.pop()
                seq.append(best_t)
            else:
                rem.remove(best_t)
                seq.append(best_t)
        return seq

    def random_init():
        s = list(all_txns)
        random.shuffle(s)
        return s

    def seeded_population():
        pop = []
        # Part random, part greedy
        n_random = max(6, pop_size // 3)
        n_greedy = max(4, pop_size // 4)
        # Random individuals
        for _ in range(n_random):
            s = random_init()
            pop.append(s)
        # Greedy-based
        for _ in range(n_greedy):
            s = greedy_append_constructor()
            pop.append(s)
        # Fill rest with random permutations
        while len(pop) < pop_size:
            pop.append(random_init())
        return pop[:pop_size]

    # ---------------- Genetic operators ----------------
    def ox_crossover(p1, p2):
        """Order crossover (OX): preserve a segment from p1, fill remaining in p2 order."""
        L = len(p1)
        a = random.randint(0, L - 2)
        b = random.randint(a + 1, L - 1)
        child = [-1] * L
        in_child = set()
        # Copy slice from p1
        for i in range(a, b + 1):
            child[i] = p1[i]
            in_child.add(p1[i])
        # Fill remaining from p2 in order
        idx = (b + 1) % L
        for x in p2:
            if x in in_child:
                continue
            child[idx] = x
            in_child.add(x)
            idx = (idx + 1) % L
        return child

    def mutate_swap(seq):
        L = len(seq)
        if L < 2:
            return seq
        i, j = random.sample(range(L), 2)
        seq[i], seq[j] = seq[j], seq[i]
        return seq

    def mutate_insert(seq):
        L = len(seq)
        if L < 2:
            return seq
        i, j = random.sample(range(L), 2)
        t = seq.pop(i)
        if j > i:
            j -= 1
        seq.insert(j, t)
        return seq

    def mutate_2opt(seq):
        L = len(seq)
        if L < 4:
            return seq
        i = random.randint(0, L - 3)
        j = random.randint(i + 2, L - 1)
        seq[i:j + 1] = reversed(seq[i:j + 1])
        return seq

    def adaptive_mutate(seq):
        r = random.random()
        if r < 0.4:
            return mutate_swap(seq)
        elif r < 0.75:
            return mutate_insert(seq)
        else:
            return mutate_2opt(seq)

    # ---------------- Local Search (VND) ----------------
    def sample_positions(base_len):
        """Stratified + random positions for reinsertion."""
        if base_len <= 25:
            return list(range(base_len + 1))
        pos = {0, base_len, base_len // 2, base_len // 4, (3 * base_len) // 4}
        # add a few random anchors
        while len(pos) < 9:
            pos.add(random.randint(0, base_len))
        return sorted(pos)

    def vnd_local_search(seq, start_cost, max_rounds=vnd_rounds):
        best_seq = list(seq)
        best_cost = start_cost
        if len(best_seq) <= 2:
            return best_seq, best_cost

        def or_opt_pass(block_len):
            nonlocal best_seq, best_cost
            L = len(best_seq)
            if L <= block_len:
                return False
            improved = False
            i = 0
            while i <= L - block_len:
                block = best_seq[i:i + block_len]
                base = best_seq[:i] + best_seq[i + block_len:]
                positions = sample_positions(len(base))
                move_best_c = best_cost
                move_best_p = None
                for p in positions:
                    cand = base[:]
                    cand[p:p] = block
                    c = eval_cost(cand)
                    if c < move_best_c:
                        move_best_c = c
                        move_best_p = p
                if move_best_p is not None:
                    # apply move
                    base = best_seq[:i] + best_seq[i + block_len:]
                    base[move_best_p:move_best_p] = block
                    best_seq = base
                    best_cost = move_best_c
                    L = len(best_seq)
                    improved = True
                    # restart after improvement
                    i = 0
                else:
                    i += 1
            return improved

        def adjacent_swap_pass():
            nonlocal best_seq, best_cost
            improved = False
            for i in range(len(best_seq) - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True
            return improved

        def sampled_2opt_pass(tries=80):
            nonlocal best_seq, best_cost
            L = len(best_seq)
            if L <= 4:
                return False
            improved = False
            best_delta = 0.0
            best_move = None
            attempts = min(tries, max(30, L))
            for _ in range(attempts):
                i = random.randint(0, L - 3)
                j = random.randint(i + 2, min(L - 1, i + 12))
                cand = best_seq[:]
                cand[i:j + 1] = reversed(cand[i:j + 1])
                c = eval_cost(cand)
                delta = best_cost - c
                if delta > best_delta:
                    best_delta = delta
                    best_move = (i, j, c)
            if best_move is not None:
                i, j, c = best_move
                cand = best_seq[:]
                cand[i:j + 1] = reversed(cand[i:j + 1])
                best_seq, best_cost = cand, c
                improved = True
            return improved

        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            any_imp = False
            for bl in (3, 2, 1):
                if or_opt_pass(bl):
                    any_imp = True
            if adjacent_swap_pass():
                any_imp = True
            if sampled_2opt_pass(tries=100 if len(best_seq) > 80 else 60):
                any_imp = True
            if not any_imp:
                break
        return best_seq, best_cost

    # ---------------- Selection ----------------
    def tournament_select(pop):
        """Tournament selection over (cost, seq) tuples."""
        k = min(tournament_k, len(pop))
        contestants = random.sample(pop, k)
        contestants.sort(key=lambda x: x[0])
        return contestants[0][1]  # return sequence

    # ---------------- Evolutionary loop ----------------
    # Initialize population
    population = []
    seen = set()
    for s in seeded_population():
        key = tuple(s)
        if key in seen:
            continue
        c = eval_cost(s)
        population.append((c, s))
        seen.add(key)
        if len(population) >= pop_size:
            break

    # Ensure we have enough unique individuals
    while len(population) < pop_size:
        s = random_init()
        key = tuple(s)
        if key in seen:
            continue
        c = eval_cost(s)
        population.append((c, s))
        seen.add(key)

    # Track best
    population.sort(key=lambda x: x[0])
    best_cost, best_seq = population[0]
    no_improve = 0

    for gen in range(generations):
        # Update LS probability toward late phase
        apply_ls_prob = ls_prob if gen < (generations * 2) // 3 else ls_prob_late

        # Elitism
        population.sort(key=lambda x: x[0])
        elites = population[:elite_keep]
        new_pop = elites[:]

        # Reproduction
        while len(new_pop) < pop_size:
            # Parent selection
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            # avoid identical parents if possible
            tries = 0
            while p2 == p1 and tries < 3:
                p2 = tournament_select(population)
                tries += 1

            # Crossover
            if random.random() < crossover_rate:
                child = ox_crossover(p1, p2)
            else:
                child = list(p1)

            # Mutation
            if random.random() < mutation_rate:
                child = adaptive_mutate(child)
                if random.random() < 0.15:
                    child = adaptive_mutate(child)  # occasional double mutation

            # Local search (memetic)
            c_child = eval_cost(child)
            if random.random() < apply_ls_prob:
                child, c_child = vnd_local_search(child, c_child, max_rounds=vnd_rounds)

            key = tuple(child)
            if key in seen:
                # slight perturb to escape duplicates
                child = mutate_insert(child[:])
                key = tuple(child)
                c_child = eval_cost(child)
            new_pop.append((c_child, child))
            seen.add(key)

        # Replacement
        new_pop.sort(key=lambda x: x[0])
        population = new_pop[:pop_size]

        # Best tracking
        if population[0][0] + 1e-9 < best_cost:
            best_cost, best_seq = population[0]
            no_improve = 0
        else:
            no_improve += 1

        # Inject immigrants on stagnation
        if no_improve >= stagnation_limit:
            num_imm = max(2, int(immigrant_frac * pop_size))
            for _ in range(num_imm):
                s = random_init()
                c = eval_cost(s)
                population.append((c, s))
            population.sort(key=lambda x: x[0])
            population = population[:pop_size]
            no_improve = 0

    # Final polish on incumbent best
    best_seq, best_cost = vnd_local_search(best_seq, best_cost, max_rounds=2)

    # Safety checks
    assert len(best_seq) == n and len(set(best_seq)) == n, "Schedule must include each transaction exactly once"

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()
    workload_size = 100

    # Workload 1: Complex mixed read/write transactions
    workload = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
    cost3 = workload3.get_opt_seq_cost(schedule3)

    total_makespan = cost1 + cost2 + cost3
    schedules = [schedule1, schedule2, schedule3]
    execution_time = time.time() - start_time

    return total_makespan, schedules, execution_time


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_scheduling():
    """Run the transaction scheduling algorithm for all workloads"""
    total_makespan, schedules, execution_time = get_random_costs()
    return total_makespan, schedules, execution_time


if __name__ == "__main__":
    total_makespan, schedules, execution_time = run_scheduling()
    print(f"Total makespan: {total_makespan}, Execution time: {execution_time:.4f}s")
    print(f"Individual workload costs: {[workload.get_opt_seq_cost(schedule) for workload, schedule in zip([Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)], schedules)]}")