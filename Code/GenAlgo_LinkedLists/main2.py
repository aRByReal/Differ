import Linkedlist as Linkedlist
import numpy as np
import vertex as v
import time
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
from heapq import heappush, heappop

# Practical 2 Evolutionary Computing

# Local search algorithms iteratively change a solution until no better solution is found in
# the neighborhood of the current solution.
# The local search algorithm used is the FiducciaMattheyses (FM) heuristic.
# MLS, ILS, and GLS are metaheuristic algorithms that improve
# the performance of the local search algorithm.

# Crossover opperator that generates 1 ofsspring list
def uniform(A, B):
    # If the hamming distance is larger than l/2  all bit values of one parent are inverted
    distance = np.count_nonzero(A != B)
    if distance > len(A) / 2:
        A = 2 ** A % 2
        distance = len(A) - distance
    S = get_random_string(distance)
    off1 = A
    j = 0
    for i in range(len(A)):
        if A[i] != B[i]:
            off1[i] = S[j]
            j += 1
    return off1


# GLS applies FM to offspring of a population
def gls(N, T):
    t_begin = time.time()
    POP = generate_data(N)

    PopOpt = []
    for e in POP:
        fit = count_edges(e)
        PopOpt.append([e, fit])

    if T != 0:
        t_end = time.time() + T
        while time.time() < t_end:

            random.shuffle(PopOpt)

            off = uniform(PopOpt[0][0], PopOpt[1][0])
            off, off_cost = fm(vertices, off)

            PopOpt.sort(key=lambda row: row[1])
            if PopOpt[0][1] >= off_cost:
                del PopOpt[0]
                PopOpt.append([off, off_cost])

    # If T=0 we run the algorithm for 10000 passes
    else:
        passes = 0
        while passes <= 10000:
            random.shuffle(PopOpt)

            off = uniform(PopOpt[0][0], PopOpt[1][0])
            off, off_cost, passes = fm3(vertices, off, passes)

            PopOpt.sort(key=lambda row: row[1])
            if PopOpt[0][1] >= off_cost:
                del PopOpt[0]
                PopOpt.append([off, off_cost])

    PopOpt.sort(key=lambda row: row[1], reverse=True)

    t = time.time() - t_begin
    return PopOpt[0], t


def count_edges(list_of_vertices):
    F = 0

    for i in range(0, 500):
        if list_of_vertices[i] == 1:
            for j in range(0, len(vertices[i]) - 2):
                a = vertices[i][j + 2]

                if list_of_vertices[i] != list_of_vertices[int(a) - 1]:
                    F = F + 1
    return F


# swaps to bits, one from one to zero and one from zero to one
def mutate(s, size):
    for i in range(0, size):
        while True:
            A = random.randint(0, 499)
            B = random.randint(0, 499)
            if s[A] != s[B]:
                a = s[A]
                s[A] = s[B]
                s[B] = a
                break
    return s


# ILS applies FM to the mutation of the previous best string
def ils(size, T):
    counter = 0
    t_begin = time.time()
    S = get_random_string(500)
    if T != 0:
        t_end = time.time() + T
        Sopt, fit1 = main.fm(vertices, S)
        while time.time() < t_end:

            Smut = mutate(Sopt, size)
            Sopt2, fit2 = main.fm(vertices, Smut)
            if Sopt is Sopt2:
                counter += 1
            else:

                if fit1 >= fit2:
                    Sopt = Sopt2
                    fit1 = fit2
    else:
        passes = 0

        Sopt, fit1, passes = fm3(vertices, S, passes)
        while passes < 10000:
            Smut = mutate(Sopt, size)
            Sopt2, fit2, passes = fm3(vertices, Smut, passes)
            if Sopt is Sopt2:
                counter += 1
            else:

                if fit1 >= fit2:
                    Sopt = Sopt2
                    fit1 = fit2

    t = time.time() - t_begin
    return [Sopt, fit1], t, counter


# applies ILS forincrementally increasing #swaps per mutation untill improvement stops
def find_pertubation_size(T, start, increment):
    count = 0
    results = []
    fit = 1000
    while count < 2:
        Soptfit, t, counter = ils(start, T)

        results.append([start, Soptfit[1], t, counter])

        start += increment

        if Soptfit[1] > fit:
            count += 1
        else:
            count = 0
        fit = Soptfit[1]
    # results.sort(key = lambda row: row[1])
    return results


def runforpasses(ILSsize):
    mls_list = []
    ils_list = []
    gls_list = []
    for i in range(0, 20):
        MLS_result = mls_list()
        mls_list.append(MLS_result)
        T = MLS_result[1]
        gls_list.append(gls_list(50, 0))

        ils_list.append(ils_list(ILSsize, 0))

    return mls_list, gls_list, ils_list


def runfortime(ILSsize, mls):
    # cutoff time is set to avg mls time:
    T = np.mean([e[1] for e in mls])

    ils_list = []
    gls_list = []
    for i in range(0, 20):
        gls_list.append(gls_list(50, T))

        ils_list.append(ils_list(ILSsize, T))

    return mls, gls_list, ils_list, T


def plotpert():
    pert = find_pertubation_size(0, 3, 3)

    # x-coordinates of left sides of bars
    X = [e[0] for e in pert]

    # naming the x-axis
    plt.xlabel('Size')
    # naming the y-axis
    plt.ylabel('Cut')
    # plot title
    plt.title('Pertubation sizing')

    Y = [e[1] for e in pert]
    Z = [e[3] for e in pert]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.2, Y, 0.4, label='cut')
    plt.bar(X_axis + 0.2, Z, 0.4, label='#Equal Optima')

    plt.xticks(X_axis, X)
    plt.legend()
    plt.show()

    pert.sort(key=lambda row: row[1])

    return pert[0][0]


def plotpasses(ILSsize):
    mls, gls, ils = runforpasses(ILSsize)
    mlscuts = [e[0][1] for e in mls]
    glscuts = [e[0][1] for e in gls]
    ilscuts = [e[0][1] for e in ils]

    mlstimes = np.mean([e[1] for e in mls])
    glstimes = np.mean([e[1] for e in gls])
    ilstimes = np.mean([e[1] for e in ils])

    plt.boxplot([mlscuts, ilscuts, glscuts])

    # naming the x-axis
    plt.xlabel('')

    plt.xticks([1, 2, 3], ["mls", "ils", "gls"])

    # naming the y-axis
    plt.ylabel('Cut')
    # plot title
    plt.title('Performance for 10000 passes')
    plt.show()

    print("avg time mls:" + str(mlstimes))
    print("avg time ils:" + str(ilstimes))
    print("avg time gls:" + str(glstimes))

    # perform the Mann-Whitney U test
    print("mls vs gls ")
    print(stats.mannwhitneyu(mlscuts, glscuts, alternative='two-sided'))
    print("mls vs ils ")
    print(stats.mannwhitneyu(mlscuts, ilscuts, alternative='two-sided'))
    print("gls vs ils ")
    print(stats.mannwhitneyu(glscuts, ilscuts, alternative='two-sided'))
    return mls


def plotruntime(ILSsize, mls):
    mls, gls, ils, T = runfortime(ILSsize, mls)
    mlscuts = [e[0][1] for e in mls]
    glscuts = [e[0][1] for e in gls]
    ilscuts = [e[0][1] for e in ils]

    mlstimes = np.mean([e[1] for e in mls])
    glstimes = np.mean([e[1] for e in gls])
    ilstimes = np.mean([e[1] for e in ils])

    plt.boxplot([mlscuts, ilscuts, glscuts])

    # naming the x-axis
    plt.xlabel('')

    plt.xticks([1, 2, 3], ["mls", "ils", "gls"])

    # naming the y-axis
    plt.ylabel('Cut')
    # plot title
    plt.title('Performance for cut-off time: ' + str(T))
    plt.show()

    print("avg time mls:" + str(mlstimes))
    print("avg time ils:" + str(ilstimes))
    print("avg time gls:" + str(glstimes))

    # perform the Mann-Whitney U test
    print("mls vs gls ")
    print(stats.mannwhitneyu(mlscuts, glscuts, alternative='two-sided'))
    print("mls vs ils ")
    print(stats.mannwhitneyu(mlscuts, ilscuts, alternative='two-sided'))
    print("gls vs ils ")
    print(stats.mannwhitneyu(glscuts, ilscuts, alternative='two-sided'))

def extractdigits(lst):
    total = []
    for el in lst:
        new = el.split(' ')
        new_list = [x for x in new if x != '']
        newlist = new_list[:1] + new_list[2:]
        final = [int(i) for i in newlist]
        total.append(final)
    return total


def read_file(filename):
    lines = open(filename).read().splitlines()
    vertices = extractdigits(lines)
    return vertices

vertices = read_file("Graph500.txt")

def get_random_string(n):
    data = np.zeros(n, dtype=int)
    f = int(n / 2)
    data[:f] = 1
    np.random.shuffle(data)
    return data


def generate_data(N):
    pop = []
    for i in range(N):
        pop.append(get_random_string(500))
    return pop

def fm2(vertices, L):

    rand_string = L
    cut = calculate_cut(vertices, rand_string)
    while True:
        last_cut = cut
        rand_string, cut = fm_pass(vertices, rand_string)


        if cut >= last_cut:
            break
    return rand_string, cut


def fm3(vertices, rand_string, passes):
    cut = calculate_cut(vertices, rand_string)
    while passes <= 10000:
        last_cut = cut
        rand_string, cut = fm_pass(vertices, rand_string)
        passes += 1

        if cut == last_cut:
            break

    return rand_string, cut, passes


def calculate_gain(rand_string, vertex):
    index = vertex[0] - 1
    relevant_part = vertex[2:]
    no_of_ones = sum([1 for i in relevant_part if rand_string[i-1] == 1])
    no_of_zeros = len(relevant_part) - no_of_ones
    gain = no_of_ones - no_of_zeros + 16 if rand_string[index] == 0 else no_of_zeros - no_of_ones + 16
    return gain


def calculate_cut(vertices, rand_string):
    connections = {}
    for vertex in vertices:
        connections[vertex[0]] = vertex[2:]

    cut = 0
    for vertex, neighbors in connections.items():
        for neighbor in neighbors:
            if rand_string[vertex - 1] != rand_string[neighbor - 1]:
                cut += 1
    cut_size = cut // 2
    return cut_size


def visualize_bucket(bucket):
    for index, list in enumerate(bucket):
        print("The index of the list is: ", index)
        list.count()
        list.print_list()


def initialization(rand_string, vertices):
    """
    # We are going to keep track of the vertices and say that we have a vertex id
    # The next and previous are the pointers for the doubly linked list
    # we have a bucket for each possible gain.
    :return:
    """
    cell_array = np.empty(len(vertices), dtype=v.vertex)
    left_bucket = [Linkedlist.DoublyLinkedList(cell_array) for i in range(33)]
    right_bucket = [Linkedlist.DoublyLinkedList(cell_array) for i in range(33)]

    # Loop over all the vertices
    for i, vert in enumerate(vertices):
        new_vert = v.vertex(vert[0])
        cell_array[i] = new_vert
        gain = calculate_gain(rand_string, vert)
        new_vert.set_gain(gain)
        if rand_string[vert[0] - 1] == 0:
            left_bucket[gain].add_to_tail(new_vert.id)
            new_vert.set_side(0)
        else:
            right_bucket[gain].add_to_tail(new_vert.id)
            new_vert.set_side(1)
        # visualize_bucket(left_bucket)
        # visualize_bucket(right_bucket)
    return left_bucket, right_bucket, cell_array

# def get_sum(bucket):
#     som = 0
#     for index, linked in enumerate(bucket):
#         som += (16 - index)*linked.count()
#     return som


# def extract_max(bucket):
#     maximum_overall = -1
#     for index in range(len(bucket)-1, -1, -1):
#         if bucket[index].count() != 0:
#             maximum_overall = bucket[index].pop_front()
#             break
#     return maximum_overall



def extract_max(bucket, cell_array):
    max_heap = []
    for index in range(len(bucket) - 1, -1, -1):
        if bucket[index].count() != 0:
            vertices = bucket[index].get_list()
            for vertex in vertices:
                heappush(max_heap, -cell_array[vertex - 1].gain)
            max_vertex = bucket[index].pop_front()
            while max_heap:
                if -max_heap[0] == cell_array[max_vertex - 1].gain:
                    heappop(max_heap)
                else:
                    break
            return max_vertex
    return -1

def maximum_gain_vertex(left_bucket, right_bucket, left_count, right_count, cell_array):
    """
    In this function the vertex with the maximum gain will be extracted from
    the bucket that has the highest gain.
    """
    if left_count > right_count:
        bucket = left_bucket
        bit_side = 0
    else: #right_count > left_count:
        bucket = right_bucket
        bit_side = 1
    # else:
    #     if get_sum(left_bucket) > get_sum(right_bucket):
    #         bucket = left_bucket
    #         bit_side = 0
    #     else:
    #         bucket = right_bucket
    #         bit_side = 1
    # We also keep track of index since this can tell us
    # from which bucket the vertex has been extracted
    maximum_overall = extract_max(bucket, cell_array)

    if maximum_overall == -1 and bit_side == 0:
        bucket = right_bucket
        maximum_overall = extract_max(bucket, cell_array)
        bit_side = 1
    elif maximum_overall == -1 and bit_side == 1:
        bucket = left_bucket
        maximum_overall = extract_max(bucket, cell_array)
        bit_side = 0
    return maximum_overall, bit_side


# def get_bucket_count(left_bucket, right_bucket):
#     count = 0
#     for index, i in enumerate(left_bucket):
#         count += left_bucket[index].count()
#     for index, i in enumerate(right_bucket):
#         count += right_bucket[index].count()
#     return count


def roll_back_to_best_observed(left_bucket, right_bucket, cut_list, track_vertices, lowest_cut, rand_string):
    for i in range(len(track_vertices) - 1, -1, -1):
        j = cut_list[i]
        if j == lowest_cut:
            break
        if track_vertices[i].side == 0:
            rand_string[track_vertices[i].id - 1] = 1
        else:
            rand_string[track_vertices[i].id - 1] = 0
    return left_bucket, right_bucket, rand_string, lowest_cut


def fm_pass(vertices, rand_string):
    cut_list = []
    track_vertices = []
    lowest_cut = 1000000
    cut = calculate_cut(vertices, rand_string)
    right_count = sum(rand_string)
    left_count = len(rand_string) - right_count
    max_vertex_while_loop = 0
    lowest_cut = cut
    left_bucket, right_bucket, cell_array = initialization(rand_string, vertices)
    while max_vertex_while_loop != -1:
        max_vertex, side = maximum_gain_vertex(left_bucket, right_bucket, left_count, right_count, cell_array)
        if max_vertex == -1:
            break
        track_vertices.append(cell_array[max_vertex - 1])
        cell_array[max_vertex - 1].set_locked(True)
        max_vertex_while_loop = max_vertex
        relevant_part = vertices[max_vertex - 1][2:]
        cut -= cell_array[max_vertex - 1].gain - 16

        if len(relevant_part) != 0:
            # Here we loop over the relevant_part to calculate the initial gains
            for i in relevant_part:
                vertex_in_question = cell_array[i - 1]
                old_gain = calculate_gain(rand_string, vertices[i - 1])
                vertex_in_question.gain = old_gain

                if vertex_in_question.side == 0:
                    left_bucket[old_gain].remove_by_list_node(vertex_in_question.list_node)
                else:
                    right_bucket[old_gain].remove_by_list_node(vertex_in_question.list_node)

            # Flip the bit of the extracted vertex
            if side == 0:
                rand_string[max_vertex - 1] = 1
                cell_array[max_vertex - 1].set_side(1)
                left_count = left_count - 1
                right_count = right_count + 1
            else:
                rand_string[max_vertex - 1] = 0
                cell_array[max_vertex - 1].set_side(0)
                left_count = left_count + 1
                right_count = right_count - 1

            # Calculate the gains after the bit flip
            for i in relevant_part:
                vertex_in_question = cell_array[i - 1]
                new_gain = calculate_gain(rand_string, vertices[i - 1])
                vertex_in_question.gain = new_gain

                if rand_string[i - 1] == 0 and vertex_in_question.locked is False:
                    left_bucket[new_gain].add_to_tail(vertex_in_question.id)
                elif not vertex_in_question.locked:
                    right_bucket[new_gain].add_to_tail(vertex_in_question.id)

        else:
            if side == 0:
                rand_string[max_vertex - 1] = 1
                cell_array[max_vertex - 1].set_side(1)
                left_count = left_count - 1
                right_count = right_count + 1
            else:
                rand_string[max_vertex - 1] = 0
                cell_array[max_vertex - 1].set_side(0)
                left_count = left_count + 1
                right_count = right_count - 1

        if cut < lowest_cut:
            lowest_cut = cut
        cut_list.append(cut)

    left_bucket, right_bucket, rand_string, lowest_cut = roll_back_to_best_observed(left_bucket, right_bucket,
                                                                                   cut_list, track_vertices, lowest_cut,
                                                                                    rand_string)
    return rand_string, lowest_cut


def fm(vertices, rand_string):
    cut = calculate_cut(vertices, rand_string)
    iterations = 0
    while True:
        last_cut = cut
        rand_string, cut = fm_pass(vertices, rand_string)
        iterations += 1
        if cut == last_cut:
            break
    return rand_string, cut

def mls():
    t_begin = time.time()
    cost = 0
    passes = 0
    while passes < 1000:
        a = get_random_string(500)
        last_cost = cost
        s, cost, passes = fm3(vertices, a, passes)
        if cost < last_cost:
            s_list = [s, cost]
        print(passes)
        passes += 1
    runtime = time.time() - t_begin
    return s_list, runtime


# def simulated_annealing():
def print_info_fm(rand_string):
    cut = calculate_cut(vertices, rand_string)
    print("The string is now:", rand_string)
    print("The cut is", cut)
    print("The number of ones", sum(rand_string))
    print("The number of zeros", len(rand_string) - sum(rand_string))

def print_info_mls(optimallist, time):
    print("Finally:", optimallist, time)
    print("The number of ones is", np.sum(optimallist[0]))
    print("The number of zeros is", len(optimallist[0]) - np.sum(optimallist[0]))

'''def main():
    vertices = read_file("Graph500.txt")
    rand_string = get_random_string(500)
    # rand_string, cut = fm(vertices, rand_string)
    optimallist, time = mls()
    print_info_mls(optimallist, time)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
'''