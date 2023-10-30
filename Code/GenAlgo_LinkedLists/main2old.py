import Linkedlist as Linkedlist
import numpy as np
import vertex as v
import time


# Practical 2 Evolutionary Computing

# Local search algorithms iteratively change a solution until no better solution is found in
# the neighborhood of the current solution.
# The local search algorithm used is the FiducciaMattheyses (FM) heuristic.
# MLS, ILS, and GLS are metaheuristic algorithms that improve
# the performance of the local search algorithm.

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
        passes += 1

        if cut == last_cut:
            break
    return rand_string, cut


def fm3(vertices, rand_string, passes):
    cut = calculate_cut(vertices, rand_string)
    while passes <= 1000:
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
    cell_array = []
    left_bucket = [Linkedlist.DoublyLinkedList(cell_array) for i in range(33)]
    right_bucket = [Linkedlist.DoublyLinkedList(cell_array) for i in range(33)]

    # Loop over all the vertices
    for vert in vertices:
        new_vert = v.vertex(vert[0])
        cell_array.append(new_vert)
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

def get_sum(bucket):
    som = 0
    for index, linked in enumerate(bucket):
        som += (16 - index)*linked.count()
    return som


def extract_max(bucket):
    maximum_overall = -1
    for index in range(len(bucket)-1, -1, -1):
        if bucket[index].count() != 0:
            maximum_overall = bucket[index].pop_front()
            break
    return maximum_overall


def maximum_gain_vertex(left_bucket, right_bucket, left_count, right_count):
    """
    In this function the vertex with the maximum gain will be extracted from
    the bucket that has the highest gain.
    """
    if left_count > right_count:
        bucket = left_bucket
        bit_side = 0
    elif right_count > left_count:
        bucket = right_bucket
        bit_side = 1
    else:
        if get_sum(left_bucket) > get_sum(right_bucket):
            bucket = left_bucket
            bit_side = 0
        else:
            bucket = right_bucket
            bit_side = 1
    # We also keep track of index since this can tell us
    # from which bucket the vertex has been extracted
    maximum_overall = extract_max(bucket)

    if maximum_overall == -1 and bit_side == 0:
        bucket = right_bucket
        maximum_overall = extract_max(bucket)
        bit_side = 1
    elif maximum_overall == -1 and bit_side == 1:
        bucket = left_bucket
        maximum_overall = extract_max(bucket)
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
    cut = 0
    right_count = sum(rand_string)
    left_count = len(rand_string) - right_count
    max_vertex_while_loop = 0
    iteration = 0
    left_bucket, right_bucket, cell_array = initialization(rand_string, vertices)
    while max_vertex_while_loop != -1:
        max_vertex, side = maximum_gain_vertex(left_bucket, right_bucket, left_count, right_count)
        if max_vertex == -1:
            break
        track_vertices.append(cell_array[max_vertex - 1])
        cell_array[max_vertex - 1].set_locked(True)
        max_vertex_while_loop = max_vertex
        relevant_part = vertices[max_vertex - 1][2:]

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

        # return the cut
        cut = calculate_cut(vertices, rand_string)
        if cut < lowest_cut:
            lowest_cut = cut
        cut_list.append(cut)

        iteration += 1
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
            s = [s, cost]
        print(passes)
    runtime = time.time() - t_begin
    return s, runtime
'''
def simulated_annealing():
  
def main():
    vertices = read_file("Graph500.txt")
    # rand_string = get_random_string(500)
    # rand_string, cut = fm(vertices, rand_string)
    optimallist, time = mls()
    print("Finally:", optimallist, time)
    print("The number of ones is", sum(optimallist))
    print("The number of zeros is", len(optimallist) - sum(optimallist))
    # print("The number of ones", sum(rand_string))
    # print("The number of zeros", len(rand_string) - sum(rand_string))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()'''