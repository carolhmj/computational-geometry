import time
import random
import sys
from math import floor

"""
Selection sort algorithm. Receives a vector and orders it in-place
"""
def selection_sort(v):
    n = len(v)
    # At each loop iteration, the invariant is that the
    # sublist from 0..i-1 is sorted and i..n is unsorted.
    for i in range(0, n):
       # Find the smallest item in the unsorted sublist
        smallest = i
        for j in range(i+1, n):
            if v[j] < v[smallest]:
               smallest = j
        # Swap the smallest element with the first one in the unsorted sublist
        if smallest != i:
            aux = v[i]
            v[i] = v[smallest]
            v[smallest] = aux
        # Grow the sorted sublist


"""
Insertion sort algorithm. Receives a vector and orders it in-place
"""
def insertion_sort(v):
    n = len(v)
    # At each loop iteration, the invariant is that the
    # sublist from 0..i-1 is sorted and i..n is unsorted.
    for i in range(0, n):
        # Move the element in i to its correct place in the sorted sublist,
        # swapping it with elements bigger than it
        j = i
        while j > 0 and v[j] < v[j-1]:
            aux = v[j-1]
            v[j-1] = v[j]
            v[j] = aux
            j -= 1
        # Grow the sorted sublist


"""
Merge sort algorithm. Receives a vector and returns an ordered
version of it
"""
def merge_sort(v):
    n = len(v)
    if n > 1:
        mid = floor(n/2)
        l1 = merge_sort(v[0:mid])
        l2 = merge_sort(v[mid:n])
        return merge(l1, l2)
    return v

def merge(l1,l2):
    # The merged list
    m = []
    i1 = 0
    n1 = len(l1)
    i2 = 0
    n2 = len(l2)
    # While there are still elements to copy from on both lists,
    # copy the smallest element
    while i1 < n1 and i2 < n2:
        if l1[i1] <= l2[i2]:
            m.append(l1[i1])
            i1 += 1
        else:
            m.append(l2[i2])
            i2 += 1
    # If only the elements of l1 remain, copy them all to the merged
    # list. Else, copy the elements of l2.        
    if i1 < n1:
        m.extend(l1[i1:n1])
    elif i2 < n2:
        m.extend(l2[i2:n2])
    return m
             
"""
Quick sort algorithm
"""

def quick_sort(v, lo, hi):
    n = hi - lo + 1
    if n > 1:
        q = partition(v, lo, hi)
        quick_sort(v, lo, q-1)
        quick_sort(v, q+1, hi)

"""
Chooses the pivot element as the last element and partitions the input
vector into smaller and bigger elements
"""
def partition(v, lo, hi):
    pivot = v[hi]
    i = lo
    # At each loop iteration, the invariant is that the elements from
    # 0..i-1 are smaller than the pivot
    for j in range(lo,hi+1):
        # If we find another element that's smaller than the pivot, place
        # into the i-th position and grow i to keep the invariant
        if v[j] < pivot:
            aux = v[i]
            v[i] = v[j]
            v[j] = aux

            i += 1

    # Place the pivot into the correct position
    aux = v[i]
    v[i] = v[hi]
    v[hi] = aux
    # Return the pivot index
    return i


def is_sorted(v):
    n = len(v)
    for i in range(0, n-1):
        if v[i] > v[i+1]:
            return False
    return True


def time_function_rand_input(fn, size, repeats, genfn):
    sum = 0
    for i in range(0, repeats):
        random_list = [genfn(size*10) for y in range(size)]
        start_time = time.time()
        fn(random_list)
        elapsed = time.time() - start_time
        sum += elapsed
    return sum/repeats


def time_function_ordered_input(fn, size, repeats, genfn):
    sum = 0
    for i in range(0, repeats):
        start_num = random.randint(0, size*10)
        ordered_list = [genfn(y) for y in range(start_num, start_num+size)]
        start_time = time.time()
        fn(ordered_list)
        elapsed = time.time() - start_time
        sum += elapsed
    return sum/repeats


def check_correctness(fn):  
    random_list = [random.randint(0, 1000) for y in range(10)]
    return_sorted_list = fn(random_list)
    # Assert the sorting's correctness
    if return_sorted_list is not None:
        assert(is_sorted(return_sorted_list))
    else:
        assert(is_sorted(random_list))

def run_sort_algorithm(l, name):
    print(f'===== Running algorithm {name} ======')
    # Check the algorithm's correctness
    check_correctness(l)

    example_10 = time_function_rand_input(l, 10, 1000, lambda s: random.randint(0, s*10))
    example_ordered = time_function_ordered_input(l, 10, 1000, lambda s: s*2)
    example_100 = time_function_rand_input(l, 100, 1000, lambda s: random.randint(0, s*10))
    example_1000 = time_function_rand_input(l, 1000, 1000, lambda s: random.randint(0, s*10))
    
    return {
        "example_10": example_10, 
        "example_ordered": example_ordered, 
        "example_100": example_100, 
        "example_1000": example_1000, 
    }


def show_polygonal_line_reduction(l, s, name):
    # Show the results of polygonal line reduction with algorithm l and size s 
    print(f'===== Show results of line reduction with algorithm {name} =====')
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        input = [[random.randint(0,s*10), random.randint(0,s*10), random.randint(0,s*10)] for i in range(0, s)]
        l(input)
        input = np.asarray(input)
        fig = plt.figure(f'{name} with {s} inputs')
        ax = fig.gca(projection='3d')
        ax.plot(input[:,0], input[:,1], input[:,2])

        plt.show()
    except ImportError:
        print("Couldn't import matplotlib module. Not showing polygonal line graphs.")


def run_polygonal_line_reduction(l, name):
    print(f'===== Running polygonal line reduction with algorithm {name} =====')

    example_10 = time_function_rand_input(l, 10, 100, lambda s: (random.randint(0,s*10), random.randint(0,s*10), random.randint(0,s*10)))
    example_100 = time_function_rand_input(l, 100, 100, lambda s: (random.randint(0,s*10), random.randint(0,s*10), random.randint(0,s*10)))
    example_1000 = time_function_rand_input(l, 1000, 100, lambda s: (random.randint(0,s*10), random.randint(0,s*10), random.randint(0,s*10)))
    example_ordered = time_function_ordered_input(l, 10, 100, lambda s: (random.randint(0,s*10), random.randint(0,s*10), random.randint(0,s*10)))
    return {
        "example_10": example_10, 
        "example_ordered": example_ordered, 
        "example_100": example_100, 
        "example_1000": example_1000, 
    }
    

def print_table_with_times(selection_times, insertion_times, merge_times, quick_times):
    # Make a comparative table
    print('{:<20} | {:^15} | {:^15} | {:^15} | {:^15}'.format('', 'Selection Sort', 'Insertion Sort', 'Merge Sort', 'Quick Sort'))
    print('{:<20} | {:^15f} | {:^15f} | {:^15f} | {:^15f}'.format('10 random entries', selection_times['example_10'], insertion_times['example_10'], merge_times['example_10'], quick_times['example_10']))
    print('{:<20} | {:^15f} | {:^15f} | {:^15f} | {:^15f}'.format('10 ordered entries', selection_times['example_ordered'], insertion_times['example_ordered'], merge_times['example_ordered'], quick_times['example_ordered']))
    print('{:<20} | {:^15f} | {:^15f} | {:^15f} | {:^15f}'.format('100 random entries', selection_times['example_100'], insertion_times['example_100'], merge_times['example_100'], quick_times['example_100']))
    print('{:<20} | {:^15f} | {:^15f} | {:^15f} | {:^15f}'.format('1000 random entries', selection_times['example_1000'], insertion_times['example_1000'], merge_times['example_1000'], quick_times['example_1000']))


def main(args):
    # Run standalone sorting algorithms
    selection_times = run_sort_algorithm(lambda l: selection_sort(l), "Selection Sort")
    insertion_times = run_sort_algorithm(lambda l: insertion_sort(l), "Insertion Sort")
    merge_times = run_sort_algorithm(lambda l: merge_sort(l), "Merge Sort")
    quick_times = run_sort_algorithm(lambda l: quick_sort(l, 0, len(l)-1), "Quick Sort")

    # Make a comparative table
    print_table_with_times(selection_times, insertion_times, merge_times, quick_times)

    print('\n')

    # Run polygonal line algorithms
    polygonal_selection_times = run_polygonal_line_reduction(lambda l: selection_sort(l), "Selection Sort")
    polygonal_insertion_times = run_polygonal_line_reduction(lambda l: insertion_sort(l), "Insertion Sort")
    polygonal_merge_times = run_polygonal_line_reduction(lambda l: merge_sort(l), "Merge Sort")
    polygonal_quick_times = run_polygonal_line_reduction(lambda l: quick_sort(l, 0, len(l)-1), "Quick Sort")
    print_table_with_times(polygonal_selection_times, polygonal_insertion_times, polygonal_merge_times, polygonal_quick_times)
    
    if len(args) > 1 and args[1] == 'true' :
        # Show ordenations on graphs
        show_polygonal_line_reduction(lambda l: selection_sort(l), 10, "Selection sort")
        show_polygonal_line_reduction(lambda l: insertion_sort(l), 10, "Insertion Sort")
        show_polygonal_line_reduction(lambda l: merge_sort(l), 10, "Merge Sort")
        show_polygonal_line_reduction(lambda l: quick_sort(l, 0, len(l)-1), 10, "Quick Sort")
    

if __name__ == '__main__':
    main(sys.argv)