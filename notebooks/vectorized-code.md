---
layout: default
title: Notes and Samples on Python Topics
description: posted by ghbcode on 2015/01/12
---

# Performance Gains Via Vectorization

As always the devil is in the details and it behooves us to dig a little. Depending on how you implement a particular calculation your code may be doing something very inefficiently. See my posting on [profiling code](https://ghbcode.github.io/website/notebooks/troubleshooting-code.html#Profiling-Code) for catching these sorts of instances. As a rule of thumb, if you are iterating over structures with loops or lambda functions, you may want to check if the same can be done via vectorized/matrix algebra. In the example below you have a list with 10 million items in the range from [0,1000] and you sum the total via a loop which takes on average close to a second. When you use the Python sum() function(vectorized) on the same list you realize almost a 10-fold reduction in time. Finally, if you turn the list into an array and use the array.sum() function, you realize over a 100-fold improvement in time. 


```python
import numpy as np
import timeit
```


```python
# Generate a list of random numbers
N = int(1e7)
num_list = [np.random.randint(0,1000) for i in range(N)] # gives you a list with elements in the [0,1000] range
num_arr = np.array(num_list)
```


```python
# Check how long it takes to sum all elements of the list using a for loop
def arr_sum(x):
    # Standard for loop accumulation
    total = 0
    for n in x:
        total += n
    return total

%timeit -r 3 arr_sum(num_list)
```

    919 ms ± 20.9 ms per loop (mean ± std. dev. of 3 runs, 1 loop each)



```python
# Check how long it takes to sum all elements of the list using the Python standard library "sum()"
def vec_sum(x):
    return sum(x)
%timeit -r 3 vec_sum(num_list)
```

    109 ms ± 1.05 ms per loop (mean ± std. dev. of 3 runs, 10 loops each)



```python
# Check how long it takes to sum all elements of the list by turning into arran and then using the sum() function
%timeit num_arr.sum()
```

    7.72 ms ± 506 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

