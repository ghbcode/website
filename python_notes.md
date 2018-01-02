---
layout: default
title: Notes and samples on Python Topics
description: posted by ghbcode on 2015/01/17
---

Data queries
  - [Fetch FRED data via fredapi](/website/notebooks/FRED-download.html)
  - [Fetch Quandl data](/website/notebooks/Quandl-download.html)

Troubleshooting Code - timing, profiling and tracing
  - [Timing your code](/website/notebooks/troubleshooting-code.html)
   - [How to profile your code](/website/notebooks/troubleshooting-code.html#Profiling-Code)
  - [Troubleshoot using pdb](/website/notebooks/troubleshooting-code.html#Tracing-Code)


Other topics
* [Virtual environments](virtual-environments.md)
* [Vectorized code for performance improvements](/website/notebooks/vectorized-code.html)
* structure code like a package
  * https://sdsawtelle.github.io/blog/output/data-science-project-standard-cookiecutter-structure.html
* tensor flow: TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and also used for machine learning applications such as neural networks.
  * http://sdsawtelle.github.io/blog/output/getting-started-with-tensorflow-in-jupyter.html
* regex
  * https://sdsawtelle.github.io/blog/output/regular-expressions-in-python.html
* iterators
  * https://sdsawtelle.github.io/blog/output/python-iterators-and-generators.html
* working with large files
  * https://sdsawtelle.github.io/blog/output/large-data-files-pandas-sqlite.html
* Good commenting examples
  * http://docs.python-guide.org/en/latest/writing/documentation/
  > #This function slows down program execution for some reason.
  >def square_and_rooter(x):
  >"""Returns the square root of self times self."""
  >For a function or class, the leading comment block is a programmer’s note. 
  >The docstring describes the operation of the  function or class:
