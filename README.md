# D-Wave Ocean plugin for IBM Qiskit

Enables [Qiskit](https://qiskit.org/) users to obtain ground state(s) of Ising Hamiltonians using [D-Wave](https://www.dwavesys.com/)'s QPU available via [Leap](https://cloud.dwavesys.com/).

The package provides an implementation of Qiskit's `MinimumEigensolver`
interface (available as `DWaveMinimumEigensolver`) which can be used directly on qubit operators, or via
`qiskit-optimization`'s `MinimumEigenOptimizer`.


## Examples

Solve a [`QuadraticProgram`](https://qiskit.org/documentation/stubs/qiskit.optimization.QuadraticProgram.html)
with [`MinimumEigenOptimizer`](https://qiskit.org/documentation/stubs/qiskit.optimization.algorithms.MinimumEigenOptimizer.html)
(see Qiskit's [tutorial](https://qiskit.org/documentation/tutorials/optimization/3_minimum_eigen_optimizer.html))
using `DWaveMinimumEigensolver`:

```python
>>> from qiskit_optimization import QuadraticProgram
>>> from qiskit_optimization.algorithms import MinimumEigenOptimizer
>>> from dwave.plugins.qiskit import DWaveMinimumEigensolver
...
>>> # Construct a simple quadratic program
>>> qp = QuadraticProgram()
>>> qp.binary_var('x')
>>> qp.binary_var('y')
>>> qp.minimize(quadratic={'xy': 1})
...
>>> # Solve using Qiskit's MinimumEigenOptimizer on D-Wave QPU as a minimum eigen solver
>>> dwave_mes = DWaveMinimumEigensolver()
>>> optimizer = MinimumEigenOptimizer(dwave_mes)
>>> result = optimizer.solve(qp)
...
>>> print(result)
optimal function value: 0.0
optimal value: [0. 1.]
status: SUCCESS
>>> result.samples
[('01', 0.0, 0.39), ('00', 0.0, 0.25), ('10', 0.0, 0.36)]
```

Solve a 6-city TSP (or [some other Ising model](https://qiskit.org/documentation/apidoc/qiskit.optimization.applications.ising.html#module-qiskit.optimization.applications.ising)).

```python
>>> from qiskit_optimization.applications import tsp
>>> from dwave.plugins.qiskit import DWaveMinimumEigensolver
...
>>> six_cities_tsp = tsp.random_tsp(6, seed=123)
>>> operator, offset = tsp.get_operator(six_cities_tsp)
...
>>> print(operator.print_details())
IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIZ	(-400141.5+0j)
IIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIII	(-400152.5+0j)
IIIIIIIIIIIIIIIIIIIIIIIIIIIIZIIIIIIZ	(12+0j)
# snipped for brevity
>>> print(operator.num_qubits)
36
...
>>> dwave_mes = DWaveMinimumEigensolver(num_reads=1000)
>>> result = dwave_mes.compute_minimum_eigenvalue(operator)
...
>>> x = sample_most_likely(result.eigenstate)
>>> tsp.tsp_feasible(x)
True
>>> tsp.get_tsp_solution(x)
[2, 3, 5, 1, 4, 0]
```

For comparison, trying this on `NumPyMinimumEigensolver` produces:

```python
>>> from qiskit_algorithms import NumPyMinimumEigensolver
>>> result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator)
# snipped for brevity
MemoryError: Unable to allocate 512. GiB for an array with shape (68719476737,) and data type uint64
```

## Installation

Compatible with Python 3.10+, `qiskit` 2.5.0+, `qiskit-algorithms` 0.4.0+,
`qiskit-optimization` 0.7.0+, and `dwave-system` 1.35.0+.

```bash
pip install dwave-qiskit-plugin
```

To install from source:
```bash
pip install -r requirements.txt
python setup.py install
```

Test requirements are in `tests/requirements.txt`.

Note: [Configured access to D-Wave API](https://docs.ocean.dwavesys.com/en/latest/overview/sapi.html) is required.


## License

Released under the Apache License 2.0. See [LICENSE](./LICENSE) file.
