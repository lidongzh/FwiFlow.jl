* download amgcl and decompress it to `TwoPhaseFlowFWI` directory with name `amgcl` (not `amgcl-master`)

* Create a CmakeLists.txt in `Ops/AMG`
```
cmake_minimum_required(VERSION 3.9)
project(amgcl_eigen)

set (CMAKE_CXX_STANDARD 11)

# --- Find AMGCL ------------------------------------------------------------
include_directories("../../amgcl")
include_directories(/usr/local/include/eigen3)
include_directories(/usr/local/include ${CONDA_INC})

# ---------------------------------------------------------------------------
add_executable(main main.cpp)
# target_link_libraries(amgcl_eigen)
```

* Replace `include_directories(/usr/local/include ${CONDA_INC})` here with include directory of `boost` library

* `cmake` and `make` in the `build` directory

* `./main ../cz308.mtx`, expected output
```
Solver
======
Type:             BiCGStab
Unknowns:         308
Memory footprint: 0.00 B

Preconditioner
==============
Number of levels:    1
Operator complexity: 1.00
Grid complexity:     1.00
Memory footprint:    106.38 K

level     unknowns       nonzeros      memory
---------------------------------------------
    0          308           3182    106.38 K (100.00%)

1 2.09874e-13
```