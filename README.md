# QTree - simple quantum circuit simulator based on the undirected graphical models.

Repository structure
--------------------
- qtree  - main source code. Contains functions for various operations
- scratchpad   - scripts for specific purposes, such as performance testing and complexity estimation
- thirdparty  - third-party programs to calculate tree decomposition
- test_circuits   - example circuits produced by Sergio Boixo for quantu supremacy experiments

Installation of tree decomposition solvers
------------------------------------------
Qtree depends on external tree decomposition 
programs, otherwise only simple heuristics will be available.
The instructions for the supported programs are provided below. We
recommend to use the solver of Tamaki for its higher speed.

- QuickBB
Clone the repository
```sh
git clone https://github.com/qbit-/quickbb.git
```
check that it works for you:
```sh
cd quickbb; ./run_quickbb_64.sh --cnffile test.cnf
```
and add it to PATH:
```sh
export PATH=$PATH:`cwd`
```

- Tamaki solver
Clone the repository
```sh
git clone https://github.com/TCS-Meiji/PACE2017-TrackA.git
```
Build the binary (requires Java compiler):
```sh
cd PACE2017-TrackA; make
chmod +x tw-exact tw-heuristic
```
and add it to PATH:
```sh
export PATH=$PATH:`cwd`
```

Installation
------------
To install run
```sh
pip install qtree
```

Working with source code
------------------------
It is also possible to work with the source code without installation.
You need to make sure that tree decomposition programs are
available in the *thirdparty* folder. Please clone the repository recursively 
and follow the instructions in the previous section. There is no need 
to include the tree decomposition programs to the PATH in this case.