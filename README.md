# ssorai_pcg
Implementation of preconditioned conjugate gradient on gpu

To build, run the following:

```mkdir -p build bin
cd build
cmake .. -DCMAKE_CXX_COMPILER=<your comiler>
make
```

To run tests, run `ctest` in the build directory.
