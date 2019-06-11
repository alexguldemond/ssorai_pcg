#include "DenseVector.hpp"
#include "SparseMatrix.hpp"
#include <iostream>

int main() {
    int dim = 8;
    DenseVector<double> v = DenseVector<double>::constant(dim, 1);
    DenseVector<double> d = DenseVector<double>::zero(dim);
    d[dim - 1] = 1;
    
    std::cout << "v " << v.toString() << "\n";
    std::cout << "d " << d.toString() << "\n";
    std::cout << "||v||^2 = " << v.normSquared() << "\n";
    std::cout << "||d||^2 = " << d.normSquared() << "\n";
    std::cout << "v.v = " << v.dot(v) << "\n";
    std::cout << "v.d = " << v.dot(d) << "\n";
    std::cout << "d.v = " << d.dot(v) << "\n";

    std::cout << "v.update(2, d)\n";
    v.updateAx(2, d);
    std::cout << "v " << v.toString() << "\n";
    std::cout << "d " << d.toString() << "\n";

    std::cout << "v = v.plusAx(-1, d)\n";
    v = v.plusAx(-1, d);
    std::cout << "v " << v.toString() << "\n";
    std::cout << "d " << d.toString() << "\n";

    SparseMatrix<double> A = SparseMatrix<double>::triDiagonal(dim, -1, 2, -1);
    DenseVector<double> x = DenseVector<double>::zero(dim);
    DenseVector<double> ax = A * x;
    
    std::cout << "x " << x.toString() << "\n";
    std::cout << "ax " << ax.toString() << "\n";
    
    v = v.plusAx(-1, A*x);
    std::cout << "v " << v.toString() << "\n";
    
}
