#ifndef LINEAR_SOLVE_HPP
#define LINEAR_SOLVE_HPP

#include "DenseVector.hpp"
#include "SparseMatrix.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

template<class U>
struct Result {
    DenseVector<U> result;
    int iterations;
    Result(const DenseVector<U>& x, int iter) : result(x), iterations(iter) { }
};

template <class T>
class LinearSolver {
public:
    virtual Result<T> solve() const =0;
};

template <class T>
class SsoraPcgSolver: LinearSolver<T> {
private:
    const SparseMatrix<T>& mat;
    const DenseVector<T>& vec;
    double threshold;
    double relaxation;
    int maxIter;

public:
    SsoraPcgSolver(const SparseMatrix<T>& matrix,
		   const DenseVector<T>& vector,
		   double thresh,
		   double relax,
		   int iters): mat(matrix),
			       vec(vector),
			       threshold(thresh),
			       relaxation(relax),
			       maxIter(iters) { }
    
    virtual Result<T> solve() const override;
};

template <class T>
Result<T> SsoraPcgSolver<T>::solve() const {
    int dim = mat.dim();
    DenseVector<T> x = DenseVector<T>::zero(dim);
    SparseMatrix<T> preconditioner = mat.ssoraInverse(relaxation);
    DenseVector<T> residual = vec.plusAx(-1, mat * x);
    DenseVector<T> nextResidual = DenseVector<T>::zero(dim);
    DenseVector<T> z = preconditioner * residual;
    DenseVector<T> nextZ = DenseVector<T>::zero(dim);
    T r_dot_z = residual.dot(z);
    DenseVector<T> direction = z;
    DenseVector<T> a_direction = mat * direction;
    
    int count = 0;
    T r_dot_r;
    while ((r_dot_r = residual.normSquared()) > threshold && count < maxIter) {
	
	boost::posix_time::ptime timeLocal = boost::posix_time::second_clock::local_time();
	std::string time = boost::posix_time::to_simple_string(timeLocal);
	std::cout << time << " k = " << count << ", r.r = " << r_dot_r  << "\n";
	
	T stepsize = r_dot_z / (direction.dot(a_direction));

	x.updateAx(stepsize, direction);

	nextResidual = residual.plusAx( -stepsize, a_direction);

	nextZ = preconditioner * nextResidual;
	
	T next_r_dot_z = nextZ.dot(nextResidual) ;
	T update = next_r_dot_z / r_dot_z;
	r_dot_z = next_r_dot_z;
	direction = nextZ.plusAx(update, direction);
	
	a_direction = mat * direction;
	
	residual = std::move(nextResidual);
	z = std::move(nextZ);
	count++;
    }
    return Result<T>(x, count);
}

#endif
