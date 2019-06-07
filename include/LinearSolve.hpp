#ifndef LINEAR_SOLVE_HPP
#define LINEAR_SOLVE_HPP

#include "DenseVector.hpp"
#include "SparseMatrix.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

template<class U>
struct Result {
    DenseVector<U> result;
    int iterations;
    U residualNormSquared;
    boost::posix_time::time_duration ssoraDuration;
    boost::posix_time::time_duration pcgDuration;
    Result(const DenseVector<U>& x, int iter, U r_dot_r,
	   const boost::posix_time::time_duration& ssora,
	   const boost::posix_time::time_duration& pcg) : result(x),
							  iterations(iter),
							  residualNormSquared(r_dot_r),
							  ssoraDuration(ssora),
							  pcgDuration(pcg) { }
};

template <class T,class Deleter = std::default_delete<T[]> >
class LinearSolver {
public:
    virtual Result<T> solve() const =0;
};

template <class T >
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
    std::cout << "Solving...\n";
    int dim = mat.dim();
    DenseVector<T> x = DenseVectorFactory::zero<T>(dim);
    std::cout << "Computing ssora inverse...\n";
    boost::posix_time::ptime time_start(boost::posix_time::microsec_clock::local_time());
    SparseMatrix<T> preconditioner = mat.ssoraInverse(relaxation);
    boost::posix_time::ptime time_end(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration ssoraDuration(time_end - time_start);

    time_start = boost::posix_time::microsec_clock::local_time();
    DenseVector<T> residual = vec.plusAx(-1, mat * x);
    DenseVector<T> nextResidual = DenseVectorFactory::zero<T>(dim);
    DenseVector<T> z = preconditioner * residual;
    DenseVector<T> nextZ = DenseVectorFactory::zero<T>(dim);
    T r_dot_z = residual.dot(z);
    DenseVector<T> direction = z;
    DenseVector<T> a_direction = mat * direction;
    
    int count = 0;
    T r_dot_r;
    while ((r_dot_r = residual.normSquared()) > threshold && count < maxIter) {
	if (count % 100 == 0) {
	    boost::posix_time::ptime timeLocal = boost::posix_time::second_clock::local_time();
	    std::string time = boost::posix_time::to_simple_string(timeLocal);
	    std::cout << time << " k = " << count << ", r.r = " << r_dot_r  << "\n";
	}
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
    time_end = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration pcgDuration(time_end - time_start);
    return Result<T>(x, count, r_dot_r, ssoraDuration , pcgDuration);
}

#endif
