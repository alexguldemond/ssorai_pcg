#ifndef LINEAR_SOLVE_HPP
#define LINEAR_SOLVE_HPP

#include "DenseVector.hpp"
#include "SparseMatrix.hpp"
#include "gpu_memory.cuh"
#include "boost/date_time/posix_time/posix_time.hpp"

template<class T, class Deleter>
struct Extractor {
    static T extract(const std::unique_ptr<T, Deleter>& data) {
	return *data;
    }
};

template<class T>
struct Extractor<T, gpu::CudaDeleter<T>> {
    static T extract(const std::unique_ptr<T, gpu::CudaDeleter<T>>& data) {
	T result = 0;
	gpu::memcpy_to_host(&result, data.get());
	return result;
    }
};

template<class U, class Deleter = std::default_delete<U[]>>
struct Result {
    DenseVector<U, Deleter> result;
    int iterations;
    U residualNormSquared;
    boost::posix_time::time_duration ssoraDuration;
    boost::posix_time::time_duration pcgDuration;
    
    Result(DenseVector<U, Deleter>&& x, int iter, U r_dot_r,
	   boost::posix_time::time_duration&& ssora,
	   boost::posix_time::time_duration&& pcg) : result(std::move(x)),
						     iterations(iter),
						     residualNormSquared(r_dot_r),
						     ssoraDuration(std::move(ssora)),
						     pcgDuration(std::move(pcg)) { }
};

template <class T,class Deleter, class IntDeleter, class ScalarDeleter>
class LinearSolver {
public:
    virtual Result<T, Deleter> solve() const =0;
};

template <class T, class Deleter = std::default_delete<T[]>, class IntDeleter = std::default_delete<int[]>, class ScalarDeleter = std::default_delete<T>>
class SsoraPcgSolver: LinearSolver<T, Deleter, IntDeleter, ScalarDeleter> {
private:
    const SparseMatrix<T, Deleter, IntDeleter>& mat;
    const DenseVector<T, Deleter>& vec;
    double threshold;
    double relaxation;
    int maxIter;

public:
    SsoraPcgSolver(const SparseMatrix<T, Deleter, IntDeleter>& matrix,
		   const DenseVector<T, Deleter>& vector,
		   double thresh,
		   double relax,
		   int iters): mat(matrix),
			       vec(vector),
			       threshold(thresh),
			       relaxation(relax),
			       maxIter(iters) { }
    
    virtual Result<T, Deleter> solve() const override;
};

template <class T, class Deleter, class IntDeleter, class ScalarDeleter>
Result<T, Deleter> SsoraPcgSolver<T, Deleter, IntDeleter, ScalarDeleter>::solve() const {
    std::cout << "Solving...\n";
    int dim = mat.dim();
    
    DenseVector<T, Deleter> x = DenseVector<T, Deleter>::zero(dim);
    
    std::cout << "Computing ssora inverse...\n";
    boost::posix_time::ptime time_start(boost::posix_time::microsec_clock::local_time());
    auto preconditioner = mat.ssoraInverse(relaxation);
    boost::posix_time::ptime time_end(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration ssoraDuration(time_end - time_start);

    time_start = boost::posix_time::microsec_clock::local_time();

    //r = b - a.x = b
    DenseVector<T, Deleter> residual = vec.plusAx(-1, mat * x);
    DenseVector<T, Deleter> nextResidual = DenseVector<T, Deleter>::zero(dim);

    //z = p.r
    DenseVector<T, Deleter> z = preconditioner * residual;
    DenseVector<T, Deleter> nextZ = DenseVector<T, Deleter>::zero(dim);

    //cache r dot z
    std::unique_ptr<T, ScalarDeleter> saved_r_dot_z = std::unique_ptr<T, ScalarDeleter>(Allocater<T, ScalarDeleter>::allocate(1), ScalarDeleter());
    residual.dot(z, saved_r_dot_z);
    T r_dot_z = Extractor<T, ScalarDeleter>::extract(saved_r_dot_z);

    //d = z
    DenseVector<T, Deleter> direction = z;
    DenseVector<T, Deleter> a_direction = mat * direction;

    //cache r dot z
    std::unique_ptr<T, ScalarDeleter> saved_r_dot_r = std::unique_ptr<T, ScalarDeleter>(Allocater<T, ScalarDeleter>::allocate(1), ScalarDeleter());
    residual.dot(residual, saved_r_dot_r);
    T r_dot_r;

    std::unique_ptr<T, ScalarDeleter> saved_d_a_d = std::unique_ptr<T, ScalarDeleter>(Allocater<T, ScalarDeleter>::allocate(1), ScalarDeleter());
    direction.dot(a_direction, saved_d_a_d);

    std::unique_ptr<T, ScalarDeleter> saved_next_r_dot_z = std::unique_ptr<T, ScalarDeleter>(Allocater<T, ScalarDeleter>::allocate(1), ScalarDeleter());
    
    int count = 0;
    while ((r_dot_r = Extractor<T, ScalarDeleter>::extract(saved_r_dot_r)) > threshold && count < maxIter) {
	if (count % 100 == 0) {
	    boost::posix_time::ptime timeLocal = boost::posix_time::second_clock::local_time();
	    std::string time = boost::posix_time::to_simple_string(timeLocal);
	    std::cout << time << " k = " << count << ", r.r = " << r_dot_r  << "\n";
	}

	//Compute alpha = r.z/d.a.d
	T stepSize = r_dot_z / Extractor<T, ScalarDeleter>::extract(saved_d_a_d);

	//x = x + alpha*d
	x.updateAx(stepSize, direction);

	//r = r - alpha*a.d
	nextResidual = residual.plusAx( -stepSize, a_direction);

	//z = p.r
	nextZ = preconditioner * nextResidual;

	//next_r.z = r.z
	nextResidual.dot(nextZ, saved_next_r_dot_z);
	T next_r_dot_z = Extractor<T, ScalarDeleter>::extract(saved_next_r_dot_z);

	//beta = rk+1 . zk+1 / rk . zk
	T update = next_r_dot_z / r_dot_z;
	r_dot_z = next_r_dot_z;

	//d = z + beta*d
	direction = nextZ.plusAx(update, direction);

	//a_d = a.d
	a_direction = mat * direction;

	//Update
	residual = std::move(nextResidual);
	z = std::move(nextZ);
	direction.dot(a_direction, saved_d_a_d);
	residual.dot(residual, saved_r_dot_r);
	count++;
    }
    time_end = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration pcgDuration(time_end - time_start);
    return Result<T, Deleter>(std::move(x), count, r_dot_r, std::move(ssoraDuration) , std::move(pcgDuration));
}

#endif
