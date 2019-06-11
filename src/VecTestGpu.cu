#include "DenseVector.hpp"
#include <iostream>

int main() {
    int dim = 1024;
    DeviceVector<float> vec1 = DeviceVector<float>::constant(dim , 1);
    DeviceVector<float> vec2 = DeviceVector<float>::zero(dim);

    DeviceVector<float> vec3 = vec2.plusAx(-1, vec1);
    DeviceVector<float> vec4(vec2);

    vec4 = vec2.plusAx(2, vec1);
    
    std::unique_ptr<float[]> entries1 = gpu::get_from_device<float>(vec1.entries(), dim);
    std::unique_ptr<float[]> entries2 = gpu::get_from_device<float>(vec2.entries(), dim);
    std::unique_ptr<float[]> entries3 = gpu::get_from_device<float>(vec3.entries(), dim);
    std::unique_ptr<float[]> entries4 = gpu::get_from_device<float>(vec4.entries(), dim);

    for (int i = 0; i < dim; i++) {
	if (entries1[i] != 1) {
	    std::cout << i << ": " << entries1[i] << " != 1\n";
	}
	if (entries2[i] != 0) {
	    std::cout << i << ": " << entries2[i] << " != 0\n";
	}
	if (entries3[i] != -1) {
	    std::cout << i << ": " << entries3[i] << " != -1\n";
	}
	if (entries4[i] != 2) {
	    std::cout << i << ": " << entries4[i] << " != 2\n";
	}
    }
    std::cout << "Done\n";
}