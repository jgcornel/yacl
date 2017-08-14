#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "yacl.hpp"

std::string mulKernelSource = R"(
    __kernel void multiply(
        __global int * destination,
        __global int * source,
                 int   factor
    )
    {
        destination[get_global_id(0)] = factor * source[get_global_id(0)];
    }
)";

std::string addKernelSource = R"(
    __kernel void add(
        __global int * destination,
        __global int * source,
                 int   term
    )
    {
        destination[get_global_id(0)] = source[get_global_id(0)] + term;
    }

)";

const size_t N{2<<20};

int main(int argc, char *argv[])
{
try {
    jc::GpuHandle GH{CL_DEVICE_TYPE_GPU};

    cl::Kernel mulKernel = GH.build_kernel(mulKernelSource, "multiply");
    cl::Kernel addKernel = GH.build_kernel(addKernelSource, "add");

    std::vector<int> input(N);
    std::vector<int> temp(N);
    std::vector<int> output(N);
    std::iota(input.begin(), input.end(), 0);

    // intermediate result, should stay on the GPU ...
    auto buffer = GH.allocate(temp);

    int factor = {2};
    int term = {1};
    
    GH.run<int>(mulKernel, cl::NDRange{N}, cl::NDRange{64},
                buffer, std::cref(input), factor);
    GH.run<int>(addKernel,  cl::NDRange{N}, cl::NDRange{64},
                std::ref(output), buffer, term);

    std::transform(input.begin(), input.end(), input.begin(),
        [=](int val) { return factor*val + term; });

    if (!std::equal(input.begin(), input.end(), output.begin()))
        throw std::runtime_error{"erroneous results"};

    return 0;
}
catch(const cl::Error & e) {
    std::cerr << e.what() << ":" << jc::readable_error(e.err()) << "\n";
    return 1;
}
catch(const std::exception & e) {
    std::cerr << e.what() << "\n";
    return 2;
}
catch(...) {
    std::cerr << "Unforeseen error occurred...\n";
    return 3;
}
}