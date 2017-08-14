#pragma once

#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace jc {

namespace detail {

bool equal_strings(const std::string & s1, const std::string & s2)
{
    const std::string & kort = s1.size() < s2.size() ? s1 : s2;
    const std::string & lang = s1.size() < s2.size() ? s2 : s1;

    return std::equal(kort.cbegin(), kort.cend(), lang.cbegin());
}

template <typename Predicate>
cl::Device get_device(const Predicate & pred)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for(const auto & platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        auto it = std::find_if(devices.cbegin(), devices.cend(), pred);
        if (it != devices.cend())
            return *it;
    }
    throw std::runtime_error("No appropriate device found");
}

struct DeviceOfType {
    cl_device_info type;

    bool operator()(const cl::Device & device)
    {
        cl_device_type t;
        device.getInfo(CL_DEVICE_TYPE, &t);
        return type == t;
    }
};

struct DeviceCalled {
    std::string name;

    bool operator()(const cl::Device & device)
    {
        std::string n;
        device.getInfo(CL_DEVICE_NAME, &n);
        return equal_strings(name, n);
    }
};

cl::Program buildProgramFromSource(
    const std::string & source,
    const cl::Context & context,
    const cl::Device & device
    )
{
    cl::Program program{context, source};

    try {
        program.build({device});
    }
    catch (cl::Error&) {
        std::string build_log;
        std::ostringstream oss;
        program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &build_log);
        oss << "Your program failed to compile: " << std::endl;
        oss << "--------------------------------" << std::endl;
        oss << build_log << std::endl;
        oss << "--------------------------------" << std::endl;
        throw std::runtime_error{oss.str()};
    }

    return program;
}

cl::Program buildProgramFromSourceFile(
    const std::string & source_file,
    const cl::Context & context,
    const cl::Device & device
    )
{
    std::string source;

    std::ifstream filestream{source_file.c_str()};
    if (!filestream) {
        std::ostringstream oss;
        oss << "Could not open a file " << source_file;
        throw std::runtime_error{oss.str()};
    }
    source.assign(std::istreambuf_iterator<char>(filestream),
                    std::istreambuf_iterator<char>());

    return buildProgramFromSource(source, context, device);
}

const std::map<cl_int, const char *> error_codes =
{
    {CL_SUCCESS,                                    "CL_SUCCESS"},
    {CL_DEVICE_NOT_FOUND,                           "CL_DEVICE_NOT_FOUND"},
    {CL_DEVICE_NOT_AVAILABLE,                       "CL_DEVICE_NOT_AVAILABLE"},
    {CL_COMPILER_NOT_AVAILABLE,                     "CL_COMPILER_NOT_AVAILABLE"},
    {CL_MEM_OBJECT_ALLOCATION_FAILURE,              "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
    {CL_OUT_OF_RESOURCES,                           "CL_OUT_OF_RESOURCES"},
    {CL_OUT_OF_HOST_MEMORY,                         "CL_OUT_OF_HOST_MEMORY"},
    {CL_PROFILING_INFO_NOT_AVAILABLE,               "CL_PROFILING_INFO_NOT_AVAILABLE"},
    {CL_MEM_COPY_OVERLAP,                           "CL_MEM_COPY_OVERLAP"},
    {CL_IMAGE_FORMAT_MISMATCH,                      "CL_IMAGE_FORMAT_MISMATCH"},
    {CL_IMAGE_FORMAT_NOT_SUPPORTED,                 "CL_IMAGE_FORMAT_NOT_SUPPORTED"},
    {CL_BUILD_PROGRAM_FAILURE,                      "CL_BUILD_PROGRAM_FAILURE"},
    {CL_MAP_FAILURE,                                "CL_MAP_FAILURE"},
    {CL_MISALIGNED_SUB_BUFFER_OFFSET,               "CL_MISALIGNED_SUB_BUFFER_OFFSET"},
    {CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,  "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"},
#ifdef CL_VERSION_1_2
    {CL_COMPILE_PROGRAM_FAILURE,                    "CL_COMPILE_PROGRAM_FAILURE"},
    {CL_LINKER_NOT_AVAILABLE,                       "CL_LINKER_NOT_AVAILABLE"},
    {CL_LINK_PROGRAM_FAILURE,                       "CL_LINK_PROGRAM_FAILURE"},
    {CL_DEVICE_PARTITION_FAILED,                    "CL_DEVICE_PARTITION_FAILED"},
    {CL_KERNEL_ARG_INFO_NOT_AVAILABLE,              "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"},
#endif
    {CL_INVALID_VALUE,                              "CL_INVALID_VALUE"},
    {CL_INVALID_DEVICE_TYPE,                        "CL_INVALID_DEVICE_TYPE"},
    {CL_INVALID_PLATFORM,                           "CL_INVALID_PLATFORM"},
    {CL_INVALID_DEVICE,                             "CL_INVALID_DEVICE"},
    {CL_INVALID_CONTEXT,                            "CL_INVALID_CONTEXT"},
    {CL_INVALID_QUEUE_PROPERTIES,                   "CL_INVALID_QUEUE_PROPERTIES"},
    {CL_INVALID_COMMAND_QUEUE,                      "CL_INVALID_COMMAND_QUEUE"},
    {CL_INVALID_HOST_PTR,                           "CL_INVALID_HOST_PTR"},
    {CL_INVALID_MEM_OBJECT,                         "CL_INVALID_MEM_OBJECT"},
    {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,            "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
    {CL_INVALID_IMAGE_SIZE,                         "CL_INVALID_IMAGE_SIZE"},
    {CL_INVALID_SAMPLER,                            "CL_INVALID_SAMPLER"},
    {CL_INVALID_BINARY,                             "CL_INVALID_BINARY"},
    {CL_INVALID_BUILD_OPTIONS,                      "CL_INVALID_BUILD_OPTIONS"},
    {CL_INVALID_PROGRAM,                            "CL_INVALID_PROGRAM"},
    {CL_INVALID_PROGRAM_EXECUTABLE,                 "CL_INVALID_PROGRAM_EXECUTABLE"},
    {CL_INVALID_KERNEL_NAME,                        "CL_INVALID_KERNEL_NAME"},
    {CL_INVALID_KERNEL_DEFINITION,                  "CL_INVALID_KERNEL_DEFINITION"},
    {CL_INVALID_KERNEL,                             "CL_INVALID_KERNEL"},
    {CL_INVALID_ARG_INDEX,                          "CL_INVALID_ARG_INDEX"},
    {CL_INVALID_ARG_VALUE,                          "CL_INVALID_ARG_VALUE"},
    {CL_INVALID_ARG_SIZE,                           "CL_INVALID_ARG_SIZE"},
    {CL_INVALID_KERNEL_ARGS,                        "CL_INVALID_KERNEL_ARGS"},
    {CL_INVALID_WORK_DIMENSION,                     "CL_INVALID_WORK_DIMENSION"},
    {CL_INVALID_WORK_GROUP_SIZE,                    "CL_INVALID_WORK_GROUP_SIZE"},
    {CL_INVALID_WORK_ITEM_SIZE,                     "CL_INVALID_WORK_ITEM_SIZE"},
    {CL_INVALID_GLOBAL_OFFSET,                      "CL_INVALID_GLOBAL_OFFSET"},
    {CL_INVALID_EVENT_WAIT_LIST,                    "CL_INVALID_EVENT_WAIT_LIST"},
    {CL_INVALID_EVENT,                              "CL_INVALID_EVENT"},
    {CL_INVALID_OPERATION,                          "CL_INVALID_OPERATION"},
    {CL_INVALID_GL_OBJECT,                          "CL_INVALID_GL_OBJECT"},
    {CL_INVALID_BUFFER_SIZE,                        "CL_INVALID_BUFFER_SIZE"},
    {CL_INVALID_MIP_LEVEL,                          "CL_INVALID_MIP_LEVEL"},
    {CL_INVALID_GLOBAL_WORK_SIZE,                   "CL_INVALID_GLOBAL_WORK_SIZE"},
#ifdef CL_VERSION_1_2
    {CL_INVALID_PROPERTY,                           "CL_INVALID_PROPERTY"},
    {CL_INVALID_IMAGE_DESCRIPTOR,                   "CL_INVALID_IMAGE_DESCRIPTOR"},
    {CL_INVALID_COMPILER_OPTIONS,                   "CL_INVALID_COMPILER_OPTIONS"},
    {CL_INVALID_LINKER_OPTIONS,                     "CL_INVALID_LINKER_OPTIONS"},
    {CL_INVALID_DEVICE_PARTITION_COUNT,             "CL_INVALID_DEVICE_PARTITION_COUNT"}
#endif
};

} // namespace jc::detail

///
/// readable_error : cl_int -> const char *
///
/// Given an OpenCL error code, return a human readable C string.
///
const char * readable_error(cl_int e)
{
    if (detail::error_codes.count(e) == 0)
        return "UNKNOWN ERROR CODE";

    return detail::error_codes.at(e);
}
///
/// best_fit : size_t -> size_t -> size_t
///
/// Given a size global and a size local, return the smallest number that is
/// greater than or equal to global and that is a multiple of local.
///
size_t best_fit(size_t global, size_t local)
{
    size_t times = global/local;
    if (local*times != global) ++times;
    return times*local;
}

template <typename U> using MemPair = std::pair<cl::Buffer, std::vector<U>&>;
///
/// GpuHandle.
///
/// A wrapper class to ease the burden of writing OpenCL host code.
///
class GpuHandle {
public:
    // Constructor
    //
    // Given an OpenCL device name try to initialize an OpenCL device,
    // context and command queue. Throw an error if no appropriate device
    // can be found.
    //
    GpuHandle(const std::string & name)
        : device_{jc::detail::get_device(jc::detail::DeviceCalled{name})}
        , context_{cl::Context{device_}}
        , queue_{cl::CommandQueue{context_, device_, CL_QUEUE_PROFILING_ENABLE}}
    {}

    // Constructor
    //
    // Given an OpenCL device type try to initialize an OpenCL device,
    // context and command queue. Throw an error if no appropriate device
    // can be found.
    //
    GpuHandle(cl_device_info type)
        : device_{jc::detail::get_device(jc::detail::DeviceOfType{type})}
        , context_{cl::Context{device_}}
        , queue_{cl::CommandQueue{context_, device_, CL_QUEUE_PROFILING_ENABLE}}
    {}

    // build_kernel
    //
    // Given a string and a kernel name, try to compile the string as OpenCL
    // code and extract a kernel with the given name. Throw an error if the
    // OpenCL code cannot be compiled or no kernel with the given name is
    // contained in the OpenCL code.
    //
    cl::Kernel build_kernel(const std::string & source, const char * name)
    {
        cl::Program program =
            jc::detail::buildProgramFromSource(source, context_, device_);
        return cl::Kernel{program, name};
    }


    // build_kernel_from_file
    //
    // Given a file name and a kernel name, try to open the corresponding
    // file, compile the OpenCL code it contains and extract a kernel
    // with the given name. Throw an error if the file cannot be opened,
    // the OpenCL code cannot be compiled or no kernel with the given name
    // is contained in the OpenCL code.
    //
    cl::Kernel build_kernel_from_file(
        const std::string & file,
        const char * name
        )
    {
        cl::Program program =
            jc::detail::buildProgramFromSourceFile(file, context_, device_);
        return cl::Kernel{program, name};
    }

    // allocate
    //
    // Given a byte size and an optional memory flag, a cl::Buffer of the
    // appropriate size is created and returned.
    //
    cl::Buffer allocate(size_t bytes, cl_mem_flags flags=CL_MEM_READ_WRITE)
    {
        cl::Buffer buffer{context_, flags, bytes};
        return buffer;
    }

    // template <typename U> allocate
    //
    // Given an std::vector<U> and an optional memory flag, a cl::Buffer of
    // the appropriate size is created and returned.
    //
    template <typename U>
    cl::Buffer allocate(
        const std::vector<U>& vector,
        cl_mem_flags flags=CL_MEM_READ_WRITE
        )
    {
        return allocate(sizeof(U)*vector.size(), flags);
    }

    // template <typename U> copy
    //
    // Given an std::vector<U>, a cl::Buffer and an optional blocking
    // indicator, copy the contents of the vector to the buffer. This boils
    // down to writing from the OpenCL host to the OpenCL device.
    //
    template <typename U>
    void copy(
        const std::vector<U>& vector,
        cl::Buffer& buffer,
        cl_bool blocking=CL_TRUE
        )
    {
        size_t bytes = sizeof(U)*vector.size();
        queue_.enqueueWriteBuffer(buffer, blocking, 0, bytes, vector.data());
    }

    // template <typename U> copy
    //
    // Given a cl::Buffer, an std::vector<U> and an optional blocking
    // indicator, copy the contents of the buffer to the vector. This boils
    // down to writing from the OpenCL device to the OpenCL host.
    //
    template <typename U>
    void copy(
        const cl::Buffer & buffer,
        std::vector<U>& vector,
        cl_bool blocking=CL_TRUE
        )
    {
        size_t bytes = sizeof(U)*vector.size();
        queue_.enqueueReadBuffer(buffer, blocking, 0, bytes, vector.data());
    }

    // template <typename U, typename T, typename... Ts> run
    //
    // Given a kernel, a global NDRange, a local NDRange and a variable non
    // zero number of arguments, try to run the kernel with the given global
    // and local NDRanges and using the variable number of arguments as the
    // kernel's arguments.
    //
    // Primitive types like int, float, ... and cl::Buffer types are simply
    // forwarded to the kernel's setArg method taking their position in the
    // argument's list into account. Special treatment is foreseen for
    // std::reference_wrapper<const std::vector<T>> and
    // std::reference_wrapper<std::vector<T>> types. Both are assumed to
    // match a __global T * or __constant T * argument, but the former is
    // assumed to be input and the latter as output. For input an appropriate
    // cl::Buffer is created that is populated with the data from the vector
    // and forwarded to the kernel's setArg method. For output an appropriate
    // cl::Buffer is created and forwarded to the kernel's setArg method.
    // When the kernel has finished, the buffer's contents is transferred to
    // the vector.
    //
    // Note: it is necessary to use the std::cref and std::ref functions to
    // create the appropriate std::reference_wrapper object.
    //
    // Note: all vectors related to output must be of the same type U!
    //
    template <typename U, typename T, typename... Ts>
    void run(
        cl::Kernel kernel,
        cl::NDRange global,
        cl::NDRange local,
        T car,
        Ts... cdr
        )
    {
        std::vector<MemPair<U>> results;
        return run_helper(results, kernel, global, local, 0, car, cdr...);
    }

private:
    cl::Device device_;
    cl::Context context_;
    cl::CommandQueue queue_;

    template <typename U>
    void run_helper(
        std::vector<MemPair<U>>&,
        cl::Kernel,
        cl::NDRange,
        cl::NDRange,
        int
        );

    template <typename U, typename T, typename... Ts>
    void run_helper(
        std::vector<MemPair<U>>&,
        cl::Kernel,
        cl::NDRange,
        cl::NDRange,
        int,
        T,
        Ts...
        );

    template <typename U, typename... Ts>
    void run_helper(
        std::vector<MemPair<U>>&,
        cl::Kernel,
        cl::NDRange,
        cl::NDRange,
        int,
        std::reference_wrapper<std::vector<U>>,
        Ts...);

    template <typename U, typename T, typename... Ts>
    void run_helper(
        std::vector<MemPair<U>>&,
        cl::Kernel,
        cl::NDRange,
        cl::NDRange,
        int,
        std::reference_wrapper<const std::vector<T>>,
        Ts...);

};
} // namespace jc

template <typename U>
void jc::GpuHandle::run_helper(
    std::vector<MemPair<U>>& results,
    cl::Kernel kernel,
    cl::NDRange global,
    cl::NDRange local,
    int
    )
{
    queue_.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
    while (!results.empty()) {
        auto result = results.back();
        size_t bytes = result.second.size() * sizeof(U);
        queue_.enqueueReadBuffer(result.first, CL_TRUE, 0, bytes, result.second.data());
        results.pop_back();
    }
}

template <typename U, typename T, typename... Ts>
void jc::GpuHandle::run_helper(
    std::vector<MemPair<U>>& results,
    cl::Kernel kernel,
    cl::NDRange global,
    cl::NDRange local,
    int num,
    T car,
    Ts... cdr
    )
{
    kernel.setArg(num, car);
    return run_helper(results, kernel, global, local, num+1, cdr...);
}

template <typename U, typename... Ts>
void jc::GpuHandle::run_helper(
    std::vector<MemPair<U>>& results,
    cl::Kernel kernel,
    cl::NDRange global,
    cl::NDRange local,
    int num,
    std::reference_wrapper<std::vector<U>> car,
    Ts... cdr
    )
{
    cl::Buffer buffer{context_, CL_MEM_READ_WRITE, car.get().size()*sizeof(U)};
    results.emplace_back(buffer, car.get());
    kernel.setArg(num, buffer);
    return run_helper(results, kernel, global, local, num+1, cdr...);
}

template <typename U, typename T, typename... Ts>
void jc::GpuHandle::run_helper(
    std::vector<MemPair<U>>& results,
    cl::Kernel kernel,
    cl::NDRange global,
    cl::NDRange local,
    int num,
    std::reference_wrapper<const std::vector<T>> car,
    Ts... cdr
    )
{
    size_t bytes = sizeof(T) * car.get().size();
    cl::Buffer buffer{context_, CL_MEM_READ_WRITE, bytes};
    queue_.enqueueWriteBuffer(buffer, CL_TRUE, 0, bytes, car.get().data());
    kernel.setArg(num, buffer);
    return run_helper(results, kernel, global, local, num+1, cdr...);
}
