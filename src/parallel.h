#ifndef NUMCPP_PARALLEL_H
#define NUMCPP_PARALLEL_H

namespace numcpp {
/**
 * Initializes all the kernels so they can be used as and when needed by Matrix class
 */
    void init_parallel();

/**
 * Releases all kernel memory allocations -> to be called at the end of any program that uses Matrix class
 */
    void static finish_parallel();
}


#endif //NUMCPP_PARALLEL_H
