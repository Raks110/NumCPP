#include "matrix.h"
#include "parallel.h"

#include <iostream>
#include <CL/cl2.hpp>

namespace numcpp {

    /**
     * variables needed by the program
     */

    //OpenCL based variables needed
    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    //These kernels have been overloaded on operators (Matrix-on-Matrix)
    cl_kernel kernel_add;
    cl_kernel kernel_subtract;
    cl_kernel kernel_multiply;
    cl_kernel kernel_gt;
    cl_kernel kernel_lt;
    cl_kernel kernel_equals;
    cl_kernel kernel_gte;
    cl_kernel kernel_lte;

    //These kernels have been overloaded on operators (Matrix-on-Scalar)
    cl_kernel scalar_kernel_multiply;
    cl_kernel scalar_kernel_gt;
    cl_kernel scalar_kernel_lt;
    cl_kernel scalar_kernel_equals;
    cl_kernel scalar_kernel_gte;
    cl_kernel scalar_kernel_lte;
    cl_kernel scalar_kernel_power;
    cl_kernel scalar_kernel_adder;
    cl_kernel scalar_kernel_subtracter;

    //These kernels have been implemented as functions
    cl_kernel matrix_kernel_multiply;
    cl_kernel matrix_kernel_transpose;

    MatrixStatus::MatrixStatus(std::string error, int code) {

        this->error_message = std::move(error);
        this->error_code = code;
    }

    std::string MatrixStatus::get_error_message() {
        return this->error_message;
    }

    int MatrixStatus::get_error_code() {
        return this->error_code;
    }

    Matrix::Matrix(size_t rows, size_t columns, int limit) {

        this->rows = rows;
        this->columns = columns;

        set_matrix(new float[get_rows() * get_columns()]);
        initialize_matrix(limit);
    }

    Matrix::Matrix(size_t rows, size_t columns, Reader* reader) {

        this->rows = rows;
        this->columns = columns;

        set_matrix(new float[get_rows() * get_columns()]);
        initialize_matrix(reader);
    }

    MatrixStatus Matrix::initialize_matrix(int limit) {

        try {

            for (long long int i = 0; i < rows; i++) {

                for (long long int j = 0; j < columns; j++) {

                    set_element(i, j, rand() % limit);
                }
            }

            return MatrixStatus("Success", 0);
        }
        catch (errno_t e) {

            return MatrixStatus("Error Reading Values. Only float Values Supported.", 1);
        }
    }

    MatrixStatus Matrix::initialize_matrix(Reader* reader) {

        try {

            for (long long int i = 0; i < rows; i++) {

                for (long long int j = 0; j < columns; j++) {

                    float element = reader->read();
                    set_element(i, j, element);
                }
            }

            return MatrixStatus("Success", 0);
        }
        catch (errno_t e) {

            return MatrixStatus("Error Reading Values. Only float Values Supported.", 1);
        }
    }

    float Matrix::get_element(size_t row, size_t column) const {
        return this->matrix[row * (this->columns) + column];
    }

    void Matrix::set_element(size_t row, size_t column, float value) {
        this->matrix[row * (this->columns) + column] = value;
    }

    float* Matrix::get_matrix() const {
        return this->matrix;
    }

    size_t Matrix::get_rows() const {

        return this->rows;
    }

    size_t Matrix::get_columns() const {

        return this->columns;
    }

    void Matrix::set_matrix(float* mat) {
        this->matrix = mat;
    }

    void Matrix::clean_up() {

        delete[] this->matrix;
    }

    MatrixStatus Matrix::ones(float multiple = 1) {

        try {
            set_matrix(new float[get_rows() * get_columns()]);

            for (long long int i = 0; i < rows; i++) {

                for (long long int j = 0; j < columns; j++) {

                    set_element(i, j, 1 * multiple);
                }
            }

            return MatrixStatus("Success", 0);
        }
        catch (errno_t e) {

            return MatrixStatus("Error Reading Values. Only float Values Supported.", 1);
        }
    }

    MatrixStatus Matrix::zeroes() {

        try {
            set_matrix(new float[get_rows() * get_columns()]);

            for (long long int i = 0; i < rows; i++) {

                for (long long int j = 0; j < columns; j++) {

                    set_element(i, j, 0);
                }
            }

            return MatrixStatus("Success", 0);
        }
        catch (errno_t e) {

            return MatrixStatus("Error Reading Values. Only float Values Supported.", 1);
        }
    }

    MatrixStatus Matrix::identity(float multiple = 1) {

        try {
            set_matrix(new float[get_rows() * get_columns()]);

            for (long long int i = 0; i < rows; i++) {

                for (long long int j = 0; j < columns; j++) {

                    if (i != j)
                        set_element(i, j, 0);
                    else
                        set_element(i, j, 1 * multiple);
                }
            }

            return MatrixStatus("Success", 0);
        }
        catch (errno_t e) {

            return MatrixStatus("Error Reading Values. Only float Values Supported.", 1);
        }
    }

    std::ostream& operator<<(std::ostream& os, Matrix const& v) {
        for (long long int i = 0; i < v.get_rows(); i++) {
            for (long long int j = 0; j < v.get_columns(); j++) {
                os << v.get_element(i, j) << "\t";
            }
            os << "\n";
        }
        return os;
    }

    int is_broadcast_possible(const Matrix* valid_a, const Matrix* valid_b, long long int* highest_cols,
                              long long int* highest_rows) {
        int flag = 0;

        if ((valid_a->get_rows() >= valid_b->get_rows() && valid_a->get_rows() % valid_b->get_rows() == 0
             && valid_a->get_columns() >= valid_b->get_columns() &&
             valid_a->get_columns() % valid_b->get_columns() == 0)) {

            flag = 2;

            *highest_cols = valid_a->get_columns();
            *highest_rows = valid_a->get_rows();
        }
        else if ((valid_a->get_rows() <= valid_b->get_rows() && valid_b->get_rows() % valid_a->get_rows() == 0
                  && valid_a->get_columns() <= valid_b->get_columns() &&
                  valid_b->get_columns() % valid_a->get_columns() == 0)) {

            flag = 1;

            *highest_cols = valid_b->get_columns();
            *highest_rows = valid_b->get_rows();
        }
        else {
            flag = 0;

            *highest_cols = -1;
            *highest_rows = -1;
        }

        return flag;
    }

    int broadcast2(const Matrix* valid_a, const Matrix* valid_b, float* output_a, float* output_b, int flag) {

        if (flag == 0)
            return flag;
        else {

            if (flag == 2) {

                for (long long int i = 0; i < valid_a->get_rows(); i++) {

                    for (long long int j = 0; j < valid_a->get_columns(); j++) {

                        output_a[i * valid_a->get_columns() + j] = valid_a->get_element(i, j);
                    }
                }

                long long int index = 0;

                for (long long int i = 0; i < valid_b->get_rows(); i++) {

                    for (long long int j = 0; j < valid_b->get_columns(); j++) {

                        output_b[index] = valid_b->get_matrix()[i * valid_b->get_columns() + j];
                        index++;
                    }

                    index -= valid_b->get_columns();
                    index += valid_a->get_columns();
                }

                long long int rows_rep = valid_a->get_rows() / valid_b->get_rows();
                long long int jump = valid_b->get_rows() * valid_a->get_columns();

                for (long long int i = 0; i < rows_rep - 1; i++) {

                    long long int k = 0;

                    for (long long int j = i * jump; j < i * jump + jump; j++) {

                        output_b[j + jump] = output_b[j];

                        k++;

                        if (k >= valid_b->get_columns()) {

                            k = 0;
                            j -= valid_b->get_columns();
                            j += valid_a->get_columns();
                        }
                    }
                }

                for (long long int i = 0; i < valid_a->get_rows(); i++) {

                    long long int curr_position = i * valid_a->get_columns();

                    for (long long int j = curr_position;
                         j <= curr_position + (valid_a->get_columns() - 1 - valid_b->get_columns()); j++) {

                        output_b[j + valid_b->get_columns()] = output_b[j];
                    }
                }

            }
            else if (flag == 1) {

                for (long long int i = 0; i < valid_b->get_rows(); i++) {

                    for (long long int j = 0; j < valid_b->get_columns(); j++) {

                        output_b[i * valid_b->get_columns() + j] = valid_b->get_element(i, j);
                    }
                }

                long long int index = 0;

                for (long long int i = 0; i < valid_a->get_rows(); i++) {

                    for (long long int j = 0; j < valid_a->get_columns(); j++) {

                        output_a[index] = valid_a->get_matrix()[i * valid_a->get_columns() + j];
                        index++;
                    }

                    index -= valid_a->get_columns();
                    index += valid_b->get_columns();
                }

                long long int rows_rep = valid_b->get_rows() / valid_a->get_rows();
                long long int jump = valid_a->get_rows() * valid_b->get_columns();

                for (long long int i = 0; i < rows_rep - 1; i++) {

                    long long int k = 0;

                    for (long long int j = i * jump; j < i * jump + jump; j++) {

                        output_a[j + jump] = output_a[j];

                        k++;

                        if (k >= valid_a->get_columns()) {

                            k = 0;
                            j -= valid_a->get_columns();
                            j += valid_b->get_columns();
                        }
                    }
                }

                for (long long int i = 0; i < valid_b->get_rows(); i++) {

                    long long int curr_position = i * valid_b->get_columns();

                    for (long long int j = curr_position;
                         j <= curr_position + (valid_b->get_columns() - 1 - valid_a->get_columns()); j++) {

                        output_a[j + valid_a->get_columns()] = output_a[j];
                    }
                }
            }

            return flag;
        }
    }

    cl_mem get_memory_buffer(size_t size, int buffer_type = CL_MEM_READ_ONLY) {

        cl_int ret;

        cl_mem buffer = clCreateBuffer(context, buffer_type,
                                       size, nullptr, &ret);

        if (ret != 0) {
            throw MatrixStatus("Memory buffer could not be created.", 92);
        }

        return buffer;
    }

    void enqueue_write(cl_mem buffer, Matrix matrix) {

        cl_int ret = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0,
                                          matrix.get_rows() * matrix.get_columns() * sizeof(float), matrix.get_matrix(),
                                          0, nullptr, nullptr);

        if (ret != 0) {

            throw MatrixStatus("Memory buffer value could not be set.", 93);
        }
    }

    void enqueue_write(cl_mem buffer, float item) {

        cl_int ret = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0,
                                          sizeof(float), &item,
                                          0, nullptr, nullptr);

        if (ret != 0) {

            throw MatrixStatus("Memory buffer value could not be set.", 93);
        }
    }

    void enqueue_write(cl_mem buffer, size_t size, float* matrix) {

        cl_int ret = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0,
                                          size, matrix,
                                          0, nullptr, nullptr);

        if (ret != 0) {

            throw MatrixStatus("Memory buffer value could not be set.", 93);
        }
    }

    void set_argument(cl_kernel kernel, int argument_position, void* argument, size_t size = sizeof(int)) {

        cl_int ret = clSetKernelArg(kernel, argument_position, size, argument);

        if (ret != 0) {

            throw MatrixStatus("Kernel arguments could not be set.", 94);
        }
    }

    void synchronize() {

        cl_int ret = clFinish(queue);

        if (ret != 0) {

            throw MatrixStatus("Error synchronizing kernel tasks.", 96);
        }
    }

    void release(cl_mem buffer) {

        cl_int ret = clReleaseMemObject(buffer);

        if (ret != 0) {

            std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
        }
    }

    Matrix matmul(Matrix a, Matrix b) {

        cl_int ret;
        auto* output = new float[a.get_rows() * b.get_columns()];

        Matrix result(a.get_rows(), b.get_columns());

        cl_mem memory_input_a = get_memory_buffer(a.get_rows() * a.get_columns() * sizeof(float));
        cl_mem memory_input_b = get_memory_buffer(b.get_rows() * b.get_columns() * sizeof(float));
        cl_mem memory_output_a = get_memory_buffer(a.get_rows() * b.get_columns() * sizeof(float), CL_MEM_WRITE_ONLY);

        enqueue_write(memory_input_a, a);
        enqueue_write(memory_input_b, b);

        size_t rows = a.get_rows(), cols = b.get_columns(), inter = b.get_rows();

        set_argument(matrix_kernel_multiply, 0, (void*)&rows);
        set_argument(matrix_kernel_multiply, 1, (void*)&cols);
        set_argument(matrix_kernel_multiply, 2, (void*)&inter);
        set_argument(matrix_kernel_multiply, 3, (void*)&memory_input_a, sizeof(cl_mem));
        set_argument(matrix_kernel_multiply, 4, (void*)&memory_input_b, sizeof(cl_mem));
        set_argument(matrix_kernel_multiply, 5, (void*)&memory_output_a, sizeof(cl_mem));

        const size_t local_work_size[2] = { 1, 1 };
        const size_t global_work_size[2] = { a.get_rows(), b.get_columns() };

        ret = clEnqueueNDRangeKernel(queue, matrix_kernel_multiply, 2, nullptr,
                                     global_work_size, local_work_size, 0, nullptr, nullptr);

        if (ret != 0)
            throw MatrixStatus("Error launching kernel.", 95);

        synchronize();

        ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                  a.get_rows() * b.get_columns() * sizeof(float), output, 0, nullptr, nullptr);

        if (ret != 0)
            throw MatrixStatus("Error reading output from kernel.", 97);

        release(memory_input_a);
        release(memory_input_b);
        release(memory_output_a);

        result.set_matrix(output);
        return result;
    }

    Matrix transpose(Matrix a) {

        cl_int ret;
        auto* output = new float[a.get_rows() * a.get_columns()];

        Matrix result(a.get_columns(), a.get_rows());

        cl_mem memory_input_a = get_memory_buffer(a.get_rows() * a.get_columns() * sizeof(float));
        cl_mem memory_output_a = get_memory_buffer(a.get_rows() * a.get_columns() * sizeof(float), CL_MEM_WRITE_ONLY);

        enqueue_write(memory_input_a, a);

        int cols = a.get_columns();

        set_argument(matrix_kernel_transpose, 0, (void*)&cols);
        set_argument(matrix_kernel_transpose, 1, (void*)&memory_input_a, sizeof(cl_mem));
        set_argument(matrix_kernel_transpose, 2, (void*)&memory_output_a, sizeof(cl_mem));

        const size_t local_work_size[2] = { 1, 1 };
        const size_t global_work_size[2] = { a.get_rows(), a.get_columns() };

        ret = clEnqueueNDRangeKernel(queue, matrix_kernel_transpose, 2, nullptr,
                                     global_work_size, local_work_size, 0, nullptr, nullptr);

        if (ret != 0) {

            throw MatrixStatus("Error launching kernel.", 95);
        }

        synchronize();

        ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                  a.get_rows() * a.get_columns() * sizeof(float), output, 0, nullptr, nullptr);

        if (ret != 0) {

            throw MatrixStatus("Error reading output from kernel.", 97);
        }

        release(memory_input_a);
        release(memory_output_a);

        result.set_matrix(output);
        return result;
    }

    Matrix operator+(Matrix& first, Matrix const& second) {

        try {

            cl_int ret;

            float* output_a = nullptr, * output_b = nullptr;
            long long int columns_highest = 0, rows_highest = 0;

            int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }
            else {

                output_a = new float[rows_highest * columns_highest];
                output_b = new float[rows_highest * columns_highest];
            }

            flag = broadcast2(&first, &second, output_a, output_b, flag);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }

            Matrix result(rows_highest, columns_highest);

            cl_mem memory_input_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_input_b = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, rows_highest * columns_highest * sizeof(float), output_a);
            enqueue_write(memory_input_b, rows_highest * columns_highest * sizeof(float), output_b);

            set_argument(kernel_add, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(kernel_add, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(kernel_add, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = rows_highest * columns_highest;
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, kernel_add, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[rows_highest * columns_highest];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);

            delete[] output_a;
            delete[] output_b;

            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator-(Matrix& first, Matrix const& second) {

        try {

            cl_int ret;

            float* output_a = nullptr, * output_b = nullptr;
            long long int columns_highest = 0, rows_highest = 0;

            int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }
            else {

                output_a = new float[rows_highest * columns_highest];
                output_b = new float[rows_highest * columns_highest];
            }

            flag = broadcast2(&first, &second, output_a, output_b, flag);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }

            Matrix result(rows_highest, columns_highest);

            cl_mem memory_input_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_input_b = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float));

            enqueue_write(memory_input_a, rows_highest * columns_highest * sizeof(float), output_a);
            enqueue_write(memory_input_b, rows_highest * columns_highest * sizeof(float), output_b);

            set_argument(kernel_subtract, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(kernel_subtract, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(kernel_subtract, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = rows_highest * columns_highest;
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, kernel_subtract, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[rows_highest * columns_highest];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);

            delete[] output_a;
            delete[] output_b;

            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator*(Matrix& first, Matrix const& second) {

        try {

            cl_int ret;

            float* output_a = nullptr, * output_b = nullptr;
            long long int columns_highest = 0, rows_highest = 0;

            int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }
            else {

                output_a = new float[rows_highest * columns_highest];
                output_b = new float[rows_highest * columns_highest];
            }

            flag = broadcast2(&first, &second, output_a, output_b, flag);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }

            Matrix result(rows_highest, columns_highest);

            cl_mem memory_input_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_input_b = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, rows_highest * columns_highest * sizeof(float), output_a);
            enqueue_write(memory_input_b, rows_highest * columns_highest * sizeof(float), output_b);

            set_argument(kernel_multiply, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(kernel_multiply, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(kernel_multiply, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = rows_highest * columns_highest;
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, kernel_multiply, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0)
                throw MatrixStatus("Error launching kernel.", 95);

            synchronize();

            auto* output_final = new float[rows_highest * columns_highest];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);

            delete[] output_a;
            delete[] output_b;

            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator>(Matrix& first, Matrix const& second) {

        try {

            cl_int ret;

            float* output_a = nullptr, * output_b = nullptr;
            long long int columns_highest = 0, rows_highest = 0;

            int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }
            else {

                output_a = new float[rows_highest * columns_highest];
                output_b = new float[rows_highest * columns_highest];
            }

            flag = broadcast2(&first, &second, output_a, output_b, flag);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }

            Matrix result(rows_highest, columns_highest);

            cl_mem memory_input_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_input_b = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, rows_highest * columns_highest * sizeof(float), output_a);
            enqueue_write(memory_input_b, rows_highest * columns_highest * sizeof(float), output_b);

            set_argument(kernel_gt, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(kernel_gt, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(kernel_gt, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = rows_highest * columns_highest;
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, kernel_gt, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[rows_highest * columns_highest];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);

            delete[] output_a;
            delete[] output_b;

            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator<(Matrix& first, Matrix const& second) {

        try {

            cl_int ret;

            float* output_a = nullptr, * output_b = nullptr;
            long long int columns_highest = 0, rows_highest = 0;

            int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }
            else {

                output_a = new float[rows_highest * columns_highest];
                output_b = new float[rows_highest * columns_highest];
            }

            flag = broadcast2(&first, &second, output_a, output_b, flag);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }

            Matrix result(rows_highest, columns_highest);

            cl_mem memory_input_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_input_b = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, rows_highest * columns_highest * sizeof(float), output_a);
            enqueue_write(memory_input_b, rows_highest * columns_highest * sizeof(float), output_b);

            set_argument(kernel_lt, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(kernel_lt, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(kernel_lt, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = rows_highest * columns_highest;
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, kernel_lt, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[rows_highest * columns_highest];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);

            delete[] output_a;
            delete[] output_b;

            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator==(Matrix& first, Matrix const& second) {

        try {

            cl_int ret;

            float* output_a = nullptr, * output_b = nullptr;
            long long int columns_highest = 0, rows_highest = 0;

            int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }
            else {

                output_a = new float[rows_highest * columns_highest];
                output_b = new float[rows_highest * columns_highest];
            }

            flag = broadcast2(&first, &second, output_a, output_b, flag);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }

            Matrix result(rows_highest, columns_highest);

            cl_mem memory_input_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_input_b = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, rows_highest * columns_highest * sizeof(float), output_a);
            enqueue_write(memory_input_b, rows_highest * columns_highest * sizeof(float), output_b);

            set_argument(kernel_equals, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(kernel_equals, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(kernel_equals, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = rows_highest * columns_highest;
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, kernel_equals, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[rows_highest * columns_highest];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);

            delete[] output_a;
            delete[] output_b;

            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator>=(Matrix& first, Matrix const& second) {

        try {

            cl_int ret;

            float* output_a = nullptr, * output_b = nullptr;
            long long int columns_highest = 0, rows_highest = 0;

            int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }
            else {

                output_a = new float[rows_highest * columns_highest];
                output_b = new float[rows_highest * columns_highest];
            }

            flag = broadcast2(&first, &second, output_a, output_b, flag);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }

            Matrix result(rows_highest, columns_highest);

            cl_mem memory_input_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_input_b = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, rows_highest * columns_highest * sizeof(float), output_a);
            enqueue_write(memory_input_b, rows_highest * columns_highest * sizeof(float), output_b);

            set_argument(kernel_gte, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(kernel_gte, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(kernel_gte, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = rows_highest * columns_highest;
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, kernel_gte, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[rows_highest * columns_highest];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);

            delete[] output_a;
            delete[] output_b;

            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator<=(Matrix& first, Matrix const& second) {

        try {

            cl_int ret;

            float* output_a = nullptr, * output_b = nullptr;
            long long int columns_highest = 0, rows_highest = 0;

            int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }
            else {

                output_a = new float[rows_highest * columns_highest];
                output_b = new float[rows_highest * columns_highest];
            }

            flag = broadcast2(&first, &second, output_a, output_b, flag);

            if (flag == 0) {

                throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
            }

            Matrix result(rows_highest, columns_highest);

            cl_mem memory_input_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_input_b = get_memory_buffer(rows_highest * columns_highest * sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(rows_highest * columns_highest * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, rows_highest * columns_highest * sizeof(float), output_a);
            enqueue_write(memory_input_b, rows_highest * columns_highest * sizeof(float), output_b);

            set_argument(kernel_lte, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(kernel_lte, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(kernel_lte, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = rows_highest * columns_highest;
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, kernel_lte, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[rows_highest * columns_highest];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);

            delete[] output_a;
            delete[] output_b;

            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

//Scalar Operations

    Matrix operator*(Matrix& first, float const& second) {

        try {

            cl_int ret;

            Matrix result(first.get_rows(), first.get_columns());

            cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                                   first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

            if (ret != 0) {

                throw MatrixStatus("Memory buffer could not be created.", 92);
            }

            cl_mem memory_input_b = get_memory_buffer(sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(first.get_rows() * first.get_columns() * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, first);
            enqueue_write(memory_input_b, second);

            set_argument(scalar_kernel_multiply, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(scalar_kernel_multiply, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(scalar_kernel_multiply, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = first.get_columns() * first.get_rows();
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, scalar_kernel_multiply, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[first.get_columns() * first.get_rows()];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);
            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator>(Matrix& first, float const& second) {

        try {

            cl_int ret;

            Matrix result(first.get_rows(), first.get_columns());

            cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                                   first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

            if (ret != 0) {

                throw MatrixStatus("Memory buffer could not be created.", 92);
            }

            cl_mem memory_input_b = get_memory_buffer(sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(first.get_rows() * first.get_columns() * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, first);
            enqueue_write(memory_input_b, second);

            set_argument(scalar_kernel_gt, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(scalar_kernel_gt, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(scalar_kernel_gt, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = first.get_columns() * first.get_rows();
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, scalar_kernel_gt, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[first.get_columns() * first.get_rows()];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);
            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator<(Matrix& first, float const& second) {

        try {

            cl_int ret;

            Matrix result(first.get_rows(), first.get_columns());

            cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                                   first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

            if (ret != 0) {

                throw MatrixStatus("Memory buffer could not be created.", 92);
            }

            cl_mem memory_input_b = get_memory_buffer(sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(first.get_rows() * first.get_columns() * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, first);
            enqueue_write(memory_input_b, second);

            set_argument(scalar_kernel_lt, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(scalar_kernel_lt, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(scalar_kernel_lt, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = first.get_columns() * first.get_rows();
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, scalar_kernel_lt, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[first.get_columns() * first.get_rows()];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);
            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator==(Matrix& first, float const& second) {

        try {

            cl_int ret;

            Matrix result(first.get_rows(), first.get_columns());

            cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                                   first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

            if (ret != 0) {

                throw MatrixStatus("Memory buffer could not be created.", 92);
            }

            cl_mem memory_input_b = get_memory_buffer(sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(first.get_rows() * first.get_columns() * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, first);
            enqueue_write(memory_input_b, second);

            set_argument(scalar_kernel_equals, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(scalar_kernel_equals, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(scalar_kernel_equals, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = first.get_columns() * first.get_rows();
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, scalar_kernel_equals, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[first.get_columns() * first.get_rows()];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);
            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator>=(Matrix& first, float const& second) {

        try {

            cl_int ret;

            Matrix result(first.get_rows(), first.get_columns());

            cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                                   first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

            if (ret != 0) {

                throw MatrixStatus("Memory buffer could not be created.", 92);
            }

            cl_mem memory_input_b = get_memory_buffer(sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(first.get_rows() * first.get_columns() * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, first);
            enqueue_write(memory_input_b, second);

            set_argument(scalar_kernel_gte, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(scalar_kernel_gte, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(scalar_kernel_gte, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = first.get_columns() * first.get_rows();
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, scalar_kernel_gte, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[first.get_columns() * first.get_rows()];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);
            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator<=(Matrix& first, float const& second) {

        try {

            cl_int ret;

            Matrix result(first.get_rows(), first.get_columns());

            cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                                   first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

            if (ret != 0) {

                throw MatrixStatus("Memory buffer could not be created.", 92);
            }

            cl_mem memory_input_b = get_memory_buffer(sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(first.get_rows() * first.get_columns() * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, first);
            enqueue_write(memory_input_b, second);

            set_argument(scalar_kernel_lte, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(scalar_kernel_lte, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(scalar_kernel_lte, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = first.get_columns() * first.get_rows();
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, scalar_kernel_lte, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[first.get_columns() * first.get_rows()];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);
            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator^(Matrix& first, float const& second) {

        try {

            cl_int ret;

            Matrix result(first.get_rows(), first.get_columns());

            cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                                   first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

            if (ret != 0) {

                throw MatrixStatus("Memory buffer could not be created.", 92);
            }

            cl_mem memory_input_b = get_memory_buffer(sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(first.get_rows() * first.get_columns() * sizeof(float), CL_MEM_WRITE_ONLY);

            enqueue_write(memory_input_a, first);
            enqueue_write(memory_input_b, second);

            set_argument(scalar_kernel_power, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(scalar_kernel_power, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(scalar_kernel_power, 2, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = first.get_columns() * first.get_rows();
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, scalar_kernel_power, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[first.get_columns() * first.get_rows()];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);
            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator+(Matrix& first, float const& second) {

        try {

            cl_int ret;

            Matrix result(first.get_rows(), first.get_columns());

            cl_mem memory_input_a = get_memory_buffer(first.get_rows() * first.get_columns() * sizeof(float));
            cl_mem memory_input_b = get_memory_buffer(sizeof(float));
            cl_mem memory_input_c = get_memory_buffer(sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(first.get_rows() * first.get_columns() * sizeof(float), CL_MEM_WRITE_ONLY);

            float num_columns = first.get_columns();

            enqueue_write(memory_input_a, first);
            enqueue_write(memory_input_b, second);
            enqueue_write(memory_input_c, num_columns);

            set_argument(scalar_kernel_adder, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(scalar_kernel_adder, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(scalar_kernel_adder, 2, (void*)&memory_input_c, sizeof(cl_mem));
            set_argument(scalar_kernel_adder, 3, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = first.get_columns() * first.get_rows();
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, scalar_kernel_adder, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[first.get_columns() * first.get_rows()];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);
            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    Matrix operator-(Matrix& first, float const& second) {

        try {

            cl_int ret;

            Matrix result(first.get_rows(), first.get_columns());

            cl_mem memory_input_a = get_memory_buffer(first.get_rows() * first.get_columns() * sizeof(float));
            cl_mem memory_input_b = get_memory_buffer(sizeof(float));
            cl_mem memory_input_c = get_memory_buffer(sizeof(float));
            cl_mem memory_output_a = get_memory_buffer(first.get_rows() * first.get_columns() * sizeof(float), CL_MEM_WRITE_ONLY);

            float num_columns = first.get_columns();

            enqueue_write(memory_input_a, first);
            enqueue_write(memory_input_b, second);
            enqueue_write(memory_input_c, num_columns);

            set_argument(scalar_kernel_subtracter, 0, (void*)&memory_input_a, sizeof(cl_mem));
            set_argument(scalar_kernel_subtracter, 1, (void*)&memory_input_b, sizeof(cl_mem));
            set_argument(scalar_kernel_subtracter, 2, (void*)&memory_input_c, sizeof(cl_mem));
            set_argument(scalar_kernel_subtracter, 3, (void*)&memory_output_a, sizeof(cl_mem));

            size_t global_work_size = first.get_columns() * first.get_rows();
            size_t local_work_size = 1;

            ret = clEnqueueNDRangeKernel(queue, scalar_kernel_subtracter, 1, nullptr,
                                         &global_work_size, &local_work_size, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error launching kernel.", 95);
            }

            synchronize();

            auto* output_final = new float[first.get_columns() * first.get_rows()];
            ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
                                      first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error reading output from kernel.", 97);
            }

            release(memory_input_a);
            release(memory_input_b);
            release(memory_output_a);

            result.set_matrix(output_final);
            return result;
        }
        catch (MatrixStatus& status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    float dominant_eigen(Matrix matrix, Matrix& eigen_vector, float tolerable_error = 0.0001) {

        if (matrix.get_rows() != matrix.get_columns()) {

            std::cerr << "108: ERROR: Eigen values supported only for square matrices.\n";
            return 0.0f;
        }

        float possible_eigen = 0.0f;

        Matrix initialGuess = Matrix(matrix.get_rows(), 1);
        Matrix newPossibleSolution = Matrix(matrix.get_rows(), 1);

        initialGuess.ones();
        newPossibleSolution.zeroes();

        float lambda_old = 1, lambda_new = 1, temp;
        up:
        for (int i = 0; i < matrix.get_rows(); i++) {
            temp = 0.0;
            for (int j = 0; j < matrix.get_rows(); j++)
            {
                temp = temp + matrix.get_element(i, j) * initialGuess.get_element(j, 0);
            }

            newPossibleSolution.set_element(i, 0, temp);
        }

        for (int i = 0; i < matrix.get_rows(); i++) {
            initialGuess.set_element(i, 0, newPossibleSolution.get_element(i, 0));
        }

        /* Finding Largest */
        lambda_new = fabs(initialGuess.get_element(0, 0));
        for (int i = 1; i < matrix.get_rows(); i++) {
            if (fabs(initialGuess.get_element(i, 0)) > lambda_new)
            {
                lambda_new = fabs(initialGuess.get_element(i, 0));
            }
        }

        /* Normalization */
        for (int i = 0; i < matrix.get_rows(); i++) {
            initialGuess.set_element(i, 0, initialGuess.get_element(i, 0) / lambda_new);
        }

        eigen_vector = Matrix(initialGuess);

        possible_eigen = lambda_new;

        /* Checking Accuracy */
        if (fabs(lambda_new - lambda_old) > tolerable_error) {
            lambda_old = lambda_new;
            goto up;
        }
        else {
            return possible_eigen;
        }
    }

    std::string kernelCode() {
        return "kernel void parallel_adder(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] + b[i];  }    kernel void parallel_subtracter(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] - b[i];  }    kernel void parallel_multiplier(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] * b[i];  }    kernel void parallel_gt(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] > b[i];  }    kernel void parallel_lt(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] < b[i];  }    kernel void parallel_equals(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] == b[i];  }    kernel void parallel_gte(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] >= b[i];  }    kernel void parallel_lte(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] <= b[i];  }    kernel void scalar_parallel_multiplier(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] * b[0];  }    kernel void scalar_parallel_gt(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] > b[0];  }    kernel void scalar_parallel_lt(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] < b[0];  }    kernel void scalar_parallel_equals(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] == b[0];  }    kernel void scalar_parallel_gte(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] >= b[0];  }    kernel void scalar_parallel_lte(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = a[i] <= b[0];  }    kernel void scalar_parallel_power(global float* a, global float* b, global float* results) {      long long int i = get_global_id(0);      results[i] = pow(a[i],b[0]);  }    kernel void scalar_parallel_adder(global float* a, global float* b, global float* col_size, global float* results) {      int i = get_global_id(0);        int r = i/(int)col_size[0];      int c = i%(int)col_size[0];        results[i] = a[i];        if(r==c)          results[i] = results[i] + b[0];  }    kernel void scalar_parallel_subtracter(global float* a, global float* b, global float* col_size, global float* results) {      int i = get_global_id(0);        int r = i/(int)col_size[0];      int c = i%(int)col_size[0];        results[i] = a[i];        if(r==c)          results[i] = results[i] - b[0];  }    kernel void parallel_matrix_multiply(const int M, const int N, const int K, const global float* A, const global float* B, global float* C) {            const int row = get_global_id(0);      const int col = get_global_id(1);        float sum = 0.0f;      for (int k=0; k<K; k++) {          sum += A[k*M + row] * B[col*K + k];      }        C[col*M + row] = sum;  }    kernel void parallel_transpose(const int N, const global float* A, global float* B) {            const int row = get_global_id(0);      const int col = get_global_id(1);        B[col*N + row] = A[row*N + col];  }  ";
    }

    void init_parallel() {

        try {
            cl_int retP, retD, retC, retQ, ret;

            retP = clGetPlatformIDs(1, &platformId, &ret_num_platforms);
            retD = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &ret_num_devices);
            context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &retC);
            queue = clCreateCommandQueueWithProperties(context, deviceId, nullptr, &retQ);

            if ((retC != 0) || (retP != 0) || (retQ != 0) || (retD != 0)) {

                throw MatrixStatus("Error detecting OpenCL supported platform.", 91);
            }

            std::string source_str = kernelCode();
            size_t source_size = source_str.size();

            program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program from source.", 99);
            }

            ret = clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);

            if (ret != 0) {

                throw MatrixStatus("Error building kernel program.", 100);
            }

            kernel_add = clCreateKernel(program, "parallel_adder", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Adder)", 101);
            }

            kernel_subtract = clCreateKernel(program, "parallel_subtracter", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Subtracter)", 101);
            }

            kernel_multiply = clCreateKernel(program, "parallel_multiplier", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Multiplier)", 101);
            }

            kernel_gt = clCreateKernel(program, "parallel_gt", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Greater Than [gt])", 101);
            }

            kernel_lt = clCreateKernel(program, "parallel_lt", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Less Than [lt])", 101);
            }

            kernel_equals = clCreateKernel(program, "parallel_equals", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Is Equal To [equals])", 101);
            }

            kernel_gte = clCreateKernel(program, "parallel_gte", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Greater Than or Equal To [gte])", 101);
            }

            kernel_lte = clCreateKernel(program, "parallel_lte", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Less Than or Equal To [lte])", 101);
            }

            scalar_kernel_multiply = clCreateKernel(program, "scalar_parallel_multiplier", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Scalar Multiplier)", 101);
            }

            scalar_kernel_gt = clCreateKernel(program, "scalar_parallel_gt", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Scalar Greater Than)", 101);
            }

            scalar_kernel_lt = clCreateKernel(program, "scalar_parallel_lt", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Scalar Less Than)", 101);
            }

            scalar_kernel_equals = clCreateKernel(program, "scalar_parallel_equals", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Scalar Is Equal To)", 101);
            }

            scalar_kernel_gte = clCreateKernel(program, "scalar_parallel_gte", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Scalar Greater Than or Equal To)", 101);
            }

            scalar_kernel_lte = clCreateKernel(program, "scalar_parallel_lte", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Scalar Less Than or Equal To)", 101);
            }

            scalar_kernel_power = clCreateKernel(program, "scalar_parallel_power", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Scalar Power)", 101);
            }

            scalar_kernel_adder = clCreateKernel(program, "scalar_parallel_adder", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Scalar Adder)", 101);
            }

            scalar_kernel_subtracter = clCreateKernel(program, "scalar_parallel_subtracter", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Scalar Subtracter)", 101);
            }

            matrix_kernel_multiply = clCreateKernel(program, "parallel_matrix_multiply", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Matrix Multiplier)", 101);
            }

            matrix_kernel_transpose = clCreateKernel(program, "parallel_transpose", &ret);

            if (ret != 0) {

                throw MatrixStatus("Error creating kernel program. (Matrix Tranpose)", 101);
            }

        }
        catch (MatrixStatus status) {

            std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
                      << "Aborting..." << std::endl;
            exit(0);
        }
    }

    void finish_parallel() {
        cl_int reta = clFlush(queue);
        cl_int retb = clReleaseKernel(kernel_add);
        cl_int retd = clReleaseKernel(kernel_multiply);
        cl_int rete = clReleaseKernel(kernel_subtract);
        cl_int retf = clReleaseKernel(kernel_gt);
        cl_int reti = clReleaseKernel(kernel_lt);
        cl_int retj = clReleaseKernel(kernel_equals);
        cl_int retk = clReleaseKernel(kernel_gte);
        cl_int retl = clReleaseKernel(kernel_lte);
        cl_int retc = clReleaseProgram(program);
        cl_int retg = clReleaseCommandQueue(queue);
        cl_int reth = clReleaseContext(context);

        if (reta != 0 || retb != 0 || retc != 0 || retg != 0 || reth != 0 || retd != 0 || rete != 0 || retf != 0 || reti != 0 || retj != 0 || retk != 0 || retl != 0) {

            std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
        }
    }
}