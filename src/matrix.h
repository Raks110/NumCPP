#ifndef NUMCPP_MATRIX_H
#define NUMCPP_MATRIX_H

#include <cstring>
#include <cmath>
#include <iostream>

namespace numcpp {

    /**
     * Reader is an abstract class that will be extended and overridden to provide 'read' function
     * For example: FileReader may implement read to read data from a file while ConsoleReader may utilize the standard input stream
     */
    class Reader {

    public:
        virtual float read() = 0;
    };

    /**
     * MatrixStatus is the class that holds the status of every operation.
     * Success/Failure message and their respective codes are enclosed in this class.
     */
    class MatrixStatus: public std::exception {

    private:

        //Error message for user readability [Scope for improvement]
        std::string error_message;

        //Error code for cleaner relationships between different errors
        int error_code;

    public:

        MatrixStatus(std::string error, int code);

        std::string get_error_message();

        int get_error_code();
    };

    /**
     * Matrix class allows all matrix operations to be performed on its objects.
     * It only supports float type matrices.
     *
     * Supported Element-wise operations: [+, -, *, <, <=, >, >=, ==, ^]
     * Supported Matrix on Scalar operations: [+, -, *, <, <=, >, >=, ==, ^]
     */
    class Matrix {

    private:

//Count of the no. of columns.
        size_t columns;

        //Count of the no. of rows.
        size_t rows;

        //The matrix itself, flattened to 1D array to reduce computational complexity.
        float* matrix{};

        //Initialize a matrix (with random values below the limit)
        MatrixStatus initialize_matrix(int limit);

        //Initialize a matrix (with user defined values)
        //reader is any object that extends Reader and implements it's own version of 'read' operation
        MatrixStatus initialize_matrix(Reader* reader);

    public:

        //initialize the matrix with random values upto `limit`
        //customize limit here if necessary
        Matrix(size_t rows, size_t columns, int limit = 10000);

        //call the initialize_matrix with a Reader object
        Matrix(size_t rows, size_t columns, Reader* reader);

        //Initialize a matrix (with all 1s)
        //multiple: defines the number to multiply to 1 during initialization
        MatrixStatus ones(float multiple);

        //Initialize a matrix (with all 0s)
        MatrixStatus zeroes();

        //Initialize a matrix (with identity matrix)
        //multiple: defines the number to multiply to 1 during initialization
        MatrixStatus identity(float multiple);

        //Respective getters and setters
        float get_element(size_t row, size_t column) const;

        void set_element(size_t row, size_t column, float value);

        float* get_matrix() const;

        size_t get_rows() const;

        size_t get_columns() const;

        void set_matrix(float* mat);

        //this function must be called at the end to ensure that the matrices are safely discarded from the memory
        void clean_up();
    };
}

#endif //NUMCPP_MATRIX_H
