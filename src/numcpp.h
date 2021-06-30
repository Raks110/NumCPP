#ifndef NUMCPP_NUMCPP_H
#define NUMCPP_NUMCPP_H

#include "parallel.h"
#include "matrix.h"

namespace numcpp {

    /**
     * all overloaded operators are here
     */
    
    //all matrix-on-matrix operations here
    Matrix operator+(Matrix &first, Matrix const &second);

    Matrix operator-(Matrix &first, Matrix const &second);

    Matrix operator*(Matrix &first, Matrix const &second);

    Matrix operator>(Matrix &first, Matrix const &second);

    Matrix operator<(Matrix &first, Matrix const &second);

    Matrix operator==(Matrix &first, Matrix const &second);

    Matrix operator<=(Matrix &first, Matrix const &second);

    Matrix operator>=(Matrix &first, Matrix const &second);


    //all matrix-on-scalar operations here
    Matrix operator*(Matrix &first, float const &second);

    Matrix operator>(Matrix &first, float const &second);

    Matrix operator<(Matrix &first, float const &second);

    Matrix operator==(Matrix &first, float const &second);

    Matrix operator>=(Matrix &first, float const &second);

    Matrix operator<=(Matrix &first, float const &second);

    Matrix operator^(Matrix &first, float const &second);

    Matrix operator+(Matrix &first, float const &second);

    Matrix operator-(Matrix &first, float const &second);

    /**
     * all non-overloaded operators implemented as functions are here
     */
    Matrix matmul(Matrix a, Matrix b);

    Matrix transpose(Matrix a);

}

#endif //NUMCPP_NUMCPP_H