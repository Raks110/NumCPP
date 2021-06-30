#include "gtest/gtest.h"
#include "numcpp.h"

TEST(MatrixOps, matmul_dim_check) {

    numcpp::init_parallel();

    auto mat1 = numcpp::Matrix(1, 2, 10);
    auto mat2 = numcpp::Matrix(2, 2, 10);
    EXPECT_TRUE(numcpp::matmul(mat1, mat2).get_columns() == 2 && numcpp::matmul(mat1, mat2).get_rows() == 1);
}

TEST(MatrixOps, transpose_dim_check) {

    numcpp::init_parallel();

    auto mat1 = numcpp::Matrix(1, 2, 10);
    EXPECT_TRUE(numcpp::transpose(mat1).get_columns() == 1 && numcpp::transpose(mat1).get_rows() == 2);
}