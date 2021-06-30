#include "gtest/gtest.h"
#include "numcpp.h"

TEST(MatrixStatus, mst_obvious) {
EXPECT_ANY_THROW(throw numcpp::MatrixStatus("Bad", 200));
}

TEST(MatrixStatus, mst_error_code) {
auto mst = numcpp::MatrixStatus("Bad", 200);
EXPECT_EQ(mst.get_error_code(), 200);
}

TEST(MatrixStatus, mst_error_msg) {
auto mst = numcpp::MatrixStatus("Bad", 200);
EXPECT_EQ(mst.get_error_message(), "Bad");
}