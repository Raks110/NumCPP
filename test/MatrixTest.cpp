#include "gtest/gtest.h"
#include "numcpp.h"

TEST(Matrix, init_random) {
    EXPECT_NO_THROW(numcpp::Matrix(2, 2, 10));
}

TEST(Matrix, init_random_df) {
    EXPECT_NO_THROW(numcpp::Matrix(2, 2));
}

TEST(Matrix, init_reader) {
    class ConsoleReader: public numcpp::Reader {
        float read() override {
            return 1;
        }
    };

    EXPECT_NO_THROW(numcpp::Reader* reader = new ConsoleReader());
}


TEST(Matrix, init_mat_wreader) {
    class ConsoleReader: public numcpp::Reader {
        float read() override {
            return 0.0f;
        }
    };

    auto mat = new numcpp::Matrix(1, 1, new ConsoleReader());
    mat->get_element(1, 1);
}