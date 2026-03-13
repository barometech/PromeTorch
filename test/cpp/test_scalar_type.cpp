#include <gtest/gtest.h>
#include "c10/core/ScalarType.h"

using namespace c10;

// ============================================================================
// ScalarType Tests
// ============================================================================

TEST(ScalarTypeTest, ElementSize) {
    EXPECT_EQ(elementSize(ScalarType::Byte), 1);
    EXPECT_EQ(elementSize(ScalarType::Char), 1);
    EXPECT_EQ(elementSize(ScalarType::Short), 2);
    EXPECT_EQ(elementSize(ScalarType::Int), 4);
    EXPECT_EQ(elementSize(ScalarType::Long), 8);
    EXPECT_EQ(elementSize(ScalarType::Half), 2);
    EXPECT_EQ(elementSize(ScalarType::Float), 4);
    EXPECT_EQ(elementSize(ScalarType::Double), 8);
    EXPECT_EQ(elementSize(ScalarType::Bool), 1);
    EXPECT_EQ(elementSize(ScalarType::BFloat16), 2);
    EXPECT_EQ(elementSize(ScalarType::ComplexFloat), 8);
    EXPECT_EQ(elementSize(ScalarType::ComplexDouble), 16);
}

TEST(ScalarTypeTest, IsFloatingType) {
    EXPECT_TRUE(isFloatingType(ScalarType::Float));
    EXPECT_TRUE(isFloatingType(ScalarType::Double));
    EXPECT_TRUE(isFloatingType(ScalarType::Half));
    EXPECT_TRUE(isFloatingType(ScalarType::BFloat16));

    EXPECT_FALSE(isFloatingType(ScalarType::Int));
    EXPECT_FALSE(isFloatingType(ScalarType::Long));
    EXPECT_FALSE(isFloatingType(ScalarType::Bool));
    EXPECT_FALSE(isFloatingType(ScalarType::ComplexFloat));
}

TEST(ScalarTypeTest, IsComplexType) {
    EXPECT_TRUE(isComplexType(ScalarType::ComplexFloat));
    EXPECT_TRUE(isComplexType(ScalarType::ComplexDouble));
    EXPECT_TRUE(isComplexType(ScalarType::ComplexHalf));

    EXPECT_FALSE(isComplexType(ScalarType::Float));
    EXPECT_FALSE(isComplexType(ScalarType::Double));
    EXPECT_FALSE(isComplexType(ScalarType::Int));
}

TEST(ScalarTypeTest, IsIntegralType) {
    EXPECT_TRUE(isIntegralType(ScalarType::Int));
    EXPECT_TRUE(isIntegralType(ScalarType::Long));
    EXPECT_TRUE(isIntegralType(ScalarType::Short));
    EXPECT_TRUE(isIntegralType(ScalarType::Byte));
    EXPECT_TRUE(isIntegralType(ScalarType::Char));

    EXPECT_FALSE(isIntegralType(ScalarType::Bool));
    EXPECT_TRUE(isIntegralType(ScalarType::Bool, true));  // with include_bool

    EXPECT_FALSE(isIntegralType(ScalarType::Float));
    EXPECT_FALSE(isIntegralType(ScalarType::Double));
}

TEST(ScalarTypeTest, ToString) {
    EXPECT_STREQ(toString(ScalarType::Float), "Float");
    EXPECT_STREQ(toString(ScalarType::Double), "Double");
    EXPECT_STREQ(toString(ScalarType::Int), "Int");
    EXPECT_STREQ(toString(ScalarType::Long), "Long");
    EXPECT_STREQ(toString(ScalarType::Bool), "Bool");
    EXPECT_STREQ(toString(ScalarType::Half), "Half");
    EXPECT_STREQ(toString(ScalarType::BFloat16), "BFloat16");
}

TEST(ScalarTypeTest, PromoteTypes) {
    // Same type
    EXPECT_EQ(promoteTypes(ScalarType::Float, ScalarType::Float), ScalarType::Float);

    // Float + Int -> Float
    EXPECT_EQ(promoteTypes(ScalarType::Float, ScalarType::Int), ScalarType::Float);
    EXPECT_EQ(promoteTypes(ScalarType::Int, ScalarType::Float), ScalarType::Float);

    // Double + Float -> Double
    EXPECT_EQ(promoteTypes(ScalarType::Double, ScalarType::Float), ScalarType::Double);

    // Complex promotion
    EXPECT_EQ(promoteTypes(ScalarType::ComplexFloat, ScalarType::Float), ScalarType::ComplexFloat);
    EXPECT_EQ(promoteTypes(ScalarType::ComplexFloat, ScalarType::Double), ScalarType::ComplexDouble);

    // Int + Long -> Long
    EXPECT_EQ(promoteTypes(ScalarType::Int, ScalarType::Long), ScalarType::Long);
}

// ============================================================================
// Half Precision Tests
// ============================================================================

TEST(HalfTest, Conversion) {
    Half h(1.5f);
    EXPECT_NEAR(static_cast<float>(h), 1.5f, 0.01f);

    Half h2(0.0f);
    EXPECT_EQ(static_cast<float>(h2), 0.0f);

    Half h3(-2.25f);
    EXPECT_NEAR(static_cast<float>(h3), -2.25f, 0.01f);
}

TEST(HalfTest, Arithmetic) {
    Half a(2.0f);
    Half b(3.0f);

    Half sum = a + b;
    EXPECT_NEAR(static_cast<float>(sum), 5.0f, 0.01f);

    Half diff = b - a;
    EXPECT_NEAR(static_cast<float>(diff), 1.0f, 0.01f);

    Half prod = a * b;
    EXPECT_NEAR(static_cast<float>(prod), 6.0f, 0.01f);

    Half quot = b / a;
    EXPECT_NEAR(static_cast<float>(quot), 1.5f, 0.01f);
}

TEST(HalfTest, Comparison) {
    Half a(1.0f);
    Half b(2.0f);
    Half c(1.0f);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a == c);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(a <= c);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b >= a);
}

// ============================================================================
// BFloat16 Tests
// ============================================================================

TEST(BFloat16Test, Conversion) {
    BFloat16 bf(1.5f);
    EXPECT_NEAR(static_cast<float>(bf), 1.5f, 0.01f);

    BFloat16 bf2(0.0f);
    EXPECT_EQ(static_cast<float>(bf2), 0.0f);

    BFloat16 bf3(-3.14f);
    EXPECT_NEAR(static_cast<float>(bf3), -3.14f, 0.1f);  // BF16 has lower precision
}

TEST(BFloat16Test, Arithmetic) {
    BFloat16 a(2.0f);
    BFloat16 b(3.0f);

    BFloat16 sum = a + b;
    EXPECT_NEAR(static_cast<float>(sum), 5.0f, 0.1f);

    BFloat16 prod = a * b;
    EXPECT_NEAR(static_cast<float>(prod), 6.0f, 0.1f);
}
