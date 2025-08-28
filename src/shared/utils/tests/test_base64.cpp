#include "gtest/gtest.h"
#include "base64.h"

// Test case for encoding a simple string
TEST(Base64Test, EncodeSimpleString) {
    const std::string original = "Hello, World!";
    const uint8_t* data = reinterpret_cast<const uint8_t*>(original.c_str());
    const std::string expected = "SGVsbG8sIFdvcmxkIQ==";
    EXPECT_EQ(base64_encode(data, original.length()), expected);
}

// Test case for decoding a simple string
TEST(Base64Test, DecodeSimpleString) {
    const std::string encoded = "SGVsbG8sIFdvcmxkIQ==";
    const std::vector<uint8_t> decoded_vec = base64_decode(encoded);
    const std::string decoded_str(decoded_vec.begin(), decoded_vec.end());
    const std::string expected = "Hello, World!";
    EXPECT_EQ(decoded_str, expected);
}

// Test case for an empty string
TEST(Base64Test, EmptyString) {
    const std::string original = "";
    const uint8_t* data = reinterpret_cast<const uint8_t*>(original.c_str());
    EXPECT_EQ(base64_encode(data, original.length()), "");

    const std::string encoded = "";
    EXPECT_TRUE(base64_decode(encoded).empty());
}

// Test case for encoding and then decoding
TEST(Base64Test, EncodeDecodeRoundtrip) {
    const std::string original = "Testing base64 encoding and decoding roundtrip.";
    const uint8_t* data = reinterpret_cast<const uint8_t*>(original.c_str());

    const std::string encoded = base64_encode(data, original.length());
    const std::vector<uint8_t> decoded_vec = base64_decode(encoded);
    const std::string roundtrip(decoded_vec.begin(), decoded_vec.end());

    EXPECT_EQ(original, roundtrip);
}
