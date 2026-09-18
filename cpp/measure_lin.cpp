#include "lin_darl.h"

#include <charconv>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

// Reject signs, fractional values, trailing characters and integer overflow.
template <typename T>
T positive_integer(const char* value)
{
    const std::string_view text(value);
    T result{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (text.empty() || text.front() < '0' || text.front() > '9'
        || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()
        || result == 0)
        throw std::invalid_argument("Expected a positive integer in range");
    return result;
}

int main(int argc, char* argv[])
{
    if (argc != 6) {
        std::cerr << "Usage: measure_lin N DEVICE EXPOSURE_US PIXELS OUTPUT_PATH\n";
        return 2;
    }
    try {
        const auto n = positive_integer<unsigned int>(argv[1]);
        const auto exposure = positive_integer<unsigned int>(argv[3]);
        const auto pixels = positive_integer<unsigned int>(argv[4]);
        if (!*argv[2] || !*argv[5])
            throw std::invalid_argument("Device and output path must not be empty");
        lin_get_and_save_signal(n, argv[2], exposure, pixels, argv[5]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Line acquisition failed: " << error.what() << '\n';
        return 1;
    }
}
