/*
* Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#pragma once
#include <string>
#include <vector>
#include <cstdint>

std::string base64_encode(const uint8_t* data, size_t len);
std::vector<uint8_t> base64_decode(const std::string& input);