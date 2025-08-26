#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>

inline std::chrono::high_resolution_clock::time_point timeNow(){return std::chrono::high_resolution_clock::now();}
inline double timeElapsed(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

const double eps=1e-10;
const size_t seed=19260817;
extern std::mt19937 rng;

struct pair_hash 
{
    std::size_t operator()(const std::pair<int, int>& p) const 
	{
		std::size_t h1 = std::hash<int>()(p.first);
        std::size_t h2 = std::hash<int>()(p.second);
        return h1 ^ (h2 * 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

// Ensure folder path ends with separator
inline std::string ensureFolderSeparator(const std::string &folder) {
    if (folder.empty()) return "./";
    char last = folder.back();
    if (last != '/' && last != '\\') {
#ifdef _WIN32
        return folder + "\\";
#else
        return folder + "/";
#endif
    }
    return folder;
}

inline double sqr(const double &x) { return x*x; }
inline double normSqr(const std::vector<double> &x)
{
	double res=0;
	for (auto &y:x) res+=sqr(y);
	return res;
}

inline std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b) 
{
	if (a.size() != b.size()) {
		throw std::invalid_argument("Size mismatch: a.size() != b.size()");
	}
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + b[i];
    return result;
}

inline std::vector<double>& operator+=(std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
		throw std::invalid_argument("Size mismatch: a.size() != b.size()");
	}
    for (size_t i = 0; i < a.size(); ++i)
        a[i] += b[i];
    return a;
}

inline std::vector<double> operator-(const std::vector<double>& a, const std::vector<double>& b) 
{
    if (a.size() != b.size()) {
		throw std::invalid_argument("Size mismatch: a.size() != b.size()");
	}
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] - b[i];
    return result;
}

inline std::vector<double>& operator-=(std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
		throw std::invalid_argument("Size mismatch: a.size() != b.size()");
	}
    for (size_t i = 0; i < a.size(); ++i)
        a[i] -= b[i];
    return a;
}
