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

const double eps=1e-8;
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

struct Matrix
{
	int n,m;
	std::vector<std::vector<double>> a;

	Matrix(){}
	Matrix(const Matrix &other)
	{
		n=other.n;m=other.m;
		a=other.a;
	}
	Matrix(Matrix &&other)
	{
		n=other.n;m=other.m;
		a=std::move(other.a);
	}
	/*
	Matrix(int x,int y) //if x==1 then eigenmatrix, or everywhere is 0
	{
		n=y;m=y;
		a.resize(n);
		for (auto &aa:a) aa.resize(n);
		if (x==1) 
		{
			for (int i=0;i<n;i++) a[i][i]=1;
		}
	}
	*/
	Matrix(int y,int z) //x do not work, y*z matrix
	{
		n=y;m=z;
		a.resize(n);
		for (auto &aa:a) aa.resize(m);
	}
	Matrix operator*(const Matrix &y)
	{
		if (m!=y.n) 
		{
			std::cerr<<"matrix multiply error"<<std::endl;
			throw std::invalid_argument("mat mul err");
		}
		Matrix res(n,y.m);
		for (int i=0;i<n;i++)
			for (int j=0;j<y.m;j++)
				for (int k=0;k<m;k++)
					res.a[i][j]+=a[i][k]*y.a[k][j];
		return res;
	}
	Matrix &operator=(Matrix &&other)
	{
		if (this!=&other)
		{
			a=std::move(other.a);
			n=other.n;
			m=other.m;
		}
		return *this;
	}
	void output()
	{
		for (int i=0;i<n;i++)
		{
			for (int j=0;j<m;j++) std::cout<<a[i][j]<<' ';
			std::cout<<std::endl;
		}
	}
	Matrix transposition()
	{
		Matrix res(m,n);
		for (int i=0;i<m;i++)
			for (int j=0;j<n;j++)
				res.a[i][j]=a[j][i];
		return res;
	}
	// n rows and m=n+1 columns
	// The first n-1 columns are coefficients, the n-th column are constants
	// the answer is saved in the n-th column (a[i=0...n-1][m-1])
	bool gauss() //if no answer return false, else true
	{
		if (m!=n+1)
		{
			std::cerr<<"gauss warning: no solution or no unique solution"<<std::endl;
		}
		for (int i=0;i<n;i++) //do for every row
		{
			int r=i; //which row to do
			for (int j=i+1;j<n;j++) if (std::abs(a[j][i])>std::abs(a[r][i])) r=j;
			if (std::abs(a[r][i])<eps) {return false;}
			if (r!=i) for (int j=0;j<m;j++) std::swap(a[r][j],a[i][j]);
			for (int j=m-1;j>=i;j--)
				for (int k=i+1;k<n;k++) a[k][j]-=a[k][i]/a[i][i]*a[i][j];
		}
		for (int i=n-1;i>=0;i--)
		{
			for (int j=i+1;j<n;j++) a[i][m-1]-=a[j][m-1]*a[i][j];
			a[i][m-1]/=a[i][i];
		}
		return true;
	}
};
