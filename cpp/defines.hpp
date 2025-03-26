#pragma once
#include <string>

const double eps=1e-8;

// Ensure folder path ends with separator
std::string ensureFolderSeparator(const std::string &folder) {
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

double sqr(const double &x) { return x*x; }

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
