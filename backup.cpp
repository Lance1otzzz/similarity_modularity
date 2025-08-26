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
// no use now. for minimum fu4gai4 hypersphere
struct Hypersphere{
	Node center; // id=-1; 
	double r;
	Hypersphere(){}
	Hypersphere(const Hypersphere &H)
	{
		r=H.r;
		center=H.center;
	}
	Hypersphere(Hypersphere &&H)
	{
		r=H.r;
		center=std::move(H.center);
	}
	Hypersphere(const Node &C,const double &R)
	{
		center=C;
		r=R;
	}
	Hypersphere(Node &&C, const double &R)
	{
		center=std::move(C);
		r=R;
	}
	void printHypersphere()
	{
		std::cout<<"Hypersphere:center point:"<<std::endl;
		center.printNode();
		std::cout<<"r="<<r<<"\nend printing Hypersphere"<<std::endl;
	}
};

//Hypersphere calcHypersphere(std::vector<Node> points)
//{
//	/// IF the points are in a same hyperplane!!!!!!!!!!!!!!!!!!!
//	int dimension=points[0].attributes.size();
//	if (points.size()!=dimension+1) 
//	{
//		std::cerr<<"cannot calculate hypershphere because the dimension and the number of points does not match"<<std::endl;
//		throw std::invalid_argument("Dimension mismatch");
//	}
//
//	Matrix equations(dimension,dimension+1);
//	for (int i=1;i<=dimension;i++) // i-th - 1st
//	{
//		for (int j=0;j<dimension;j++) 
//			equations.a[i-1][dimension]+=sqr(points[0].attributes[j])-sqr(points[i].attributes[j]);
//		for (int j=0;j<dimension;j++)
//			equations.a[i-1][j]=2*(points[i].attributes[j]-points[0].attributes[j]);
//	}
//	if (!equations.gauss())
//	{
//		std::cerr<<"gauss err"<<std::endl;
//		throw std::invalid_argument("Gauss Error");
//	}
//	std::vector<double> ans(dimension);
//	for (int i=0;i<dimension;i++) ans[i]=equations.a[i][dimension];
//	Node center(-1,std::move(ans));
//	double r=0;
//	for (int i=0;i<dimension;i++) r+=sqr(points[0].attributes[i]-center.attributes[i]);
//	r=sqrt(r);
//	Hypersphere res(std::move(center),r);
//	return res;
//}


