#include <iostream>
#include <functional>

#include <Eigen/Dense>
#include <EigenRand/EigenRand>

using namespace std;
using namespace Eigen;

int main()
{
	Rand::Vmt19937_64 urng{4212};
	Vector4f mean{0, 1, 2, 3};
	Matrix4f cov;
	cov << 1, 1, 0, 0,
		1, 2, 0, 0,
		0, 0, 3, 1,
		0, 0, 1, 2;

	// constructs MvNormalGen with Scalar=float, Dim=4
	Rand::MvNormalGen<float, 4> gen1{mean, cov};

	// or you can use `make-` helper function. It can deduce the type of generator to be created.
	auto gen2 = Rand::makeMvNormGen(mean, cov);

	// generates one sample ( shape (4, 1) )
	Vector4f sample = gen1.generate(urng);

	// generates 10 samples ( shape (4, 10) )
	MatrixXf samples = gen1.generate(urng, 10);
	// or you can just use `MatrixXf` type

	cout << sample << endl;
	cout << samples << endl;

	Rand::StdNormalGen<float> stdnorm;
	for (int i = 0; i < 10; ++i)
	{
		cout << stdnorm.generate<Matrix<float, 4, -1>>(4, 1, urng) << endl;
	};
};
