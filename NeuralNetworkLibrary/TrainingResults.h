#pragma once
#include <vector>

namespace NeuralNetworkNative
{
	struct __LENETLIB_DLLEXPORT TrainingResults
	{
		double Error;
		bool Correct;
	};

	TrainingResults CreateResults(const std::vector<double> &results, int correctClass);
}