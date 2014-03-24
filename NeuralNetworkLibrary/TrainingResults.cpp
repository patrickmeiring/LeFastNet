#include "stdafx.h"
#include "TrainingResults.h"

using namespace NeuralNetworkNative;

TrainingResults NeuralNetworkNative::CreateResults(const std::vector<double> &results, int correctClass)
{
	TrainingResults result;
	result.Error = results[correctClass];
	double minimumNonCorrectError = std::numeric_limits<double>::max();

	int index = 0;
	for (auto vi = results.begin(), ve = results.end(); vi != ve; ++vi) {
		if (index++ == correctClass) continue;
		if (*vi < minimumNonCorrectError)
			minimumNonCorrectError = *vi;
	}
	result.Correct = minimumNonCorrectError > result.Error;
	return result;
};