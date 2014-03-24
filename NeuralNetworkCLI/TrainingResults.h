#pragma once

namespace NeuralNetworkCLI
{
	public ref class TrainingResults
	{
	public:
		TrainingResults();

		double Error;
		bool Correct;
	};

}