#pragma once
#include "nnlib.h"
#include <vector>

namespace NeuralNetworkNative
{
	class __LENETLIB_DLLEXPORT LeNetConfiguration
	{
	public:
		LeNetConfiguration(const LeNetConfiguration &config);
		LeNetConfiguration(int classCount);

		int ClassCount;
		double* ClassDefinitions;
		wchar_t* Characters;

		~LeNetConfiguration();
	};
}