#pragma once
#include <vector>

namespace NeuralNetworkNative
{
	struct __LENETLIB_DLLEXPORT DataSetItem
	{
		double* Inputs;
		wchar_t Character;
	};
}