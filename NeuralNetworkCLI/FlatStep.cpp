#include "stdafx.h"
#include "FlatStep.h"

using namespace System;
using namespace NeuralNetworkCLI;

FlatStep::FlatStep(LeNetNetwork^ network, const NeuralNetworkNative::FlatStep* nativeStep) : Step(network), nativeStep(nativeStep)
{
	if (nativeStep == nullptr)
	{
		throw gcnew ArgumentNullException("nativeStep");
	}
}

const NeuralNetworkNative::Step* FlatStep::getNativeStep()
{
	return nativeStep;
}