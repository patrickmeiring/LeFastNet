#include "stdafx.h"
#include "RectangularStep.h"

using namespace NeuralNetworkCLI;
using namespace System;

RectangularStep::RectangularStep(LeNetNetwork^ network, const NeuralNetworkNative::RectangularStep* nativeStep) : Step(network), nativeStep(nativeStep)
{
	if (nativeStep == nullptr)
	{
		throw gcnew ArgumentNullException("nativeStep");
	}
}

int RectangularStep::Height::get()
{
	ThrowOnDisposed();
	return nativeStep->Height;
}

int RectangularStep::Width::get()
{
	ThrowOnDisposed();
	return nativeStep->Width;
}

const NeuralNetworkNative::Step* RectangularStep::getNativeStep()
{
	return nativeStep;
}