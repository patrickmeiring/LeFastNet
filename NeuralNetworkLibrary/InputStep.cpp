#include "stdafx.h"
#include "InputStep.h"

using namespace NeuralNetworkNative;

InputStep::InputStep(int width, int height) : RectangularStep(width, height, std::vector<RectangularStep*>(0))
{
}

void InputStep::setInputs(const double* inputs)
{
	memcpy(&Output[0], inputs, sizeof(double) * Length);
}

Weights* InputStep::getWeights()
{
	return nullptr;
}

InputStep::~InputStep()
{
}
