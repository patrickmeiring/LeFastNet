#include "stdafx.h"
#include "LeNetConfiguration.h"

using namespace NeuralNetworkNative;

LeNetConfiguration::LeNetConfiguration(int classCount)
{
	ClassCount = classCount;
	Characters = new wchar_t[classCount];
	ClassDefinitions = new double[classCount * LeNetNetwork::OutputFeedForwardNeurons];
}

LeNetConfiguration::LeNetConfiguration(const LeNetConfiguration &config) 
{
	ClassCount = config.ClassCount;
	Characters = new wchar_t[ClassCount];
	memcpy(Characters, config.Characters, ClassCount * sizeof(wchar_t));
	ClassDefinitions = new double[ClassCount * LeNetNetwork::OutputFeedForwardNeurons];
	memcpy(ClassDefinitions, config.ClassDefinitions, ClassCount * LeNetNetwork::OutputFeedForwardNeurons * sizeof(double));
}

LeNetConfiguration::~LeNetConfiguration()
{
	delete[] Characters;
	Characters = nullptr;
	delete[] ClassDefinitions;
	ClassDefinitions = nullptr;
}