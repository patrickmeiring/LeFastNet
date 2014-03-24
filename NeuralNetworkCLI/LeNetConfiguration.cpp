#include "stdafx.h"
#include "LeNetConfiguration.h"

using namespace NeuralNetworkCLI;

LeNetConfiguration::LeNetConfiguration(int classCount)
{
	ClassCount = classCount;
	Characters = gcnew array<System::Char>(classCount);
	ClassDefinitions = gcnew array<double>(classCount * LeNetNetwork::OutputFeedForwardNeurons);

}
