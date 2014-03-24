#include "stdafx.h"
#include "LeNetNetwork.h"

using namespace NeuralNetworkNative;

LeNetNetwork::LeNetNetwork(const LeNetConfiguration &configuration) : configuration(configuration)
{
	learningRate = 0.0;
	mu = 0.0;
	preTraining = false;
	CreateNetwork();
}

void LeNetNetwork::CreateNetwork()
{
	InstanciateSteps();
	CreateStepLists();
}

const int LeNetNetwork::FirstConvolutionCount = 6;
const int LeNetNetwork::FirstConvolutionSize = 5;

const int LeNetNetwork::SecondConvolutionCount = 16;
const int LeNetNetwork::SecondConvolutionSize = 5;

const std::vector<bool> LeNetNetwork::SecondConvolutionConnections = {
	true, true, true, false, false, false,
	false, true, true, true, false, false,
	false, false, true, true, true, false,
	false, false, false, true, true, true,
	true, false, false, false, true, true,
	true, true, false, false, false, true,

	true, true, true, true, false, false,
	false, true, true, true, true, false,
	false, false, true, true, true, true,
	true, false, false, true, true, true,
	true, true, false, false, true, true,
	true, true, true, false, false, true,

	true, true, false, true, true, false,
	false, true, true, false, true, true,
	true, false, true, true, false, true,
	true, true, true, true, true, true
};
const int LeNetNetwork::ConsolidationNeurons = 120;
const int LeNetNetwork::OutputFeedForwardNeurons = 84;

void LeNetNetwork::InstanciateSteps()
{
	inputLayer = new InputStep(32, 32);
	firstConvolutions = std::vector<ConvolutionStep*>(FirstConvolutionCount);
	firstSubsampling = std::vector<SubsamplingStep*>(FirstConvolutionCount);
	for (int i = 0; i < FirstConvolutionCount; i++)
	{
		ConvolutionStep* convolutionStep = new ConvolutionStep(*inputLayer, FirstConvolutionSize);
		firstConvolutions[i] = convolutionStep;
		firstSubsampling[i] = new SubsamplingStep(*convolutionStep, 2);	
	}

	secondConvolutions = std::vector<ConvolutionStep*>(SecondConvolutionCount);
	secondSubsampling = std::vector<SubsamplingStep*>(SecondConvolutionCount);
	for (int i = 0; i < SecondConvolutionCount; i++)
	{
		std::vector<RectangularStep*> inputs = std::vector<RectangularStep*>();
		for (int j = 0; j < FirstConvolutionCount; j++)
		{
			if (SecondConvolutionConnections[i * FirstConvolutionCount + j])
				inputs.push_back(firstSubsampling[j]);
		}
		ConvolutionStep* convolutionStep = new ConvolutionStep(inputs, SecondConvolutionSize);
		secondConvolutions[i] = convolutionStep;
		secondSubsampling[i] = new SubsamplingStep(*convolutionStep, 2);
	}

	consolidation = new FeedForwardStep(*((std::vector<Step*>*)&secondSubsampling), 120);
	output = new FeedForwardStep(*consolidation, OutputFeedForwardNeurons);
	marking = new MarkingStep(*output, configuration);
}

void LeNetNetwork::CreateStepLists()
{
	allSteps = std::vector<Step*>();
	allSteps.reserve(firstConvolutions.size() + firstSubsampling.size() + secondConvolutions.size() + secondSubsampling.size());
	allSteps.insert(allSteps.end(), firstConvolutions.begin(), firstConvolutions.end());
	allSteps.insert(allSteps.end(), firstSubsampling.begin(), firstSubsampling.end());
	allSteps.insert(allSteps.end(), secondConvolutions.begin(), secondConvolutions.end());
	allSteps.insert(allSteps.end(), secondSubsampling.begin(), secondSubsampling.end());
	allSteps.push_back(consolidation);
	allSteps.push_back(output);
	allSteps.push_back(marking);
}

bool LeNetNetwork::isPreTraining()
{
	return preTraining;
}

void LeNetNetwork::setPreTraining(bool value)
{
	if (preTraining != value)
	{
		preTraining = value;
		for (auto si = allSteps.begin(), se = allSteps.end(); si != se; ++si)
			(*si)->setPreTraining(value);
	}
}

void LeNetNetwork::PropogateForward(DataSetItem &inputs)
{
	inputLayer->setInputs(inputs.Inputs);
	for (auto si = allSteps.begin(), se = allSteps.end(); si != se; ++si)
		(*si)->PropogateForward();
}

TrainingResults LeNetNetwork::Train(DataSetItem &inputs)
{
	PropogateForward(inputs);
	int correctClass = -1;
	wchar_t* characters = configuration.Characters;
	for (int i = 0; i < configuration.ClassCount; i++){
		if (characters[i] == inputs.Character)
		{
			correctClass = i;
			break;
		}
	}
	marking->setCorrectClass(correctClass);
	for (auto si = allSteps.rbegin(), se = allSteps.rend(); si != se; ++si)
		(*si)->PropogateBackwards();
	return CreateResults(marking->Output, correctClass);
}

double LeNetNetwork::getLearningRate()
{
	return learningRate;
}

void LeNetNetwork::setLearningRate(double learningRate)
{
	this->learningRate = learningRate;
	for (auto si = allSteps.begin(), se = allSteps.end(); si != se; ++si)
		(*si)->getWeights()->setLearningRate(learningRate);
}

double LeNetNetwork::getMu()
{
	return mu;
}

void LeNetNetwork::setMu(double mu)
{
	this->mu = mu;
	for (auto si = allSteps.begin(), se = allSteps.end(); si != se; ++si)
		(*si)->getWeights()->setMu(mu);
}

LeNetNetwork::~LeNetNetwork()
{
//	configuration.~LeNetConfiguration();
	delete inputLayer;
	inputLayer = nullptr;
	for (auto si = allSteps.begin(), se = allSteps.end(); si != se; ++si)
	{
		delete *si;
		*si = nullptr;
	}
}
