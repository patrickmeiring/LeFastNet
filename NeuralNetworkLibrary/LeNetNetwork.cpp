#include "stdafx.h"
#include "LeNetNetwork.h"

using namespace NeuralNetworkNative;

LeNetNetwork::LeNetNetwork(const LeNetConfiguration &configuration) : configuration(configuration), allSteps(), firstConvolutions(FirstConvolutionCount), firstSubsampling(FirstConvolutionCount), secondConvolutions(SecondConvolutionCount), secondSubsampling(SecondConvolutionCount)
{
	learningRate = 0.0;
	mu = 0.0;
	preTraining = false;
	CreateNetwork();
}

void LeNetNetwork::CreateNetwork()
{
	CreateInputStep();
	CreateFirstConvolutionStep();
	CreateFirstSubsamplingStep();
	CreateSecondConvolutionStep();
	CreateSecondSubsamplingStep();
	CreateConsolidationAndOutputSteps();
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

void LeNetNetwork::CreateInputStep()
{
	inputLayer = new InputStep(32, 32);

}

void LeNetNetwork::CreateFirstConvolutionStep()
{
	for (int i = 0; i < FirstConvolutionCount; i++)
	{
		ConvolutionStep* convolutionStep = new ConvolutionStep(*inputLayer, FirstConvolutionSize);
		firstConvolutions[i] = convolutionStep;
	}

	allSteps.reserve(allSteps.size() + firstConvolutions.size());
	allSteps.insert(allSteps.end(), firstConvolutions.begin(), firstConvolutions.end());
}

void LeNetNetwork::CreateFirstSubsamplingStep()
{
	for (int i = 0; i < FirstConvolutionCount; i++)
	{
		ConvolutionStep* convolutionStep = firstConvolutions[i];
		firstSubsampling[i] = new SubsamplingStep(*convolutionStep, 2);
	}

	allSteps.reserve(allSteps.size() + firstSubsampling.size());
	allSteps.insert(allSteps.end(), firstSubsampling.begin(), firstSubsampling.end());
}

void LeNetNetwork::CreateSecondConvolutionStep()
{
	for (int i = 0; i < SecondConvolutionCount; i++)
	{
		std::vector<RectangularStep*> stepInputs = std::vector<RectangularStep*>();
		for (int j = 0; j < FirstConvolutionCount; j++)
		{
			if (SecondConvolutionConnections[i * FirstConvolutionCount + j])
				stepInputs.push_back(firstSubsampling[j]);
		}
		ConvolutionStep* convolutionStep = new ConvolutionStep(stepInputs, SecondConvolutionSize);
		secondConvolutions[i] = convolutionStep;
	}

	allSteps.reserve(allSteps.size() + secondConvolutions.size());
	allSteps.insert(allSteps.end(), secondConvolutions.begin(), secondConvolutions.end());
}

void LeNetNetwork::CreateSecondSubsamplingStep()
{
	for (int i = 0; i < SecondConvolutionCount; i++)
	{
		ConvolutionStep* convolutionStep = secondConvolutions[i];
		secondSubsampling[i] = new SubsamplingStep(*convolutionStep, 2);
	}

	allSteps.reserve(allSteps.size() + secondSubsampling.size());
	allSteps.insert(allSteps.end(), secondSubsampling.begin(), secondSubsampling.end());
}

void LeNetNetwork::CreateConsolidationAndOutputSteps()
{
	std::vector<Step*> consolidationInputs = std::vector<Step*>(SecondConvolutionCount);
	for (int i = 0; i < SecondConvolutionCount; i++)
	{
		consolidationInputs[i] = secondSubsampling[i];
	}

	consolidation = new FeedForwardStep(consolidationInputs, 120);
	output = new FeedForwardStep(*consolidation, OutputFeedForwardNeurons);
	marking = new MarkingStep(*output, configuration);

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

const InputStep* LeNetNetwork::getInputStep()
{
	return inputLayer;
}

const ConvolutionStep *const * LeNetNetwork::getFirstConvolutions()
{
	const ConvolutionStep** result = new const ConvolutionStep*[FirstConvolutionCount];
	for (int i = 0; i < FirstConvolutionCount; i++)
		result[i] = firstConvolutions[i];
	return result;
}

const SubsamplingStep *const * LeNetNetwork::getFirstSubsampling()
{
	const SubsamplingStep** result = new const SubsamplingStep*[FirstConvolutionCount];
	for (int i = 0; i < FirstConvolutionCount; i++)
		result[i] = firstSubsampling[i];
	return result;
}

const ConvolutionStep *const * LeNetNetwork::getSecondConvolutions()
{
	const ConvolutionStep** result = new const ConvolutionStep*[SecondConvolutionCount];
	for (int i = 0; i < SecondConvolutionCount; i++)
		result[i] = secondConvolutions[i];
	return result;
}

const SubsamplingStep*const * LeNetNetwork::getSecondSubsampling()
{
	const SubsamplingStep** result = new const SubsamplingStep*[SecondConvolutionCount];
	for (int i = 0; i < SecondConvolutionCount; i++)
		result[i] = secondSubsampling[i];
	return result;
}

const FeedForwardStep* LeNetNetwork::getConsolidationStep()
{
	return consolidation;
}

const FeedForwardStep* LeNetNetwork::getOutputStep()
{
	return output;
}

LeNetNetwork::~LeNetNetwork()
{
	delete inputLayer;
	inputLayer = nullptr;
	for (auto si = allSteps.begin(), se = allSteps.end(); si != se; ++si)
	{
		delete *si;
		*si = nullptr;
	}
}
