#include "stdafx.h"
#include "LeNetNetwork.h"

using namespace NeuralNetworkCLI;

LeNetNetwork::LeNetNetwork(LeNetConfiguration^ configuration)
{
	NeuralNetworkNative::LeNetConfiguration nativeConfig(configuration->ClassCount);
	pin_ptr<System::Char> character_pin = &(configuration->Characters[0]);
	memcpy(nativeConfig.Characters, character_pin, sizeof(System::Char) * (configuration->ClassCount));
	character_pin = nullptr;

	pin_ptr<double> classDefinition_pin = &(configuration->ClassDefinitions[0]);
	memcpy(nativeConfig.ClassDefinitions, classDefinition_pin, sizeof(double) * (configuration->ClassDefinitions->Length));
	classDefinition_pin = nullptr;

	nativeNetwork = new NeuralNetworkNative::LeNetNetwork(nativeConfig);
}

void LeNetNetwork::PropogateForward(DataSetItem^ item)
{
	NeuralNetworkNative::DataSetItem nativeItem;
	nativeItem.Character = item->Character;

	pin_ptr<double> inputs_pin = &(item->Inputs[0]);
	nativeItem.Inputs = inputs_pin;
	nativeNetwork->PropogateForward(nativeItem);
	inputs_pin = nullptr;
}

TrainingResults^ LeNetNetwork::Train(DataSetItem^ item)
{
	NeuralNetworkNative::DataSetItem nativeItem;
	nativeItem.Character = item->Character;

	pin_ptr<double> inputs_pin = &(item->Inputs[0]);
	nativeItem.Inputs = inputs_pin;
	NeuralNetworkNative::TrainingResults nativeResults = nativeNetwork->Train(nativeItem);
	inputs_pin = nullptr;

	TrainingResults^ results = gcnew TrainingResults();
	results->Correct = nativeResults.Correct;
	results->Error = nativeResults.Error;
	return results;
}

double LeNetNetwork::Mu::get()
{
	return nativeNetwork->getMu();
}

void LeNetNetwork::Mu::set(double mu)
{
	nativeNetwork->setMu(mu);
}

double LeNetNetwork::LearningRate::get()
{
	return nativeNetwork->getLearningRate();
}

void LeNetNetwork::LearningRate::set(double learningRate)
{
	nativeNetwork->setLearningRate(learningRate);
}

bool LeNetNetwork::IsPreTraining::get()
{
	return nativeNetwork->isPreTraining();
}

void LeNetNetwork::IsPreTraining::set(bool value)
{
	nativeNetwork->setPreTraining(value);
}

LeNetNetwork::!LeNetNetwork()
{
	delete nativeNetwork;
	nativeNetwork = nullptr;
}

LeNetNetwork::~LeNetNetwork()
{
	this->!LeNetNetwork();
}
