#include "stdafx.h"
#include "LeNetNetwork.h"

using namespace NeuralNetworkCLI;
using namespace System;
using namespace System::Collections::Generic;
using namespace System::Collections::ObjectModel;

LeNetNetwork::LeNetNetwork(LeNetConfiguration^ configuration)
{
	if (configuration == nullptr)
	{
		throw gcnew ArgumentNullException("configuration");
	}

	NeuralNetworkNative::LeNetConfiguration nativeConfig(configuration->ClassCount);
	pin_ptr<Char> character_pin = &(configuration->Characters[0]);
	memcpy(nativeConfig.Characters, character_pin, sizeof(Char) * (configuration->ClassCount));
	character_pin = nullptr;

	pin_ptr<double> classDefinition_pin = &(configuration->ClassDefinitions[0]);
	memcpy(nativeConfig.ClassDefinitions, classDefinition_pin, sizeof(double) * (configuration->ClassDefinitions->Length));
	classDefinition_pin = nullptr;

	nativeNetwork = new NeuralNetworkNative::LeNetNetwork(nativeConfig);

	CreateSteps();
}

void LeNetNetwork::CreateSteps()
{
	CreateInputStep();
	CreateFirstConvolutions();
	CreateFirstSubsampling();
	CreateSecondConvolutions();
	CreateSecondSubsampling();
	CreateConsolidationAndOutputSteps();
}

void LeNetNetwork::CreateInputStep()
{
	inputStep = gcnew RectangularStep(this, nativeNetwork->getInputStep());
}

void LeNetNetwork::CreateFirstConvolutions()
{
	int count = nativeNetwork->FirstConvolutionCount;
	List<RectangularStep^>^ convolutions = gcnew List<RectangularStep^>(count);

 	const NeuralNetworkNative::ConvolutionStep*const * convolutionSteps = nativeNetwork->getFirstConvolutions();
	for (int i = 0; i < count; i++)
	{
		RectangularStep^ step = gcnew RectangularStep(this, convolutionSteps[i]);
		convolutions->Add(step);
	}
	firstConvolutions = convolutions->AsReadOnly();
}

void LeNetNetwork::CreateFirstSubsampling()
{
	int count = nativeNetwork->FirstConvolutionCount;
	List<RectangularStep^>^ subsampling = gcnew List<RectangularStep^>(count);

	const NeuralNetworkNative::SubsamplingStep*const * subsamplingSteps = nativeNetwork->getFirstSubsampling();
	for (int i = 0; i < count; i++)
	{
		RectangularStep^ step = gcnew RectangularStep(this, subsamplingSteps[i]);
		subsampling->Add(step);
	}
	firstSubsampling = subsampling->AsReadOnly();
}

void LeNetNetwork::CreateSecondConvolutions()
{
	int count = nativeNetwork->SecondConvolutionCount;
	List<RectangularStep^>^ convolutions = gcnew List<RectangularStep^>(count);

	const NeuralNetworkNative::ConvolutionStep*const * convolutionSteps = nativeNetwork->getSecondConvolutions();
	for (int i = 0; i < count; i++)
	{
		RectangularStep^ step = gcnew RectangularStep(this, convolutionSteps[i]);
		convolutions->Add(step);
	}
	secondConvolutions = convolutions->AsReadOnly();
}

void LeNetNetwork::CreateSecondSubsampling()
{
	int count = nativeNetwork->SecondConvolutionCount;
	List<RectangularStep^>^ subsampling = gcnew List<RectangularStep^>(count);

	const NeuralNetworkNative::SubsamplingStep*const * subsamplingSteps = nativeNetwork->getSecondSubsampling();
	for (int i = 0; i < count; i++)
	{
		RectangularStep^ step = gcnew RectangularStep(this, subsamplingSteps[i]);
		subsampling->Add(step);
	}
	secondSubsampling = subsampling->AsReadOnly();
}

void LeNetNetwork::CreateConsolidationAndOutputSteps()
{
	consolidationStep = gcnew FlatStep(this, nativeNetwork->getConsolidationStep());
	outputStep = gcnew FlatStep(this, nativeNetwork->getOutputStep());
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

bool LeNetNetwork::IsDisposed::get()
{
	return nativeNetwork == nullptr;
}

RectangularStep^ LeNetNetwork::InputStep::get()
{
	return inputStep;
}

ReadOnlyCollection<RectangularStep^>^ LeNetNetwork::FirstConvolutions::get()
{
	return firstConvolutions;
}

ReadOnlyCollection<RectangularStep^>^ LeNetNetwork::FirstSubsampling::get()
{
	return firstSubsampling;
}

ReadOnlyCollection<RectangularStep^>^ LeNetNetwork::SecondConvolutions::get()
{
	return secondConvolutions;
}

ReadOnlyCollection<RectangularStep^>^ LeNetNetwork::SecondSubsampling::get()
{
	return secondSubsampling;
}

FlatStep^ LeNetNetwork::ConsolidationStep::get()
{
	return consolidationStep;
}

FlatStep^ LeNetNetwork::OutputStep::get()
{
	return outputStep;
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
