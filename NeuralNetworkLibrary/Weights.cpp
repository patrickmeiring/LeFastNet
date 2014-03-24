#include "stdafx.h"
#include "Weights.h"

using namespace NeuralNetworkNative;

Weights::Weights(int size) : Size(size)
{
	learningRate = 0.0;
	mu = 0.0;
}

void Weights::StartPreTraining()
{
	preTrainingSamples = 0;
	StartPreTrainingCore();
}

void Weights::PreTrain(Step& downstream)
{
	preTrainingSamples += 1;
	PreTrainCore(downstream);
}

void Weights::CompletePreTraining()
{
	assert(preTrainingSamples > 0);
	CompletePreTrainingCore();
}

void Weights::PropogateForward(Step& downstream)
{
	PropogateForwardCore(downstream);
}

void Weights::Train(Step& downstream)
{
	TrainCore(downstream);
}

void Weights::Randomise(std::vector<double>& weights, int fanIn)
{
	for (auto vi = weights.begin(), ve = weights.end(); vi != ve; ++vi) {
		*vi = RandomWeight(fanIn);
	}
}

void Weights::Clear(std::vector<double>& vector)
{
	for (auto vi = vector.begin(), ve = vector.end(); vi != ve; ++vi) {
		*vi = 0.0;
	}
}

std::mt19937_64 Weights::random = std::mt19937_64(std::mt19937_64::default_seed);

double Weights::getLearningRate()
{
	return learningRate;
}

void Weights::setLearningRate(double learningRate)
{
	this->learningRate = learningRate;
}

double Weights::getMu()
{
	return mu;
}

void Weights::setMu(double mu)
{
	this->mu = mu;
}


double Weights::RandomWeight(int fanIn)
{
	std::uniform_real<double> uniform(-2.4 / fanIn, 2.4 / fanIn);
	return uniform(random);
}

Weights::~Weights()
{

}