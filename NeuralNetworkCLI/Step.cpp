#include "stdafx.h"
#include "Step.h"

using namespace NeuralNetworkCLI;
using namespace System;

Step::Step(LeNetNetwork^ network) : network(network)
{
	if (network == nullptr)
	{
		throw gcnew ArgumentNullException("network");
	}
}

void Step::ThrowOnDisposed()
{
	if (network->IsDisposed)
	{
		throw gcnew ObjectDisposedException(network->GetType()->FullName);
	}
}

int Step::Length::get()
{
	ThrowOnDisposed();
	return getNativeStep()->Length;
}

LeNetNetwork^ Step::Network::get()
{
	return network;
}

void Step::CopyOutputs(array<double>^ destination)
{
	ThrowOnDisposed();
	const NeuralNetworkNative::Step* nativeStep = getNativeStep();
	if (destination->Length != nativeStep->Length)
	{
		throw gcnew ArgumentOutOfRangeException("destination", "The length of the destination array must be the same as the step output.");
	}
	pin_ptr<double> destination_pin = &destination[0];
	nativeStep->CopyOutputs(destination_pin);
	destination_pin = nullptr;
}