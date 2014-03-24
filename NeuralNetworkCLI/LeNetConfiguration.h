#pragma once

namespace NeuralNetworkCLI
{
	public ref class LeNetConfiguration
	{
	public:
		LeNetConfiguration(int classCount);
		array<System::Char>^ Characters;
		array<double>^ ClassDefinitions;
		int ClassCount;


	};

}