#pragma once

namespace NeuralNetworkCLI
{
	public ref class DataSetItem
	{
	public:
		DataSetItem();

		array<double>^ Inputs;
		System::Char Character;
	};

}