using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworkCLI;

namespace NeuralNetworkOCR
{
    class LeNetTrainer
    {
        public LeNetTrainer()
        {

        }

        public void Initialise()
        {
            Console.WriteLine("Loading Training Data Set...");
            
            TrainingDataSet = DataSets.GetTrainingSet().Randomise(0);
            Console.WriteLine("Loading Generalisation Data Set...");
            GeneralisationDataSet = DataSets.GetGeneralisationSet().Randomise(1);

            Console.WriteLine("Creating LeNet...");
            //Network = new LeNetNetwork('0', '1', '2', '3', '4', '5', '6', '7', '8', '9');
            //Snapshot = new LeNetSnapshot(Network);           
            LeNetConfiguration configuration = LeNetConfigurations.FromCharacters('0', '1', '2', '3', '4', '5', '6', '7', '8', '9');
            Network = new LeNetNetwork(configuration);

            Network.LearningRate = 0.0005 / 8.0;
            Network.Mu = 0.02;
            bool isPreTraining = Network.IsPreTraining;
        }

        public async Task TrainAsync()
        {
            await Task.Run(new Action(() =>
            {
                Train();
            }));
        }

        public void Train()
        {
            Console.WriteLine();
            for (int i = 0; i < 50; i++)
            {
                Console.WriteLine("Run Epoch {0} with LC {1}", i, Network.LearningRate);

                Network.IsPreTraining = true;
                DoEpoch(TrainingDataSet.Take(500));
                Network.IsPreTraining = false;
                DoEpoch(TrainingDataSet);
                Network.LearningRate *= 0.90;
            }
            Console.WriteLine("Complete.");
        }

        protected void DoEpoch(IEnumerable<DataSetItem> trainItems)
        {
            int correct = 0;
            int total = 0;
            foreach (DataSetItem item in trainItems)
            {
                TrainingResults result = Network.Train(item);

                if (result.Correct) correct++;
                total++;

                if (total % 10 == 0) UpdateStatus(correct, total);
            }
            Console.WriteLine();
        }

        private void UpdateStatus(int itemsCorrect, int itemsProcessed)
        {
            double currentAccuracy = (itemsCorrect * 100.0) / ((double)itemsProcessed);
            Console.CursorLeft = 0;
            Console.CursorTop -= 1;
            Console.WriteLine(currentAccuracy.ToString("000.00") + "% on " + itemsProcessed.ToString() + " items.       ");


        }

        LeNetNetwork Network;
        IList<DataSetItem> TrainingDataSet;
        IList<DataSetItem> GeneralisationDataSet;
    }
}
