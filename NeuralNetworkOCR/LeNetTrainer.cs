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
        public LeNetNetwork Network { get; private set; }
        IList<DataSetItem> trainingDataSet;
        IList<DataSetItem> generalisationDataSet;
        public LeNetSnapshot Snapshot { get; private set; }

        int epochCorrect;
        int epochTotal;

        public LeNetTrainer()
        {

        }


        public void Initialise()
        {
            Console.WriteLine("Loading Training Data Set...");
            
            trainingDataSet = DataSets.GetTrainingSet().Randomise(0);
            Console.WriteLine("Loading Generalisation Data Set...");
            generalisationDataSet = DataSets.GetGeneralisationSet().Randomise(1);

            Console.WriteLine("Creating LeNet...");         
            LeNetConfiguration configuration = LeNetConfigurations.FromCharacters('0', '1', '2', '3', '4', '5', '6', '7', '8', '9');
            Network = new LeNetNetwork(configuration);
            Snapshot = new LeNetSnapshot(Network);

            Network.LearningRate = 0.0005 / 16.0;
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
                DoEpoch(trainingDataSet.Take(500));
                Console.WriteLine();

                Network.IsPreTraining = false;
                DoEpoch(trainingDataSet);
                Console.WriteLine();

                Network.LearningRate *= 0.90;
            }
            Console.WriteLine("Complete.");
        }

        protected void DoEpoch(IEnumerable<DataSetItem> trainItems)
        {
            StartEpoch();
            foreach (DataSetItem item in trainItems)
            {
                TrainingResults result = Network.Train(item);

                ItemTrained(result);
                UpdateSnapshot();
            }
        }

        private void StartEpoch()
        {
            epochCorrect = 0;
            epochTotal = 0;
        }

        private void ItemTrained(TrainingResults result)
        {
            if (result.Correct)
            {
                epochCorrect++;
            }
            epochTotal++;

            if (epochTotal % 10 == 0)
            {
                UpdateStatus();
            }
        }

        private void UpdateSnapshot()
        {
            if (Snapshot.UpdateRequested)
            {
                Snapshot.UpdateSnapshot();
            }
        }

        private void UpdateStatus()
        {
            double currentAccuracy = (epochCorrect * 100.0) / ((double)epochTotal);
            Console.CursorLeft = 0;
            Console.CursorTop -= 1;
            Console.WriteLine(currentAccuracy.ToString("000.00") + "% on " + epochTotal.ToString() + " items.       ");
        }

    }
}
