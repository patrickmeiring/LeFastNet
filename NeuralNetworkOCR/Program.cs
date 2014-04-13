using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworkCLI;
using System.Windows.Forms;

namespace NeuralNetworkOCR
{
    class Program
    {
        static void Main(string[] args)
        {
            LeNetTrainer trainer = new LeNetTrainer();
            trainer.Initialise();
            trainer.TrainAsync();

            ObservationForm observationForm = new ObservationForm(trainer.Snapshot);
            Application.Run(observationForm);
        }
    }
}
