using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworkCLI;

namespace NeuralNetworkOCR
{
    class Program
    {
        static void Main(string[] args)
        {
            LeNetTrainer trainer = new LeNetTrainer();
            trainer.Initialise();
            trainer.Train();
            
        }
    }
}
