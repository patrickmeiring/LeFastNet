using NeuralNetworkCLI;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkOCR
{
    class StepSnapshot
    {
        public StepSnapshot(RectangularStep step)
            : this(step, step.Width)
        {
        }

        public StepSnapshot(Step step, int width)
        {
            this.Step = step;
            OutputSnapshot = new double[step.Length];
            Width = width;
        }

        public readonly Step Step;
        public readonly int Width;
        public readonly double[] OutputSnapshot;
        public Bitmap OutputBitmap;

        public void UpdateOutputBitmap()
        {
            OutputBitmap = OutputSnapshot.ToBitmap(Width);
        }

        public event EventHandler Updated;

        protected void OnUpdated()
        {
            EventHandler handler = Updated;
            if (handler != null) handler(this, EventArgs.Empty);
        }

        public virtual void UpdateSnapshot()
        {
            Step.CopyOutputs(OutputSnapshot);
            OnUpdated();
        }
    }
}
