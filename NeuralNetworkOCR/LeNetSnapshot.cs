using NeuralNetworkCLI;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkOCR
{
    class LeNetSnapshot
    {
        public LeNetSnapshot(LeNetNetwork network)
        {
            Network = network;
            Input = new StepSnapshot(network.InputStep);
            FirstConvolutions = network.FirstConvolutions.Select(step => new StepSnapshot(step)).ToList().AsReadOnly();
            FirstSubsampling = network.FirstSubsampling.Select(step => new StepSnapshot(step)).ToList().AsReadOnly();
            SecondConvolutions = network.SecondConvolutions.Select(step => new StepSnapshot(step)).ToList().AsReadOnly();
            SecondSubsampling = network.SecondSubsampling.Select(step => new StepSnapshot(step)).ToList().AsReadOnly();
            Consolidation = new StepSnapshot(network.ConsolidationStep, 1);
            Output = new StepSnapshot(network.OutputStep, LeNetConfigurations.OutputWidth);
        }

        public readonly LeNetNetwork Network;
        public StepSnapshot Input { get; protected set; }
        public ReadOnlyCollection<StepSnapshot> FirstConvolutions { get; protected set; }
        public ReadOnlyCollection<StepSnapshot> FirstSubsampling { get; protected set; }
        public ReadOnlyCollection<StepSnapshot> SecondConvolutions { get; protected set; }
        public ReadOnlyCollection<StepSnapshot> SecondSubsampling { get; protected set; }
        public StepSnapshot Consolidation { get; protected set; }
        public StepSnapshot Output { get; protected set; }


        protected IEnumerable<StepSnapshot> All()
        {
            yield return Input;
            foreach (StepSnapshot snapshot in FirstConvolutions)
                yield return snapshot;
            foreach (StepSnapshot snapshot in FirstSubsampling)
                yield return snapshot;
            foreach (StepSnapshot snapshot in SecondConvolutions)
                yield return snapshot;
            foreach (StepSnapshot snapshot in SecondSubsampling)
                yield return snapshot;
            yield return Consolidation;
            yield return Output;
        }

        public bool UpdateRequested { get; protected set; }

        public void RequestUpdate()
        {
            if (!UpdateRequested)
            {
                UpdateRequested = true;
            }
        }

        public event EventHandler Updated;

        protected void OnUpdated()
        {
            UpdateRequested = false;
            EventHandler handler = Updated;
            if (handler != null) handler(this, EventArgs.Empty);
        }

        public void UpdateSnapshot()
        {
            if (!UpdateRequested) return;
            foreach (StepSnapshot snapshot in All())
                snapshot.UpdateSnapshot();
            Task.Run(new Action(UpdateOutputBitmaps));
        }

        protected void UpdateOutputBitmaps()
        {
            foreach (StepSnapshot snapshot in All())
                snapshot.UpdateOutputBitmap();
            OnUpdated();
        }
    }
}
