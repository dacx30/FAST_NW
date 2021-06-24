package com.FAST_NW.Losses;

import com.FAST_NW.Entity.Sample;
import com.FAST_NW.Entity.Layer;
import com.FAST_NW.Entity.Neuron;

public class LossXEntropy implements Loss {
    @Override
    public void cost_delta(Layer outputLayer, Sample sample) {
        for(int i = 0; i<outputLayer.neurones.length; i++){
            Neuron neuron = outputLayer.neurones[i];
            neuron.dC_dZ = sample.outputs_forecast[i]-sample.outputs_actual[i];
        }
    }

    @Override
    public double score(Sample sample) {
        return sample.negLogLoss();
    }

}
