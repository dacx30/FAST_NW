package com.FAST_NW.Losses;

import com.FAST_NW.Entity.Layer;
import com.FAST_NW.Entity.Sample;

public class LossMAE implements Loss {

    @Override
    public void cost_delta(Layer outputLayer, Sample sample) {
        throw new RuntimeException("not implemented yet");
    }

    @Override
    public double score(Sample sample) {
        return sample.MAE(false);
    }

}
