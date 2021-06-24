package com.FAST_NW.Losses;

import com.FAST_NW.Entity.Sample;
import com.FAST_NW.Entity.Layer;

public enum LossEnum implements Loss {
    MAE(new LossMAE()),
    MSE(new LossMSE()),
    CROSS_ENTROPY(new LossXEntropy()),
    ;

    private final Loss l;

    private LossEnum(Loss loss) {
        this.l = loss;
    }

    @Override
    public void cost_delta(Layer outputLayer, Sample sample) {
        this.l.cost_delta(outputLayer, sample);
    }

    @Override
    public double score(Sample sample) {
        return this.l.score(sample);
    }
}
