package com.FAST_NW.Losses;

import com.FAST_NW.Entity.Sample;
import com.FAST_NW.Entity.Layer;

public interface Loss {

    public void cost_delta(Layer outputLayer, Sample sample);

    public double score(Sample sample);



}
