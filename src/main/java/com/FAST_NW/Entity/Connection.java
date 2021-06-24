
package com.FAST_NW.Entity;

import java.util.Random;

public class Connection {
    public final Neuron from;
    public final Neuron to;
    public double weight;
    public double dC_dweight;
    public Connection(Neuron from, Neuron to, Random rnd) {
        this.from = from;        
        this.to = to;
        this.weight = rnd.nextGaussian()*0.1;
    }
    public Connection(Neuron from, Neuron to, double weight) {
        this.from   = from;
        this.to     = to;
        this.weight = weight;
    }
}