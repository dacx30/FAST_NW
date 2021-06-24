/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.FAST_NW.Activations;

import com.FAST_NW.Entity.Neuron;

public enum ActivationEnum implements Activation{
    LEAKY_RELU(new ActivationLeakyRELU()),
    LINEAR(new ActivationLinear()),
    SIGMOID(new ActivationSigmoid()),
    HARDSIGMOID(new ActivationHardSigmoid()),
    SOFTMAX(new ActivationSoftMax()),
    TANH(new ActivationTANH()),
    ELU(new ActivationELU()),
    DUMMY(null);
    private final Activation act;

    private ActivationEnum(Activation act) {
        this.act = act;
    }

    @Override
    public void func(Neuron[] f) {
        act.func(f);
    }

    @Override
    public void derivative(Neuron[] f) {
        act.derivative(f);
    }
     
}
