/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.FAST_NW.Activations;

import com.FAST_NW.Entity.Neuron;

public interface Activation {
    public void func(Neuron[] f);
    public void derivative(Neuron[] f);
    
}
