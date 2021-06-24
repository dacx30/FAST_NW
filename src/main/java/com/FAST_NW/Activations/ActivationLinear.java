/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.FAST_NW.Activations;

import com.FAST_NW.Entity.Neuron;

public class ActivationLinear implements Activation{

    @Override
    public void func(Neuron[] f) {
        for(int i = 0; i<f.length; i++){
            Neuron neurone = f[i];            
            neurone.set_A(neurone.z);
        }
    }    

    @Override
    public void derivative(Neuron[] f) {
        for(int i = 0; i<f.length; i++){
            Neuron neurone = f[i];            
            neurone.set_da_dz(1d);
        }
        
    }
}
