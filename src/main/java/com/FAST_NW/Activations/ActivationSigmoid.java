/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.FAST_NW.Activations;

import com.FAST_NW.Entity.Neuron;
import org.apache.commons.math3.util.FastMath;

public class ActivationSigmoid implements Activation{

    public static double sigmoid(double value){
        return 1f/(1f+FastMath.exp(-value));
    }
    private static double sigmoid_prime(double value){
        double s = sigmoid(value);
        return s * (1-s);
    }
    @Override
    public void func(Neuron[] f) {
        for(int i = 0; i<f.length; i++){
            Neuron neurone = f[i];            
            neurone.set_A(sigmoid(neurone.z)); 
        }
    }    

    @Override
    public void derivative(Neuron[] f) {
        for(int i = 0; i<f.length; i++){
            Neuron neurone = f[i];            
            neurone.set_da_dz(sigmoid_prime(neurone.z)); 
        }
        
    }
}
