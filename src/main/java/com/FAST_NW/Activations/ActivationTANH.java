/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.FAST_NW.Activations;

import com.FAST_NW.Entity.Neuron;
import org.apache.commons.math3.util.FastMath;

public class ActivationTANH implements Activation{

    public static double rational_tanh(double x)
    {
        if( x < -3 ) {
            return -1;
        } else if( x > 3 ) {
            return 1;
        } else {
            return x * ( 27 + x * x ) / ( 27 + 9 * x * x );
        }
    }
    public static double tanh(double value){
//        return rational_tanh(value);
        return FastMath.tanh(value);
    }
    private static double tanh_prime(double value){
        double s = tanh(value);
        return (1-s*s);
    }
    @Override
    public void func(Neuron[] f) {
        for (Neuron neurone : f) {
            neurone.set_A(tanh(neurone.z)); 
        }
    }    

    @Override
    public void derivative(Neuron[] f) {
        for (Neuron neurone : f) {
            neurone.set_da_dz(tanh_prime(neurone.z)); 
        }
        
    }
}
