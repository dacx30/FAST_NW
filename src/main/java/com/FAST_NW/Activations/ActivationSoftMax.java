/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.FAST_NW.Activations;

import com.FAST_NW.Entity.Neuron;
import org.apache.commons.math3.util.FastMath;

public class ActivationSoftMax implements Activation{

    @Override
    public void func(Neuron[] f) {
        double max = Double.NEGATIVE_INFINITY;
        for(int i = 0; i<f.length; i++){
            Neuron neurone = f[i];
            if( neurone.z>max){
                max = neurone.z;
            }            
        }
        
        double sum = 0;
        for(int i = 0; i<f.length; i++){
            Neuron neurone = f[i];
            double q = Math.exp(neurone.z-max);
            neurone.set_A(q);
            sum+=q;
        }
        for(int i = 0; i<f.length; i++){       
            Neuron n = f[i];
            n.set_A(n.get_A()/sum);
        }         
    }

    @Override
    public void derivative(Neuron[] f) {
        double sum = 0;
        for(int i = 0; i<f.length; i++){
            Neuron neurone = f[i];
            double q = FastMath.exp(neurone.z);
            neurone.set_exp_forSoftMax(q);
            sum+=q;
        }
        double sumSq = sum * sum;        
        for(int i = 0; i<f.length; i++){       
            Neuron n = f[i];
            double derivative = n.exp_forSoftMax()*(sum-n.exp_forSoftMax())/sumSq;
            n.set_da_dz(derivative);            
        } 
    }

    
    
}
