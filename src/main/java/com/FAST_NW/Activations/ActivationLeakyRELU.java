
package com.FAST_NW.Activations;

import com.FAST_NW.Entity.Neuron;


public class ActivationLeakyRELU implements Activation{

    @Override
    public void func(Neuron[] f) {
        for(int i = 0; i<f.length; i++){
            Neuron neurone = f[i];            
            double v = neurone.z;            
            if( v >0){
                neurone.set_A(v);
            }
            else{
                neurone.set_A(v*0.01f);
            }
        }
    }    

    @Override
    public void derivative(Neuron[] f) {
        for(int i = 0; i<f.length; i++){
            Neuron neurone = f[i];            
            double v = neurone.z;            
            if( v >0){
                neurone.set_da_dz(1);
            }
            else{
                neurone.set_da_dz(0.01f);
            }
        }
    }
}
