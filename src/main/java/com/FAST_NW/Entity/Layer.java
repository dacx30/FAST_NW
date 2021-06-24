
package com.FAST_NW.Entity;

import com.FAST_NW.Activations.ActivationEnum;

import java.util.Random;

public class Layer {
    public final ActivationEnum activation;
    public final Neuron[] neurones;
    public Layer previousLayer;
    public Layer nextLayer;
    public final boolean hasBias;
    public Layer(int n, ActivationEnum act, Random rnd, boolean hasbias) {
        this.hasBias = hasbias;
        this.neurones = new Neuron[n];
        for(int i = 0; i<n; i++){
            neurones[i]=new Neuron(rnd, hasBias);
        }
        this.activation = act;        
    }
    public Layer(int n, ActivationEnum act, double[] biases, boolean hasbias) {
        this.hasBias = hasbias;
        this.neurones = new Neuron[n];
        if(biases.length!=n) throw new RuntimeException("dimension mistake: " + biases.length + " " + n);
        for(int i = 0; i<n; i++){
            neurones[i]=new Neuron(biases[i], hasBias);
        }
        this.activation = act;
    }
    public void setNextLayer(Layer next, Random rnd){
        if(nextLayer!=null){
            throw new RuntimeException("wrong");
        }
        this.nextLayer = next;
        nextLayer.previousLayer = this;
        for(int i = 0; i<next.neurones.length; i++){
            Neuron to = next.neurones[i];            
            for(int j = 0; j<neurones.length; j++){
                Neuron from = neurones[j];
                Connection connection = new Connection(from, to, rnd);
                to.temp_connections_2_prev.add(connection);
                from.temp_connections_2_next.add(connection);
            }            
        }
    }
    public void setNextLayer(Layer next, double[][] weights){
        if(nextLayer!=null){
            throw new RuntimeException("wrong");
        }
        this.nextLayer = next;
        nextLayer.previousLayer = this;
        if(weights.length!=next.neurones.length) throw new RuntimeException("dimension mistake: " + weights.length + " " + next.neurones.length);
        if(weights[0].length!=neurones.length) throw new RuntimeException("dimension mistake: " + weights[0].length + " " + neurones.length);
        for(int i = 0; i<next.neurones.length; i++){
            Neuron to = next.neurones[i];
            for(int j = 0; j<neurones.length; j++){
                Neuron from = neurones[j];
                Connection connection = new Connection(from, to, weights[i][j]);
                to.temp_connections_2_prev.add(connection);
                from.temp_connections_2_next.add(connection);
            }
        }
    }

    public void feedFwd(double[] inputs){
        if(inputs.length != neurones.length) throw new RuntimeException("problem with inputs dimension: " + inputs.length + " vs " + neurones.length);
        for(int i = 0; i<neurones.length; i++){
            neurones[i].set_A(inputs[i]);
        }
    }
    public void feedFwd(Sample sample){
        for(int i = 0; i<neurones.length; i++){
            neurones[i].set_A(sample.inputs[i]);
        }        
    }    
    public void feedFwd(){        
        for(Neuron n : neurones){
            n.calcZ();
        }
        activation.func(neurones);
    }

    public double[] biases(){
        double[] biases = new double[neurones.length];
        for (int i = 0; i < biases.length; i++) {
            biases[i] = neurones[i].bias;
        }
        return biases;
    }
    public double[][] weights(){
        if(nextLayer == null) throw new RuntimeException("mistake");
        double[][] weights = new double[nextLayer.neurones.length][neurones.length];
        for(int i = 0; i<nextLayer.neurones.length; i++){
            Neuron n = nextLayer.neurones[i];
            for (int j = 0; j < n.connections_2_prev.length; j++) {
                weights[i][j] = n.connections_2_prev[j].weight;
            }
        }
        return weights;
    }
}
