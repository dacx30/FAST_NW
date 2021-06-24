
package com.FAST_NW.Entity;

import com.FAST_NW.Entity.Connection;

import java.util.ArrayList;
import java.util.Random;

public class Neuron {
    public double z;
    public double a;
    public double bias;
    public double dC_dBias;
    public double dC_dA;
    public double dA_dZ;
    public double dC_dZ;
    public ArrayList<Connection> temp_connections_2_prev = new ArrayList<>();
    public ArrayList<Connection> temp_connections_2_next = new ArrayList<>();
    
    public Connection[] connections_2_prev;
    public Connection[] connections_2_next;
    public final boolean hasBias;
    
    public Neuron(Random rnd, boolean hasbias) {
        this.hasBias = hasbias;
        if(hasbias){
            this.bias = rnd.nextGaussian()*0.1;
        }
    }
    public Neuron(double bias, boolean hasbias) {
        this.hasBias = hasbias;
        if(hasbias){
            this.bias = bias;
        }
    }
    public void calcZ(){
        if(connections_2_prev==null){
            throw new RuntimeException("something is wrong");
        }
        
        if(hasBias){
            this.z = bias;
        }
        else{
            this.z = 0;
        }
        for(Connection c : connections_2_prev){
            z += c.from.a*c.weight;
        }
    }    
    public double exp_forSoftMax(){
        return exp_forSoftMax;
    }
    private double exp_forSoftMax;
    public void set_exp_forSoftMax(double e){
        if(Double.isFinite(e)){
            this.exp_forSoftMax = e;
        }
        else{
            throw new RuntimeException("wrong " + e);
        }
    }
    public void set_A(double activation){
        if(Double.isFinite(activation)){
            this.a = activation;
        }
        else{
            throw new RuntimeException("wrong " + activation);
        }
    }
     public void set_da_dz(double derive){
        if(Double.isFinite(derive)){
            this.dA_dZ = derive;
        }
        else{
            throw new RuntimeException("wrong " + derive);
        }
    }
    public double get_A(){
        return a;
    }
}
