package com.FAST_NW.Entity;

import com.FAST_NW.Losses.LossEnum;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NWCONF{

    public LossEnum loss;
    public ArrayList<Object[]> conf;
    public List<double[]> biases;
    public List<double[][]> weights;

    public NWCONF(){ }

    public void print(){
        System.out.println("loss: " + loss);
        System.out.println("nLayer: " + conf.size());
        System.out.println("conf:");
        for (int i = 0; i < conf.size(); i++) {
            System.out.println("layer " + i);
            Object[] conf_i = conf.get(i);
            System.out.println(conf_i[0] + " " + conf_i[1]);
            double[] bias_i = biases.get(i);
            System.out.println("bias:    " + Arrays.toString(bias_i));
            if(i < weights.size()){
                double[][] weights_i = weights.get(i);
                System.out.println("weights: ");
                for (int j = 0; j < weights_i.length; j++) {
                    System.out.println(Arrays.toString(weights_i[j]));
                }
            }
        }
    }

}
