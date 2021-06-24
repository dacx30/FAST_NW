
package com.FAST_NW.Entity;

import java.util.Random;

import static com.FAST_NW.Entity.NW.XOR_IDEAL;
import static com.FAST_NW.Entity.NW.XOR_INPUT;

public class Sample {
    public final double[] inputs;
    public final double[] outputs_actual;
    public final double[] outputs_forecast;
    public boolean outputIsOneHotEncoded = false;
    public long time_yn;
    public Sample(double[] ins, double[] outs) {
        this.inputs = ins;
        this.outputs_actual = outs;
        this.outputs_forecast = new double[outs.length];
    }

    public double negLogLoss(){
        if(!outputIsOneHotEncoded) throw new RuntimeException("cannot calculate logLoss if output is not one hot encoded");
        if(outputs_forecast.length != 2) throw new RuntimeException("not implemented for multi cross entropy, only binary. " + outputs_forecast.length);
        double pred_class1 = Math.max(1e-10, outputs_forecast[0]);
        double act  = outputs_actual[0];
        if(act == 1){
            return - Math.log(pred_class1);
        }
        else if(act == 0){
            return - Math.log(1 - pred_class1);
        }
        else{
            throw new RuntimeException("y labels must be binarised (0 or 1), not : " + act);
        }
    }

    public double MSE(boolean print){
        double sum = 0;
        for(int i = 0; i<outputs_actual.length; i++){
            double act   = outputs_actual[i];
            double forec = outputs_forecast[i];
            double delta = act-forec;
            if(print){
                System.out.println(act + " " + forec);
            }
            sum += delta*delta;
        }
        return sum;
    }
    public double MAE(boolean print){
        double sum = 0;
        for(int i = 0; i<outputs_actual.length; i++){
            double act = outputs_actual[i];
            double forec = outputs_forecast[i];
            double delta = act-forec;
            if(print){
                System.out.println(act + " " + forec);
            }
            sum += Math.abs(delta);
        }
        return sum;
    }
    public static Sample random(int n_inputs, int n_outs, Random rnd){
        double[] ins = new double[n_inputs];
        for(int i=0; i<ins.length; i++){
            ins[i]=rnd.nextGaussian();
        }
        double[] outs = new double[n_outs];
        for(int i=0; i<outs.length; i++){
            outs[i]=rnd.nextFloat();
        }
        return new Sample(ins, outs);     
    }
    public static Sample randomXOR(Random rnd){
        int index = rnd.nextInt(4);
        double[] ins    = XOR_INPUT[index];
        double[] outs   = XOR_IDEAL[index];
        Sample sample = new Sample(ins, outs);
        sample.outputIsOneHotEncoded = true;
        return sample;
    }
    public static int countXOR = 0;
    public static Sample nextXOR(){
        double[] ins    = XOR_INPUT[countXOR];
        double[] outs   = XOR_IDEAL[countXOR];
        Sample sample = new Sample(ins, outs);
        sample.outputIsOneHotEncoded = true;
        countXOR++;
        countXOR = countXOR%XOR_IDEAL.length;
        return sample;
    }

    public static Sample[] subSamples(Sample[] samples, int[] indices_feature){
        Sample[] samples2use = new Sample[samples.length];
        for (int i = 0; i < samples.length; i++) {
            double[] x_i = new double[indices_feature.length];
            for (int j = 0; j < indices_feature.length; j++) {
                int index_j = indices_feature[j];
                x_i[j] = samples[i].inputs[index_j];
            }
            samples2use[i] = new Sample(x_i, samples[i].outputs_actual);
            samples2use[i].outputIsOneHotEncoded = samples[i].outputIsOneHotEncoded;
        }
        return samples2use;
    }
}
