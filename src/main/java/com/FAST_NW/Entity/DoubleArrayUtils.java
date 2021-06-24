
package com.FAST_NW.Entity;

public class DoubleArrayUtils {
    
    public static double sum(double[] array){
        double sum = 0;
        for(double d: array){
            sum+=d;
        }
        return sum;
    }
    public static double prod(double[] array){
        double prod = 1;
        for(double d: array){
            prod*=d;
        }
        return prod;
    }
    
}
