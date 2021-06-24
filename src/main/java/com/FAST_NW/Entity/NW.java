
package com.FAST_NW.Entity;

import com.FAST_NW.Activations.ActivationEnum;
import com.FAST_NW.Activations.ActivationSigmoid;
import com.FAST_NW.Losses.LossEnum;

import java.util.*;

public class NW {
    public final Layer[] layers;
    public final Layer outputLayer;
    public final LossEnum lossEnum;
    private final ArrayList<Object[]> conf;

    public NW(ArrayList<Object[]> conf, LossEnum loss, Random rnd) {
        this.lossEnum = loss;
        this.conf = conf;
        int nLayers= conf.size();
        if(nLayers<2){
            throw new RuntimeException("too small");
        }
        this.layers = new Layer[nLayers];                
        for(int i = 0; i<nLayers; i++){
            //DC addition on 10 March 2021
            boolean hasBias = conf.get(i).length == 3 ? (boolean)conf.get(i)[2] : i!=0;
            this.layers[i]=new Layer((int)conf.get(i)[1], (ActivationEnum)conf.get(i)[0], rnd, hasBias);
//            this.layers[i]=new Layer((int)conf.get(i)[1], (ActivationEnum)conf.get(i)[0], rnd, i!=0);
        }
        this.outputLayer = this.layers[layers.length-1];
    }
    public NW(ArrayList<Object[]> conf, LossEnum loss, List<double[]> biases) {
        this.lossEnum = loss;
        this.conf = conf;
        int nLayers= conf.size();
        if(nLayers<2){
            throw new RuntimeException("too small");
        }
        if(conf.size()!=biases.size()) throw new RuntimeException("dimension mistake: "  + conf.size() + " " + biases.size());
        this.layers = new Layer[nLayers];
        for(int i = 0; i<nLayers; i++){
            boolean hasBias = conf.get(i).length == 3 ? (boolean)conf.get(i)[2] : i!=0;
            this.layers[i]=new Layer((int)conf.get(i)[1], (ActivationEnum)conf.get(i)[0], biases.get(i), hasBias);
        }
        this.outputLayer = this.layers[layers.length-1];
    }

    public void connect_init(Random rnd){
        for(int i = 0; i<layers.length-1; i++){
            Layer li = layers[i];
            Layer lip1 = layers[i+1];                        
            li.setNextLayer(lip1, rnd);
        }
        connect_array();
    }
    public void connect_init(List<double[][]> weights){
        for(int i = 0; i<layers.length-1; i++){
            Layer li = layers[i];
            Layer lip1 = layers[i+1];
            li.setNextLayer(lip1, weights.get(i));
        }
        connect_array();
    }

    private void connect_array(){
        for(int i = 0; i<layers.length; i++){
            Layer li = layers[i];
            for(Neuron neu : li.neurones){
                neu.connections_2_prev = neu.temp_connections_2_prev.toArray(new Connection[neu.temp_connections_2_prev.size()]);
                neu.temp_connections_2_prev = null;

                neu.connections_2_next = neu.temp_connections_2_next.toArray(new Connection[neu.temp_connections_2_next.size()]);
                neu.temp_connections_2_next = null;
            }
        }
    }

    public NWCONF getconf(){
        NWCONF nwconf = new NWCONF();
        nwconf.conf = conf;
        nwconf.loss = lossEnum;
        List<double[]> biases    = new ArrayList<>();
        List<double[][]> weights = new ArrayList<>();
        nwconf.weights = weights;
        nwconf.biases = biases;
        for(Layer layer : layers){
            biases.add(layer.biases());
        }
        for (int i = 0; i < layers.length - 1; i++) {
            Layer layer = layers[i];
            weights.add(layer.weights());
        }
        return nwconf;
    }

    public int n_inputs(){
        return layers[0].neurones.length;
    }

    public void fit(double learningRate, Sample... samples){
        clear();
        for(Sample sample : samples){
            feedFwd(sample);
            if(useGenericLoss){
                outputError_NEW(sample);
                backProp_NEW(sample);
            }
            else{
                throw new RuntimeException("not implemented anymore");
//                outputError(sample);
//                backProp(sample);
            }
        }
        gradientDescent(samples, learningRate);
    }
    public void clear(){
        for(Layer layer : layers){
            for(Neuron neuron : layer.neurones){
                neuron.dC_dBias = 0;
                for(Connection con : neuron.connections_2_next){
                    con.dC_dweight = 0;
                }
            }
        }
    }
    public void feedFwd(Sample sample){
//        Feedforward: For each l=2,3,…,L compute zx,l=wlax,l−1+bl and ax,l=σ(zx,l).
        layers[0].feedFwd(sample);                
        for(int i = 1; i<layers.length; i++){
            layers[i].feedFwd();
        }        
        for(int i = 0; i<outputLayer.neurones.length; i++){
            sample.outputs_forecast[i] = outputLayer.neurones[i].a;
        }
    }

    public Double forecast(double[] x){
        layers[0].feedFwd(x);
        for(int i = 1; i<layers.length; i++){
            layers[i].feedFwd();
        }
        return outputLayer.neurones[0].a;
    }

    public boolean isFitted = false;

    public static boolean useGenericLoss = true;


    public void outputError_NEW(Sample sample){

        lossEnum.cost_delta(outputLayer, sample);

    }

    public void backProp_NEW(Sample sample){

        for (Neuron neuron : outputLayer.neurones) {
            neuron.dC_dBias += neuron.dC_dZ;
        }
        for(int L = layers.length-2; L>=0; L--){
            Layer layer = layers[L];
            for(Neuron neuron : layer.neurones){
                for(Connection con : neuron.connections_2_next){
                    con.dC_dweight += con.to.dC_dZ*con.from.a;// OK
                }
            }
            for(Neuron neuron : layer.neurones){
                double sum = 0;
                for(Connection con : neuron.connections_2_next){
                    sum += con.to.dC_dZ * con.weight;
                }
                neuron.dC_dA = sum;
            }
            if( L != 0){
                layer.activation.derivative(layer.neurones);
            }
            for(Neuron neuron : layer.neurones){
                neuron.dC_dZ = neuron.dC_dA * neuron.dA_dZ;
            }

            if(layer.hasBias){
                for(Neuron neuron : layer.neurones){
                    neuron.dC_dBias += neuron.dC_dZ;
                }
            }
        }
    }



    public void outputError(Sample sample){
        for(int i = 0; i<outputLayer.neurones.length; i++){
            Neuron neuron = outputLayer.neurones[i];
            neuron.dC_dA = sample.outputs_forecast[i]-sample.outputs_actual[i];
//            System.out.println(neuron.a);
        }

        outputLayer.activation.derivative(outputLayer.neurones);
        for (Neuron neuron : outputLayer.neurones) {
            neuron.dC_dBias += neuron.dC_dA*neuron.dA_dZ;
        }
    }
    public void backProp(Sample sample){
//        Backpropagate the error: For each l=L−1,L−2,…,2 compute δx,l=((wl+1)Tδx,l+1)⊙σ′(zx,l).
        for(int L = layers.length-2; L>=0; L--){
            Layer layer = layers[L];
            for(Neuron neuron : layer.neurones){
                for(Connection con : neuron.connections_2_next){
                    con.dC_dweight += con.to.dC_dA*con.to.dA_dZ*con.from.a;// OK
                }
            }
            for(Neuron neuron : layer.neurones){
                double sum = 0;
                for(Connection con : neuron.connections_2_next){
                    sum += con.to.dC_dA * con.to.dA_dZ * con.weight;
                }
                neuron.dC_dA = sum;
            }
            if( L != 0){
                layer.activation.derivative(layer.neurones);
            }
            if(layer.hasBias){
                for(Neuron neuron : layer.neurones){
                    neuron.dC_dBias += neuron.dC_dA*neuron.dA_dZ;
                }
            }
        }
    }
    public void gradientDescent(Sample[] samples, double learningRate){
//        Gradient descent: For each l=L,L−1,…,2 update the weights according to the rule wl→wl−ηm∑xδx,l(ax,l−1)T, and the biases according to the rule bl→bl−ηm∑xδx,l.
        int n_sample = samples.length;
        for(Layer layer : layers){
            for(Neuron neurone : layer.neurones){
                if(neurone.hasBias){
                    double gradient = neurone.dC_dBias/n_sample;
                    double step = gradient * learningRate;
                    neurone.bias -= step;
                }
                for(Connection con : neurone.connections_2_next){
                    double gradient = con.dC_dweight/n_sample;
                    double step = gradient * learningRate;
                    con.weight -= step;
                }
            }
        }
    }


    public double prevalence(Sample[] samples){
        int count_class1 = 0;
        for(Sample sample : samples){
            if(!sample.outputIsOneHotEncoded) {
                return Double.NaN;
//                throw new RuntimeException("cannot calculate prevalence of class 1 if output is not one hot encoded");
            }
            if(sample.outputs_actual[0] == 1){
                count_class1++;
            }
        }
        return (double)count_class1 / (double)samples.length;
    }
    public double scoreRandomGuess(Sample[] samples){
        if(!this.lossEnum.equals(LossEnum.CROSS_ENTROPY)) {
            return Double.NaN;
//            throw new RuntimeException("cannot calculate baseline score for loss " + this.lossEnum);
        }
        double prevalence = prevalence(samples);
        return -prevalence * Math.log(prevalence) - (1-prevalence) * Math.log(1-prevalence);
    }

    public double score(Sample[] samples){
        return scoreForLoss(samples, this.lossEnum);
    }
    public double scoreForLoss(Sample[] samples, LossEnum loss){
        double sum = 0;
        for(Sample sample : samples){
            feedFwd(sample);
            sum+=loss.score(sample);
        }
        return sum/samples.length;
    }

    public double[] getDensityArray(Sample[] samples, int index_y, int n, double step, double min){
        double[] dist = new double[n];
        for(Sample sample : samples){
            feedFwd(sample);
            double out = sample.outputs_forecast[index_y];
            int bucket = (int) ((out - min) / step);
            bucket = Math.max(0, bucket);
            bucket = Math.min(n - 1, bucket);
            dist[bucket] += 1f;
        }
        return dist;
    }

    private static void printLine(ArrayList<Object> lines, ArrayList<String> formats){
        if( lines.size()!=formats.size()){
            throw new RuntimeException("wrnog");
        }
        String format = "";
        Object[] arr = new Object[lines.size()+1];
        for(int i = 0; i<lines.size(); i++){
            arr[i]=lines.get(i);
            format+=formats.get(i)+ " ";
        }
        System.out.printf(format+"\n",arr);
        lines.clear();
        formats.clear();
    }

    public double sumWeights(){
        double sumw = 0;
        for(Layer layer : layers){
            for(Neuron n : layer.neurones){
                for(Connection con : n.connections_2_next){
                    sumw += con.weight;
                }
            }
        }
        return sumw;
    }

    public int numberParams(){
        int count = 0;
        for(Layer layer : layers){
            for(Neuron n : layer.neurones){
                if(n.hasBias){
                    count++;
                }
                for(Connection con : n.connections_2_next){
                    count++;
                }
            }
        }
        return count;
    }

    public void printInfo(){
        System.out.println("---------------------------------------");
        ArrayList<Object> lines = new ArrayList<>();
        ArrayList<String> formats = new ArrayList<>();
        Integer L = 0;
        for(Layer layer : layers){
            formats.add("L=%4d");
            lines.add(L);
            for(Neuron n : layer.neurones){                
                formats.add("Bias=%7.5f");
                lines.add(n.bias);
//                System.out.printf(n.bias + " ");
                for(Connection con : n.connections_2_next){
                    formats.add("w=%7.5f");
                    lines.add(con.weight);
//                    System.out.printf(con.weight + " ");
                }
            }
            printLine(lines, formats);
            L++;
        }
    }
    public static double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 1.0 } };
    public static double XOR_IDEAL[][] = { { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 } };



    public static void exampleClassification(int maxEpoch){
        Random rnd = new Random(1);
        int n_inputs = 2;
        int n_outs = 2;
        ArrayList<Object[]> conf = new ArrayList<>();
        conf.add(new Object[]{ActivationEnum.DUMMY, n_inputs});
//        conf.add(new Object[]{ActivationEnum.LEAKY_RELU, 10});
//        conf.add(new Object[]{ActivationEnum.LEAKY_RELU, 10});
        conf.add(new Object[]{ActivationEnum.SIGMOID, 4});
        conf.add(new Object[]{ActivationEnum.SOFTMAX, n_outs});
        NW nw = new NW(conf, LossEnum.CROSS_ENTROPY, rnd);
        nw.connect_init(rnd);
        nw.printInfo();
//        System.exit(0);
        int batchSize = 100000;
        Sample[] full_trainSet = new Sample[batchSize];
        for(int i = 0; i<batchSize; i++){
            Sample sample = Sample.randomXOR(rnd);
//            System.out.println(Arrays.toString(sample.inputs) + " | " + Arrays.toString(sample.outputs_actual));
            full_trainSet[i]=sample;
        }

        Sample[] full_testSet = new Sample[batchSize];
        for(int i = 0; i<batchSize; i++){
            Sample sample = Sample.randomXOR(rnd);
            full_testSet[i]=sample;
        }
        double LR_base = 10d;
        double LR = LR_base;
        int epoch = 0;
        int freq = 100;
        nw.printInfo();
        NWCONF nwconf = nw.getconf();
        nwconf.print();
        System.out.println("\nSTART TRAINING");
        double scoreTR = nw.score(full_trainSet);
        double scoreTE = nw.score(full_testSet);
        double scoreTRRandomGuess = nw.scoreRandomGuess(full_trainSet);
        double scoreTERandomGuess = nw.scoreRandomGuess(full_testSet);
        System.out.printf("%,10d %9.7f (%9.7f) %9.7f (%9.7f) %9.7f\n", epoch, scoreTR, scoreTRRandomGuess, scoreTE, scoreTERandomGuess, LR);
        double prevalTR = nw.prevalence(full_trainSet);
        double prevalTE = nw.prevalence(full_testSet);
        System.out.printf("%,10d %9.7f %9.7f %9.7f\n", epoch, prevalTR, prevalTE, LR);
        while(true){
            LR = LR_base * rnd.nextDouble();
            nw.fit(LR, full_trainSet);
            if(epoch % freq ==0){
//                nw.printInfo();
                scoreTR = nw.score(full_trainSet);
                scoreTE = nw.score(full_testSet);
                System.out.printf("%,10d %9.7f (%9.7f) %9.7f (%9.7f) %9.7f\n", epoch, scoreTR, scoreTRRandomGuess, scoreTE, scoreTERandomGuess, LR);
            }
            epoch++;
            if(epoch >= maxEpoch){
                break;
            }
        }
        System.out.println("\nEND TRAINING");
        nw.printInfo();

        for (int i = 0; i < 5; i++) {
            Sample sample = Sample.nextXOR();
            nw.feedFwd(sample);
            System.out.println("inputs: " + Arrays.toString(sample.inputs)
                    + " predict = " + Arrays.toString(sample.outputs_forecast)
                    + " actual = " + Arrays.toString(sample.outputs_actual)
                    + " score = " + nw.lossEnum.score(sample));
        }

    }


    public static void exampleRegression(){
        Random rnd = new Random(1);
        int n_inputs = 3;
        int n_outs = 1;
        ArrayList<Object[]> conf = new ArrayList<>();
        conf.add(new Object[]{ActivationEnum.DUMMY, n_inputs});
//        conf.add(new Object[]{ActivationEnum.LEAKY_RELU, 10});
//        conf.add(new Object[]{ActivationEnum.LEAKY_RELU, 10});
        conf.add(new Object[]{ActivationEnum.SIGMOID, n_outs});
//        conf.add(new Object[]{ActivationEnum.LINEAR, n_outs});
        NW nw = new NW(conf, LossEnum.MSE, rnd);
        nw.connect_init(rnd);
        int batchSize = 100_000;
        Sample[] full_trainSet = new Sample[batchSize];
        for(int i = 0; i<batchSize; i++){
            Sample sample = Sample.random(n_inputs, n_outs, rnd);
            double OUT = ActivationSigmoid.sigmoid(-1+3.14* DoubleArrayUtils.sum(sample.inputs));
//            double OUT = -1+3.14*DoubleArrayUtils.sum(sample.inputs);
//            System.out.println(OUT);
            sample.outputs_actual[0]=OUT;
            full_trainSet[i]=sample;
        }

        Sample[] full_testSet = new Sample[batchSize];
        for(int i = 0; i<batchSize; i++){
            Sample sample = Sample.random(n_inputs, n_outs, rnd);
            double OUT = ActivationSigmoid.sigmoid(-1+3.14*DoubleArrayUtils.sum(sample.inputs));
//            double OUT = -1+3.14*DoubleArrayUtils.sum(sample.inputs);
//            System.out.println(OUT);
            sample.outputs_actual[0]=OUT;
            full_testSet[i]=sample;
        }
//        double LR_base = 1d;
        double LR_base = 10d;
        double LR = LR_base;
        int epoch = 0;
        int freq = 100;
        nw.printInfo();
        double scoreTR = nw.score(full_trainSet);
        double scoreTE = nw.score(full_testSet);
        System.out.printf("%,10d %9.7f %9.7f %9.7f\n", epoch, scoreTR, scoreTE, LR);
        while(true){
            LR = LR_base * (0.5+0.5*Math.sin(epoch*0.1));
            LR = LR_base * rnd.nextDouble();
            nw.fit(LR, full_trainSet);
            if(epoch % freq ==0){
                nw.printInfo();
                scoreTR = nw.score(full_trainSet);
                scoreTE = nw.score(full_testSet);
                System.out.printf("%,10d %9.7f %9.7f %9.7f\n", epoch, scoreTR, scoreTE, LR);
            }
            epoch++;
            if(epoch >= 100){
                break;
            }
        }
//        int a = 1;
//        System.out.println(score);
//        int chunk = 1000;
//        for(int k = 0; k<1000; k++){
//            double dt = -System.currentTimeMillis();
//            for(int i = 0; i<chunk; i++){
//                nw.output(sample);
//            }
//            dt +=System.currentTimeMillis();
//            System.out.println("dt " + dt/chunk);
//        }

    }

    public static void main(String[] args) {

        exampleClassification(1000);

    }

}