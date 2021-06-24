package com.FAST_NW;

import com.FAST_NW.Activations.ActivationEnum;
import com.FAST_NW.Entity.NW;
import com.FAST_NW.Entity.Sample;
import com.FAST_NW.Losses.LossEnum;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.Random;


public class Main {

    public static void main(String[] args) {

        //instanciation networks - fast vs baseline dp4j
        NW nw_fast                  = get_trainedNW_FAST(100_000);
        MultiLayerNetwork nw_dp4j   = get_trainedNW_DP4J(100_000);


        //feedforward latency comparison
        System.out.println("\nFEEDFORWARD LATENCIES");
        int n   = 100_000;

        Sample[] samples_fast   = new Sample[n];
        INDArray[] samples_dp4j = new INDArray[n];
        for (int i = 0; i < n; i++) {
            samples_fast[i] = Sample.nextXOR();
            samples_dp4j[i] = nextXOR_dp4j();
        }

        //FAST NW
        System.out.println("\nFAST NW...");
        long dt = 0;
        long time0 = System.currentTimeMillis();
        for (int i = 0; i < n; i++) {
            Sample sample = samples_fast[i];
            nw_fast.feedFwd(sample);
        }
        dt += System.currentTimeMillis() - time0;
        System.out.println("N =      " + n);
//        System.out.println("SUM DT = " + dt + " ms");
        System.out.println("AVG DT = " + Math.round(dt / (double)n * 1000000) + " ns");


        //DP4j NW
        System.out.println("\nDP4j NW...");
        time0 = System.currentTimeMillis();
        dt = 0;
        for (int i = 0; i < n; i++) {
            INDArray sample = samples_dp4j[i];
            nw_dp4j.output(sample);
        }
        dt += System.currentTimeMillis() - time0;
        System.out.println("N =      " + n);
//        System.out.println("SUM DT = " + dt + " ms");
        System.out.println("AVG DT = " + Math.round(dt / (double)n * 1000000) + " ns");

    }


    public static INDArray[] XORInputs_DP4J(){

        INDArray input_0 = Nd4j.zeros(1, 2);
        input_0.putScalar(new int[]{0, 0}, 0);
        input_0.putScalar(new int[]{0, 1}, 0);

        INDArray input_1 = Nd4j.zeros(1, 2);
        input_1.putScalar(new int[]{0, 0}, 1);
        input_1.putScalar(new int[]{0, 1}, 1);

        INDArray input_2 = Nd4j.zeros(1, 2);
        input_2.putScalar(new int[]{0, 0}, 0);
        input_2.putScalar(new int[]{0, 1}, 1);

        INDArray input_3 = Nd4j.zeros(1, 2);
        input_3.putScalar(new int[]{0, 0}, 1);
        input_3.putScalar(new int[]{0, 1}, 0);

        return new INDArray[]{input_0, input_1, input_2, input_3};
    }




    public static NW get_trainedNW_FAST(int nEpoch){

        Random rnd = new Random(1);
        int n_inputs = 2;
        int n_outs = 2;
        int freq = nEpoch / 10;
        ArrayList<Object[]> conf = new ArrayList<>();
        conf.add(new Object[]{ActivationEnum.DUMMY, n_inputs});
        conf.add(new Object[]{ActivationEnum.SIGMOID, 4});
        conf.add(new Object[]{ActivationEnum.SOFTMAX, n_outs});
        NW nw = new NW(conf, LossEnum.CROSS_ENTROPY, rnd);
        nw.connect_init(rnd);

        int batchSize = 4;
        Sample[] full_trainSet = new Sample[batchSize];
        for(int i = 0; i<batchSize; i++){
            Sample sample = Sample.nextXOR();
            full_trainSet[i]=sample;
        }

        double lr = 0.1d;
        int epoch = 0;
//        nw.printInfo();
        System.out.println("\nSTART TRAINING FAST...");
        double scoreTR = nw.score(full_trainSet);
        double scoreTRRandomGuess = nw.scoreRandomGuess(full_trainSet);
        System.out.printf("EPOCH %,10d | score_train %9.7f (baseline %9.7f)\n", epoch, scoreTR, scoreTRRandomGuess);
        while(true){
            epoch++;
            nw.fit(lr, full_trainSet);
            if(epoch % freq == 0){
                scoreTR = nw.score(full_trainSet);
                System.out.printf("EPOCH %,10d | score_train %9.7f (baseline %9.7f)\n", epoch, scoreTR, scoreTRRandomGuess);
            }
            if(epoch >= nEpoch){
                break;
            }
        }
        System.out.println("END TRAINING");
//        nw.printInfo();
        return nw;
    }

    public static final INDArray[] inputs_DP4J = XORInputs_DP4J();
    public static int countXOR_dp4j = 0;
    public static INDArray nextXOR_dp4j(){
        INDArray ins = inputs_DP4J[countXOR_dp4j];
        countXOR_dp4j++;
        countXOR_dp4j = countXOR_dp4j% inputs_DP4J.length;
        return ins;
    }


    public static MultiLayerNetwork get_trainedNW_DP4J(int nEpochs){

        int seed = 1234;        // number used to initialize a pseudorandom number generator.

        // list off input values, 4 training samples with data for 2
        // input-neurons each
        INDArray input = Nd4j.zeros(4, 2);

        // correspondending list with expected output values, 4 training samples
        // with data for 2 output-neurons each
        INDArray labels = Nd4j.zeros(4, 2);

        // create first dataset
        // when first input=0 and second input=0
        input.putScalar(new int[]{0, 0}, 0);
        input.putScalar(new int[]{0, 1}, 0);
        // then the first output fires for false, and the second is 0 (see class comment)
        labels.putScalar(new int[]{0, 0}, 1);
        labels.putScalar(new int[]{0, 1}, 0);

        // when first input=1 and second input=0
        input.putScalar(new int[]{1, 0}, 1);
        input.putScalar(new int[]{1, 1}, 0);
        // then xor is true, therefore the second output neuron fires
        labels.putScalar(new int[]{1, 0}, 0);
        labels.putScalar(new int[]{1, 1}, 1);

        // same as above
        input.putScalar(new int[]{2, 0}, 0);
        input.putScalar(new int[]{2, 1}, 1);
        labels.putScalar(new int[]{2, 0}, 0);
        labels.putScalar(new int[]{2, 1}, 1);

        // when both inputs fire, xor is false again - the first output should fire
        input.putScalar(new int[]{3, 0}, 1);
        input.putScalar(new int[]{3, 1}, 1);
        labels.putScalar(new int[]{3, 0}, 1);
        labels.putScalar(new int[]{3, 1}, 0);

        // create dataset object
        DataSet ds = new DataSet(input, labels);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Sgd(0.1))
                .seed(seed)
                .biasInit(0) // init the bias with 0 - empirical value, too
                // from "http://deeplearning4j.org/architecture": The networks can
                // process the input more quickly and more accurately by ingesting
                // minibatches 5-10 elements at a time in parallel.
                // this example runs better without, because the dataset is smaller than
                // the mini batch size
                .miniBatch(false)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(4)
                        .activation(Activation.SIGMOID)
                        // random initialize weights with values between 0 and 1
                        .weightInit(new UniformDistribution(0, 1))
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .weightInit(new UniformDistribution(0, 1))
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // add an listener which outputs the error every 100 parameter updates
        net.setListeners(new ScoreIterationListener(100));

        // C&P from LSTMCharModellingExample
        // Print the number of parameters in the network (and for each layer)
//        System.out.println(net.summary());

        // here the actual learning takes place
        System.out.println("\nSTART TRAINING DP4J...");
        for( int i=0; i < nEpochs; i++ ) {
            net.fit(ds);
        }
        System.out.println("FINAL SCORE = " + net.score(ds));
        System.out.println("END TRAINING");
        return net;
    }

}
