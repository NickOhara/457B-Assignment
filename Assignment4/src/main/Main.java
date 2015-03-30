package main;

import java.util.ArrayList;
import java.util.Random;

import bpl.Neuron;

public class Main {
	private static int Nodes = 5;
	private double hiddenLayers = 1;
	private static int layerDepth = 0;
	private static Random generator = new Random(System.currentTimeMillis());
	private static double learningRate = 0.1;
	static double randomGenerator() {
	        return generator.nextDouble()*0.5;
	}
	public static void main( String[] args ) throws Exception{
		
		
		
		setupFunction(1);
		
		setupFunction(2);
		
	}
	
	public static double outputFunction1(double x){
		return Math.pow(Math.E, -x*x);
	}
	public static double outputFunction2(double x){
		return Math.atan(x);
	}
	public static void getEstimate(Neuron [] neurons,Neuron outputNeuron,int _function, double [] inputWeights,double [] outputWeights){
		
		double [] xEst = new double [Nodes];
	
		for(int i=0;i<xEst.length;i++){
			xEst[i] = randomGenerator();
		}
		double []  yEst = new double [xEst.length];
		double [] xProp = new double[1]; // Propagating input
		double [] o = new double[Nodes];
		for(int i=0;i<xEst.length;i++){
			if(_function == 1){
				yEst[i] = outputFunction1(xEst[i]);
			}
			else{
				yEst[i] = outputFunction2(xEst[i]);
			}
		}
		double predict = 0;
		for(int i=0;i<xEst.length;i++){
			xProp[0] = xEst[i];
			for ( int j = 0; j < Nodes; j++ ) {
				o[j] = neurons[j].predictOutput(inputWeights, xProp); //Predict Neuron Output
			}
	
			predict = 0.0;
			
	
			predict = outputNeuron.predictOutput(outputWeights,o); //Predict Network Output
			
			System.out.println("Actual: " + xProp[0] + " Estimate: " + predict);
		}
		
	}
	public static void setupFunction(int function){
		System.out.println("Starting "+function+" function:");
		System.out.println("-----------------------------------------");
		System.out.println();
		
		int k = 2; //number of inputs
		double [] input = new double [k];
		double [] output = new double [input.length];
		double [] error = new double [Nodes];
		double [] inputWeights = new double [Nodes];
		double [] outputWeights = new double [Nodes];
		
		for(int i=0;i<input.length;i++){
			input[i] = randomGenerator();
		}	
		for(int i=0;i<Nodes;i++){
			inputWeights[i] = 0.2;
			outputWeights[i] = 0.2;
		}	
		double errorTotal = 5;

		for(int i=0;i<input.length;i++){
			if(function == 1){
				output[i] = outputFunction1(input[i]);
			}
			else{
				output[i] = outputFunction2(input[i]);
			}
		}
		
		Neuron [] neurons = new Neuron [Nodes];
		for(int i = 0;i<Nodes;i++){
			Neuron neuron = new Neuron ( i, Nodes,"input") ;
			neurons[i] = neuron;
		}
		Neuron outputNeuron = new Neuron(0,1,"output");
		
		//double bias = 0;
		double[] xProp = new double[1]; // Propagating input
		double [] os = new double[(int) (input.length)]; // Array of outputs

		
		double predict=0;
		double [] o = new double[Nodes];
		while(true){
			if(errorTotal < 0.01 && errorTotal >= 0){
				break;
			}
			for(int index=0;index<input.length;index++){
				xProp[0] = input[index];
				errorTotal = 0;
				
				// Prop forward
				for( int bob = 0; bob < Nodes; bob++ ) {
					o[bob] = 0.0;
				}
				for ( int j = 0; j < Nodes; j++ ) {
					o[j] = neurons[j].predictOutput(inputWeights, xProp); //Predict Neuron Output
				}
		
				predict = 0.0;
				

				predict = outputNeuron.predictOutput(outputWeights,o); //Predict Network Output
					

				errorTotal = Neuron.deltaError(predict, output[index]);
				double outSigErr = Neuron.outputSignalError(predict,output[index]);
				for( int i = 0; i < Nodes; i++ ) {
					outputWeights[i] = outputWeights[i] + learningRate*outSigErr*o[i];					
				}
				double [] inSigErr = new double [Nodes];
				for(int i = 0;i<Nodes-1;i++){
					inSigErr[i] = o[i]*(1-o[i])*outputWeights[i]*outSigErr;
				}
				for( int i = 0; i < Nodes; i++ ) {
					inputWeights[i] = inputWeights[i] + learningRate*inSigErr[i]*o[i];					
				}
				//back prop
				/*double dOut = 0;
				double [] d = new double[Nodes];;
				for(int i=0;i<Nodes;i++){
					dOut = output[i] - neurons[i].predictedoutput;
					double [] tempD = new double [1];
					tempD[0] = dOut;
					for (int j = 0; i < Nodes; i++) {
						double [] w = new double [1];
						w[0] = outputWeights[j];
						d[j] = neurons[j].signalError(w, tempD);
					}
				}*/
				//changeError = errorTotal - prevError;
			}
		}
		getEstimate(neurons,outputNeuron,function,inputWeights,outputWeights);
	}
	
		
}