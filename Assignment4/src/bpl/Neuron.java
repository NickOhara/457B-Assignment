package bpl;

public class Neuron {
	private String title;
	private int neuron;
	private int numOfNeurons;
	private double [] weight;
	private double inputWeight;
	private double outputWeight;
	private double initWeight = 0.5; // Set
	private double learningRateOutput = 0.1; // Set
	private double learningRateInput = 0.1;
	public double predictedoutput = 0;
	public String getTitle(){
		return title;
	}
	
	public int GetNumberOfNeurons() {
		return numOfNeurons;
	}
	
	public Neuron ( int index, int _numOfNeurons, String _title ) {
		title = _title;
		neuron = index;
		numOfNeurons = _numOfNeurons;
		initWeights();
	}
	
	public void initWeights () {
		inputWeight = initWeight;
		outputWeight = initWeight;
	}
	
	public double updateNeuronWeights( double weight, double err, double neuronVal, double inputVal, String neuronType) {
		if ( neuronType.equals( "output" ) ){
			outputWeight = outputWeight - err * neuronVal * learningRateOutput;
			return outputWeight;
		}else {
//			double x = 1 - ( neuronVal * neuronVal );
//			x = x * outputWeight * err * learningRateInput;
//			x = x * inputVal;
			inputWeight = weight - learningRateInput * err * neuronVal * (1 - neuronVal);
			return inputWeight;
		}
	}
	
	public double [] updateWeights (double sigErr, double o) {
		for ( int i = 0; i < numOfNeurons; i++ ) {
		//	weight[i] = weight[i] + learningRate*sigErr*o;
		}
		return weight;
	}
	public static double sigmoid (double z) {
		return 1 / (1 + Math.exp((-1)*z));
	}
	
	public static double inverseTanh( double z ) {
		return 0.5 *( Math.log( 1 + z ) - Math.log( 1 - z ) );
	}
	
	public static double tanh( double z ) {
		if ( z > 500 )
			return 1;
		else if( z < -500 )
			return -1;
		else {
			double a = Math.exp(z);
			double b = Math.exp(-z);
			return (a-b)/(a+b);
		}
	}
	
	public double predictNeuronOutput( double inputVal ) {
		return sigmoid(inputVal * inputWeight);
	}
	
	public double predictOutputNew( double neuronVal ) {
		return sigmoid(neuronVal * outputWeight);
	}
	
	public double predictOutput ( double [] w , double [] x ) {
		double o = 0;
		for ( int i = 0; i < x.length; i++ ) {
			o += w[i] * x[i];
		}
		predictedoutput = o;
		return predictedoutput;
	}
	
	public static double deltaError ( double o, double t ) {
		return ((Math.pow((o - t),2)) / 2);
	}
	
	public static double outputSignalError ( double o, double t ) {
		return o * (1 - o) * (t - o);
	}
	
	public double inputSignalError ( double o, double sum ) {
		return o * (1 - o) * sum;
	}
	public double signalError (double [] w, double [] d) {
		return summation(w, d);
	}
	public double summation (double [] w, double [] x) {
		double y = 0;
		for (int i = 0; i < x.length; i++) {
			y += w[i] * x[i];

		}
		return y;
	}
}
