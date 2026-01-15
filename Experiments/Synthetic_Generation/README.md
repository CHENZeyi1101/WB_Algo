For 2d experiments:
Step1: Select the source measure (i.e., the ground-truth barycenter measure) and the auxiliary measures by running input_measure_select.py
Step2: Set up and generate samples from input measures by running input_sample_dim2.py. Samples are saved in CSV files.
Step3: Visualize source measures, auxiliary measures, and input measures by running visualize_measures_dim2.py
Step4: Prepare a set of samples from the barycenter for evaluation by running samples_for_evaluation_dim2.py

For 10d experiments:
Step1: Set up and generate samples from input measures by running input_sample_dim10.py. Samples are saved in CSV files.
Step2: Prepare a set of samples from the barycenter for evaluation by running samples_for_evaluation_dim10.py