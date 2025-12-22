make ./experiments/bike_sharing/bike_sharing_PoissonGLM

./experiments/bike_sharing/bike_sharing_PoissonGLM sample num_warmup=10000 num_samples=10000000  data file=./experiments/bike_sharing/data_full.json output file=./experiments/bike_sharing/posterior_full.csv random seed=9000

./experiments/bike_sharing/bike_sharing_PoissonGLM sample num_warmup=10000 num_samples=10000000  data file=./experiments/bike_sharing/data_split_0.json output file=./experiments/bike_sharing/posterior_split_0.csv random seed=1000

./experiments/bike_sharing/bike_sharing_PoissonGLM sample num_warmup=10000 num_samples=10000000  data file=./experiments/bike_sharing/data_split_1.json output file=./experiments/bike_sharing/posterior_split_1.csv random seed=1001

./experiments/bike_sharing/bike_sharing_PoissonGLM sample num_warmup=10000 num_samples=10000000  data file=./experiments/bike_sharing/data_split_2.json output file=./experiments/bike_sharing/posterior_split_2.csv random seed=1002

./experiments/bike_sharing/bike_sharing_PoissonGLM sample num_warmup=10000 num_samples=10000000  data file=./experiments/bike_sharing/data_split_3.json output file=./bike_sharing/experiments/posterior_split_3.csv random seed=1003

./experiments/bike_sharing/bike_sharing_PoissonGLM sample num_warmup=10000 num_samples=10000000  data file=./experiments/bike_sharing/data_split_4.json output file=./experiments/bike_sharing/posterior_split_4.csv random seed=1004