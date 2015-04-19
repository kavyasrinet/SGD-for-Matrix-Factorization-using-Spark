# SGD-for-Matrix-Factorization-using-Spark
The program dsgd_mf.py is the main program to be run.

The syntax to run the file is :
spark-submit dsgd_mf.py #_of_factors #_of_workers #_of_iterations beta_value lambda_value input_file_containing_data outputW_file outputH_file


Using the given parameters, the program writes out to the given outputW_file and outputH_file

If checking for convergence, uncomment the lines 105-113, if not checking for convergence, and want to run to completion, comment those lines.
