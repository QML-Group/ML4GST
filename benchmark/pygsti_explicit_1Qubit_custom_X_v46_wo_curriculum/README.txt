***You need to change file path to load the csv dataset file***

Benchmarking functions (run all cells before # Training loop, model is not required for benchmark, instead a static list is fed into benchmark functions)
-	def apply_gate_sequence_ground_truth
-	def validate_step_ground_truth
-	def sample_and_evaluate_ground_truth
-	ground_truth_list = [0.01, 0.02, 0.1, 0.2] # This is the ground-truth parameter list
-	predicted_list = [0.009888766, 0.019712888, 0.100903568, 0.198058618] # This is the predicted parameter list WITHOUT curriculum learning (from the EPOCH of min MSE)
-	predicted_list_curriculum_learning = [0.010365579, 0.020106427, 0.100261497, 0.199429939] # This is the predicted parameter list WITH curriculum learning (from the EPOCH of min MSE)

RUN sample_and_evaluate_ground_truth 