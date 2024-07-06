import os
import hydra

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    output_dir = cfg["output_file_path"]["local"]
    bash_list = []
    instance_query_strategies = ['random', 'uncertainty', 'coreset']
    annotator_query_strategies = ['random', 'round-robin']
    n_annototar_per_instance_list = [1, 2]
    dataset = 'letter'
    batch_size = 256
    n_cycles = 25
    seed_list = [0, 1, 2, 3, 4]

    file_path = "srun python/mnt/stud/home/jcheng/scikit-activeml/experiment/experiment.py"

    for instance_query_strategy in instance_query_strategies:
        for annotator_query_strategy in annotator_query_strategies:
            for n_annototar_per_instance in n_annototar_per_instance_list:
                for seed in seed_list:
                    _bash = (f"{file_path} {dataset} {instance_query_strategy} "
                             f"{annotator_query_strategy} {batch_size} "
                             f"{n_annototar_per_instance} {n_cycles} {seed}")
                    bash_list.append(_bash)

    with open(f'{output_dir}/bash.txt', 'w', newline='') as bash_file:
        for line in bash_list:
            bash_file.write(f"{line}\n")


if __name__ == "__main__":
    main()



