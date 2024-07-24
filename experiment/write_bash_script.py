import os
import hydra


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    output_dir = cfg["output_file_path"]["local"]
    bash_list = []
    instance_query_strategies = ['random', 'uncertainty', 'coreset']
    annotator_query_strategies = ['random', 'round-robin', 'trace-reg', 'geo-ref-f', 'geo-ref-w']
    learning_strategies = ['majority-vote', 'trace-reg', 'geo-ref-f', 'geo-ref-w']
    n_annototar_per_instance_list = [1, 2]
    dataset = 'letter'
    batch_size = 256
    n_cycles = 25
    seed_list = [0, 1, 2, 3, 4]

    file_path = "srun python /mnt/stud/home/jcheng/scikit-activeml/experiment/experiment.py"

    for instance_query_strategy in instance_query_strategies:
        for annotator_query_strategy in annotator_query_strategies:
            for learning_strategy in learning_strategies:
                for n_annototar_per_instance in n_annototar_per_instance_list:
                    for seed in seed_list:
                        if (annotator_query_strategy in ['trace-reg', 'geo-ref-f', 'geo-ref-w'] and
                                learning_strategy != annotator_query_strategy):
                            continue
                        _bash = (f"{file_path} "
                                 f"+dataset={dataset} "
                                 f"+instance_query_strategy={instance_query_strategy} "
                                 f"+annotator_query_strategy={annotator_query_strategy} "
                                 f"+learning_strategy={learning_strategy} "
                                 f"+batch_size={batch_size} "
                                 f"+n_annotator_per_instance={n_annototar_per_instance} "
                                 f"+n_cycles={n_cycles} "
                                 f"+seed={seed}")
                        bash_list.append(_bash)

    with open(f'{output_dir}bash.txt', 'w', newline='') as bash_file:
        for line in bash_list:
            bash_file.write(f"{line}\n")


if __name__ == "__main__":
    main()