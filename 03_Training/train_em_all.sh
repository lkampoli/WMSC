for d in training_data_cluster_3_partition_0_5e3 training_data_cluster_3_partition_1_7e1 training_data_cluster_3_partition_2_4e4 training_data_cluster_6_partition_0_7e2 training_data_cluster_6_partition_1_7e1 training_data_cluster_6_partition_2_2e3 training_data_cluster_6_partition_3_1e3 training_data_cluster_6_partition_4_9e3 training_data_cluster_6_partition_5_1e4
do
    cd $d
    echo $PWD
    cp ../eve3_study_A.slurm .
    cp ../eve3_study_R.slurm .
    cp ../my_eve3_model_A.py .
    cp ../my_eve3_model_R.py .
    cd ..
done
echo "All slurms copied!"
do
    cd $d
    echo $PWD
    sbatch eve3_study_A.slurm 
    sbatch eve3_study_R.slurm 
    cd ..
done
echo "All GEP training lunched!"
