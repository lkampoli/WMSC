for d in training_data_cluster_3_partition_0_5e3 training_data_cluster_3_partition_1_7e1 training_data_cluster_3_partition_2_4e4 training_data_cluster_6_partition_0_7e2 training_data_cluster_6_partition_1_7e1 training_data_cluster_6_partition_2_2e3 training_data_cluster_6_partition_3_1e3 training_data_cluster_6_partition_4_9e3 training_data_cluster_6_partition_5_1e4
do
    cd $d
    echo $PWD
    tail -n 10 log.my_eve3_model_A.py > model_A
    tail -n 10 log.my_eve3_model_R.py > model_R
    cd ..
done
