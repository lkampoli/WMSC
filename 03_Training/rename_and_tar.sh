for d in training_data_cluster_3_partition_0_5e3 training_data_cluster_3_partition_1_7e1 training_data_cluster_3_partition_2_4e4 training_data_cluster_6_partition_0_7e2 training_data_cluster_6_partition_1_7e1 training_data_cluster_6_partition_2_2e3 training_data_cluster_6_partition_3_1e3 training_data_cluster_6_partition_4_9e3 training_data_cluster_6_partition_5_1e4
do
    cd $d
    echo $PWD
    mv V1.edf V01.edf
    mv V2.edf V02.edf
    mv V3.edf V03.edf
    mv V4.edf V04.edf
    mv V5.edf V05.edf
    mv V6.edf V06.edf
    mv V7.edf V07.edf
    mv V8.edf V08.edf
    mv V9.edf V09.edf
    mv aij.edf A.edf
    mv RtermNorm.edf R.edf
    gzip *.edf
    cd ..
done
