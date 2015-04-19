import sys 
from pyspark import SparkContext,SparkConf
import os
import random
from functools import partial
import numpy as np
import new
import math
import time

#this is the main method and takes as an input num_factors num_workers num_itrns beta_value lambda_value input_file w_file h_file
def main():
    num_factors  = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_itrns = int(sys.argv[3])
    beta_value = float(sys.argv[4])
    lambda_value = float(sys.argv[5])
    inputV_file = sys.argv[6]
    outputW_file = sys.argv[7]
    outputH_file = sys.argv[8]

    #initialize spark context
    #using conf 
    #conf = new SparkConf().setAppName("My application").setMaster("local")
    #sc = new SparkContext(conf=conf)
    #using only SparkContext
    sc = SparkContext("local", "dsgd_mf")
    #use accumulator to update values of teh loss over every iteration
    L_NZSL = sc.accumulator(0)
    print "L_NZSL value is ",str(L_NZSL.value)

    #read directory or file   
    if os.path.isdir(inputV_file):
        #read data and get it from the input files
        RDD = get_data_in_RDD_folder(sc,inputV_file)
        #construct the V matrix, the user and movie hashmaps
        RDD_matrix_V,user_Hashmap, movie_Hashmap, N_j, N_i = get_matrix_V_from_netflix(RDD)
    
    elif os.path.isfile(inputV_file):
        #read data and get it from the input files
        RDD = get_data_in_RDD_file(sc,inputV_file)
        #construct the V matrix, the user and movie hashmaps
        RDD_matrix_V,user_Hashmap, movie_Hashmap, N_j, N_i = get_matrix_V_from_autolab(RDD)
        
    start_time = time.time()
    print "Time is ",str(start_time)
    
    #persist the ratings matrix as we will be using this throughout
    RDD_matrix_V = RDD_matrix_V.persist()
    

    #construct initial W and H matrices with random value between 0 and 1
    RDD_matrix_W , RDD_matrix_H = construct_initial_factors(sc, user_Hashmap,movie_Hashmap, num_factors)

    #now partition the data ( V and W)
    partition_W = RDD_matrix_W.partitionBy(num_workers).persist()
    partition_V = RDD_matrix_V.partitionBy(num_workers).persist()

    l_values = [] #stores the value of loss over every iteration
    block_size = len(movie_Hashmap.keys())/num_workers #size of the block is number of movies/ num of workers
    
    total_n = 0
    #here my iteration below are over each diagonal , and not over the whole matrix at once, hence my iterations
    #is the product of the iteraion over the whole data once times the number of workers
    
    new_itrns = num_itrns*num_workers 
    #Following are te initial values to keep track of convergence of W and H for autolab data
    prev_W_intersect = 1000000000
    prev_H_intersect = 1000000000
    #Now for each iteration
    for i in range(0,new_itrns):
        #get strata for this iteration by joining with W        
        partitions = partition_V.join(partition_W, numPartitions=num_workers)
        #now map partition with index
        strata_for_this = partitions.mapPartitionsWithIndex(get_partition, preservesPartitioning=True)
        #Filter the strata now
        new_block_strata = strata_for_this.filter(partial(get_block,i,block_size,num_workers))
        #now get movie id and data
        #get the map for H (movies)
        new_map = RDD_matrix_H.collectAsMap()
        
        #here, call the update_WH function now which performs the gradient update 
        updated_W_and_H = new_block_strata.mapPartitions(partial(update_WH,lambda_value,N_i, N_j, num_workers,new_map,beta_value,total_n,L_NZSL),preservesPartitioning=True)
        #got updated maps from different blocks in parallel
        total_n = total_n + new_block_strata.count()
        #get RDD of new updated W and H
        W_list = updated_W_and_H.flatMap(lambda x: x[0]).collect()
        H_list = updated_W_and_H.flatMap(lambda x: x[1]).collect()
        RDD_matrix_W_new = sc.parallelize(W_list).sortByKey()
        RDD_matrix_H_new = sc.parallelize(H_list).sortByKey()
        #Compute the square of difference on W and H , for convergence
        RDD_intersect_W = sum(RDD_matrix_W_new.join(RDD_matrix_W).map(lambda x : (x[1][0]-x[1][1])**2).sum())
        RDD_intersect_H = sum(RDD_matrix_H_new.join(RDD_matrix_H).map(lambda x : (x[1][0]-x[1][1])**2).sum())
       
        #Update and construct the new W and H for next iteration , some entries are replicated when a user appears multiple times in a strats
        RDD_matrix_W = RDD_matrix_W_new.union(RDD_matrix_W.subtractByKey(RDD_matrix_W_new))
        RDD_matrix_H = RDD_matrix_H_new.union(RDD_matrix_H.subtractByKey(RDD_matrix_H_new))
        RDD_matrix_W = RDD_matrix_W.sortByKey()
        RDD_matrix_H = RDD_matrix_H.sortByKey()
        partition_W = RDD_matrix_W.partitionBy(num_workers)
        
        #if the whole matrix has been covered (the input value of iteration)
        if (i+1)%num_workers == 0:
            #update and check for convergence
            if prev_W_intersect - RDD_intersect_W <= 0.00001 and prev_H_intersect - RDD_intersect_H <= 0.00001:
                print "W diff "+str(RDD_intersect_W - prev_W_intersect)
                print "H diff "+str(RDD_intersect_H - prev_H_intersect)
                print "converged"
                break
            prev_W_intersect = RDD_intersect_W
            prev_H_intersect = RDD_intersect_H
            print "W diff new "+str(prev_W_intersect)
            print "H diff new "+str(prev_H_intersect)
            #record the loss for this iteration and reset accumulator to zero
            l_values.append(L_NZSL)
            L_NZSL = sc.accumulator(0)
    #Now write out the new W and H to the files mentioned    
    for l in l_values:
        print "iteration: "+str(i)+" L_NZSL value is "+str(l.value)
        i= i+1
    print 
    print
    
    #Now write out the final W and H matrices to the mentioned files
    write_matrix(RDD_matrix_H,RDD_matrix_W,outputW_file,outputH_file,user_Hashmap,movie_Hashmap,num_factors) 
    
    print "time taken  was ",str(start_time -    time.time())
  
#perform DSGD and update W and H and return the updated maps   
def update_WH(lamda, N_i, N_j,num_workers,H_map,beta_value,total_n,L_NZSL,new_updated_H):
    #extract elements required for computation of SGD updates for W and H
    #update the new dictionaries for W and H
    W_i_dict = {}
    H_j_dict = {}
    cnt  =0
    for i in new_updated_H:
        #get user and movie ids
        user_id = int(i[1])
        movie_id = int(i[2][0][0])
        if user_id not in W_i_dict:
            W_i_dict[user_id] = np.array(i[2][1])
        if movie_id not in H_j_dict:
            H_j_dict[movie_id] = np.array(H_map[movie_id])
        
        #get the rating, and other useful entries for computation of the update
        V_ij = int(i[2][0][1])
        W_i = W_i_dict[user_id]
        H_j = H_j_dict[movie_id]
        dot_p = np.dot(W_i,H_j)
        bracket = V_ij  - dot_p
        #compute loss for non-zero entries and update
        if V_ij !=0:
            update = bracket*bracket
            L_NZSL.add(update)
        element_1 = -2*(bracket)
        
        #get Ni and Nj
        n_i = N_i[user_id]
        n_j = N_j[movie_id]
        #update for W and H
        grad_update_W = np.add(element_1*H_j , (2*lamda/n_i)*W_i)
        #print "N_j is ",str(n_j)
        grad_update_H = np.add(element_1*W_i, (2*lamda/n_j)*H_j)
        #compute epsilon
        epsilon = math.pow((100 + total_n+cnt), -beta_value)
        
        W_i_dict[user_id] = W_i - (epsilon*grad_update_W)
        H_j_dict[movie_id] = H_j - (epsilon*grad_update_H)
        cnt  = cnt +1
    #return back the two updated dictionaries
    yield (W_i_dict.items(),H_j_dict.items())   
    
    
    
     
#returns data in the format : partition number, key, value   
def get_partition(splitIndex, iterator):
    for i in iterator:
        yield (splitIndex, i[0],i[1])

#returns whether or not the block will be a part of the stratum (used in filter function)       
def get_block(itr, block_size, num_workers, entry):
    partition_no = entry[0]
    column_block = (partition_no + itr) % num_workers
    start = column_block*block_size
    end = (column_block+1) *block_size
    #check explicitly for the last worker and 
    if column_block == num_workers -1:
        end = (num_workers+1)*block_size
    
    #return given the following conditions
    return start <= entry[2][0][0] and entry[2][0][0] < end 
    
#get the data from netflix    
def get_matrix_V_from_netflix(RDD):
    #get movie id from the file name and get each line associated with it
    new_formatted_RDD = RDD.map(lambda (movie_id,content): (int(os.path.basename(movie_id)[3:-4]),content.strip().split("\n")[1:]))
    #now split each line by "," to get userid and rating
    updated_RDD = new_formatted_RDD.map(lambda (movie_id,content) : (movie_id, [line.split(",") for line in content]))
    #create new tuples with movie_id : userid, rating
    RDD_matrix_V = updated_RDD.flatMap(lambda (movie_id,user_info) : [(movie_id, int(line[0]), int(line[1])) for line in user_info])
    #since the data is sparse, we map the userids and movie ids to work in sparse domain
    user_Hashmap ={}
    movie_Hashmap = {}
    
    #construct the maps
    movie_Hashmap = RDD_matrix_V.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
    user_Hashmap = RDD_matrix_V.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
    #get N_j
    RDD_matrix_2 = RDD_matrix_V.map(lambda x: (x[0],[x[1],x[2]]))
    N_j = RDD_matrix_2.countByKey()
    
   
    #Store data as userxmoviexrating
    RDD_matrix_V = RDD_matrix_V.map(lambda x: (user_Hashmap[x[1]],[x[0],x[2]])).persist()
    
    #get N_i values
    N_i = RDD_matrix_V.countByKey()
    
    return RDD_matrix_V, user_Hashmap, movie_Hashmap, N_j, N_i

 
#Read the data format from autolab and for experiments, as the data forat is different
def get_matrix_V_from_autolab(RDD):
    
    RDD_matrix_V = RDD.map(lambda x: (x.strip().split(",")[0:]))
    RDD_matrix_V = RDD_matrix_V.map(lambda (x): (int(x[0]),int(x[1]), int(x[2])))
    user_Hashmap ={}
    movie_Hashmap = {}
    
    #construct the maps
    movie_Hashmap = RDD_matrix_V.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
    user_Hashmap = RDD_matrix_V.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
    #get N_j
    RDD_matrix_2 = RDD_matrix_V.map(lambda x: (x[1],[x[0],x[2]]))
    N_j = RDD_matrix_2.countByKey()
    #get N_i values
    
    RDD_matrix_V = RDD_matrix_V.map(lambda x: (user_Hashmap[x[0]],[x[1],x[2]]))
    N_i = RDD_matrix_V.countByKey()
    return RDD_matrix_V, user_Hashmap, movie_Hashmap, N_j, N_i

#get data from file    
def get_data_in_RDD_file(sc,file):
    return sc.textFile(file) 
       
    
#get the files from the folder
def get_data_in_RDD_folder(sc,folder):
    return sc.wholeTextFiles(folder)

#construct the initial W and H
def construct_initial_factors(sc, user_Hashmap,movie_Hashmap, num_factors):
    matrix_W_map = {}
    for key in user_Hashmap:
        this_row = []
        for col in range(0,num_factors):
            rnd = random.uniform(0,1)
            this_row.append(rnd)
        matrix_W_map[user_Hashmap[key]] = this_row

    RDD_matrix_W = sc.parallelize(matrix_W_map.items()).sortByKey()
    
    matrix_H_map = {}
    for key in movie_Hashmap:
        this_row = []
        for col in range(0,num_factors):
            rnd = random.uniform(0,1)
            this_row.append(rnd)
        matrix_H_map[key] = this_row

    RDD_matrix_H = sc.parallelize(matrix_H_map.items()).sortByKey()
    return RDD_matrix_W, RDD_matrix_H

#write the matrices W and H to the files mentioned    
def write_matrix(RDD_matrix_H,RDD_matrix_W,file_W,file_H,user_map,movie_map,num_factors):
    #get the mapping from sparse representation to the actual user ids
    inv_user_map = {v: k for k, v in user_map.items()}
    #sort W and H by keys
    sorted_W = RDD_matrix_W.sortByKey()
    sorted_H = RDD_matrix_H.sortByKey()
    
    fw = open(file_W,"w")
    prev_user = 0
    #the following loop checks for gaps between userids, if there is a gap, it writes out an array of 0s for the missing users
    #if there is no gap, it writes out the final updated W for that user
    for x in sorted_W.toLocalIterator():
        curr_user = int(inv_user_map[x[0]])
        if curr_user == prev_user +1 :
            prev_user = curr_user
            y=1
            for i in x[1]:
                if y==len(x[1]):
                    fw.write(str(i))
                else:
                    fw.write(str(i)+",")
                y = y+1
            fw.write("\n")
        elif curr_user > (prev_user +1):
            diff = curr_user - (prev_user+1)
            for i in range(0,diff):
                n = np.array([0 for i in range(num_factors)])
                y=1
                for k in n:
                    if y==len(x[1]):
                        fw.write(str(k))
                    else:
                        fw.write(str(k)+",")
                    y = y+1
                fw.write("\n")
            y=1
            for i in x[1]:
                if y==len(x[1]):
                    fw.write(str(i))
                else:
                    fw.write(str(i)+",")
                y = y+1
            fw.write("\n")
            prev_user = curr_user
    fw.close()
    
    sorted_H_list = []
    prev_movie = 0
    #the following loop checks for gaps in movie ids, if there are gaps, we output a vector of 0s
    #otherwise constrcut the matrix with actual values of H for that movie
    for x in sorted_H.toLocalIterator():
        curr_movie = int(x[0])
        if curr_movie == prev_movie +1:
            sorted_H_list.append(x[1])
            prev_movie = curr_movie
        else:
            diff = curr_movie - (prev_movie+1)
            for i in range(0,diff):
                n = np.array([0 for i in range(0,num_factors)])
                sorted_H_list.append(n)
            sorted_H_list.append(x[1])
            prev_movie = curr_movie
            
    H_matrix  = np.array(sorted_H_list)
    print len(H_matrix),len(H_matrix[0])
    #transpose the matrix to be a factor x movie matrix
    new_H = H_matrix.transpose()
    print len(new_H),len(new_H[0])
    #write out the matrix to a file
    np.savetxt(file_H, new_H, delimiter = ',', newline='\n')
       

main()