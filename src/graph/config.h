//
// Created by sc on 10/9/20.
//

#ifndef CONFIG_H
#define CONFIG_H
#include <string>
#include <fstream>

//#include <inseg_lib/segmentation.h>

    struct ConfigPSLAM {
        ConfigPSLAM();
        explicit ConfigPSLAM(const std::string &path);
        ~ConfigPSLAM();
        bool use_thread;

        bool use_fusion;

        /// Use graph predict or not
        bool graph_predict;

        /// model path
        std::string pth_model;

        int filter_num_node;

//        int predict_when_have_at_least_n_node;
        size_t n_pts;

        float neighbor_margin; // mm

        /// Only update the information of a node if the node has size change to N percent
        float update_thres_node_size;
        /// Update a node if its time stamp is older than n
        int update_thres_time_stamp;

        std::string pth;



    };




#endif //CONFIG_H
