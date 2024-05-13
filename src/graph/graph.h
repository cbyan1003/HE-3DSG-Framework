#ifndef GRAPH_H_
#define GRAPH_H_
#include <gui/Gui.h>
#include <map>
#include <graph/MemoryBlock.h>
#include <graph/edge.h>
#include <graph/node.h>
#include <Core/InstanceFusion.h>
#include "EatGCN.h"
#include "/home/ycb/SceneGraphFusion/external/include/x86_64-linux-gnu/json11.hpp"
#include "config.h"
#include <utilities/Types.h>
// CC=`which gcc-9` CXX=`which g++-9` cmake ..
typedef std::tuple<MemoryBlock3D,MemoryBlock2D,MemoryBlock2D, std::map<size_t,size_t>,std::map<int, std::pair<size_t,size_t>> > GCNINPUTDATA;

class Graph{

    public:
        typedef std::tuple< std::map<size_t,std::map<std::string,float>>, std::vector<EdgePtr>,std::map<int, std::pair<size_t,size_t>> > Prediction;
        Graph();
        std::map<int, NodePtr> nodes;
        std::map<std::pair<int,int>, EdgePtr> edges;
        std::set<int> node_add_to_edge;
        std::map<int,int>real_node2all_node;
        std::map<int,int>all_node2real_node;
        std::map<int,int>no_use2all_node;
        std::map<int,int>all_node2no_use;
        // std::shared_ptr<Prediction> Predict(const std::shared_ptr<GCNINPUTDATA>& data);
        // const std::map<std::string, json11::Json>& GetParams();
        // std::map<std::string,std::vector<std::pair<int,float>>> GetTimes();
        
        std::map<size_t,std::string> mLabels, mRelationships;
        std::map<std::string,size_t> mLabelsName2Idx,mRelationshipsName2Idx;
        std::shared_ptr<Prediction> Predict(
                const std::shared_ptr<GCNINPUTDATA>& data);

        // std::vector<NodeList> NodeLists;
        // std::vector<ClassColour> ColorList;
        
        // std::map<std::vector,std::vector> summary;
    public:
        void Update(const std::unique_ptr<Gui>& gui,const std::unique_ptr<InstanceFusion>& instanceFusion);
        void AddEdge(const int node_from_idx,const int node_to_idx);
        void Debug();
        void UpdatePrediction();
        void UpdateNodeFeature(const std::unique_ptr<InstanceFusion>& instanceFusion);
        void UpdateEdgeFeature();
        void UpdateGCNFeature();
        int UpdateNodes(const std::unique_ptr<Gui>& gui,const std::unique_ptr<InstanceFusion>& instanceFusion);
    private:
        bool upflag = false;
        bool debug_flag_node = true;
        bool debug_flag = false;
        bool source_to_target;
        const ConfigPSLAM *mConfigPslam;
        std::unique_ptr<Ort::MemoryInfo> mMemoryInfo;
        std::unique_ptr<EatGCN> mEatGCN;
        // static MemoryBlock2D ConcatObjFeature(size_t n_node, size_t dim_obj_feature, Ort::Value &obj_feature,const MemoryBlock2D &descriptor);


};

#endif /* GRAPH_H_ */