#include "node.h"
Node::Node(){}
Node::Node(int label){
    idx = label;
    class_idx = 0;
    // feature =0;
    centroid.setZero();
    bbx_volume = 0;
    bbox_max.setZero();
    bbox_min.setZero();
    stdev.setZero();
    // this->mFeatures = node.mFeatures;

}
// const int last_node_num = 0;

void Node::UpdatePrediction(const std::map<std::string, float> &pd, const std::map<std::string,std::pair<size_t, size_t>> &sizeAndEdge) {
    for(const auto &pair:pd) {
        auto name = pair.first;
        auto value = pair.second;//value from GNN

        if(mClsProb.find(name) == mClsProb.end()) {//找不到则添加
            // std::cout <<"找不到则添加"<<std::endl;
            mClsProb[name] = value; // stores the class and its probability
            mSizeAndEdge[name] = sizeAndEdge.at(name);
            mClsWeight[name] = 1;
            std::cout <<"点"<< this->idx<<" 类 "<<name << " 概率 " <<mClsProb.at(name)<< " 权重 " <<mClsWeight.at(name) <<std::endl;
        }
        else {
            // std::cout <<"找到"<<std::endl;
            static const float max_w = 100;
            const static float new_w = 1;
            const auto &old_value = mClsProb.at(name);
            const auto &old_w = mClsWeight.at(name);
            mClsProb.at(name) = (old_value * old_w + value * new_w) / (old_w + new_w);
            mClsWeight.at(name) = std::min(max_w, mClsWeight.at(name) + new_w);
            std::cout <<"点"<< this->idx<<" 类 "<<name << " 概率 " <<mClsProb.at(name)<< " 权重 " <<mClsWeight.at(name) <<std::endl;
        }
    }

    // find the newest with maximum prob
    if(mClsProb.empty()) return;

    float max_v=mClsProb.begin()->second;
    std::string max_label = GetLabel();
    for(const auto& pair:mClsProb){
        if(pair.second>=max_v){
            max_v = pair.second;
            max_label = pair.first;
        }
    }
    last_class_predicted = max_label;
    std::cout <<"点"<< this->idx<<" 类 "<<last_class_predicted <<std::endl;
    // std::cout <<"点"<< this->idx<<" 类 "<<last_class_predicted << " 概率 " <<mClsProb.at(last_class_predicted)<< " 权重 " <<mClsWeight.at(last_class_predicted) <<std::endl;

}

const std::string Node::GetLabel() const {
    return last_class_predicted;
}


bool Node::Add(int idx,const std::unique_ptr<Gui>& gui,int size,std::vector<int> color){
    if(gui->bbx_centroid.size() != gui->bbx_minmax.size())
    {
        std::cout<<"ERROR bbx_centroid != bbx_minmax"<<std::endl;
        return false;
    }
    else
    {
        // std::cout<<"else:"<<idx<<std::endl;
        this->idx = idx;
        // std::cout<<"this->idx:"<<this->idx<<std::endl;
        this->class_idx = gui->bbx_instanceTable_idx.at(idx);
        // std::cout<<"class_idx:"<<this->class_idx<<std::endl;
        this->size = size;
        for(int i = 0;i < 3;i ++)
        {
            this->centroid[i] = gui->bbx_centroid.at(idx).at(i);
            this->bbox_min[i] = gui->bbx_minmax.at(idx).first.at(i);
            this->bbox_max[i] = gui->bbx_minmax.at(idx).second.at(i);
            this->color[i] = color[i];
        }
        bbx_volume = (bbox_max[0]-bbox_min[0])*(bbox_max[1]-bbox_min[1])*(bbox_max[2]-bbox_min[2]);
        return true;
    }
}

// void Node::findAdd(const std::unique_ptr<Gui>& gui,const std::unique_ptr<InstanceFusion>& instancefusion){
//     int node_num = gui->bbx_instanceTable_idx.size();
//     for(int i=0;i<(node_num - last_node_num);i++)
//     {
//         if(Add(last_node_num,gui,instancefusion))
//         {
//             last_node_num++;
//         }
//     }
// }

// std::vector<std::pair<std::set<float>,std::set<float>>> bbx_minmax;存储的是当前帧每个bbx的min_x,min_y,min_z,max_x,max_y,max_z
// std::vector<int> bbx_instanceTable_idx;存储的是当前帧的instance信息
// std::vector<std::set<float>> bbx_centroid;存储的是当前帧bbx的质心set
// std::map<std::string,std::pair<size_t, size_t>> sizeAndEdge;存储的是
// std::map<std::string,float> pd;存储的是每个点的对应概率
// void Node::Update(const std::unique_ptr<Gui>& gui,const std::unique_ptr<InstanceFusion>& instancefusion){
//     if(gui->bbx_centroid.size() != gui->bbx_minmax.size())
//     {
//         std::cout<<"ERROR bbx_centroid != bbx_minmax"<<std::endl;
//     }
//     else
//     {
//         int node_num = gui->bbx_centroid.size();
//         if(node_num > last_node_num)
//         {
//             std::cout<<"new node add it"<<std::endl;
//             findAdd(gui,instancefusion);
//         }
//         else
//         {
//             std::cout<<"old node update it"<<std::endl;
//             // UpdataPrediction();
//         }

//     }
// }
