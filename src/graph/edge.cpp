#include "/home/ycb/InstanceFusion/src/graph/edge.h"

Edge::Edge(){}
Edge::Edge(Edge &edge) {
    this->nodefrom = edge.nodefrom;
    this->nodeto = edge.nodeto;
    this->labelProp = edge.labelProp;
    this->mClsWeight = edge.mClsWeight;
    this->mFeatures = edge.mFeatures;
}

void Edge::UpdatePrediction(const std::map<std::string, float> &prop){
    for (const auto &pair:prop) {
        const auto &name = pair.first;
        auto new_v = pair.second;
        // std::cout <<"边类 "<<name << " 新值 "<<new_v <<std::endl;
        if (labelProp.find(name) == labelProp.end()) {//找不到则新添加边
            // std::cout <<"新添加边 "<< name << " 新值 "<<new_v <<std::endl;
            labelProp.insert({name, new_v});
            mClsWeight.insert({name, 1});
            // std::cout <<"边"<< this->nodefrom<< " " <<this->nodeto<<" 初更新，更新为 "<<name << " 概率 " <<labelProp.at(name)<< " 权重 " <<mClsWeight.at(name) <<std::endl;
        } 
        else{
            const static float max_w = 100;
            const static float new_w = 1;
            const auto old_v = labelProp.at(name);
            const auto old_w = mClsWeight.at(name);
            labelProp.at(name) = (old_v * old_w + new_v * new_w) / float(old_w + new_w);
            mClsWeight.at(name) = (old_w + new_w);
            mClsWeight.at(name) = std::min(max_w, mClsWeight.at(name));
            // std::cout <<"边"<< this->nodefrom<< " " <<this->nodeto<<" 已存在，更新为 "<<name << " 概率 " <<labelProp.at(name)<< " 权重 " <<mClsWeight.at(name) <<std::endl;
        
        }
    }

    float max_prop = labelProp.begin()->second;
    std::string ll;
    for(const auto &pair:labelProp){
        if(pair.second>=max_prop){
            max_prop = pair.second;
            ll = pair.first;
        }
    }
    label = ll;
    std::cout <<"边"<< this->nodefrom<< " -> " <<this->nodeto<<"  "<<label <<std::endl;

}