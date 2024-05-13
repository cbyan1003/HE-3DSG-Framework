#ifndef NODE_H_
#define NODE_H_
#include <iostream>
#include "/home/ycb/InstanceFusion/external/include/eigen3/Eigen/Dense"  //<Eigen/Dense>
#include <opencv2/core.hpp>
#include <memory>
#include "/home/ycb/InstanceFusion/src/gui/Gui.h"
#include <unordered_set>
#include <Core/InstanceFusion.h>
#include "/home/ycb/InstanceFusion/src/graph/edge.h"
#include "/home/ycb/InstanceFusion/src/graph/MemoryBlock.h"
#include "/home/ycb/InstanceFusion/src/Core/InstanceFusion.h"

// InstanceFusion instancefusion(96,640,480,false,1);
// Gui gui(true,instancefusion.getInstanceTable(),640,480);

class Node {
    public:
        Node();
        Node(int label);
        // InstanceFusion *instancefusion;
        
        // void UpdataPrediction(const std::unique_ptr<InstanceFusion>& instancefusion);
        // void findAdd(const std::unique_ptr<Gui>& gui,const std::unique_ptr<InstanceFusion>& instancefusion);
        bool Add(int idx,const std::unique_ptr<Gui>& gui,int size,std::vector<int> color);
        void UpdatePrediction(const std::map<std::string, float> &pd, const std::map<std::string,std::pair<size_t, size_t>> &sizeAndEdge);
        // void Update(const std::unique_ptr<Gui>& gui,const std::unique_ptr<InstanceFusion>& instancefusion);
    public:
        int idx;
        int label;
        int class_idx;
        int size;
        Eigen::Vector3f centroid,bbox_max,bbox_min,stdev;
        int color[3];
        float bbx_volume;
        std::unordered_set<EdgePtr> edges;
        std::map<std::string, std::shared_ptr<MemoryBlock>> mFeatures;

        inline static std::string Unknown() { return "unknown"; };
        std::string last_class_predicted = Unknown();
        const std::string GetLabel() const;
    private:
        std::map<std::string, float> mClsProb;
        std::map<std::string, float> mClsWeight;
        std::map<std::string, std::pair<size_t, size_t>> mSizeAndEdge;
};
    typedef std::shared_ptr<Node> NodePtr;
    static inline NodePtr MakeNodePtr(int label) { return std::make_shared<Node>(label); }
#endif /* NODE_H_ */