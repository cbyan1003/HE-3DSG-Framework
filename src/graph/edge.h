#ifndef EDGE_H_
#define EDGE_H_
#include <map>
#include <memory>
#include "/home/ycb/InstanceFusion/src/graph/MemoryBlock.h"
class Edge{
    public:
        inline static std::string None() {return "none";}
        Edge();
        Edge(Edge &edge);
        int nodeto,nodefrom;
        std::map<std::string, float> labelProp;
        std::map<std::string, float> mClsWeight;
        std::map<std::string, std::shared_ptr<MemoryBlock>> mFeatures;
        void UpdatePrediction(const std::map<std::string, float> &prop);
        std::string label = None();
};
    typedef std::shared_ptr<Edge> EdgePtr;
    static inline EdgePtr MakeEdgePtr(){return std::make_shared<Edge>();}

#endif /* EDGE_H_ */