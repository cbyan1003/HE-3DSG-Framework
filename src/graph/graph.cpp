#include "graph.h"
#include "MemoryBlock.h"

static MemoryBlock2D ConcatObjFeature(size_t n_node, size_t dim_obj_feature, Ort::Value &obj_feature,
                                      const MemoryBlock2D &descriptor) {
    /// Concat descriptor to the object feature
    MemoryBlock2D out(MemoryBlock::DATA_TYPE::FLOAT, {static_cast<unsigned long>(n_node), dim_obj_feature + 8});
    MemoryBlock2D tmp_out_enc_obj(obj_feature.GetTensorMutableData<float>(), MemoryBlock::FLOAT,
                                  {static_cast<unsigned long>(n_node), dim_obj_feature});
    for (size_t n = 0; n < n_node; ++n) {
        MemoryBlock tmp (descriptor.Get<float>(n,3),descriptor.type(),8);
//                SCLOG(VERBOSE) << tmp;
        size_t offset= 0;
        memcpy(out.Get<float>(n, offset), tmp_out_enc_obj.Get<float>(n,0), sizeof(float)*dim_obj_feature);
        offset += dim_obj_feature;
        memcpy(out.Get<float>(n, offset), tmp.Get<float>(), sizeof(float)*6);
        offset += 6;
        out.at<float>(n, offset) = std::log( tmp.at<float>(6) );
        offset+=1;
        out.at<float>(n, offset) = std::log( tmp.at<float>(7) );
    }
    return out;
}

Graph::Graph(){

    source_to_target = false;
    std::string path_onnx = "/home/ycb/InstanceFusion/traced/";
    mMemoryInfo = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    mEatGCN = std::make_unique<EatGCN>(path_onnx, mMemoryInfo.get());

    /// Read labels
    {
        std::fstream file(path_onnx+"classes.txt", std::ios::in);
        if(!file.is_open())std::cout << "Cannot open class file at path " << path_onnx+"classes.txt";
        std::string line;
        std::vector<std::string> labels;
        while (std::getline(file, line)) labels.push_back(line);

        std::cout << "Labels";
        for(size_t i=0;i<labels.size();++i) {
            mLabels.insert({i, labels[i]});
            mLabelsName2Idx.insert({labels[i],i});
            std::cout << i << ": " << labels[i];
        }
    }
    {
        std::fstream file(path_onnx+"relationships.txt", std::ios::in);
        if(!file.is_open())std::cout << "Cannot open relationship file at path " << path_onnx+"relationships.txt";
        std::string line;
        std::vector<std::string> labels;
        while (std::getline(file, line)) labels.push_back(line);
        std::cout << "Relationships";
        for(size_t i=0;i<labels.size();++i) {
            mRelationships.insert({i,labels[i]});
            mRelationshipsName2Idx.insert({labels[i],i});
            std::cout << i << ": " << labels[i];
        }
    }

}
int last_no_use_node_num = -1;
int last_node_num = -1;
int all_node_cout = -1;
void Graph::Update(const std::unique_ptr<Gui>& gui,const std::unique_ptr<InstanceFusion>& instanceFusion){

    int node_num = gui->bbx_instanceTable_idx.size(); //all_node size
    int selected_node_num = nodes.size(); // real_node size


    std::cout<<"=================================================="<<std::endl;
    std::cout<<"all node_num:"<<node_num<<std::endl;
    std::cout<<"selected_node_num = nodes.size():"<<selected_node_num<<std::endl;
    for(auto n:nodes)
    {
        std::cout<<"real node idx:"<<n.first<<std::endl;
    }
    for(auto n : real_node2all_node)
    {
        std::cout<<"real node idx:"<<n.first<<" all node idx: "<<n.second<<std::endl;
    }
    // std::cout<<"node_num: "<<node_num<<std::endl;
    // std::cout<<"selected_node_num: "<<nodes.size()<<std::endl;
    if(nodes.size() != 0)UpdateNodes(gui,instanceFusion);

    if((node_num > selected_node_num) && upflag == false)//|| upflag == true)
    {
        //add node
        // if(upflag)std::cout<<"666666666666666"<<std::endl;
        std::cout<<"all_node_cout:"<<all_node_cout<<std::endl;
        int cout = node_num - (all_node_cout + 1);
        std::cout<<"found new node so add it"<<std::endl;
        for(int i=0;i<cout;i++)
        {   
            NodePtr node;
            int index = last_node_num + 1;
            int size = 0;
            std::cout<<"size "<<instanceFusion->getFeatureMap(all_node_cout+1).size()<<" all node index:"<<all_node_cout+1<<std::endl;
            if(instanceFusion->getFeatureMap(all_node_cout+1).size()>(512*3))
            {
                nodes.insert({index,std::make_shared<Node>(index)});
                node_add_to_edge.insert(index);
                node = nodes.at(index);
                size = instanceFusion->getFeatureMap(all_node_cout+1).size();
                std::vector<int>color = instanceFusion->getInstanceColor(all_node_cout+1);
                if(node->Add(index,gui,size,color));
                {
                    int classid = gui->bbx_instanceTable_idx.at(index);
                    std::string class_name = (classid == -1) ? "Unused" : class_names[classid];
                    std::cout<<"To Add Node index: "<<index<<" class name: "<< class_name <<std::endl;
                }
                ++all_node_cout;
                ++last_node_num;
                real_node2all_node.insert({index,all_node_cout});
                all_node2real_node.insert({all_node_cout,index});
            }
            else{
                ++all_node_cout;
                ++last_no_use_node_num;
                no_use2all_node.insert({all_node_cout,all_node_cout});
                std::cout<<"add: node no use:"<<all_node_cout<<", all node idx: "<<all_node_cout<<std::endl; 
                all_node2no_use.insert({all_node_cout,all_node_cout});
            }
            
        }
        selected_node_num = nodes.size();

        //add edge
        if(selected_node_num > 1 && node_add_to_edge.size() > 0)
        {
            std::cout<<"add edge"<<std::endl;
            for(auto n:nodes)
            {   
                // std::cout<<"n.first: "<<n.first<<std::endl;
                for(auto nn:node_add_to_edge)
                {
                    // std::cout<<"nn: "<<nn<<std::endl;
                    if(n.first == nn)
                        continue;
                    else
                    {   
                        // std::cout<<"n.first: "<<n.first<<" nn: "<<nn<<std::endl;
                        AddEdge(n.first,nn);
                        AddEdge(nn,n.first);
                        upflag = true;
                    }
                }
            }
           
        }
        upflag = true;
        node_add_to_edge.clear();
        //update new prediction
        Debug();

    }
    else if(upflag)
    {   
        std::cout<<"old node update it"<<std::endl;
        
        std::cout<<"=================================================="<<std::endl;
        UpdateNodeFeature(instanceFusion);
        UpdateEdgeFeature();
        UpdateGCNFeature();
        UpdatePrediction();
        upflag = false;
        Debug();
    }
    else
    {
        upflag = true;
    }

   

}


std::shared_ptr<Graph::Prediction> Graph::Predict(
        const std::shared_ptr<GCNINPUTDATA>& data) {
    {
        std::tuple<MemoryBlock2D, MemoryBlock2D> result;
        auto& edge_index = std::get<2>(*data);
        auto& idx2seg = std::get<3>(*data);
        result = mEatGCN->Run(std::get<0>(*data), std::get<1>(*data),std::get<2>(*data));
        auto& cls_obj = std::get<0>(result);
        auto& cls_rel = std::get<1>(result);


        std::map<size_t,std::map<std::string,float>> output_objs;
        for(size_t i=0;i<cls_obj.mDims.x;++i){
            std::map<std::string,float> m;
            for(size_t j=0;j<cls_obj.mDims.y;++j){
                m[mLabels.at(j)] = cls_obj.at<float>(i,j);
            }
            output_objs[idx2seg.at(i)] = std::move(m);
        }
//        std::map<size_t,size_t> output_objs;
//        for(size_t i=0;i<cls_obj.size();++i){
//            output_objs[idx2seg.at(i)] = cls_obj.at(i)+1;
//        }

        std::vector<EdgePtr> output_edges;
        for(size_t i=0;i<cls_rel.mDims.x;++i){
            output_edges.emplace_back(new Edge());
            auto &edge = output_edges.back();
            edge->nodefrom = idx2seg.at( edge_index.at<int64_t>(0,i) );
            edge->nodeto = idx2seg.at( edge_index.at<int64_t>(1,i) );
            for(size_t j=0;j<cls_rel.mDims.y;++j) {
                edge->labelProp[mRelationships.at(j)] = cls_rel.at<float>(i,j);
            }
        }

        return std::make_shared<Prediction>(
                std::move(output_objs), std::move(output_edges), std::move(std::get<std::map<int, std::pair<size_t,size_t>>>(*data)));
    }
}

void Graph::UpdateGCNFeature(){
    std::cout <<"GCN feature start"<<std::endl;
    size_t n_nodes = nodes.size();
    std::map<size_t, size_t> idx2seg;
    std::map<size_t, size_t> seg2idx;
    if(n_nodes==0){
        std::cout << "got 0 nodes. return"<<std::endl;
        return;
    }
    size_t k=0;
    for (const auto &pair:nodes) {
        idx2seg[k] = pair.second->idx;
        seg2idx[pair.second->idx] = k;
        k++;
    }

    // count edges
    std::cout << "Update countEdges"<<std::endl;
    std::vector<EdgePtr> selected_edges;
    for(auto & n : nodes)
    {
        for(auto & nn : nodes)
        {
            if(n != nn)selected_edges.emplace_back(edges.at({n.first,nn.first}));
        }
    }

    size_t n_edges = selected_edges.size();
    if(n_edges==0){
        std::cout << "got 0 edges. return"<<std::endl;
        return;

        
    }
    
    // Compute Edge descriptor
    MemoryBlock2D edge_index(MemoryBlock2D::INT64_T, { 2, n_edges });
    for(size_t i=0;i<selected_edges.size();++i) {
        edge_index.at<int64_t>(0, i) = seg2idx.at(selected_edges.at(i)->nodefrom);
        edge_index.at<int64_t>(1, i) = seg2idx.at(selected_edges.at(i)->nodeto);
    }

    if(debug_flag){
        std::stringstream ss;
        ss << "edges\n";
        for(size_t i=0;i<n_edges;++i)
            ss << edge_index.at<int64>(0,i) << ", " << edge_index.at<int64_t>(1,i) << "\n";
        std::cout << ss.str()<<std::endl;
    }

    // build node input
    int64_t dim_obj_f = 512;
    MemoryBlock2D nodeFeatures(MemoryBlock::FLOAT, {nodes.size(), static_cast<unsigned long>(dim_obj_f)});
    size_t i=0;
    for(const auto &pair:nodes){
        auto &node = pair.second;
        const auto *feature = node->mFeatures.at("0")->Get<float>();
        // std::cout << "feature "<<feature<<std::endl;
        memcpy(nodeFeatures.Row<float>(i),feature,sizeof(float)*dim_obj_f);
        i++;
    }

    /// Build Edge Input
    int64_t dim_rel_f = 256;
    MemoryBlock2D edgeFeatures(MemoryBlock::FLOAT, {n_edges, static_cast<unsigned long>(dim_rel_f)});
    for(size_t i=0;i<n_edges;++i){
        auto &edge = selected_edges.at(i);
        const auto *feature = edge->mFeatures["0"]->Get<float>();
        memcpy(edgeFeatures.Row<float>(i),feature,sizeof(float)*dim_rel_f);

        if(debug_flag)
        {//TODO: remove me
            std::stringstream ss;
            ss.precision(4);
            ss << "["<<edge->nodefrom<<","<<edge->nodeto<<"]:";
            for(size_t j=0;j<edge->mFeatures["0"]->size();++j) ss << feature[j] << " ";
            std::cout << ss.str()<<std::endl;
        }
        if(debug_flag)
        {//TODO: remove me
            std::stringstream ss;
            ss.precision(4);
            ss << "["<<edge->nodefrom<<","<<edge->nodeto<<"]:";
            for(size_t j=0;j<edge->mFeatures["0"]->size();++j) ss << edgeFeatures.Row<float>(i)[j] << " ";
            std::cout << ss.str()<<std::endl;
        }
    }

    if(debug_flag)
    ONNX::PrintVector<float,unsigned long>("edgeFeatures", edgeFeatures.Get<float>(), {n_edges, static_cast<unsigned long>(dim_rel_f)});

    /// Compute
    size_t i_i = source_to_target ? 1 : 0;
    size_t i_j = source_to_target ? 0 : 1;

    std::vector<float> obj_f_i = DataUtil::Collect(nodeFeatures.Get<float>(),
                                                    edge_index.Row<int64_t>(i_i), dim_obj_f, n_edges);
    std::vector<float> obj_f_j = DataUtil::Collect(nodeFeatures.Get<float>(),
                                                    edge_index.Row<int64_t>(i_j), dim_obj_f, n_edges);
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), obj_f_i.data(),
            {static_cast<long>(n_edges), dim_obj_f}));
    inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), edgeFeatures.Get<float>(),
            {static_cast<long>(n_edges), dim_rel_f}));
    inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), obj_f_j.data(),
            {static_cast<long>(n_edges), dim_obj_f}));
    auto output_atten = mEatGCN->ComputeGCN(EatGCN::OP_GCN_ATTEN, 0, inputs);

    if (debug_flag) {
            ONNX::PrintVector("output_atten", output_atten[0].GetTensorMutableData<float>(),
                        std::vector<int64_t>{output_atten[0].GetTensorTypeAndShapeInfo().GetShape()[0],
                                             output_atten[0].GetTensorTypeAndShapeInfo().GetShape()[1]});
            ONNX::PrintVector("output_edge", output_atten[1].GetTensorMutableData<float>(),
                        std::vector<int64_t>{output_atten[1].GetTensorTypeAndShapeInfo().GetShape()[0],
                                             output_atten[1].GetTensorTypeAndShapeInfo().GetShape()[1]});
        }

    auto dim_hidden = output_atten[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    auto xx = DataUtil::IndexAggr(output_atten[0].GetTensorMutableData<float>(),
                                    edge_index.Row<int64_t>(i_i), dim_hidden, n_edges, n_nodes, 1);
//        if (bVerbose) PrintVector("xx_aggr", xx.data(), std::vector<int64_t>{n_node, dim_hidden});

    auto xxx = DataUtil::Concat(nodeFeatures.Get<float>(), xx.data(), n_nodes,
                                dim_obj_f, dim_hidden,
                                dim_obj_f, dim_hidden);

    if (debug_flag) ONNX::PrintVector("xxx_concat", xxx.data(), std::vector<int64_t>{static_cast<long>(n_nodes), dim_obj_f + dim_hidden});


    std::vector<float *> gcn_nn_inputs = {xxx.data()};
    std::vector<std::vector<int64_t>> gcn_nn_input_size = {{static_cast<long>(n_nodes), dim_obj_f + dim_hidden}};
    auto gcn_nn2_outputs = mEatGCN->ComputeGCN(EatGCN::OP_GCN_PROP, 0, gcn_nn_inputs, gcn_nn_input_size);
    if ( 1 < size_t(2)) {
        DataUtil::Relu(gcn_nn2_outputs[0].GetTensorMutableData<float>(), n_nodes * dim_obj_f);
        DataUtil::Relu(output_atten[1].GetTensorMutableData<float>(), n_edges * dim_rel_f);
    }

    /// Copy Node feature
    size_t n=0;
    for(const auto & pair : nodes) {
        auto& node_feature = pair.second->mFeatures[std::to_string(1)];
        if(!node_feature) node_feature.reset(new MemoryBlock(MemoryBlock::FLOAT, dim_obj_f));
        auto *feature = gcn_nn2_outputs[0].GetTensorMutableData<float>() + n * dim_obj_f;
        memcpy(node_feature->Get<float>(), feature, sizeof(float) * dim_obj_f);
        n++;
    }

    /// Copy Edge feature
    for(size_t n=0;n<n_edges;++n) {
        auto& edge_feature = selected_edges.at(n)->mFeatures[std::to_string(1)];
        if(!edge_feature) edge_feature.reset(new MemoryBlock(MemoryBlock::FLOAT, dim_rel_f));
        auto *feature = output_atten[1].GetTensorMutableData<float>() + n * dim_rel_f;
        memcpy(edge_feature->Get<float>(), feature, sizeof(float) * dim_rel_f);
    }

    std::cout<<"Update GCN feature done"<<std::endl;
}

void Graph::UpdateEdgeFeature(){
    std::cout<<"edge feature start"<<std::endl;
    size_t d_edge = 11;

    std::cout<< "Build descriptor for these nodes"<<std::endl;
    MemoryBlock2D descriptor_full(MemoryBlock::DATA_TYPE::FLOAT, {nodes.size(), d_edge});
    size_t i=0;
    for(const auto &pair:nodes)
    {
        std::cout<<"nodes: "<<pair.first<<std::endl;
        const auto &node = pair.second;
        const auto centroid = node->centroid*1e1;
        const auto bbox_max = node->bbox_max*1e1;
        const auto bbox_min = node->bbox_min*1e1;
        const auto stdev    = node->stdev*1e-3;//点位置的标准差

        Eigen::Vector3f dim = (bbox_max-bbox_min);//bbx三维数据
        float volume = 1;
        for (size_t d = 0; d < 3; ++d) volume *= dim[d];//bbx的体积
        float length = *std::max_element(dim.begin(), dim.end());//找到bbx中最大的bbx_max

        for (size_t d = 0; d < 3; ++d) 
        {
            descriptor_full.Row<float>(i)[d] = centroid[d];
            descriptor_full.Row<float>(i)[d + 3] = stdev[d];
            descriptor_full.Row<float>(i)[d + 6] = dim[d];
        }

        descriptor_full.Row<float>(i)[9] = volume;
        descriptor_full.Row<float>(i)[10] = length;
        
        i++;
    }

    std::map<int,int> sorted_nodes;
    int k = 0;
    for(auto n:nodes)
    {
        sorted_nodes.insert({n.first,k});
        if(debug_flag)
        {
            std::cout<<"sorted_nodes"<<std::endl;
            std::cout<<"sorted_nodes: "<<n.first<<" -> "<<sorted_nodes.at(n.first)<<std::endl;
        }
        k++;
    }

    // cout edge
    std::vector<std::pair<int,int>> selected_edges;
    for(const auto & pair:edges)
    {

        selected_edges.emplace_back(sorted_nodes.at(pair.first.first),sorted_nodes.at(pair.first.second));
        if(debug_flag)
        {
            std::cout<<"selected_edges: "<<pair.first.first<<" -> "<<pair.first.second<<std::endl;
            std::cout<<"sorted_edges: "<<sorted_nodes.at(pair.first.first)<<" -> "<<sorted_nodes.at(pair.first.second)<<std::endl;
        }
    }
    size_t n_edges = selected_edges.size();

    /// Compute Edge descriptor
    MemoryBlock2D edge_index(MemoryBlock2D::INT64_T, { 2, n_edges });
    for(size_t i=0;i<selected_edges.size();++i) {
        // std::cout<< "selected_edges.at(i).first: "<<selected_edges.at(i).first<<std::endl;
        // std::cout<< "selected_edges.at(i).second: "<<selected_edges.at(i).second<<std::endl;
        edge_index.at<int64_t>(0, i) = selected_edges.at(i).first;
        edge_index.at<int64_t>(1, i) = selected_edges.at(i).second;
    }

    // Edge feature
    auto edge_descriptor = DataUtil::compute_edge_descriptor(descriptor_full, edge_index, 0);
    std::vector<float *> rel_inputs = {static_cast<float *>(edge_descriptor.data())};
    std::vector<std::vector<int64_t>> rel_input_sizes = {{static_cast<long>(n_edges), static_cast<long>(d_edge), 1}};
    auto out_enc_rel = mEatGCN->Compute(EatGCN::OP_ENC_REL, rel_inputs, rel_input_sizes);
    
    //TODO: remove me
    if(debug_flag)
    {
        ONNX::PrintVector<float,size_t>("edge_descriptor",edge_descriptor.Get<float>(), {edge_descriptor.mDims.x,edge_descriptor.mDims.y});
        // for(auto n:out_enc_rel)
        // {
        //     std::cout<<*n<<std::endl;
        // }
        ONNX::PrintTensorShape("out_enc_rel",out_enc_rel[0]);
        ONNX::PrintVector<float>("out_enc_rel",out_enc_rel[0].GetTensorMutableData<float>(),
                std::vector<size_t>{static_cast<unsigned long>(out_enc_rel[0].GetTensorTypeAndShapeInfo().GetShape()[0]),
                 static_cast<unsigned long>(out_enc_rel[0].GetTensorTypeAndShapeInfo().GetShape()[1])});
    }

    int64_t size_edge_feature = out_enc_rel[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    size_t j=0;
    for(const auto & pair:edges)
    {
        auto &edge = pair.second;
        auto &edge_feature = edge->mFeatures["0"];

        // std::cout<< "old edge feature:"<<edge_feature<<std::endl;
        // get edge feature
        if (!edge_feature) edge_feature.reset(new MemoryBlock(MemoryBlock::FLOAT, size_edge_feature));
        float *feature = out_enc_rel[0].GetTensorMutableData<float>()+j*size_edge_feature;
        memcpy(edge_feature->Get<float>(),feature,sizeof(float)*size_edge_feature);

        // std::cout<< "edge feature:"<<feature<<std::endl;
        j++;
    }
        
    std::cout<< "edge feature end"<<std::endl;
}
// real_node2all_node
int Graph::UpdateNodes(const std::unique_ptr<Gui>& gui,const std::unique_ptr<InstanceFusion>& instanceFusion){
    bool reset = false;
    bool needadd = false;
    std::vector<int>delete_node;
    std::vector<int>add_node;
    std::map<int,int>buffer;

    if(gui->bbx_instanceTable_idx.size() < all_node_cout)
    {
        std::cout<<"need delete node:" << all_node_cout-1 <<std::endl;
        reset = true;
        delete_node.emplace_back(all_node2real_node.at(all_node_cout-1));
        all_node2real_node.erase(all_node_cout-1);
        --all_node_cout;
    }

    for(int i = 0; i<gui->bbx_instanceTable_idx.size();i++)
    {
        if(instanceFusion->getFeatureMap(i).size() < 512*3 && all_node2real_node.find(i) != all_node2real_node.end())
        {
            reset = true;
            delete_node.emplace_back(all_node2real_node.at(i));
            std::cout<<"need delete real node:" << all_node2real_node.at(i) <<std::endl; 
            all_node2real_node.erase(i);
        }
        if((instanceFusion->getFeatureMap(i).size() > 512*3) && all_node2no_use.find(i) != all_node2no_use.end())
        {
            needadd = true;
            add_node.emplace_back(all_node2no_use.at(i));
            upflag = false;
            std::cout<<"need delete no use node:" << all_node2no_use.at(i) <<std::endl; 
            all_node2no_use.erase(i);
        }
    }

    if((gui->bbx_instanceTable_idx.size() <= all_node2real_node.size()) || (gui->bbx_instanceTable_idx.size() <= real_node2all_node.size()))
    {
        
        for(auto n : all_node2real_node)
        {
            int k = 0;
            std::cout<<"find "<< n.second<<" in nodes"<<std::endl; 

            for(int i = 0;i<gui->bbx_instanceTable_idx.size();i++)
            {
                std::cout<<"i "<<i<<std::endl; 
                std::vector<int>colorlist = instanceFusion->getInstanceColor(i);
                if(nodes.at(n.second)->color[0] != colorlist[0]  && nodes.at(n.second)->color[1] != colorlist[1])k++;
                if(k == gui->bbx_instanceTable_idx.size())
                {
                    reset = true;
                    delete_node.emplace_back(n.second);
                    std::cout<<"node suddenly disappear, need delete real node:" << n.second << " all node : " <<n.first <<std::endl; 
                    all_node2real_node.erase(n.first);
                }
            }
        }
       
    }

    if((!reset) && (!needadd))
    {   
        std::cout<<"no need add or delete return "<<std::endl; 
        // upflag = true;
        return 0;
    }
    for(int i = 0;i < delete_node.size();i++)
    {   
        // --all_node_cout;
        int all_node_idx = real_node2all_node.at(delete_node[i]);
        real_node2all_node.erase(delete_node[i]);
        nodes.erase(delete_node[i]);
        std:cout<<"delete nodes: "<< delete_node[i]<<std::endl;
        for(auto n:edges)
        {
            if(n.first.first == delete_node[i] || n.first.second == delete_node[i])
            {
                edges.erase({n.first.first,n.first.second});
                std::cout<<"delete edges: "<< n.first.first<<" -> "<<n.first.second<<std::endl;
            }

        }
        buffer.insert({delete_node[i],all_node_idx});//no_use2all_node
        std::cout<<"wait to add no use node:"<<delete_node[i]<<", all node idx: "<<all_node_idx<<std::endl; 

    }

    for(int i = 0;i < add_node.size();i++)
    {   
        // std::cout<<"add no use node"<<std::endl;
        int all_node_idx = no_use2all_node.at(add_node[i]);
        no_use2all_node.erase(add_node[i]);
        std::cout<<"last_node_num:"<<last_node_num<<std::endl;
        std::cout<<"all_node_cout:"<<all_node_cout<<std::endl;
        std::cout<<"all_node_idx:"<<all_node_idx<<std::endl;
        node_add_to_edge.insert(last_node_num + 1);
        NodePtr node;
        nodes.insert({last_node_num + 1,std::make_shared<Node>(last_node_num + 1)});
        node = nodes.at(last_node_num + 1);
        int size = instanceFusion->getFeatureMap(all_node_idx).size();
        std::vector<int>color = instanceFusion->getInstanceColor(all_node_idx);
        if(node->Add(last_node_num + 1,gui,size,color));
        {
            int classid = gui->bbx_instanceTable_idx.at(last_node_num + 1);
            std::string class_name = (classid == -1) ? "Unused" : class_names[classid];
            std::cout<<"To Add Node index: "<<last_node_num + 1<<" class name: "<< class_name <<std::endl;
        }
        
        real_node2all_node.insert({last_node_num + 1,all_node_idx});
        all_node2real_node.insert({all_node_idx,last_node_num + 1});
        // --all_node_cout;
        ++last_node_num;
    }
    add_node.clear();
    delete_node.clear();
    for(auto n:buffer)
    {
        no_use2all_node.insert({n.first,n.second});
        all_node2no_use.insert({n.second,n.first});
        std::cout<<"add no use node:"<<n.first<<", all node idx: "<<n.second<<std::endl; 
    }
    // if(upflag)Update(gui,instanceFusion);
}

void Graph::UpdateNodeFeature(const std::unique_ptr<InstanceFusion>& instanceFusion){
    std::cout<<"node feature start"<<std::endl;
    auto n_nodes = nodes.size();   
    size_t n_pts = 512;//一个node只考虑512个点以上的
    size_t d_pts = 9;//6位长度，存pos，color
    size_t d_edge = 11;
    //TODO 一次更更新的不一定是所有的点，应该是最新获取到的点，修改n_node
    MemoryBlock3D input_nodes(MemoryBlock::DATA_TYPE::FLOAT,{n_nodes,d_pts,n_pts});

    float dist,max_dist=0;
    size_t n = 0;
    std::cout<< "for"<<std::endl;
    for(const auto &pair : nodes)
    {
        //将点的特征抽象成一个长度为10的向量，0-2是质心，3-5是位置标准差，6-8是bbx尺寸，9是bbx体积，10是bbx_max长度
        //保存在MemoryBlock2D中
        std::cout<< "node id: "<<pair.first <<std::endl;
        auto &node = pair.second;
        const auto &centroid = node->centroid;
        
        float stdev_x=0,stdev_y=0,stdev_z=0;
        float stdev_xx=0,stdev_yy=0,stdev_zz=0;

        std::cout<< "对应的点: "<<real_node2all_node.at(pair.first)<<std::endl;
        auto pointcloud = instanceFusion->getFeatureMap(real_node2all_node.at(pair.first));
        std::cout<< "pointcloud: "<<pointcloud.size()<<std::endl;
        std::cout<< "centroid: "<<centroid[0] <<" "<<centroid[1]<< " "<<centroid[2] <<std::endl;

        int step = pointcloud.size()/(512*3);
        std::cout<< "step: "<<step<<std::endl;
        for(size_t p = 0; p < n_pts;p=p+step)
        {
            
            // std::cout<< "pointcloud: "<<pointcloud[3*p] <<" "<<pointcloud[3*p+1]<< " "<<pointcloud[3*p+2] <<std::endl;
            // std::cout<< "1.f: "<<pointcloud[3*p]*1.f <<" "<<pointcloud[3*p+1]*1.f<< " "<<pointcloud[3*p+2]*1.f <<std::endl;
            const auto &color = instanceFusion->getColorMap(real_node2all_node.at(pair.first));
            dist = 0;

            input_nodes.at<float>(n,0,p) = pointcloud[3*p]*1.f - centroid[0]*1.f;
            input_nodes.at<float>(n,1,p) = pointcloud[3*p+1]*1.f - centroid[1]*1.f;
            input_nodes.at<float>(n,2,p) = pointcloud[3*p+2]*1.f - centroid[2]*1.f;

            // stdev_xx = pointcloud[3*p]*1000.f - centroid[0]*1000.f;
            // stdev_yy = pointcloud[3*p+1]*1000.f - centroid[1]*1000.f;
            // stdev_zz = pointcloud[3*p+2]*1000.f - centroid[2]*1000.f;
            stdev_x += pow(fabs(pointcloud[3*p]*1.f - centroid[0]*1.f), 2);
            stdev_y += pow(fabs(pointcloud[3*p+1]*1.f - centroid[1]*1.f), 2);
            stdev_z += pow(fabs(pointcloud[3*p+2]*1.f - centroid[2]*1.f), 2);

            // std::cout<< "pointcloud: "<<pointcloud[3*p]<<" "<<pointcloud[3*p+1]<< " "<<pointcloud[3*p+2] <<std::endl;
            // std::cout<< "fabs: "<<fabs(pointcloud[3*p] - centroid[0])<<" "<<fabs(pointcloud[3*p+1] - centroid[1])<< " "<<fabs(pointcloud[3*p+2] - centroid[2]) <<std::endl;
            // std::cout<< "stdev_: "<<stdev_x <<" "<<stdev_y<< " "<<stdev_z <<std::endl;

            input_nodes.at<float>(n,3,p) = color[3*p+2]/ 255.f * 2.f - 1.f;
            input_nodes.at<float>(n,4,p) = color[3*p+1]/ 255.f * 2.f - 1.f;
            input_nodes.at<float>(n,5,p) = color[3*p]/ 255.f * 2.f - 1.f;

            // should be normal but not finish
            input_nodes.at<float>(n,6,p) = pointcloud[3*p]*1.f - centroid[0]*1.f;
            input_nodes.at<float>(n,7,p) = pointcloud[3*p+1]*1.f - centroid[1]*1.f;
            input_nodes.at<float>(n,8,p) = pointcloud[3*p+2]*1.f - centroid[2]*1.f;
            dist += std::pow(input_nodes.at<float>(n,0,p),2) + std::pow(input_nodes.at<float>(n,1,p),2) + std::pow(input_nodes.at<float>(n,2,p),2);
            if(dist > max_dist)
            {
                max_dist = dist;
            }

            // std::cout<<std::endl;
        }
        stdev_x = std::sqrt(fabs(stdev_x / (n_pts-1)));
        stdev_y = std::sqrt(fabs(stdev_y / (n_pts-1)));
        stdev_z = std::sqrt(fabs(stdev_z / (n_pts-1)));
        
        Eigen::Vector3f biaozhuncha;

        biaozhuncha(0) = stdev_x;
        biaozhuncha(1) = stdev_y;
        biaozhuncha(2) = stdev_z;
        std::cout<< "biaozhuncha: "<<biaozhuncha(0) <<" "<<biaozhuncha(1)<< " "<<biaozhuncha(2) <<std::endl;
        node->stdev = biaozhuncha;


        max_dist = std::sqrt(max_dist);
        for(size_t p = 0;p<n_pts;p++)
        {
            input_nodes.at<float>(n, 0, p) /= max_dist;
            input_nodes.at<float>(n, 1, p) /= max_dist;
            input_nodes.at<float>(n, 2, p) /= max_dist;
        }
        n++;
    }

    std::cout<< "input_nodes.size: "<<input_nodes.size() <<std::endl;
    //TODO descriptor full
    MemoryBlock2D descriptor_full(MemoryBlock::DATA_TYPE::FLOAT, {n_nodes, d_edge});
    
    size_t i=0;
    for(const auto &pair:nodes)
    {
        const auto &node = pair.second;
        const auto centroid = node->centroid;
        const auto bbox_max = node->bbox_max;
        const auto bbox_min = node->bbox_min;
        const auto stdev    = node->stdev;//点位置的标准差

        for(size_t d = 0; d < 3; ++d)
        {
            std::cout<<"bbox_max:"<<bbox_max[d]<<std::endl;
            std::cout<<"bbox_min:"<<bbox_min[d]<<std::endl;
        }
        Eigen::Vector3f dim = (bbox_max-bbox_min);//bbx三维数据
        float volume = 1;
        for(size_t d = 0; d < 3; ++d) volume *= dim[d];//bbx的体积
        float length = *std::max_element(dim.begin(), dim.end());//找到bbx中最大的bbx_max

        for (size_t d = 0; d < 3; ++d) 
        {
            descriptor_full.Row<float>(i)[d] = centroid[d];
            descriptor_full.Row<float>(i)[d + 3] = stdev[d];
            descriptor_full.Row<float>(i)[d + 6] = dim[d];
        }

        descriptor_full.Row<float>(i)[9] = volume;
        descriptor_full.Row<float>(i)[10] = length;
        i++;
    }

    if(debug_flag_node)
        ONNX::PrintVector<float,size_t>("descriptor_full", descriptor_full.Get<float>(),
                                         {descriptor_full.mDims.x,descriptor_full.mDims.y});

    //compute
    std::vector<float *> obj_inputs = {reinterpret_cast<float *>(input_nodes.data())};//返回一个指向input node的直接指针
    std::vector<std::vector<int64_t>> obj_input_sizes = {{static_cast<long>(n_nodes), static_cast<long>(d_pts), static_cast<long>(n_pts)}};
    auto out_enc_obj = mEatGCN->Compute(EatGCN::OP_ENC_OBJ, obj_inputs, obj_input_sizes);
    size_t dim_obj_feature = out_enc_obj[0].GetTensorTypeAndShapeInfo().GetShape()[1];//获取维度
    auto out = ConcatObjFeature(n_nodes,dim_obj_feature, out_enc_obj[0], descriptor_full);

    if(debug_flag_node)
        ONNX::PrintVector<float,int64_t>("nodeFeature", out_enc_obj[0].GetTensorMutableData<float>(),
                {out_enc_obj[0].GetTensorTypeAndShapeInfo().GetShape()[0], out_enc_obj[0].GetTensorTypeAndShapeInfo().GetShape()[1]});


    // Save Obj and Rel feature to nodes
    size_t size_node_feature = out.mDims.y;
    size_t j = 0;
    for (const auto &pair : nodes) {
        auto &node = pair.second;
        auto &node_feature = node->mFeatures["0"];
        if (!node_feature) node_feature.reset(new MemoryBlock(MemoryBlock::FLOAT, size_node_feature));
        auto *feature = out.Row<float>(j);
        memcpy(node_feature->Get<float>(), feature, sizeof(float) * size_node_feature);
        j++;
    }
    std::cout<< "node feature end"<<std::endl;
}

void Graph::UpdatePrediction(){
    std::vector<NodePtr> selected_nodes;
    std::vector<std::pair<size_t,size_t>> filtered_SizeAndEdge;
    std::map<int,std::pair<size_t,size_t>> sizeAndEdge;

    for(const auto &pair : nodes) 
    {
        auto i = pair.first;
        const auto &node = pair.second;
        selected_nodes.push_back(node);
        sizeAndEdge[i].first = node->size;
        sizeAndEdge[i].second = nodes.size();
        filtered_SizeAndEdge.push_back(sizeAndEdge.at(i));
    }
    size_t n_nodes = selected_nodes.size();

    /// Node 
    std::cout << "=============== node ===============" << std::endl;
    int64_t dim_obj_f = 512;
    if (!selected_nodes.empty()) {
        MemoryBlock2D nodeFeatures(MemoryBlock::FLOAT,
                                       {selected_nodes.size(), static_cast<unsigned long>(dim_obj_f)});
        for (size_t i = 0; i < n_nodes; ++i) {
            // std::cout << "selected_nodes: "<< selected_nodes.at(i) <<std::endl;
            const auto &last_node_feature = selected_nodes.at(i)->mFeatures.at("0")->Get<float>();
            memcpy(nodeFeatures.Row<float>(i), last_node_feature, sizeof(float) * dim_obj_f);
        }
        
        if (debug_flag) ONNX::PrintVector("nodeFeatures", nodeFeatures.Get<float>(), std::vector<size_t>{nodeFeatures.mDims.x, nodeFeatures.mDims.y});
        std::vector<Ort::Value> node_features;
        node_features.push_back(ONNX::CreateTensor(mMemoryInfo.get(), nodeFeatures.Get<float>(),
                                                       {static_cast<long>(n_nodes),
                                                        static_cast<long>(dim_obj_f)}));
        auto objcls_prob = mEatGCN->Compute(EatGCN::OP_CLS_OBJ, node_features);
        size_t num_cls = objcls_prob[0].GetTensorTypeAndShapeInfo().GetShape()[1];
        assert(mLabels.size() == num_cls);
        for (size_t i = 0; i < n_nodes; ++i) 
        {
            const auto &node = selected_nodes.at(i);
            std::map<std::string, float> m;
            std::map<std::string, std::pair<size_t, size_t>> mm;
            float *data = objcls_prob[0].GetTensorMutableData<float>() + i * num_cls;
            for (size_t j = 0; j < num_cls; ++j) {
                m[mLabels.at(j)] = expf(data[j]); // logsoftmax -> softmax
                mm[mLabels.at(j)] = filtered_SizeAndEdge.at(i);
                //m存储的是<类别，概率>
                //mm存储的是<类别，<点，点>(所有的边)>,
                // std::cout << "mLabels.at(j) "<< mLabels.at(j) <<std::endl;
                // std::cout << "m[mLabels.at(j)]: "<< m[mLabels.at(j)] <<std::endl;
                // std::cout << "mm[mLabels.at(j)].first : "<< mm[mLabels.at(j)].first <<std::endl;//node的点云size
                // std::cout << "mm[mLabels.at(j)].second : "<< mm[mLabels.at(j)].second <<std::endl;//边
            }
            node->UpdatePrediction(m, mm);
        }
    }

    /// Edge
    std::cout << "=============== edge ===============" << std::endl;
    std::vector<EdgePtr> selected_edges;
    int64_t dim_rel_f = 256;
    for (auto &n : edges) {
        selected_edges.emplace_back(n.second);
    }
    MemoryBlock2D edgeFeatures(MemoryBlock::FLOAT,
                                   {selected_edges.size(), static_cast<unsigned long>(dim_rel_f)});
    for (size_t i = 0; i < selected_edges.size(); ++i) {
        const auto &last_node_feature = selected_edges.at(i)->mFeatures.at("0")->Get<float>();
        memcpy(edgeFeatures.Row<float>(i), last_node_feature, sizeof(float) * dim_rel_f);
    }

    if (debug_flag) ONNX::PrintVector("edgeFeatures", edgeFeatures.Get<float>(), std::vector<size_t>{edgeFeatures.mDims.x, edgeFeatures.mDims.y});

    std::vector<Ort::Value> edge_features;
    edge_features.push_back(ONNX::CreateTensor(mMemoryInfo.get(), edgeFeatures.Get<float>(),
                                                   {static_cast<long>(selected_edges.size()),
                                                    static_cast<long>(dim_rel_f)}));
    auto relcls_prob = mEatGCN->Compute(EatGCN::OP_CLS_REL, edge_features);
    size_t num_rel = relcls_prob[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    assert(num_rel == mRelationships.size());

    for(size_t i=0;i<selected_edges.size();++i)
    {
        auto &edge = selected_edges.at(i);
        float *data = relcls_prob[0].GetTensorMutableData<float>() + i * num_rel;
        std::map<std::string, float> prop;

        for(size_t j=0;j<num_rel;++j) 
        {
            prop[mRelationships.at(j)] = expf(data[j]); // logsoftmax->softmax  FAT
            //预测每一种关系的概率
        }

        edge->UpdatePrediction(prop);
    }

    /// Update Instances
}

void Graph::Debug(){
    std::cout<<"=====================nodes======================="<<std::endl;
    for(auto n:nodes)
    {
        std::cout<<"real_idx: "<<n.first <<" bbx_volume: "<<n.second->bbx_volume << " color: "<<n.second->color[0]<<","<<n.second->color[1]<<","<<n.second->color[2];
        std::cout<<std::endl;
    }
    std::cout<<"==================edges:from->to=================="<<std::endl;
    for(auto n:edges)
    {
        std::cout<<n.first.first <<" -> "<<n.first.second;
        std::cout<<std::endl;
    }
    std::cout<<"===============real_node2all_node================"<<std::endl;
    for(auto n:real_node2all_node)
    {
        std::cout<<"real:"<<n.first<<" -> all_node:"<<n.second;
        std::cout<<std::endl;
    }
    std::cout<<"===============all_node2real_node================"<<std::endl;
    for(auto n:all_node2real_node)
    {
        std::cout<<"all_node:"<<n.first<<" -> real:"<<n.second;
        std::cout<<std::endl;
    }
    std::cout<<"================no_use2all_node=================="<<std::endl;
    for(auto n:no_use2all_node)
    {
        std::cout<<"no use:"<<n.first<<" -> all_node:"<<n.second;
        std::cout<<std::endl;
    }
    std::cout<<"================all_node2no_use=================="<<std::endl;
    for(auto n:all_node2no_use)
    {
        std::cout<<"all_node:"<<n.first<<" -> no use:"<<n.second;
        std::cout<<std::endl;
    }
    std::cout<<"================================================="<<std::endl;
}

void Graph::AddEdge(const int node_from_idx,const int node_to_idx){
    NodePtr node_from,node_to;
    node_to = nodes.at(node_to_idx);
    node_from = nodes.at(node_from_idx);
    node_from->idx = node_from_idx;
    node_to->idx = node_to_idx;
    std::cout<<"add edge: "<< node_from_idx<<" -> "<< node_to_idx<<std::endl;

    EdgePtr edge;
    edges.insert({{node_from_idx,node_to_idx},std::make_shared<Edge>()});
    auto pair = std::make_pair(node_from_idx, node_to_idx);
    edge = edges.at(pair);
    edge->nodefrom = node_from_idx;
    edge->nodeto = node_to_idx;
    nodes.at(edge->nodefrom)->edges.insert(edge);
    nodes.at(edge->nodeto)->edges.insert(edge);
}


