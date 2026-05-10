//
// Created by redamancyguy on 23-8-12.
// Simplified: removed baseline comparisons (b+tree, alex, pgm, lipp)
//

#include <iostream>
#include <random>
#include<functional>
#include <iomanip>
#include <torch/torch.h>


//#define CB
#define using_small_network

#include "../include/DEFINE.h"
#include "../index/include/Parameter.h"

#include "../index/include/Index.hpp"
#include "../include/DataSet.hpp"
#include "../index/include/experience.hpp"
#include "../index/include/Controller.hpp"

double train_proportion = 0.1;
auto train_size = 0;


GlobalController controller;

class EvaluationTask {
public:
    std::string dataset_name;
    int start = 0;
    int length = 0;
};

template<typename T>
std::pair<std::vector<T>, std::vector<T>> split_dataset(std::vector<T> dataset, std::pair<double, double> proportion) {
    auto train_size = std::size_t(double(dataset.size()) * proportion.first / (proportion.first + proportion.second));
    std::vector<std::size_t> indices(dataset.size());
    for (std::size_t i = 0; i < dataset.size(); i++) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), e);
    std::vector<T> train_dataset;
    train_dataset.reserve(train_size);
    std::vector<T> test_dataset;
    test_dataset.reserve(dataset.size() - train_size);
    for (std::size_t i = 0; i < train_size; i++) {
        train_dataset.push_back(dataset[indices[i]]);
    }
    for (std::size_t i = train_size; i < dataset.size(); i++) {
        test_dataset.push_back(dataset[indices[i]]);
    }
    return {train_dataset, test_dataset};
}

int steps = 4;

auto evaluate_cha(
        std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> bulkload_dataset,
        std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    std::vector<float> get_result;
    std::vector<float> add_result;
    std::vector<float> memory_result;
    std::vector<float> erase_result;
    std::vector<float> memory_result2;
    std::vector<float> get_result2;
    auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());
    auto min_max2 = get_min_max<KEY_TYPE, VALUE_TYPE>(bulkload_dataset.begin(), bulkload_dataset.end());
    min_max.first = std::min(min_max.first,min_max2.first);
    min_max.second = std::min(min_max.second,min_max2.second);
    auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(bulkload_dataset.begin(), bulkload_dataset.end(), min_max.first, min_max.second,
                                             BUCKET_SIZE);
    experience_t exp_chosen;
    std::copy(pdf.begin(), pdf.end(), exp_chosen.distribution);
    exp_chosen.data_size = float(bulkload_dataset.size() + dataset.size());
    auto best_gen = controller.get_best_action_GA(exp_chosen);
    auto index = new Hits::Index<KEY_TYPE, VALUE_TYPE>(best_gen.conf, min_max.first, min_max.second);
    index->bulk_load(bulkload_dataset.begin(), bulkload_dataset.end());
    int step_size = dataset.size() / steps;
    for (int i = 0; i < steps; i++) {
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = i * step_size + j;
            if (!index->add(dataset[id].first, dataset[id].second)) {
                puts("hits add error");
            }
        }
        add_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        memory_result.push_back(((Hits::Index<KEY_TYPE, VALUE_TYPE> *) index)->memory_occupied() / (1024.0 * 1024.0));
        VALUE_TYPE v;
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = e() % ((i + 1) * step_size);
            if (!index->get(dataset[id].first, v) ||
                v != dataset[id].second) {
                puts("hits get error 1");
            }
        }
        get_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
    }
/////////////////
    for (int i = 0; i < steps; i++) {
        VALUE_TYPE v;
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto max = steps * step_size;
            auto min = (i * step_size);
            auto id = (e() % (max - min)) + min;
            if (!index->get(dataset[id].first, v) ||
                v != dataset[id].second) {
                puts("hits get error");
            }
        }
        get_result2.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        tc.synchronization();
        for (int j = 0; j < step_size; j++) {
            auto id = i * step_size + j;
            if (!index->erase(dataset[id].first)) {
                puts("hits erase error");
            }
        }
        erase_result.push_back(double(tc.get_timer_nanoSec()) / double(step_size));
        memory_result2.push_back(((Hits::Index<KEY_TYPE, VALUE_TYPE> *) index)->memory_occupied() / (1024.0 * 1024.0));
    }
    delete index;
    return std::vector<std::vector<float>>(
            {add_result, get_result, memory_result, erase_result, get_result2, memory_result2});
}

//int all_size = 40000000;
int all_size = 200000000;
int bulkload_size = 4000000;


auto evaluate_cha_none_exist_key(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> bulkload_dataset,std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset) {
    Cost cost;
    auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());
    auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end(), min_max.first,
                                             min_max.second,BUCKET_SIZE);
    experience_t exp_chosen;
    std::copy(pdf.begin(), pdf.end(), exp_chosen.distribution);
    exp_chosen.data_size = float(dataset.size() + bulkload_dataset.size());
    auto best_gen = controller.get_best_action_GA(exp_chosen);
    auto index = new Hits::Index<KEY_TYPE, VALUE_TYPE>(best_gen.conf, min_max.first, min_max.second);
    long long opt_count = 0;
    VALUE_TYPE v;
    tc.synchronization();
    for(auto i:dataset){
        if(!index->get(i.first,v)){
            index->add(i.first,i.second);
            opt_count++;
        }
        opt_count++;
    }
    delete index;
    cost.add = opt_count / tc.get_timer_second();
    return cost;
}


void exp2(){
    controller.load_in();

    for (const auto &dataset_name: std::vector<std::string>(
            {"logn.data", })) {
        std::cout << " dataset_name:" << dataset_name << std::endl;
        auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
        if(dataset.size() > all_size){
            dataset.resize(all_size);
        }
        bulkload_size = dataset.size() * 0.25;
        std::cout <<"bulkload_size:"<<bulkload_size<< std::endl;
        auto bulkload_dataset = std::vector(dataset.begin(), dataset.begin() + bulkload_size);
        std::sort(bulkload_dataset.begin(), bulkload_dataset.end(),
                  [&](std::pair<KEY_TYPE, VALUE_TYPE> &a, std::pair<KEY_TYPE, VALUE_TYPE> &b) {
                      return a.first < b.first; });
        dataset.erase(dataset.begin(), dataset.begin() + bulkload_size);

        auto result = evaluate_cha(bulkload_dataset, dataset);
        puts("cha");
        for (int i = 0; i < steps; ++i) {
            std::cout << "step:" << i << " add:" << result[0][i] << " get:" << result[1][i]
                      << " erase:" << result[3][i] << " get:" << result[4][i] << std::endl;
        }
    }
}


int main() {
    for (const auto &dataset_name: std::vector<std::string>(
            {"uden.data","logn.data","osmc.data", "face.data"})) {
        std::cout << " dataset_name:" << dataset_name << std::endl;
        auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
        std::sort(dataset.begin(), dataset.end(),
                  [&](std::pair<KEY_TYPE, VALUE_TYPE> &a, std::pair<KEY_TYPE, VALUE_TYPE> &b) {
                      return a.first < b.first; });
        double result = 0;
        int batch_size = 1000;
        for(int b = 1,end = dataset.size() / batch_size;b < end;++b){
            auto sub_dataset = std::vector(dataset.begin() + (b-1) * batch_size,dataset.begin() + b * batch_size);
            auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(sub_dataset.begin(),sub_dataset.end());
            result += local_skew<KEY_TYPE,VALUE_TYPE>(sub_dataset.begin() ,sub_dataset.end(),min_max.first,min_max.second);
        }
        std::cout<<"skew batch:"<<result/(dataset.size()/batch_size)<<std::endl;
        result = 0;
        auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end());
            result += local_skew<KEY_TYPE,VALUE_TYPE>(dataset.begin() ,dataset.end(),min_max.first,min_max.second);
        std::cout<<"skew all:"<<result<<std::endl;
    }
    return 0;
    controller.load_in();
    e.seed(1000);
    exp2();
    return 0;

    for (const auto &dataset_name: std::vector<std::string>(
            {"wiki.data","logn.data","logn.data","osmc.data", "face.data"})) {
        std::cout << " dataset_name:" << dataset_name << std::endl;
        auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
        std::shuffle(dataset.begin(), dataset.end(), e);
        auto bulkload_dataset = std::vector(dataset.begin() + all_size,dataset.begin()+all_size + 4000000);
        dataset.resize(all_size);
        dataset.insert(dataset.end(),dataset.begin(),dataset.end());
        controller.query_weight = 52428.8;
        for(int i = 0;i<100;i++){
            auto cost = evaluate_cha_none_exist_key(bulkload_dataset,dataset);
            std::cout <<"cha"<<cost<<"  controller.query_weight:"<<controller.query_weight<< std::endl;
            controller.query_weight *= 2;
        }
    }
    return 0;
}
