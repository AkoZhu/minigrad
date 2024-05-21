#ifndef NN_HPP
#define NN_HPP

#include "engine.hpp"
#include <random>
#include <iostream>

using std::ostream;

// Module is the base class for all neural network modules
class Module {
    public:
    ~Module() {
        for (auto& param: this->parameters()) {
            delete &param;
        }
    }

    void zero_grad() {
        for (auto& param: this->parameters()) {
            param->grad = 0;
        }
    }

    virtual vector<Value*> parameters() {
        return {};
    }
};

// Neuron is the base class for all neural network neurons
class Neuron : public Module {
    public:
    vector<Value> w;
    Value b;
    bool nonlin;

    Neuron() : w({}), b(0), nonlin(false) {};

    Neuron(int nin, bool nonlin = true) {
        w = {};
        std::random_device rd;  
        std::mt19937 gen(rd()); 
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for (int i = 0; i < nin; i++) {
            w.push_back(Value(dis(gen)));
        }
        b = Value(0);
        this->nonlin = nonlin;
    }

    vector<Value*> parameters() override {
        vector<Value*> params;
        params.insert(params.end(), this->w.begin(), this->w.end());
        params.push_back(&(this->b));
        return params;
    }


    // call the forward function
    Value forward(vector<Value> x) {
        if (x.size() != this->w.size()) {
            throw std::invalid_argument("Input size mismatch");
        }

        double sum = 0;
        for (int i = 0; i < x.size(); i++) {
            sum += x[i].data * this->w[i].data;
        }

        Value act = sum + this->b.data;
        return this->nonlin ? act.relu() : act;
    };

    friend ostream& operator<<(ostream& os, const Neuron& n) {
        os << "Neuron(";
        for (const Value& w: n.w) {
            os << w.data << ", ";
        }
        os << n.b.data << ")";
        return os;
    };
};

// Layer
class Layer : public Module {
    public:
    vector<Neuron> neurons;

    Layer(): neurons({}) {};

    Layer(int nin, int nout, bool nonlin = true) {
        neurons = {};
        for (int i = 0; i < nout; i++) {
            neurons.push_back(Neuron(nin, nonlin));
        }
    }

    vector<Value> forward(vector<Value> x) {
        vector<Value> out;
        for (Neuron& neuron: this->neurons) {
            out.push_back(neuron.forward(x));
        }
        return out;
    }

    vector<Value*> parameters() override {
        vector<Value*> params;
        for (Neuron& neuron: this->neurons) {
            vector<Value*> neuron_params = neuron.parameters();
            params.insert(params.end(), neuron_params.begin(), neuron_params.end());
        }
        return params;
    }

    friend ostream& operator<<(ostream& os, const Layer& l) {
        os << "Layer(";
        for (const Neuron& n: l.neurons) {
            os << n << ", ";
        }
        os << ")";
        return os;
    };
};

// MLP
class MLP : Module {
    public: 
    vector<Layer> layers;

    MLP(): layers({}) {};

    MLP(int nin, vector<int>& nouts) {
        vector<int> sz;
        sz.push_back(nin);
        sz.insert(sz.end(), nouts.begin(), nouts.end());

        for (size_t i = 0; i < nouts.size(); i++) {
            bool nonlin = i != nouts.size() - 1;
            layers.push_back(Layer(sz[i], sz[i+1], nonlin));
        }
    }

    vector<Value> forward(vector<Value> x) {
        vector<Value> out = x;
        for (Layer& layer: this->layers) {
            out = layer.forward(out);
        }
        return out;
    }

    vector<Value*> parameters() override {
        vector<Value*> params;
        for (Layer& layer: this->layers) {
            vector<Value*> layer_params = layer.parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

    friend ostream& operator<<(ostream& os, const MLP& m) {
        os << "MLP(";
        for (const Layer& l: m.layers) {
            os << l << ", ";
        }
        os << ")";
        return os;
    };
};

#endif