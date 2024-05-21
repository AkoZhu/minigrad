#ifndef ENGINE_HPP
#define ENGINE_HPP


#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <set>

using std::set;
using std::string;
using std::function;
using std::vector;

using std::stringstream;

/*
    Store a single scalar value and its gradient
*/
class Value {
    private:
    set<Value> _prev;
    string _op;
    function<void()> _backward;

    public:
    double data;
    double grad;

    Value(double data, set<Value> _children={}, string _op="") 
        : _op(_op), _backward([]() {return 0.0;}), data(data), grad(0)
    {
        this->_prev = {};
        this->_prev.insert(_children.begin(), _children.end());
    };

    Value() : Value(0) {} ;

    // to_string
    friend std::ostream& operator<<(std::ostream& os, const Value& value) {
        os << "Value(data=" << value.data << ", grad=" << value.grad << ")";
        return os;
    }

    // redefine the + operator
    template<typename T>
    Value operator+(const T& other) {
        Value other_value;
        if constexpr(std::is_same<T, Value>::value) {
            other_value = other;
        } else {
            other_value = Value(other);
        }

        Value out = Value(this->data + other_value.data, { *this, other_value }, "+");
        
        function<void()> backward = [&]() {
            this->grad += out.grad;
            other_value.grad += out.grad;
        };

        out._backward = backward;
        return out;
    }

    // redefine the mul
    template<typename T>
    Value operator*(const T& other) {
        Value other_value;
        if constexpr(std::is_same<T, Value>::value) {
            other_value = other;
        } else {
            other_value = Value(other);
        }

        Value out = Value(this->data * other_value.data, { *this, other_value }, "*");
        
        function<void()> backward = [&]() {
            this->grad += other_value.data * out.grad;
            other_value.grad += this->data * out.grad;
        };

        out._backward = backward;
        return out;
    }

    // redefine the pow
    template<typename T>
    Value pow(const T& other) {
        if constexpr(!(std::is_same<T, double>::value || std::is_same<T, int>::value)) {
            throw std::invalid_argument("The power should be a scalar value");
        }

        Value out = Value(std::pow(this->data, other), { *this }, "**{" + std::to_string(other) + "}");
        
        function<void()> backward = [&]() {
            this->grad += other * std::pow(this->data, other - 1) * out.grad;
        };

        out._backward = backward;
        return out;
    }

    // redefine unary - 
    Value operator-() {
        return (*this) * -1;
    };

    // redefine the - operator
    template<typename T>
    Value operator-(const T& other) {
        return (*this) + (-other);
    }

    // redefine the / operator
    template<typename T>
    Value operator/(const T& other) {
        Value other_value;
        if constexpr(std::is_same<T, Value>::value) {
            other_value = other;
        } else {
            other_value = Value(other);
        }

        return (*this) * other_value.pow(-1);
    }

    // redefine exp
    Value exp() {
        double x = this->data;
        double t = std::exp(x);
        Value out = Value(t, { *this }, "exp");

        function<void()> backward = [&]() {
            this->grad += out.data * out.grad;
        };

        out._backward = backward;
        return out;
    }

    // Define the comparison operator
    bool operator<(const Value& other) const {
        return this->data < other.data;
    }


    // ================ Here is the activation function =================
    /**
     * @brief ReLU activation function
    */
    Value relu() {
        Value out = Value(std::max(0.0, this->data), { *this }, "reLU");
        
        function<void()> backward = [&]() {
            this->grad += (out.data > 0) ? out.grad : 0;
        };
        out._backward = backward;
        return out;
    }

    /*
        @brief Sigmoid activation function
    */
    Value tanh() {
        double x = this->data;
        double t = std::tanh(x);
        Value out = Value(t, { *this }, "tanh");

        function<void()> backward = [&]() {
            this->grad += (1 - out.data * out.data) * out.grad;
        };

        out._backward = backward;
        return out;
    }


    // ================= Here is the backward function =================
    void backward() {
        vector<Value> topo = {};
        set<Value> visited = {};

        // topological sort
        function<void(Value&)> topo_sort = [&](Value& node) {
            if (visited.find(node) != visited.end()) {
                return;
            }
            visited.insert(node);
            for (Value child : node._prev) {
                topo_sort(child);
            }
            topo.push_back(node);
        };
        topo_sort(*this);
        this->grad = 1;
        auto iter = topo.rbegin();
        while (iter != topo.rend()) {
            Value node = *iter;
            node._backward();
            iter++;
        }
    }
};


#endif // ENGINE_HPP